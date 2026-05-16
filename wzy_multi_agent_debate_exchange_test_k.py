"""
聚类标签修正后的 Exchange 流程（工具模块，由 wzy_multi_agent_debate_math.py 编排调用）

提供的核心能力：
1. PCA 降维 + KMeans/DBSCAN 聚类，用聚类标签修正 step 的 is_correct 标签
2. 以 agent 为单位排序（全错 agent → 混合 agent → 全对 agent）
3. 为每个 agent 构建 exchange prompt（排除自身 step，避免自我强化）
4. 并发调用 API，让每个 agent 参考他人推理步骤更新答案

对外接口：
- run_exchange1_from_expand_outputs: 在 expand 向量数据基础上执行单轮 exchange1
- run_exchange2_from_exchange1_outputs: 在 exchange1 完成后，基于最新回复重建 step 向量并再跑一轮聚类 exchange
- run_exchange_bidirectional_1_from_expand_outputs / run_exchange_bidirectional_2_from_bidirectional_1_outputs:
  流程同 exchange1/2，但 Step 4 为双向修正：correct 类簇全置 True，wrong 类簇全置 False，
  no_majority/noise 不修改。
"""

import sys
import asyncio
import copy

# Windows 控制台默认 GBK 编码，无法输出 emoji/特殊 Unicode 字符。
# 强制使用 UTF-8 编码，并将仍无法编码的字符以 ? 代替，避免 UnicodeEncodeError。
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

# 从 expand 模块获取数据及工具函数
from wzy_multi_agent_debate_expand import (
    ExpandConfig,
    agenerate_answer,
    construct_assistant_message,
    extract_steps_from_response,
    _safe_str,
    MODEL_TAG,
    expand_compute_majority_and_agent_results_from_latest,
    expand_flatten_all_steps,
    expand_print_step4_agent_vs_majority,
    expand_print_step5_steps,
    expand_print_step6_all_steps,
    expand_run_embedding,
)

# 从聚类模块导入核心函数、配置常量及答案处理工具函数
from wzy_multi_agent_debate_clustering import (
    _reduce_dimensions_pca,
    _cluster_kmeans,
    _cluster_kmeans_fixed_k,
    _cluster_dbscan,
    _label_clusters_by_majority,
    _apply_cluster_labels_to_steps,
    MAJORITY_THRESHOLD,
    TARGET_DIM,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    KMEANS_K_MIN,
    KMEANS_K_MAX,
    extract_answer_from_text,
    is_correct_answer,
    get_majority_answer_from_latest,
    IS_MATH,
)

# 并发配置
EXCHANGE_CONCURRENT_LIMIT = 5  # exchange 阶段每批并发的 agent 数量上限


# ============================================================
# 双向聚类标签修正（correct→全 True，wrong→全 False，其余不动）
# ============================================================


def _apply_cluster_labels_to_steps_bidirectional(
    cluster_labels: np.ndarray,
    cluster_label_results: dict,
    step_indices: list,
    all_steps: list,
) -> tuple:
    """
    根据聚类标签更新 step 的 is_correct（双向）。

    规则：
    - cluster_tag == "correct"：该类簇内所有 step → is_correct True
    - cluster_tag == "wrong"：该类簇内所有 step → is_correct False
    - no_majority / noise：保持原标签

    Returns:
        (steps_copy, modified_count, change_records) 与 clustering._apply_cluster_labels_to_steps 相同结构。
    """
    steps_copy = copy.deepcopy(all_steps)
    modified_count = 0
    change_records: list = []

    for vec_i, cid in enumerate(cluster_labels):
        cluster_tag = cluster_label_results.get(cid, "no_majority")
        step_idx = step_indices[vec_i]
        step = steps_copy[step_idx]
        original_label = step.get("is_correct")

        if cluster_tag == "correct":
            new_label = True
            changed = original_label is not True
            if changed:
                step["is_correct"] = True
                modified_count += 1
        elif cluster_tag == "wrong":
            new_label = False
            changed = original_label is not False
            if changed:
                step["is_correct"] = False
                modified_count += 1
        else:
            new_label = original_label
            changed = False

        change_records.append({
            "cluster_id": int(cid),
            "cluster_tag": cluster_tag,
            "agent_id": step.get("agent_id"),
            "step_number": step.get("step_number"),
            "original_label": original_label,
            "new_label": new_label,
            "changed": changed,
        })

    return steps_copy, modified_count, change_records


# ============================================================
# Step 3: 排序 —— 按聚类结果排序策略
# ============================================================

def _sort_steps_by_cluster(
    steps_modified: list,
    cluster_labels_raw: np.ndarray,
    step_indices: list,
    cluster_tag_results: dict,
) -> list:
    """
    按聚类标签修正后的 is_correct，以 agent 为单位对 step 排序。

    排序规则：
    A档（全错 agent）：该 agent 所有 step 的 is_correct 均为 False → 排最前面
        agent 之间按 agent_id 升序，agent 内按 step_number 升序。
    B档（混合 agent）：该 agent 的 step 同时存在 True 和 False → 排中间，分三段：
        1) 错误 step：所有 B 档 agent 的 is_correct=False step 打散，按 (agent_id, step_number) 升序
        2) 正确 step（末步错的 agent）：agent 的最后一步 is_correct=False → 该 agent 的正确 step 作为一组排前
           agent 之间按 agent_id 升序，agent 内按 step_number 升序
        3) 正确 step（末步对的 agent）：agent 的最后一步 is_correct=True → 该 agent 的正确 step 作为一组排后
           agent 之间按 agent_id 升序，agent 内按 step_number 升序
    C档（全对 agent）：该 agent 所有 step 的 is_correct 均为 True → 排最后面
        agent 之间按 agent_id 升序，agent 内按 step_number 升序。

    Args:
        steps_modified: 聚类标签修正后的 step 列表（is_correct 已更新）
        cluster_labels_raw: 存储每个向量所属的聚类编号（cluster id）
        step_indices: 向量下标 -> all_steps 下标 的映射列表
        cluster_tag_results: {cluster_id: "correct" | "wrong" | "no_majority" | "noise"}

    Returns:
        排序后的新 step 列表（深拷贝，不修改原始数据）
    """
    steps = copy.deepcopy(steps_modified)

    for vec_i, cid in enumerate(cluster_labels_raw):
        step_idx = step_indices[vec_i]
        steps[step_idx]["cluster_id"] = int(cid)
        steps[step_idx]["cluster_tag"] = cluster_tag_results.get(int(cid), "no_majority")

    for step in steps:
        if "cluster_id" not in step:
            step["cluster_id"] = -1
            step["cluster_tag"] = "noise"

    # --- 按 agent_id 分组 ---
    agent_groups: dict[int, list] = {}
    for step in steps:
        aid = step.get("agent_id")
        if aid not in agent_groups:
            agent_groups[aid] = []
        agent_groups[aid].append(step)

    # --- 找出每个 agent 的最大 step_number（即「最后一步」） ---
    agent_max_step: dict[int, int] = {}
    for aid, agent_steps in agent_groups.items():
        agent_max_step[aid] = max(s.get("step_number", 0) for s in agent_steps)

    # --- 按 agent 分档 ---
    tier_a: list[int] = []  # 全错
    tier_b: list[int] = []  # 混合
    tier_c: list[int] = []  # 全对

    for aid, agent_steps in agent_groups.items():
        has_true = any(s.get("is_correct") is True for s in agent_steps)
        has_false = any(s.get("is_correct") is False for s in agent_steps)
        if has_false and not has_true:
            tier_a.append(aid)
        elif has_true and not has_false:
            tier_c.append(aid)
        else:
            tier_b.append(aid)

    tier_a.sort()
    tier_b.sort()
    tier_c.sort()

    sorted_steps: list = []

    # A档：全错 agent，按 agent_id 升序，内部按 step_number 升序
    for aid in tier_a:
        sorted_steps.extend(
            sorted(agent_groups[aid], key=lambda s: s.get("step_number", 0))
        )

    # B档：分三段
    # 段1：所有 B 档 agent 的错误 step（打散）
    b_wrong: list = []
    for aid in tier_b:
        for s in agent_groups[aid]:
            if s.get("is_correct") is False:
                b_wrong.append(s)
    b_wrong.sort(key=lambda s: (s.get("agent_id", 0), s.get("step_number", 0)))
    sorted_steps.extend(b_wrong)

    # 段2+段3：正确 step 以 agent 为单位连续输出
    #   末步错的 agent → 段2（前）；末步对的 agent → 段3（后）
    tier_b_last_wrong: list[int] = []
    tier_b_last_correct: list[int] = []
    for aid in tier_b:
        last_step = next(
            (s for s in agent_groups[aid] if s.get("step_number", 0) == agent_max_step[aid]),
            None,
        )
        if last_step is not None and last_step.get("is_correct") is True:
            tier_b_last_correct.append(aid)
        else:
            tier_b_last_wrong.append(aid)
    tier_b_last_wrong.sort()
    tier_b_last_correct.sort()

    for aid in tier_b_last_wrong + tier_b_last_correct:
        correct_steps = [s for s in agent_groups[aid] if s.get("is_correct") is True]
        correct_steps.sort(key=lambda s: s.get("step_number", 0))
        sorted_steps.extend(correct_steps)

    # C档：全对 agent，按 agent_id 升序，内部按 step_number 升序
    for aid in tier_c:
        sorted_steps.extend(
            sorted(agent_groups[aid], key=lambda s: s.get("step_number", 0))
        )

    return sorted_steps


def _print_sorted_steps_summary(sorted_steps: list):
    """打印排序后 step 的顺序摘要，按 A/B/C 三档和 B 档三段展示（供调试用）"""
    print(f"\n{'═'*80}")
    print(f"  [排序结果] 共 {len(sorted_steps)} 个 step，排序后顺序（以 agent 分档）")
    print(f"{'═'*80}")

    if not sorted_steps:
        print("  (无 step)")
        print(f"{'═'*80}")
        return

    # 重建分档信息：按 agent_id 统计全对/全错/混合
    agent_steps_map: dict[int, list] = {}
    for s in sorted_steps:
        aid = s.get("agent_id")
        if aid not in agent_steps_map:
            agent_steps_map[aid] = []
        agent_steps_map[aid].append(s)

    agent_max_step: dict[int, int] = {}
    agent_tier: dict[int, str] = {}
    for aid, asteps in agent_steps_map.items():
        agent_max_step[aid] = max(s.get("step_number", 0) for s in asteps)
        has_true = any(s.get("is_correct") is True for s in asteps)
        has_false = any(s.get("is_correct") is False for s in asteps)
        if has_false and not has_true:
            agent_tier[aid] = "A"
        elif has_true and not has_false:
            agent_tier[aid] = "C"
        else:
            agent_tier[aid] = "B"

    current_section = None
    idx = 0
    for s in sorted_steps:
        aid = s.get("agent_id")
        tier = agent_tier.get(aid, "?")
        is_correct = s.get("is_correct")
        step_num = s.get("step_number", 0)
        correct_mark = "对" if is_correct is True else ("错" if is_correct is False else "未定")

        if tier == "A":
            section = "A档 (全错 agent)"
        elif tier == "C":
            section = "C档 (全对 agent)"
        else:
            if is_correct is False:
                section = "B档·错误 step"
            elif step_num == agent_max_step.get(aid, -1):
                section = "B档·正确 step (最后一步/含答案)"
            else:
                section = "B档·正确 step (非最后一步)"

        if section != current_section:
            current_section = section
            print(f"\n  -- {section} --")

        idx += 1
        cid = s.get("cluster_id", "?")
        ctag = s.get("cluster_tag", "?")
        print(f"  {idx:>3}. Agent{aid}-Step{step_num} [{correct_mark}]  (聚类{cid} [{ctag}])")

    print(f"{'═'*80}")


# ============================================================
# Step 4: 构建 Exchange Prompt
# ============================================================

def _build_exchange_prompt_for_agent(
    current_agent_idx: int,
    sorted_steps: list,
) -> dict:
    """
    为指定 agent 构建 exchange prompt

    核心逻辑：排除当前 agent 自身的 step（避免自我强化），
    只将其他 agent 已排好序的 step 组织进 prompt。
    排序不变；连续同一 agent 的 step 合并为一段展示，段首标注 agent（与内部 agent_id 一致），
    段内仅保留 original Step。

    Args:
        current_agent_idx: 当前 agent 的内部索引（0-based），其 step 将被过滤掉
        sorted_steps: 已按 true-thinking-last 排序的全部 step 列表

    Returns:
        {"role": "user", "content": prompt 文本}
    """
    # 过滤掉当前 agent 自己的 step，只保留其他 agent 的 step
    other_steps = [s for s in sorted_steps if s.get("agent_id") != current_agent_idx]

    if not other_steps:
        # 极端情况：过滤后没有其他 agent 的 step，退化为仅要求逐步重写与给答案
        return {
            "role": "user",
            "content": (
                "Please review your reasoning carefully. "
                "Please structure your updated reasoning step by step in the format: "
                "Step 1: ... Step 2: ... and so on. "
                "Write your final answer in the format: The answer is: <ANSWER> at the end of your response."
            )
        }

    # 构建 prompt：排序不变；按「连续同 agent」分段，无全局 Step 序号，仅保留 original Step
    prefix = "These are the reasoning steps from other agents: "

    n = len(other_steps)
    # --- 将 other_steps 按「排序不变」划分为连续 run：相邻且 agent_id 相同的一段为同一 run ---
    # 例：顺序为 A,A,B,A 时 → 三个 run：[A,A]、[B]、[A]。同一 agent 在全局排序中不连续则会出现多段。
    runs = []
    i0 = 0
    while i0 < n:
        aid = other_steps[i0].get("agent_id")
        i1 = i0 + 1
        # 向右扩展，直到 agent 变化或到达列表末尾；[i0, i1) 即本 run 的下标区间
        while i1 < n and other_steps[i1].get("agent_id") == aid:
            i1 += 1
        runs.append((i0, i1, aid))
        i0 = i1

    # --- 按 run 顺序拼入 prefix：每段先写 Agent {id}，再写该段内各步的 original Step 与正文 ---
    for start, end, agent_id in runs:
        n_in_run = end - start

        if n_in_run == 1:
            # 单步 run：一行 Agent 标题 + 一条 Original Step（换行与多步首行对齐）
            st = other_steps[start]
            sn = st.get("step_number")
            content = st.get("content", "")
            prefix += (
                f"\n\nAgent {agent_id}:\n"
                f"  Step {sn}: ```{content}```"
            )
        else:
            # 多步 run：仅一行 Agent 标题，其下连续列出该 agent 在本段中的多个 original Step（仍按全局顺序）
            prefix += f"\n\nAgent {agent_id}:"
            for k in range(start, end):
                st = other_steps[k]
                sn = st.get("step_number")
                content = st.get("content", "")
                prefix += f"\n Step {sn}: ```{content}```"

    # 引导语：中性表述，不引用正确性标签
    suffix = (
        "\n\nUsing the reasoning from other agents as additional advice, "
        "can you give an updated answer? "
        "Examine your solution and that other agents step by step. "
        "Please structure your updated reasoning step by step in the format: "
        "Step 1: ... Step 2: ... and so on. "
        "Put your answer in the form (X) at the end of your response."
    )

    return {"role": "user", "content": prefix + suffix}


def _print_exchange_prompts_preview(agent_prompts: list, max_agents: int = 2, max_content_len: int = 800):
    """
    打印 exchange prompt 的预览（只打印前 max_agents 个 agent，避免输出过长）

    Args:
        agent_prompts: 每个 agent 的 prompt 消息列表
        max_agents: 最多打印几个 agent 的 prompt
        max_content_len: 每个 prompt 最多显示的字符数
    """
    print(f"\n{'='*70}")
    print(f"[Exchange Prompt 预览] 共 {len(agent_prompts)} 个 agent（展示前 {min(max_agents, len(agent_prompts))} 个）")
    print("="*70)

    for agent_idx, prompt in enumerate(agent_prompts[:max_agents]):
        content = prompt.get("content", "")
        preview = content[:max_content_len] + ("..." if len(content) > max_content_len else "")
        print(f"\n--- Agent {agent_idx} 的 Exchange Prompt （共 {len(content)} 字符）---")
        print(preview)

    if len(agent_prompts) > max_agents:
        print(f"\n... 另有 {len(agent_prompts) - max_agents} 个 agent 的 prompt（内容已略去）")
    print("="*70)


# ============================================================
# Step 5: 并发调用 API 执行 exchange
# ============================================================

async def _run_exchange(agent_contexts: list, agent_prompts: list) -> list:
    """
    并发调用 API，为每个 agent 执行 exchange

    流程：
    1. 将 exchange user prompt 追加到 agent 上下文
    2. 调用 API 生成新的 assistant 回复
    3. 将 assistant 回复追加到上下文（原地修改 agent_contexts）

    Args:
        agent_contexts: 每个 agent 的对话上下文（会被原地追加 user + assistant 消息）
        agent_prompts: 每个 agent 对应的 exchange prompt 消息

    Returns:
        [(agent_idx, success, error_or_None), ...] 结果列表
    """
    async def _exchange_one(agent_idx: int):
        context = agent_contexts[agent_idx]
        prompt_msg = agent_prompts[agent_idx]
        try:
            # 将 exchange 的 user prompt 追加到上下文
            context.append(prompt_msg)
            # 调用 API 生成新回复（agenerate_answer 参照 expand.py 实现）
            completion = await agenerate_answer(context)
            assistant_msg = construct_assistant_message(completion)
            # 将 assistant 回复追加到上下文
            context.append(assistant_msg)
            return agent_idx, True, None
        except Exception as e:
            return agent_idx, False, e

    all_results = []
    for batch_start in range(0, len(agent_contexts), EXCHANGE_CONCURRENT_LIMIT):
        batch_end = min(batch_start + EXCHANGE_CONCURRENT_LIMIT, len(agent_contexts))
        tasks = [_exchange_one(i) for i in range(batch_start, batch_end)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=False)
        all_results.extend(batch_results)
        # 批次间等待，避免触发 API 速率限制
        if batch_end < len(agent_contexts):
            await asyncio.sleep(1)

    return all_results


# ============================================================
# Expand1：聚类 → 类簇标签 → 修正 step 标签 → 排序 → 构造 prompt → API（单轮，供 math 编排）
# ============================================================


def _print_exchange_round_results(agent_contexts: list, round_num: int):
    """聚类+exchange 调用完成后：打印 Step 8–10；返回本轮 majority_answer。"""
    _round_tag = f"Exchange Round {round_num}"
    print(f"\n{'═'*80}")
    print(f"  [{_round_tag} - Step 8] 各 Agent 推理结果")
    print(f"{'═'*80}")
    for _aidx, _ctx in enumerate(agent_contexts):
        _last_msg = _ctx[-1] if _ctx else {}
        _ex_resp = _last_msg.get("content", "")
        print(f"\n  \u250c\u2500 Agent {_aidx} (回复长度: {len(_ex_resp)} 字符) {'\u2500'*40}")
        if _ex_resp:
            for _line in _ex_resp.split("\n"):
                print(f"  \u2502  {_line}")
        else:
            print(f"  \u2502  (无内容)")
        print(f"  \u2514{'\u2500'*70}")
    print(f"{'═'*80}")

    _round_majority = _get_latest_exchange_majority(agent_contexts)
    _n_agents = len(agent_contexts)

    print(f"\n{'═'*80}")
    print(f"  [{_round_tag} - Step 9] 答案提取与多数投票")
    print(f"{'═'*80}")
    _extracted_answers = []
    for _aidx, _ctx in enumerate(agent_contexts):
        _resp = _ctx[-1].get("content", "") if _ctx else ""
        _ans = extract_answer_from_text(_resp, is_math=IS_MATH)
        _extracted_answers.append(_ans)
        print(f"    Agent {_aidx}: 提取答案 = {_ans}")
    print(f"\n  {'─'*70}")
    print(f"    答案列表: {_extracted_answers}")
    print(f"    majority_answer (多数投票结果) = {_round_majority}")
    if _round_majority is None:
        print("    [警告] 所有 agent 均未提取到有效答案，majority_answer 为 None")
    print(f"{'═'*80}")

    print(f"\n{'═'*80}")
    print(f"  [{_round_tag} - Step 10] 各 Agent 答案与 majority_answer 对比")
    print(f"{'═'*80}")
    for _aidx, _ans in enumerate(_extracted_answers):
        _ans_disp = _ans if _ans is not None else "(未提取到)"
        _ok = is_correct_answer(_ans, _round_majority, is_math=IS_MATH) if _ans else False
        _mark = "[正确]" if _ok else "[错误]"
        print(f"    Agent {_aidx}: 答案={_ans_disp}, 与 majority_answer={_round_majority} 对比 → {_mark}")
    _round_maj_cnt = sum(
        1
        for _ans in _extracted_answers
        if _ans and is_correct_answer(_ans, _round_majority, is_math=IS_MATH)
    )
    print(f"\n    与 majority_answer 一致的 agent 数: {_round_maj_cnt}/{_n_agents}")
    print(f"{'═'*80}")
    return _round_majority


async def run_exchange1_from_expand_outputs(
    step_vectors: np.ndarray,
    step_indices: list,
    all_steps: list,
    agent_contexts: list,
    *,
    use_method: str = "kmeans",
    round_num: int = 1,
) -> dict:
    """
    在已完成 expand 且已得到向量与 all_steps 的前提下，执行单轮 exchange1：
    PCA + 聚类 → 类簇标签 → 修正 step 标签 → 排序 → 构造 exchange prompt → API。
    会原地修改 agent_contexts（追加 user + assistant）。

    Returns:
        {"majority_answer": str | None, "agent_contexts": list}
    """
    await _run_single_cluster_exchange_round(
        step_vectors,
        step_indices,
        all_steps,
        agent_contexts,
        round_num,
        use_method,
        bidirectional=False,
    )
    maj = _print_exchange_round_results(agent_contexts, round_num)
    return {"majority_answer": maj, "agent_contexts": agent_contexts}


async def run_exchange1_multi_k_from_expand_outputs(
    step_vectors: np.ndarray,
    step_indices: list,
    all_steps: list,
    agent_contexts: list,
    *,
    k_values: list | None = None,
    round_num: int = 1,
    bidirectional: bool = False,
) -> dict:
    """
    多 k 对比实验：对同一批 expand 向量，分别用 k=k_values[0], k_values[1], ... 运行
    完整的「聚类→标签修正→排序→构建 prompt→API 调用」流程，
    最后汇总各 k 对应的 majority_answer 并打印对比表。

    关键设计：每个 k 使用 agent_contexts 的独立深拷贝，各轮实验互不污染。
    原始 agent_contexts 不被修改。

    Args:
        step_vectors: expand 阶段产出的 step 向量矩阵
        step_indices: 向量下标 → all_steps 下标的映射
        all_steps: 含 is_correct 标签的 step 列表
        agent_contexts: 所有 agent 的对话上下文（本函数不修改原始列表）
        k_values: 要实验的 k 取值列表，默认使用 clustering 配置的 KMEANS_K_MIN..KMEANS_K_MAX
        round_num: 轮次编号（仅用于日志标注）
        bidirectional: 标签修正策略
            False → 单向修正（仅 correct 类簇置 True）
            True  → 双向修正（correct 类簇全 True、wrong 类簇全 False）

    Returns:
        {
            "k_results": {k: {"majority_answer": str|None, "agent_contexts": list}},
            "summary_table": [(k, majority_answer), ...]   # 按 k 排序
        }
    """
    if k_values is None:
        k_values = list(range(KMEANS_K_MIN, KMEANS_K_MAX + 1))

    mode_tag = "Bidirectional" if bidirectional else "Unidirectional"

    k_results: dict = {}
    summary_table: list = []

    for k in k_values:
        print("\n" + "★" * 80)
        print(f"  [多 k 实验·{mode_tag}] KMeans k={k}  (round_num={round_num})")
        print("★" * 80)

        # 每个 k 独立深拷贝，防止上下文互相污染
        contexts_copy = copy.deepcopy(agent_contexts)

        await _run_single_cluster_exchange_round(
            step_vectors,
            step_indices,
            all_steps,
            contexts_copy,
            round_num,
            use_method="kmeans",
            bidirectional=bidirectional,
            kmeans_k=k,
        )
        maj = _print_exchange_round_results(contexts_copy, round_num)
        k_results[k] = {"majority_answer": maj, "agent_contexts": contexts_copy}
        summary_table.append((k, maj))

    # 汇总对比表
    print("\n" + "★" * 80)
    print(f"  [多 k 实验·{mode_tag}·汇总] 各 k 下的 majority_answer 对比")
    print("★" * 80)
    print(f"  {'k':>4}  {'majority_answer'}")
    print("  " + "-" * 30)
    for k, maj in summary_table:
        print(f"  {k:>4}  {maj!r}")
    print("★" * 80 + "\n")

    return {"k_results": k_results, "summary_table": summary_table}


async def run_exchange2_multi_k_from_multi_k1_outputs(
    k1_results: dict,
    cfg: "ExpandConfig",
    *,
    k_values: list | None = None,
    round_num: int = 2,
    bidirectional: bool = False,
) -> dict:
    """
    策略 A（同 k 配对）的第二轮多 k 实验：
    对 round1 中每个 k 产生的 agent_contexts，基于其最新回复重新向量化，
    再用**相同的 k** 进行第二轮聚类 + exchange。

    k 严格一一对应：
        round1 k=3 → contexts_k3 → 重新向量化 → round2 k=3
        round1 k=4 → contexts_k4 → 重新向量化 → round2 k=4
        ...

    关键设计：
    1. 向量来源：从 k1_results[k]["agent_contexts"] 的 context[-1]（round1 新回复）提取 step
    2. 向量化：调用与 run_exchange2_from_exchange1_outputs 完全相同的重新向量化流程
    3. 深拷贝：对 round1 的 contexts 深拷贝后再追加 round2 消息，不污染 round1 结果
    4. 同 k：round2 的 kmeans_k == round1 的 k，严格对应
    5. 同策略：bidirectional 应与 round1 保持一致，否则数据流错位

    Args:
        k1_results    : run_exchange1_multi_k_from_expand_outputs 返回值中的 "k_results" 字段
                        结构：{k: {"majority_answer": str|None, "agent_contexts": list}}
        cfg           : ExpandConfig，用于重新向量化（num_agents / embedding 参数等）
        k_values      : 要实验的 k 列表，必须是 k1_results 的子集；
                        None 时取 k1_results 的所有 k（按升序）
        round_num     : 轮次编号（仅用于日志标注，建议传 2）
        bidirectional : Round2 的标签修正策略
            False → 单向修正
            True  → 双向修正（应与 Round1 一致，否则两轮策略不同会破坏对比性）

    Returns:
        {
            "k_results": {
                k: {
                    "majority_answer": str | None,
                    "agent_contexts" : list,       # round2 结束后的 contexts（round1 的深拷贝 + round2 追加）
                    "step_vectors"   : np.ndarray, # round2 使用的向量（源自 round1 回复）
                }
            },
            "summary_table": [(k, majority_answer), ...]  # 按 k 升序
        }
    """
    if k_values is None:
        k_values = sorted(k1_results.keys())

    mode_tag = "Bidirectional" if bidirectional else "Unidirectional"

    k2_results: dict = {}
    summary_table: list = []

    for k in k_values:
        # ── 0. 检查 round1 是否有该 k 的结果 ──────────────────────────────
        if k not in k1_results:
            print(f"[警告] exchange2_multi_k: round1 中不存在 k={k} 的结果，跳过")
            continue

        print("\n" + "◆" * 80)
        print(f"  [Round2 多 k 实验·{mode_tag}] k={k}（与 Round1 k={k} 严格对应）")
        print("◆" * 80)

        # ── 1. 取 round1 k 对应的 agent_contexts ────────────────────────
        #    注意：不直接使用，而是深拷贝后再操作，确保 round1 结果不被污染
        contexts_round1_k = k1_results[k]["agent_contexts"]

        # ── 2. 从 round1 的最新回复（context[-1]）重新提取 step 并向量化 ──
        #    与 run_exchange2_from_exchange1_outputs 完全相同的向量化流程
        print(f"\n  [k={k}] Step A: 从 Round1 k={k} 的最新回复重新提取 step 并向量化")
        majority_r1, agent_results_r1 = expand_compute_majority_and_agent_results_from_latest(
            contexts_round1_k, cfg
        )
        expand_print_step4_agent_vs_majority(agent_results_r1, majority_r1, cfg.num_agents)
        expand_print_step5_steps(agent_results_r1)

        all_steps_k = expand_flatten_all_steps(agent_results_r1)
        expand_print_step6_all_steps(all_steps_k, cfg.num_agents)

        step_vectors_k, step_indices_k = expand_run_embedding(all_steps_k, cfg)

        if step_vectors_k is None or step_vectors_k.shape[0] < 2:
            n = step_vectors_k.shape[0] if step_vectors_k is not None else 0
            print(f"  [k={k}] 向量数量不足（{n}），跳过 Round2 k={k}")
            k2_results[k] = {
                "majority_answer": majority_r1,
                "agent_contexts": contexts_round1_k,
                "step_vectors": step_vectors_k,
            }
            summary_table.append((k, majority_r1))
            continue

        print(f"  [k={k}] 向量化完成: {step_vectors_k.shape[0]} 个 step，{step_vectors_k.shape[1]} 维")

        # ── 3. 深拷贝 round1 的 contexts，作为 round2 的起点 ─────────────
        #    必须深拷贝：_run_single_cluster_exchange_round 会原地追加消息
        contexts_round2_k = copy.deepcopy(contexts_round1_k)

        # ── 4. 用相同的 k 运行 round2 聚类 + exchange ────────────────────
        print(f"\n  [k={k}] Step B: Round2 聚类（kmeans_k={k}，与 Round1 严格对应，修正策略={mode_tag}）+ Exchange")
        await _run_single_cluster_exchange_round(
            step_vectors_k,
            step_indices_k,
            all_steps_k,
            contexts_round2_k,    # round2 的 contexts（round1 深拷贝，不影响 round1）
            round_num,
            use_method="kmeans",
            bidirectional=bidirectional,  # ★ 关键：与 Round1 保持一致
            kmeans_k=k,           # ★ 关键：round2 的 k == round1 的 k
        )
        maj_r2 = _print_exchange_round_results(contexts_round2_k, round_num)

        k2_results[k] = {
            "majority_answer": maj_r2,
            "agent_contexts": contexts_round2_k,
            "step_vectors": step_vectors_k,
        }
        summary_table.append((k, maj_r2))

    # ── 汇总对比表 ──────────────────────────────────────────────────────
    print("\n" + "◆" * 80)
    print(f"  [Round2 多 k 实验·{mode_tag}·汇总] 各 k 下 Round2 的 majority_answer 对比")
    print("◆" * 80)
    print(f"  {'k':>4}  {'Round2 majority_answer'}")
    print("  " + "-" * 35)
    for k, maj in summary_table:
        print(f"  {k:>4}  {maj!r}")
    print("◆" * 80 + "\n")

    return {"k_results": k2_results, "summary_table": summary_table}


async def run_exchange_bidirectional_1_multi_k_from_expand_outputs(
    step_vectors: np.ndarray,
    step_indices: list,
    all_steps: list,
    agent_contexts: list,
    *,
    k_values: list | None = None,
    round_num: int = 1,
) -> dict:
    """
    与 run_exchange1_multi_k_from_expand_outputs 完全相同的多 k 流程，但使用
    **双向标签修正**（correct 类簇全 True、wrong 类簇全 False）。

    用于多 k 实验中的 bidirectional Round1。
    """
    return await run_exchange1_multi_k_from_expand_outputs(
        step_vectors,
        step_indices,
        all_steps,
        agent_contexts,
        k_values=k_values,
        round_num=round_num,
        bidirectional=True,
    )


async def run_exchange_bidirectional_2_multi_k_from_multi_k1_outputs(
    k1_results: dict,
    cfg: "ExpandConfig",
    *,
    k_values: list | None = None,
    round_num: int = 2,
) -> dict:
    """
    与 run_exchange2_multi_k_from_multi_k1_outputs 完全相同的同 k 配对流程，但
    使用**双向标签修正**。

    用于多 k 实验中的 bidirectional Round2，要求 k1_results 来自
    run_exchange_bidirectional_1_multi_k_from_expand_outputs（即 Round1 也是 bidirectional），
    保证两轮策略一致、k 严格配对。
    """
    return await run_exchange2_multi_k_from_multi_k1_outputs(
        k1_results,
        cfg,
        k_values=k_values,
        round_num=round_num,
        bidirectional=True,
    )


async def run_exchange2_from_exchange1_outputs(
    agent_contexts: list,
    cfg: ExpandConfig,
    *,
    use_method: str = "kmeans",
    round_num: int = 2,
) -> dict:
    """
    在 exchange1 已完成（agent_contexts 已含最新 assistant）的前提下执行 exchange2：
    从各 agent 最后一条 assistant 提取 step，按多数票继承 is_correct，向量化后执行
    与 exchange1 相同的 PCA/聚类/修正/排序/Exchange API 流程。
    会原地修改 agent_contexts（再追加一轮 user + assistant）。

    Returns:
        {"majority_answer": str | None, "agent_contexts": list}
    """
    majority_answer, agent_results = expand_compute_majority_and_agent_results_from_latest(
        agent_contexts, cfg
    )
    expand_print_step4_agent_vs_majority(agent_results, majority_answer, cfg.num_agents)
    expand_print_step5_steps(agent_results)

    all_steps = expand_flatten_all_steps(agent_results)
    expand_print_step6_all_steps(all_steps, cfg.num_agents)

    step_vectors, step_indices = expand_run_embedding(all_steps, cfg)
    if step_vectors is None or len(step_vectors) < 2:
        n = step_vectors.shape[0] if step_vectors is not None else 0
        print(f"[错误] exchange2 需要至少 2 个 step 向量，当前 {n}，跳过聚类 exchange")
        return {"majority_answer": majority_answer, "agent_contexts": agent_contexts}

    await _run_single_cluster_exchange_round(
        step_vectors,
        step_indices,
        all_steps,
        agent_contexts,
        round_num,
        use_method,
        bidirectional=False,
    )
    maj = _print_exchange_round_results(agent_contexts, round_num)
    return {"majority_answer": maj, "agent_contexts": agent_contexts}


async def run_exchange_bidirectional_1_from_expand_outputs(
    step_vectors: np.ndarray,
    step_indices: list,
    all_steps: list,
    agent_contexts: list,
    *,
    use_method: str = "kmeans",
    round_num: int = 1,
) -> dict:
    """
    与 run_exchange1_from_expand_outputs 相同数据流，但 Step 4 使用双向标签修正
    （correct 类簇全 True，wrong 类簇全 False）。
    """
    await _run_single_cluster_exchange_round(
        step_vectors,
        step_indices,
        all_steps,
        agent_contexts,
        round_num,
        use_method,
        bidirectional=True,
    )
    maj = _print_exchange_round_results(agent_contexts, round_num)
    return {"majority_answer": maj, "agent_contexts": agent_contexts}


async def run_exchange_bidirectional_2_from_bidirectional_1_outputs(
    agent_contexts: list,
    cfg: ExpandConfig,
    *,
    use_method: str = "kmeans",
    round_num: int = 2,
) -> dict:
    """
    与 run_exchange2_from_exchange1_outputs 相同，但在 bidirectional_1 之后执行，
    聚类后 Step 4 为双向标签修正。
    """
    majority_answer, agent_results = expand_compute_majority_and_agent_results_from_latest(
        agent_contexts, cfg
    )
    expand_print_step4_agent_vs_majority(agent_results, majority_answer, cfg.num_agents)
    expand_print_step5_steps(agent_results)

    all_steps = expand_flatten_all_steps(agent_results)
    expand_print_step6_all_steps(all_steps, cfg.num_agents)

    step_vectors, step_indices = expand_run_embedding(all_steps, cfg)
    if step_vectors is None or len(step_vectors) < 2:
        n = step_vectors.shape[0] if step_vectors is not None else 0
        print(
            f"[错误] exchange_bidirectional_2 需要至少 2 个 step 向量，当前 {n}，跳过聚类 exchange"
        )
        return {"majority_answer": majority_answer, "agent_contexts": agent_contexts}

    await _run_single_cluster_exchange_round(
        step_vectors,
        step_indices,
        all_steps,
        agent_contexts,
        round_num,
        use_method,
        bidirectional=True,
    )
    maj = _print_exchange_round_results(agent_contexts, round_num)
    return {"majority_answer": maj, "agent_contexts": agent_contexts}


# ============================================================
# 辅助：获取最新一轮 exchange 的多数答案（轮次无关版本）
# ============================================================

def _get_latest_exchange_majority(agent_contexts: list):
    """
    获取最新一轮 exchange 后的多数答案。

    使用 context[-1] 提取每个 agent 最新的 assistant 回复，多数投票确定答案。
    逻辑委托给 clustering.get_majority_answer_from_latest（严格参照 expand.py 实现）。

    Args:
        agent_contexts: 所有 agent 的对话上下文列表

    Returns:
        str or None: 多数答案，若无有效答案则返回 None
    """
    return get_majority_answer_from_latest(agent_contexts, is_math=IS_MATH)


# ============================================================
# 单轮聚类 + Exchange 核心流程（可循环复用）
# ============================================================

async def _run_single_cluster_exchange_round(
    step_vectors: np.ndarray,
    step_indices: list,
    all_steps: list,
    agent_contexts: list,
    round_num: int,
    use_method: str = "kmeans",
    *,
    bidirectional: bool = False,
    kmeans_k: int | None = None,
) -> None:
    """
    执行单轮「聚类 → 排序 → 构建 prompt → exchange」流程。

    agent_contexts 被原地追加（每个 agent 追加 user + assistant 两条消息），
    调用方通过读取 agent_contexts 获取本轮结果，无需返回值。

    Args:
        step_vectors: 待聚类的 step 向量矩阵（N, dim）
        step_indices: 向量下标 → all_steps 下标的映射列表
        all_steps: 含 is_correct 标签的 step 列表
        agent_contexts: 所有 agent 的对话上下文（原地追加）
        round_num: 当前轮次编号（仅用于日志标注）
        use_method: "kmeans" 或 "dbscan"
        bidirectional: True 时 correct 类簇全置 True、wrong 类簇全置 False；False 时仅 correct 类簇置 True
        kmeans_k: 仅 use_method=="kmeans" 时生效。
                  None → 用轮廓系数自动选 k；
                  整数 → 固定使用该 k 值（用于多 k 对比实验）
    """
    tag = (
        f"Bidirectional Exchange Round {round_num}"
        if bidirectional
        else f"Exchange Round {round_num}"
    )

    # ══════════════════════════════════════════════════════════
    # Step 1: PCA 降维 + L2 归一化
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  [{tag} - Step 1] PCA 降维 + L2 归一化")
    print(f"{'═'*80}")
    print(f"    输入向量: {step_vectors.shape[0]} 个, {step_vectors.shape[1]} 维")
    print(f"    目标维度: {TARGET_DIM}")
    vectors_reduced = _reduce_dimensions_pca(step_vectors, target_dim=TARGET_DIM)
    norms = np.linalg.norm(vectors_reduced, axis=1, keepdims=True)
    vectors_reduced = vectors_reduced / (norms + 1e-12)
    print(f"    L2 归一化完成")
    print(f"    输出向量: {vectors_reduced.shape[0]} 个, {vectors_reduced.shape[1]} 维")
    print(f"{'═'*80}")

    # ══════════════════════════════════════════════════════════
    # Step 2: 聚类
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  [{tag} - Step 2] {use_method.upper()} 聚类")
    print(f"{'═'*80}")
    if use_method == "dbscan":
        print(f"    参数: eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}")
        cluster_labels_raw = _cluster_dbscan(
            vectors_reduced, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES
        )
    elif kmeans_k is not None:
        print(f"    模式: 固定 k={kmeans_k}（多 k 对比实验，绕过轮廓系数）")
        cluster_labels_raw = _cluster_kmeans_fixed_k(vectors_reduced, kmeans_k)
    else:
        print(f"    模式: 轮廓系数自动选 k（范围 2..{min(KMEANS_K_MAX, vectors_reduced.shape[0]-1)}）")
        cluster_labels_raw = _cluster_kmeans(vectors_reduced)
    print(f"{'═'*80}")

    # ══════════════════════════════════════════════════════════
    # Step 3: 类簇标签判定
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  [{tag} - Step 3] 类簇标签判定")
    print(f"{'═'*80}")
    print(f"  [判定规则] MAJORITY_THRESHOLD = {MAJORITY_THRESHOLD}")
    print(f"    correct_ratio >= {MAJORITY_THRESHOLD} → correct")
    print(f"    wrong_ratio   >= {MAJORITY_THRESHOLD} → wrong")
    print(f"    否则 → no_majority")

    cluster_tag_results = _label_clusters_by_majority(
        cluster_labels_raw, step_indices, all_steps, majority_threshold=MAJORITY_THRESHOLD
    )

    _unique_cids = sorted(set(int(c) for c in cluster_labels_raw))
    for _cid in _unique_cids:
        if _cid == -1:
            _noise_cnt = sum(1 for c in cluster_labels_raw if int(c) == -1)
            print(f"\n    类簇 {_cid} (噪声): {_noise_cnt} 个 step → [noise]")
            continue
        _vec_indices = [i for i in range(len(cluster_labels_raw)) if int(cluster_labels_raw[i]) == _cid]
        _c_correct = sum(1 for vi in _vec_indices if all_steps[step_indices[vi]].get("is_correct") is True)
        _c_wrong = sum(1 for vi in _vec_indices if all_steps[step_indices[vi]].get("is_correct") is False)
        _c_total = len(_vec_indices)
        _c_labeled = _c_correct + _c_wrong
        _c_cr = _c_correct / _c_labeled if _c_labeled > 0 else 0.0
        _c_wr = _c_wrong / _c_labeled if _c_labeled > 0 else 0.0
        _c_tag = cluster_tag_results.get(_cid, "?")
        print(f"\n    类簇 {_cid}: {_c_total} 个 step, correct={_c_correct} wrong={_c_wrong}")
        print(f"      correct_ratio={_c_cr:.2f}, wrong_ratio={_c_wr:.2f} → [{_c_tag}]")
        _step_labels = []
        for vi in _vec_indices:
            _si = step_indices[vi]
            _s = all_steps[_si]
            _mk = "对" if _s.get("is_correct") else "错"
            _step_labels.append(f"Agent{_s.get('agent_id')}-Step{_s.get('step_number')}[{_mk}]")
        if len(_step_labels) <= 20:
            print(f"      包含: {', '.join(_step_labels)}")
        else:
            print(f"      包含: {', '.join(_step_labels[:20])}, ... (共 {len(_step_labels)} 个)")
    print(f"{'═'*80}")

    # ══════════════════════════════════════════════════════════
    # Step 4: 标签修正
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  [{tag} - Step 4] 标签修正")
    print(f"{'═'*80}")
    print(f"  [修正规则]")
    if bidirectional:
        print(f"    correct 类簇 → 内部所有 step 的 is_correct 置为 True")
        print(f"    wrong 类簇 → 内部所有 step 的 is_correct 置为 False")
        print(f"    no_majority / noise 类簇 → 保持原标签不修改")
    else:
        print(f"    correct 类簇 → 内部所有 step 的 is_correct 置为 True")
        print(f"    wrong / no_majority / noise 类簇 → 保持原标签不修改")
        print(f"    (单向修正: 只把 False 改为 True，不会把 True 改为 False)")

    if bidirectional:
        steps_modified, n_modified, change_records = _apply_cluster_labels_to_steps_bidirectional(
            cluster_labels_raw, cluster_tag_results, step_indices, all_steps
        )
    else:
        steps_modified, n_modified, change_records = _apply_cluster_labels_to_steps(
            cluster_labels_raw, cluster_tag_results, step_indices, all_steps
        )

    _lbl = {True: "True", False: "False", None: "None"}
    _cr_by_cluster = {}
    for _rec in change_records:
        _rcid = _rec["cluster_id"]
        if _rcid not in _cr_by_cluster:
            _cr_by_cluster[_rcid] = []
        _cr_by_cluster[_rcid].append(_rec)
    for _rcid in sorted(_cr_by_cluster.keys()):
        _recs = _cr_by_cluster[_rcid]
        _rtag = _recs[0]["cluster_tag"]
        print(f"\n    类簇 {_rcid} [{_rtag}]:")
        for _rec in sorted(_recs, key=lambda r: (r["agent_id"], r["step_number"])):
            _orig = _lbl.get(_rec["original_label"], str(_rec["original_label"]))
            _new = _lbl.get(_rec["new_label"], str(_rec["new_label"]))
            if _rec["changed"]:
                print(f"      Agent{_rec['agent_id']}-Step{_rec['step_number']}: {_orig} → {_new} [已修改]")
            else:
                print(f"      Agent{_rec['agent_id']}-Step{_rec['step_number']}: {_orig} (不变)")
    _changed_cnt = sum(1 for r in change_records if r["changed"])
    print(f"\n    汇总: 共 {len(change_records)} 个 step，其中 {_changed_cnt} 个被修改")
    print(f"{'═'*80}")

    # ══════════════════════════════════════════════════════════
    # Step 5: 排序
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  [{tag} - Step 5] 经聚类结果标签修正后，以agent为单位进行排序")
    sorted_steps = _sort_steps_by_cluster(
        steps_modified, cluster_labels_raw, step_indices, cluster_tag_results
    )
    _print_sorted_steps_summary(sorted_steps)
    print(f"{'═'*80}")

    # ══════════════════════════════════════════════════════════
    # Step 6: 构建 Exchange Prompt
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  [{tag} - Step 6] Exchange Prompt 构建")
    print(f"{'═'*80}")
    print(f"  [构建规则]")
    print(f"    1. 排除当前 agent 自身的 step（避免自我强化）")
    print(f"    2. 其余 step 按 Step 5 的排序组织")
    print(f"    3. 连续同 agent 分段、仅 original Step，无全局 Step 序号；不向模型展示正确性标注")
    print(f"    4. 引导 agent 参考他人推理并更新答案")
    agent_prompts = [
        _build_exchange_prompt_for_agent(agent_idx, sorted_steps)
        for agent_idx in range(len(agent_contexts))
    ]
    if agent_prompts:
        _p0_content = agent_prompts[0].get("content", "")
        print(f"\n  ┌─ Agent 0 的完整 Exchange Prompt (共 {len(_p0_content)} 字符) {'─'*20}")
        for _line in _p0_content.split("\n"):
            print(f"  │  {_line}")
        print(f"  └{'─'*70}")
    if len(agent_prompts) > 1:
        print(f"\n    其余 agent prompt 长度:")
        for _i, _p in enumerate(agent_prompts[1:], start=1):
            print(f"      Agent {_i}: {len(_p.get('content', ''))} 字符")
    print(f"{'═'*80}")

    # ══════════════════════════════════════════════════════════
    # Step 7: 执行 Exchange API 调用
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  [{tag} - Step 7] Exchange API 调用")
    print(f"{'═'*80}")
    print(f"    并发上限: {EXCHANGE_CONCURRENT_LIMIT}, 调用模型: {MODEL_TAG}")
    results = await _run_exchange(agent_contexts, agent_prompts)
    _success_cnt = sum(1 for _, ok, _ in results if ok)
    _fail_cnt = len(results) - _success_cnt
    print(f"    调用完成: 成功={_success_cnt}, 失败={_fail_cnt}")
    for agent_idx, ok, err in results:
        if not ok:
            print(f"    [失败] Agent {agent_idx}: {err}")
    print(f"{'═'*80}")
