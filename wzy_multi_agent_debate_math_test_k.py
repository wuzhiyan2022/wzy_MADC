"""
数学多 Agent 辩论编排入口：按动作列表依次执行（如 expand → exchange1）。

- expand：独立推理、多数票、step 标签、向量化（run_expand_pipeline）。
- exchange1 / exchange2 / exchange_bidirectional_*：见各分支说明。
- 支持单题或批量：批量时在每题每个 action 结束后更新累计正确率并打印。

expand 结束后会 deepcopy 一份 expand_pack 快照；exchange_bidirectional_1/2 只从快照解包，
可与 exchange1/2 同 pipeline 串行，便于在相同 expand 起点上对比两种 exchange。
"""

import os

# 必须在 numpy / sklearn / MKL 相关模块 import 之前设置，
# 防止 Windows + MKL 环境下 KMeans 小数据集时触发 OMP 线程数 > chunk 数的内存泄漏警告。
os.environ.setdefault("OMP_NUM_THREADS", "1")

import contextlib
import copy
import io
import json
import sys


def _init_stdio_utf8() -> None:
    """Windows 控制台默认 GBK，仅 reconfigure 仍可能乱码；先切 UTF-8 代码页再绑定 stdio。"""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "buffer"):
            continue
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            try:
                setattr(
                    sys,
                    name,
                    io.TextIOWrapper(
                        stream.buffer,
                        encoding="utf-8",
                        errors="replace",
                        line_buffering=name == "stdout",
                    ),
                )
            except Exception:
                pass


_init_stdio_utf8()

import asyncio
from typing import Any, Dict, List, Optional

from wzy_multi_agent_debate_expand import (
    VERIFY_VECTORIZATION,
    SAVE_DEBUG_FILES,
    default_expand_config,
    expand_save_cache_if_enabled,
    get_cache_path,
    is_correct_answer,
    run_expand_pipeline,
)
from wzy_multi_agent_debate_exchange import (
    run_exchange1_from_expand_outputs,
    run_exchange1_multi_k_from_expand_outputs,
    run_exchange2_multi_k_from_multi_k1_outputs,
    run_exchange2_from_exchange1_outputs,
    run_exchange_bidirectional_1_from_expand_outputs,
    run_exchange_bidirectional_1_multi_k_from_expand_outputs,
    run_exchange_bidirectional_2_from_bidirectional_1_outputs,
    run_exchange_bidirectional_2_multi_k_from_multi_k1_outputs,
)

# 与 wzy_multi_agent_debate_expand 模块默认一致；可在此覆盖字段
CONFIG = default_expand_config()

# 依次执行的动作名（示例）：
#   ["expand", "exchange1", "exchange2"]
#   ["expand", "exchange_bidirectional_1", "exchange_bidirectional_2"]
#   ["expand", "exchange1", "exchange2"]
#   亦可合并：bidirectional 使用 expand 快照，与 exchange1/2 互不污染 agent_contexts
# 可用动作：
#   expand / exchange1 / exchange2 / exchange_bidirectional_1 / exchange_bidirectional_2
#   exchange1_multi_k                  - 单向多 k 实验（Round1，遍历 KMEANS_EXPERIMENT_K_VALUES）
#   exchange2_multi_k                  - 单向多 k 实验（Round2，必须跟在 exchange1_multi_k 之后，同 k 配对）
#   exchange_bidirectional_1_multi_k   - 双向多 k 实验（Round1，与 exchange1_multi_k 流程一致但用双向修正）
#   exchange_bidirectional_2_multi_k   - 双向多 k 实验（Round2，必须跟在 exchange_bidirectional_1_multi_k 之后，同 k 配对）
ACTION_PIPELINE: List[str] = ["expand", "exchange_bidirectional_1_multi_k", "exchange_bidirectional_2_multi_k"]

# exchange1 聚类方法："kmeans" | "dbscan"
EXCHANGE1_CLUSTER_METHOD = "kmeans"

# ---------- 多 k 实验配置 ----------
# 仅在 ACTION_PIPELINE 包含 "exchange1_multi_k" 时生效
# 指定要对比的 KMeans 聚类数取值列表（k 会逐一跑完整 exchange 流程）
KMEANS_EXPERIMENT_K_VALUES: List[int] = list(range(3, 11))  # k = 2, 4, …, 10

# ---------- 批量配置 ----------
# True：按 BATCH_QUESTION_IDS / BATCH_MAX_QUESTIONS 从数据集取多题依次跑
# False：单题（沿用 CONFIG.fixed_question_id 为 None 时随机一题，否则固定 id）
RUN_BATCH: bool = True
# 非 None 时只跑这些 question_id（与数据集中类型一致即可，内部转 str 匹配）
BATCH_QUESTION_IDS: Optional[List[Any]] = [18,37,124,247,352,392,395,460,467,495]
# BATCH_QUESTION_IDS: Optional[List[Any]] = [18]
# 全表模式下最多跑多少题（None 表示不截断）；在 BATCH_QUESTION_IDS 为 None 时生效
BATCH_MAX_QUESTIONS: Optional[int] = None

# ---------- 日志详细度 ----------
# 0 = 静默（仅关键摘要 + 累计正确率），1 = 全量日志（调试单题）
# 批量时建议 0；单题调试时设 1
BATCH_VERBOSE: int = 0


@contextlib.contextmanager
def _suppress_stdout():
    """临时将 sys.stdout 重定向到 devnull，屏蔽所有 print 输出。"""
    old = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def _quiet() -> bool:
    """当前是否处于静默模式。"""
    return BATCH_VERBOSE < 1


_STAGES = (
    "expand",
    "exchange1",
    "exchange1_multi_k",
    "exchange2_multi_k",
    "exchange2",
    "exchange_bidirectional_1",
    "exchange_bidirectional_1_multi_k",
    "exchange_bidirectional_2",
    "exchange_bidirectional_2_multi_k",
)


def _init_stage_stats() -> Dict[str, Dict[str, Any]]:
    return {a: {"attempted": 0, "correct": 0, "correct_ids": []} for a in _STAGES}


def _bump_stage(
    stats: Dict[str, Dict[str, Any]],
    action_label: str,
    match_gt: bool,
    question_id: str,
) -> None:
    if action_label not in stats:
        return
    stats[action_label]["attempted"] += 1
    if match_gt:
        stats[action_label]["correct"] += 1
        ids: List[str] = stats[action_label]["correct_ids"]
        ids.append(str(question_id))


def _print_cumulative_stage(action_label: str, stats: Dict[str, Dict[str, Any]]) -> None:
    s = stats.get(action_label)
    if not s:
        return
    att, cor = s["attempted"], s["correct"]
    pct = (cor / att * 100.0) if att else 0.0
    print(
        f"\n[正确率·{action_label}·累计] {cor}/{att} = {pct:.2f}% "
        f"（多数票与 GT 一致题数 / 已进入该阶段的题数）"
    )
    ids = s.get("correct_ids") or []
    if ids:
        print(f"  正确题 question_id: {', '.join(ids)}")
    else:
        print("  正确题 question_id: (无)")


def _print_batch_summary(stats: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("[汇总 · 各阶段累计正确率]")
    for a in ACTION_PIPELINE:
        if a in stats:
            _print_cumulative_stage(a, stats)
    print("=" * 72)


def _print_question_stage_results(summary: Dict[str, Any]) -> None:
    """流水线全部 action 跑完后，按 ACTION_PIPELINE 顺序打印各阶段与 GT 是否一致。"""
    qid = summary.get("question_id", "?")
    gt = summary.get("ground_truth")
    print(f"\n[本题各阶段结果] id={qid}  GT={gt!r}")
    for action in ACTION_PIPELINE:
        entry = summary.get(action)
        if entry is None:
            print(f"  {action}: skipped")
            continue
        if action in (
            "exchange1_multi_k",
            "exchange2_multi_k",
            "exchange_bidirectional_1_multi_k",
            "exchange_bidirectional_2_multi_k",
        ):
            # 多 k 实验：打印各 k 的 majority_answer 及与 GT 的对比
            summary_table = entry.get("summary_table", [])
            print(f"  {action}:")
            for k, maj in summary_table:
                match = is_correct_answer(maj, gt, is_math=CONFIG.is_math) if maj and gt else False
                mark = "correct" if match else "incorrect"
                print(f"    k={k}: majority={maj!r}  [{mark}]")
        elif isinstance(entry, dict) and "match_gt" in entry:
            label = "correct" if entry["match_gt"] else "incorrect"
            print(f"  {action}: {label}")
        else:
            print(f"  {action}: skipped")


def load_batch_question_items(cfg: Any) -> List[Dict[str, Any]]:
    """从 {model}/data/{task}.json 读取 examples，过滤方式与 expand_load_question 中 is_hard 一致。"""
    path = f"{cfg.model_name}/data/{cfg.task_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["examples"]

    if cfg.is_hard:
        hard_id = {str(i) for i in range(1, 101)}
        data = [d for d in data if str(d.get("question_id")) in hard_id]

    if BATCH_QUESTION_IDS is not None:
        want = {str(x) for x in BATCH_QUESTION_IDS}
        data = [d for d in data if str(d.get("question_id")) in want]

    if BATCH_MAX_QUESTIONS is not None:
        data = data[: int(BATCH_MAX_QUESTIONS)]

    return data


async def run_question_pipeline(
    question_item: Optional[Dict[str, Any]],
    stats: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    对单道题执行 ACTION_PIPELINE；更新 stats中各阶段 attempted/correct，并打印本题摘要与累计正确率。
    question_item 为 None 时与原先单题行为一致（随机或 fixed_question_id）。
    """
    summary: Dict[str, Any] = {
        "actions": list(ACTION_PIPELINE),
        "expand": None,
        "exchange1": None,
        "exchange1_multi_k": None,
        "exchange2_multi_k": None,
        "exchange2": None,
        "exchange_bidirectional_1": None,
        "exchange_bidirectional_1_multi_k": None,
        "exchange_bidirectional_2": None,
        "exchange_bidirectional_2_multi_k": None,
        "ground_truth": None,
        "question_id": str(question_item.get("question_id", "?")) if question_item else None,
    }

    if question_item is not None and question_item.get("question_id") is not None:
        run_qid = str(question_item["question_id"])
    else:
        run_qid = "?"

    expand_pack = None
    snapshot_expand_pack: Any = None

    for action in ACTION_PIPELINE:
        if action == "expand":
            if not _quiet():
                print("\n" + "#" * 72)
                print("# 动作: expand")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    expand_pack = await run_expand_pipeline(
                        cfg=CONFIG,
                        question_item=question_item,
                        save_cache=True,
                        return_vectorization_data=True,
                        verify_vectorization=VERIFY_VECTORIZATION,
                        save_debug_files=SAVE_DEBUG_FILES,
                    )
            else:
                expand_pack = await run_expand_pipeline(
                    cfg=CONFIG,
                    question_item=question_item,
                    save_cache=True,
                    return_vectorization_data=True,
                    verify_vectorization=VERIFY_VECTORIZATION,
                    save_debug_files=SAVE_DEBUG_FILES,
                )

            if expand_pack is None:
                print("[错误] expand 未返回数据，本题后续动作跳过")
                _bump_stage(stats, "expand", False, run_qid)
                _print_cumulative_stage("expand", stats)
                summary["error"] = "expand_failed"
                return summary

            (
                _sv,
                _si,
                _steps,
                _ctx,
                expand_majority,
                ground_truth,
                _question,
                _question_id,
            ) = expand_pack
            snapshot_expand_pack = copy.deepcopy(expand_pack)
            run_qid = str(_question_id)
            summary["ground_truth"] = ground_truth
            summary["question_id"] = run_qid
            match_gt = (
                is_correct_answer(expand_majority, ground_truth, is_math=CONFIG.is_math)
                if expand_majority and ground_truth
                else False
            )
            summary["expand"] = {
                "majority_answer": expand_majority,
                "match_gt": match_gt,
            }
            print(
                f"\n[摘要·expand] id={_question_id} majority={expand_majority!r}  "
                f"与 GT 一致: {match_gt}  GT={ground_truth!r}"
            )
            _bump_stage(stats, "expand", match_gt, run_qid)
            _print_cumulative_stage("expand", stats)

        elif action == "exchange1":
            if expand_pack is None:
                print("[错误] exchange1 需要先有 expand 结果，本题跳过")
                return summary
            step_vectors, step_indices, all_steps, agent_contexts, _, ground_truth, question, question_id = expand_pack
            if step_vectors is None or len(step_vectors) < 2:
                n = step_vectors.shape[0] if step_vectors is not None else 0
                print(f"[错误] exchange1 需要至少 2 个 step 向量，当前 {n}，本题跳过 exchange")
                summary["error"] = "exchange1_skip_few_steps"
                return summary

            if not _quiet():
                print("\n" + "#" * 72)
                print("# 动作: exchange1（聚类 + 排序 + exchange prompt + 新回复）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    out = await run_exchange1_from_expand_outputs(
                        step_vectors,
                        step_indices,
                        all_steps,
                        agent_contexts,
                        use_method=EXCHANGE1_CLUSTER_METHOD,
                        round_num=1,
                    )
            else:
                out = await run_exchange1_from_expand_outputs(
                    step_vectors,
                    step_indices,
                    all_steps,
                    agent_contexts,
                    use_method=EXCHANGE1_CLUSTER_METHOD,
                    round_num=1,
                )
            exchange1_maj = out.get("majority_answer")
            match_gt = (
                is_correct_answer(exchange1_maj, ground_truth, is_math=CONFIG.is_math)
                if exchange1_maj and ground_truth
                else False
            )
            summary["exchange1"] = {
                "majority_answer": exchange1_maj,
                "match_gt": match_gt,
            }
            print(
                f"\n[摘要·exchange1] id={question_id} majority={exchange1_maj!r}  "
                f"与 GT 一致: {match_gt}  GT={ground_truth!r}"
            )
            _bump_stage(stats, "exchange1", match_gt, str(question_id))
            _print_cumulative_stage("exchange1", stats)

            exchange1_cache_path = get_cache_path(CONFIG, "exchange1")
            if _quiet():
                with _suppress_stdout():
                    expand_save_cache_if_enabled(
                        save_cache=True,
                        question=question,
                        agent_contexts=agent_contexts,
                        ground_truth=ground_truth,
                        question_id=question_id,
                        cache_path=exchange1_cache_path,
                        cfg=CONFIG,
                    )
            else:
                expand_save_cache_if_enabled(
                    save_cache=True,
                    question=question,
                    agent_contexts=agent_contexts,
                    ground_truth=ground_truth,
                    question_id=question_id,
                    cache_path=exchange1_cache_path,
                    cfg=CONFIG,
                )

        elif action == "exchange2":
            if summary["exchange1"] is None:
                print("[错误] exchange2 依赖 exchange1，本题跳过")
                return summary
            if expand_pack is None:
                print("[错误] 缺少 expand 数据包，本题跳过 exchange2")
                return summary
            _sv2, _si2, _st2, agent_contexts, _, ground_truth, question, question_id = expand_pack

            if not _quiet():
                print("\n" + "#" * 72)
                print("# 动作: exchange2（基于 exchange1 最新回复重建向量 + 聚类 exchange）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    out2 = await run_exchange2_from_exchange1_outputs(
                        agent_contexts,
                        CONFIG,
                        use_method=EXCHANGE1_CLUSTER_METHOD,
                        round_num=2,
                    )
            else:
                out2 = await run_exchange2_from_exchange1_outputs(
                    agent_contexts,
                    CONFIG,
                    use_method=EXCHANGE1_CLUSTER_METHOD,
                    round_num=2,
                )
            exchange2_maj = out2.get("majority_answer")
            match_gt = (
                is_correct_answer(exchange2_maj, ground_truth, is_math=CONFIG.is_math)
                if exchange2_maj and ground_truth
                else False
            )
            summary["exchange2"] = {
                "majority_answer": exchange2_maj,
                "match_gt": match_gt,
            }
            print(
                f"\n[摘要·exchange2] id={question_id} majority={exchange2_maj!r}  "
                f"与 GT 一致: {match_gt}  GT={ground_truth!r}"
            )
            _bump_stage(stats, "exchange2", match_gt, str(question_id))
            _print_cumulative_stage("exchange2", stats)

            exchange2_cache_path = get_cache_path(CONFIG, "exchange2")
            if _quiet():
                with _suppress_stdout():
                    expand_save_cache_if_enabled(
                        save_cache=True,
                        question=question,
                        agent_contexts=agent_contexts,
                        ground_truth=ground_truth,
                        question_id=question_id,
                        cache_path=exchange2_cache_path,
                        cfg=CONFIG,
                    )
            else:
                expand_save_cache_if_enabled(
                    save_cache=True,
                    question=question,
                    agent_contexts=agent_contexts,
                    ground_truth=ground_truth,
                    question_id=question_id,
                    cache_path=exchange2_cache_path,
                    cfg=CONFIG,
                )

        elif action == "exchange_bidirectional_1":
            if snapshot_expand_pack is None:
                print("[错误] exchange_bidirectional_1 需要先有 expand 快照，本题跳过")
                return summary
            step_vectors, step_indices, all_steps, agent_contexts, _, ground_truth, question, question_id = (
                snapshot_expand_pack
            )
            if step_vectors is None or len(step_vectors) < 2:
                n = step_vectors.shape[0] if step_vectors is not None else 0
                print(f"[错误] exchange_bidirectional_1 需要至少 2 个 step 向量，当前 {n}，本题跳过")
                summary["error"] = "exchange_bidirectional_1_skip_few_steps"
                return summary

            if not _quiet():
                print("\n" + "#" * 72)
                print("# 动作: exchange_bidirectional_1（双向标签修正 + 聚类 exchange）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    out_bd1 = await run_exchange_bidirectional_1_from_expand_outputs(
                        step_vectors,
                        step_indices,
                        all_steps,
                        agent_contexts,
                        use_method=EXCHANGE1_CLUSTER_METHOD,
                        round_num=1,
                    )
            else:
                out_bd1 = await run_exchange_bidirectional_1_from_expand_outputs(
                    step_vectors,
                    step_indices,
                    all_steps,
                    agent_contexts,
                    use_method=EXCHANGE1_CLUSTER_METHOD,
                    round_num=1,
                )
            bd1_maj = out_bd1.get("majority_answer")
            match_gt = (
                is_correct_answer(bd1_maj, ground_truth, is_math=CONFIG.is_math)
                if bd1_maj and ground_truth
                else False
            )
            summary["exchange_bidirectional_1"] = {
                "majority_answer": bd1_maj,
                "match_gt": match_gt,
            }
            print(
                f"\n[摘要·exchange_bidirectional_1] id={question_id} majority={bd1_maj!r}  "
                f"与 GT 一致: {match_gt}  GT={ground_truth!r}"
            )
            _bump_stage(stats, "exchange_bidirectional_1", match_gt, str(question_id))
            _print_cumulative_stage("exchange_bidirectional_1", stats)

            bd1_cache_path = get_cache_path(CONFIG, "exchange_bidirectional_1")
            if _quiet():
                with _suppress_stdout():
                    expand_save_cache_if_enabled(
                        save_cache=True,
                        question=question,
                        agent_contexts=agent_contexts,
                        ground_truth=ground_truth,
                        question_id=question_id,
                        cache_path=bd1_cache_path,
                        cfg=CONFIG,
                    )
            else:
                expand_save_cache_if_enabled(
                    save_cache=True,
                    question=question,
                    agent_contexts=agent_contexts,
                    ground_truth=ground_truth,
                    question_id=question_id,
                    cache_path=bd1_cache_path,
                    cfg=CONFIG,
                )

        elif action == "exchange_bidirectional_2":
            if summary["exchange_bidirectional_1"] is None:
                print("[错误] exchange_bidirectional_2 依赖 exchange_bidirectional_1，本题跳过")
                return summary
            if snapshot_expand_pack is None:
                print("[错误] 缺少 expand 快照，本题跳过 exchange_bidirectional_2")
                return summary
            _svb, _sib, _stb, agent_contexts, _, ground_truth, question, question_id = snapshot_expand_pack

            if not _quiet():
                print("\n" + "#" * 72)
                print("# 动作: exchange_bidirectional_2（双向标签修正，基于 bidirectional_1 后回复）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    out_bd2 = await run_exchange_bidirectional_2_from_bidirectional_1_outputs(
                        agent_contexts,
                        CONFIG,
                        use_method=EXCHANGE1_CLUSTER_METHOD,
                        round_num=2,
                    )
            else:
                out_bd2 = await run_exchange_bidirectional_2_from_bidirectional_1_outputs(
                    agent_contexts,
                    CONFIG,
                    use_method=EXCHANGE1_CLUSTER_METHOD,
                    round_num=2,
                )
            bd2_maj = out_bd2.get("majority_answer")
            match_gt = (
                is_correct_answer(bd2_maj, ground_truth, is_math=CONFIG.is_math)
                if bd2_maj and ground_truth
                else False
            )
            summary["exchange_bidirectional_2"] = {
                "majority_answer": bd2_maj,
                "match_gt": match_gt,
            }
            print(
                f"\n[摘要·exchange_bidirectional_2] id={question_id} majority={bd2_maj!r}  "
                f"与 GT 一致: {match_gt}  GT={ground_truth!r}"
            )
            _bump_stage(stats, "exchange_bidirectional_2", match_gt, str(question_id))
            _print_cumulative_stage("exchange_bidirectional_2", stats)

            bd2_cache_path = get_cache_path(CONFIG, "exchange_bidirectional_2")
            if _quiet():
                with _suppress_stdout():
                    expand_save_cache_if_enabled(
                        save_cache=True,
                        question=question,
                        agent_contexts=agent_contexts,
                        ground_truth=ground_truth,
                        question_id=question_id,
                        cache_path=bd2_cache_path,
                        cfg=CONFIG,
                    )
            else:
                expand_save_cache_if_enabled(
                    save_cache=True,
                    question=question,
                    agent_contexts=agent_contexts,
                    ground_truth=ground_truth,
                    question_id=question_id,
                    cache_path=bd2_cache_path,
                    cfg=CONFIG,
                )
        elif action == "exchange1_multi_k":
            if expand_pack is None:
                print("[错误] exchange1_multi_k 需要先有 expand 结果，本题跳过")
                return summary
            step_vectors, step_indices, all_steps, agent_contexts, _, ground_truth, question, question_id = expand_pack
            if step_vectors is None or len(step_vectors) < 2:
                n = step_vectors.shape[0] if step_vectors is not None else 0
                print(f"[错误] exchange1_multi_k 需要至少 2 个 step 向量，当前 {n}，本题跳过")
                summary["error"] = "exchange1_multi_k_skip_few_steps"
                return summary

            if not _quiet():
                print("\n" + "#" * 72)
                print(f"# 动作: exchange1_multi_k（遍历 k={KMEANS_EXPERIMENT_K_VALUES}）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    multi_k_out = await run_exchange1_multi_k_from_expand_outputs(
                        step_vectors,
                        step_indices,
                        all_steps,
                        agent_contexts,
                        k_values=KMEANS_EXPERIMENT_K_VALUES,
                        round_num=1,
                    )
            else:
                multi_k_out = await run_exchange1_multi_k_from_expand_outputs(
                    step_vectors,
                    step_indices,
                    all_steps,
                    agent_contexts,
                    k_values=KMEANS_EXPERIMENT_K_VALUES,
                    round_num=1,
                )

            summary["exchange1_multi_k"] = multi_k_out

            # 打印各 k 摘要
            print(f"\n[摘要·exchange1_multi_k] id={question_id}  GT={ground_truth!r}")
            for k, maj in multi_k_out.get("summary_table", []):
                match_gt = (
                    is_correct_answer(maj, ground_truth, is_math=CONFIG.is_math)
                    if maj and ground_truth
                    else False
                )
                mark = "correct" if match_gt else "incorrect"
                print(f"  k={k}: majority={maj!r}  [{mark}]")

            # 为 stats 统计：取 majority_answer 出现次数最多的 k 作为代表（或可按需调整）
            from collections import Counter as _Counter
            maj_votes = [maj for _, maj in multi_k_out.get("summary_table", []) if maj]
            rep_majority = _Counter(maj_votes).most_common(1)[0][0] if maj_votes else None
            rep_match_gt = (
                is_correct_answer(rep_majority, ground_truth, is_math=CONFIG.is_math)
                if rep_majority and ground_truth
                else False
            )
            _bump_stage(stats, "exchange1_multi_k", rep_match_gt, str(question_id))
            _print_cumulative_stage("exchange1_multi_k", stats)

        elif action == "exchange2_multi_k":
            # ── 前置检查：仅依赖 exchange1_multi_k 的输出 ──────────────────
            # exchange2_multi_k 完全基于 round1 各 k 的 contexts 工作，
            # 重新向量化在 run_exchange2_multi_k_from_multi_k1_outputs 内部完成，
            # 因此不需要 expand_pack（ground_truth/question_id 直接从 summary 读）
            if summary["exchange1_multi_k"] is None:
                print("[错误] exchange2_multi_k 依赖 exchange1_multi_k，请先在 ACTION_PIPELINE 中配置并运行 exchange1_multi_k")
                return summary

            # ── 取 round1 的 k_results（按 k 索引的 contexts 和 majority） ──
            # 关键：k_results[k]["agent_contexts"] 是 round1 用 k 聚类后，
            #       每个 agent 已追加了 round1 回复的 contexts 深拷贝
            r1_k_results = summary["exchange1_multi_k"]["k_results"]
            ground_truth = summary.get("ground_truth")
            question_id = summary.get("question_id", run_qid)

            if not _quiet():
                print("\n" + "#" * 72)
                print(f"# 动作: exchange2_multi_k（同 k 配对，k={KMEANS_EXPERIMENT_K_VALUES}）")
                print(f"#   向量来源: Round1 各 k 对应 contexts 的最新回复（context[-1]）")
                print(f"#   聚类 k:   与 Round1 严格一一对应（k1 == k2）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    multi_k2_out = await run_exchange2_multi_k_from_multi_k1_outputs(
                        r1_k_results,
                        CONFIG,
                        k_values=KMEANS_EXPERIMENT_K_VALUES,
                        round_num=2,
                    )
            else:
                multi_k2_out = await run_exchange2_multi_k_from_multi_k1_outputs(
                    r1_k_results,
                    CONFIG,
                    k_values=KMEANS_EXPERIMENT_K_VALUES,
                    round_num=2,
                )

            summary["exchange2_multi_k"] = multi_k2_out

            # ── 打印各 k 的 Round2 结果摘要（与 GT 对比） ──────────────────
            print(f"\n[摘要·exchange2_multi_k] id={question_id}  GT={ground_truth!r}")
            r1_table = {k: maj for k, maj in summary["exchange1_multi_k"].get("summary_table", [])}
            for k, maj_r2 in multi_k2_out.get("summary_table", []):
                maj_r1 = r1_table.get(k)
                match_r2 = (
                    is_correct_answer(maj_r2, ground_truth, is_math=CONFIG.is_math)
                    if maj_r2 and ground_truth else False
                )
                mark_r2 = "correct" if match_r2 else "incorrect"
                # 同时打出 round1 该 k 的结果，方便对比两轮变化
                match_r1 = (
                    is_correct_answer(maj_r1, ground_truth, is_math=CONFIG.is_math)
                    if maj_r1 and ground_truth else False
                )
                mark_r1 = "correct" if match_r1 else "incorrect"
                changed = "→ 变化" if match_r1 != match_r2 else "  持平"
                print(
                    f"  k={k}: Round1={maj_r1!r}[{mark_r1}]  "
                    f"Round2={maj_r2!r}[{mark_r2}]  {changed}"
                )

            # ── stats 统计：取各 k 的 Round2 majority 中出现最多的作为代表 ──
            from collections import Counter as _Counter2
            maj_votes2 = [maj for _, maj in multi_k2_out.get("summary_table", []) if maj]
            rep_majority2 = _Counter2(maj_votes2).most_common(1)[0][0] if maj_votes2 else None
            rep_match_gt2 = (
                is_correct_answer(rep_majority2, ground_truth, is_math=CONFIG.is_math)
                if rep_majority2 and ground_truth else False
            )
            _bump_stage(stats, "exchange2_multi_k", rep_match_gt2, str(question_id))
            _print_cumulative_stage("exchange2_multi_k", stats)

        elif action == "exchange_bidirectional_1_multi_k":
            if expand_pack is None:
                print("[错误] exchange_bidirectional_1_multi_k 需要先有 expand 结果，本题跳过")
                return summary
            # 与 exchange1_multi_k 完全相同的数据来源；这里直接读取 snapshot_expand_pack，
            # 与 exchange_bidirectional_1 保持一致（避免被 exchange1 系列污染）
            if snapshot_expand_pack is None:
                print("[错误] exchange_bidirectional_1_multi_k 需要 expand 快照，本题跳过")
                return summary
            step_vectors, step_indices, all_steps, agent_contexts, _, ground_truth, question, question_id = (
                snapshot_expand_pack
            )
            if step_vectors is None or len(step_vectors) < 2:
                n = step_vectors.shape[0] if step_vectors is not None else 0
                print(f"[错误] exchange_bidirectional_1_multi_k 需要至少 2 个 step 向量，当前 {n}，本题跳过")
                summary["error"] = "exchange_bidirectional_1_multi_k_skip_few_steps"
                return summary

            if not _quiet():
                print("\n" + "#" * 72)
                print(f"# 动作: exchange_bidirectional_1_multi_k（双向修正 + 遍历 k={KMEANS_EXPERIMENT_K_VALUES}）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    bd1_multi_k_out = await run_exchange_bidirectional_1_multi_k_from_expand_outputs(
                        step_vectors,
                        step_indices,
                        all_steps,
                        agent_contexts,
                        k_values=KMEANS_EXPERIMENT_K_VALUES,
                        round_num=1,
                    )
            else:
                bd1_multi_k_out = await run_exchange_bidirectional_1_multi_k_from_expand_outputs(
                    step_vectors,
                    step_indices,
                    all_steps,
                    agent_contexts,
                    k_values=KMEANS_EXPERIMENT_K_VALUES,
                    round_num=1,
                )

            summary["exchange_bidirectional_1_multi_k"] = bd1_multi_k_out

            print(f"\n[摘要·exchange_bidirectional_1_multi_k] id={question_id}  GT={ground_truth!r}")
            for k, maj in bd1_multi_k_out.get("summary_table", []):
                match_gt = (
                    is_correct_answer(maj, ground_truth, is_math=CONFIG.is_math)
                    if maj and ground_truth else False
                )
                mark = "correct" if match_gt else "incorrect"
                print(f"  k={k}: majority={maj!r}  [{mark}]")

            from collections import Counter as _CounterBd1
            maj_votes_bd1 = [maj for _, maj in bd1_multi_k_out.get("summary_table", []) if maj]
            rep_majority_bd1 = _CounterBd1(maj_votes_bd1).most_common(1)[0][0] if maj_votes_bd1 else None
            rep_match_gt_bd1 = (
                is_correct_answer(rep_majority_bd1, ground_truth, is_math=CONFIG.is_math)
                if rep_majority_bd1 and ground_truth else False
            )
            _bump_stage(stats, "exchange_bidirectional_1_multi_k", rep_match_gt_bd1, str(question_id))
            _print_cumulative_stage("exchange_bidirectional_1_multi_k", stats)

        elif action == "exchange_bidirectional_2_multi_k":
            # ── 前置检查：仅依赖 exchange_bidirectional_1_multi_k 的输出 ──
            if summary["exchange_bidirectional_1_multi_k"] is None:
                print(
                    "[错误] exchange_bidirectional_2_multi_k 依赖 exchange_bidirectional_1_multi_k，"
                    "请先在 ACTION_PIPELINE 中配置并运行它"
                )
                return summary

            # 关键：取 round1 双向修正版的 k_results，保证两轮策略一致 + k 严格配对
            r1_bd_k_results = summary["exchange_bidirectional_1_multi_k"]["k_results"]
            ground_truth = summary.get("ground_truth")
            question_id = summary.get("question_id", run_qid)

            if not _quiet():
                print("\n" + "#" * 72)
                print(f"# 动作: exchange_bidirectional_2_multi_k（同 k 配对，k={KMEANS_EXPERIMENT_K_VALUES}）")
                print(f"#   向量来源: Round1 各 k 对应 contexts 的最新回复（context[-1]）")
                print(f"#   聚类 k:   与 Round1 严格一一对应（k1 == k2）")
                print(f"#   修正策略: 双向（与 Round1 保持一致）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    bd2_multi_k_out = await run_exchange_bidirectional_2_multi_k_from_multi_k1_outputs(
                        r1_bd_k_results,
                        CONFIG,
                        k_values=KMEANS_EXPERIMENT_K_VALUES,
                        round_num=2,
                    )
            else:
                bd2_multi_k_out = await run_exchange_bidirectional_2_multi_k_from_multi_k1_outputs(
                    r1_bd_k_results,
                    CONFIG,
                    k_values=KMEANS_EXPERIMENT_K_VALUES,
                    round_num=2,
                )

            summary["exchange_bidirectional_2_multi_k"] = bd2_multi_k_out

            print(f"\n[摘要·exchange_bidirectional_2_multi_k] id={question_id}  GT={ground_truth!r}")
            r1_table_bd = {
                k: maj
                for k, maj in summary["exchange_bidirectional_1_multi_k"].get("summary_table", [])
            }
            for k, maj_r2 in bd2_multi_k_out.get("summary_table", []):
                maj_r1 = r1_table_bd.get(k)
                match_r2 = (
                    is_correct_answer(maj_r2, ground_truth, is_math=CONFIG.is_math)
                    if maj_r2 and ground_truth else False
                )
                mark_r2 = "correct" if match_r2 else "incorrect"
                match_r1 = (
                    is_correct_answer(maj_r1, ground_truth, is_math=CONFIG.is_math)
                    if maj_r1 and ground_truth else False
                )
                mark_r1 = "correct" if match_r1 else "incorrect"
                changed = "→ 变化" if match_r1 != match_r2 else "  持平"
                print(
                    f"  k={k}: Round1={maj_r1!r}[{mark_r1}]  "
                    f"Round2={maj_r2!r}[{mark_r2}]  {changed}"
                )

            from collections import Counter as _CounterBd2
            maj_votes_bd2 = [maj for _, maj in bd2_multi_k_out.get("summary_table", []) if maj]
            rep_majority_bd2 = _CounterBd2(maj_votes_bd2).most_common(1)[0][0] if maj_votes_bd2 else None
            rep_match_gt_bd2 = (
                is_correct_answer(rep_majority_bd2, ground_truth, is_math=CONFIG.is_math)
                if rep_majority_bd2 and ground_truth else False
            )
            _bump_stage(stats, "exchange_bidirectional_2_multi_k", rep_match_gt_bd2, str(question_id))
            _print_cumulative_stage("exchange_bidirectional_2_multi_k", stats)

        else:
            print(f"[警告] 未知动作 {action!r}，已跳过")

    _print_question_stage_results(summary)
    if _quiet():
        print(f"[本题流水线结束] id={summary.get('question_id')}")
    else:
        print("\n" + "=" * 72)
        print("[本题流水线结束]", summary)
        print("=" * 72)
    return summary


async def main_math() -> Optional[Dict[str, Any]]:
    """单题或批量：返回 summary 或包含 batch 列表与 stats 的字典。"""
    stats = _init_stage_stats()

    if RUN_BATCH:
        items = load_batch_question_items(CONFIG)
        if not items:
            print("[错误] 批量列表为空，请检查 BATCH_QUESTION_IDS / 数据路径 / is_hard 过滤")
            return None
        print(f"\n[批量] 共 {len(items)} 道题，ACTION_PIPELINE={ACTION_PIPELINE}")
        summaries: List[Optional[Dict[str, Any]]] = []
        for idx, item in enumerate(items):
            qid = item.get("question_id", "?")
            print(f"\n{'#' * 72}\n# 批量进度 {idx + 1}/{len(items)}  question_id={qid}\n{'#' * 72}")
            summ = await run_question_pipeline(item, stats)
            summaries.append(summ)
        _print_batch_summary(stats)
        return {
            "batch": True,
            "n_questions": len(items),
            "stats": stats,
            "summaries": summaries,
        }

    summ = await run_question_pipeline(None, stats)
    _print_batch_summary(stats)
    return summ


if __name__ == "__main__":
    asyncio.run(main_math())
