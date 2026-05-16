"""
上下文增强 Step 向量化模块

功能：
1. 对每个 step 构造上下文增强的 embedding_text（Previous + Current + Next）。
2. 按 agent_id 分组，每个 agent 内部按 step_number 排序后确定前后 step。
3. 调用现有 StepClusteringRefiner 的 embedding API 批量向量化。
4. 返回值与原有 expand_run_embedding 保持兼容：
   - step_vectors: np.ndarray，形状 (有效 step 数, embedding_dim)
   - step_indices: list[int]，表示每个向量对应 all_steps 的原始下标

集成方式（二选一）：

方式一：在 wzy_multi_agent_debate_expand.py 中直接替换 expand_run_embedding

    from wzy_context_step_vectorization import expand_run_embedding_contextual

    def expand_run_embedding(all_steps, cfg):
        return expand_run_embedding_contextual(all_steps, cfg)

方式二：在 run_expand_pipeline 中局部替换调用

    from wzy_context_step_vectorization import expand_run_embedding_contextual
    step_vectors, step_indices = expand_run_embedding_contextual(all_steps, cfg)

约束：
- 不将 question/task_type/agent_id/step_number 加入 embedding_text。
- agent_id 和 step_number 仅作为 metadata 保留。
- 不改变 all_steps 原有字段顺序。
- 每道题内部按 agent 独立处理。
"""

import sys
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# 复用现有 embedding 工具
from wzy_step_clustering import StepClusteringRefiner


# ============================================================
# 常量
# ============================================================

START_OF_REASONING = "[START_OF_REASONING]"
END_OF_REASONING = "[END_OF_REASONING]"


# ============================================================
# 1. 文本清理
# ============================================================

def clean_step_text(text: str) -> str:
    """
    清理 step 文本。

    - 去掉首尾空白。
    - 将连续多个空白字符（空格、制表符、换行）压缩为单个空格。
    - 保留数学公式、箭头、数字、符号等原始内容。
    - 如果输入为空或 None，返回空字符串。

    Args:
        text: 原始 step 文本。

    Returns:
        清理后的字符串。
    """
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    # 将任意连续空白字符压缩为单个空格
    import re
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# 2. 构造单个 step 的上下文文本
# ============================================================

def build_single_context_text(
    prev_text: str,
    cur_text: str,
    next_text: str,
) -> str:
    """
    构造单个 step 的 embedding_text。

    格式：
        Previous step:
        <prev_text，第一个 step 时为 [START_OF_REASONING]>

        Current step:
        <cur_text>

        Next step:
        <next_text，最后一个 step 时为 [END_OF_REASONING]>

    Args:
        prev_text: 前一个 step 的内容，或 [START_OF_REASONING]（第一个 step）。
        cur_text: 当前 step 的清理后内容。
        next_text: 后一个 step 的内容，或 [END_OF_REASONING]（最后一个 step）。

    Returns:
        拼接后的上下文文本。
    """
    prev = clean_step_text(prev_text)
    cur = clean_step_text(cur_text)
    nxt = clean_step_text(next_text)

    # 注意：保留换行结构，让 embedding 模型能感知段落边界
    lines = [
        "Previous step:",
        prev,
        "",
        "Current step:",
        cur,
        "",
        "Next step:",
        nxt,
    ]
    return "\n".join(lines)


# ============================================================
# 3. 为所有 step 构造上下文文本
# ============================================================

def build_context_step_texts(
    all_steps: List[Dict[str, Any]],
) -> Tuple[List[str], List[int]]:
    """
    为 all_steps 中每个有效 step 构造上下文增强的 embedding_text。

    处理流程：
      1. 按 agent_id 分组。
      2. 每个 agent 内部按 step_number 升序排序。
      3. 对每个 step 取同 agent 的前一个和后一个 step 的 content 作为上下文。
      4. 将构造好的 embedding_text 写回 step["embedding_text"]。
      5. 收集所有有效 step 的 embedding_text 和原始下标。

    Args:
        all_steps: 展平后的 step 列表，每个元素通常包含：
            {
                "agent_id": int,
                "step_number": int,
                "content": str,
                "is_correct": bool,
                ...
            }

    Returns:
        (embedding_texts, step_indices)
        - embedding_texts: list[str]，每个元素对应一个有效 step 的上下文文本。
        - step_indices: list[int]，每个元素是 embedding_texts 中同位置文本在 all_steps 中的原始下标。

    注意：
        - 不改变 all_steps 的原始顺序。
        - content 为空的 step 会被跳过，不会出现在返回列表中。
        - 单 step agent 时 prev=[START_OF_REASONING], next=[END_OF_REASONING]。
    """
    # 按 agent_id 分组收集 (原始下标, step_dict)
    agent_groups: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for raw_idx, step in enumerate(all_steps):
        aid = step.get("agent_id")
        if aid is None:
            continue
        if aid not in agent_groups:
            agent_groups[aid] = []
        agent_groups[aid].append((raw_idx, step))

    embedding_texts: List[str] = []
    step_indices: List[int] = []

    for aid in sorted(agent_groups.keys()):
        group = agent_groups[aid]
        # 按 step_number 升序排序
        group_sorted = sorted(group, key=lambda x: x[1].get("step_number", 0))
        n = len(group_sorted)

        for i, (raw_idx, step) in enumerate(group_sorted):
            cur_content = step.get("content", "")
            if not clean_step_text(cur_content):
                # content 为空，跳过
                continue

            # 第一个 step 的 previous 用固定占位符，最后一个 step 的 next 用固定占位符
            # 单 step agent 时 first=last=0，同时满足两个条件
            if i == 0:
                prev_content = START_OF_REASONING
            else:
                prev_content = group_sorted[i - 1][1].get("content", "")

            if i == n - 1:
                next_content = END_OF_REASONING
            else:
                next_content = group_sorted[i + 1][1].get("content", "")

            embedding_text = build_single_context_text(
                prev_text=prev_content,
                cur_text=cur_content,
                next_text=next_content,
            )

            # 写回原始 step 字典（不改变 all_steps 原始顺序）
            step["embedding_text"] = embedding_text

            embedding_texts.append(embedding_text)
            step_indices.append(raw_idx)

    return embedding_texts, step_indices


# ============================================================
# 4. 上下文增强向量化（直接替换 expand_run_embedding）
# ============================================================

def expand_run_embedding_contextual(
    all_steps: List[Dict[str, Any]],
    cfg,
    *,
    batch_size: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """
    上下文增强版 expand_run_embedding，可直接替换原有函数。

    流程：
      1. 调用 build_context_step_texts(all_steps) 构造 embedding_texts。
      2. 使用现有 StepClusteringRefiner 批量调用 embedding API 向量化。
      3. 返回 step_vectors, step_indices。

    Args:
        all_steps: 展平后的 step 列表。
        cfg: ExpandConfig 实例，需包含 api_url, api_key, embedding_model 等字段。
        batch_size: 批量 embedding 的批次大小；None 时使用 refiner 默认值。

    Returns:
        (step_vectors, step_indices)
        - step_vectors: np.ndarray，形状 (有效 step 数, embedding_dim)。
        - step_indices: list[int]，对应 all_steps 原始下标。
        如果没有有效向量，返回 (np.empty((0, 0)), [])。
    """
    print(f"\n{{'='*80}}")
    print("  [Expand Step 7 - Contextual Embedding] 上下文增强步骤向量化")
    print(f"{{'='*80}}")

    # 构造上下文文本
    embedding_texts, step_indices = build_context_step_texts(all_steps)
    n_valid = len(embedding_texts)
    print(f"\n[上下文构造] 共 {len(all_steps)} 个 step，有效 step {n_valid} 个")

    if n_valid < 2:
        print(f"\n[错误] 有效 step 数 {n_valid} < 2，无法进行向量化")
        print(f"{{'='*80}}")
        return np.empty((0, 0)), []

    # 复用现有 StepClusteringRefiner
    embedding_model = getattr(cfg, "embedding_model", "qwen3-embedding-8b")
    api_url = getattr(cfg, "api_url", "https://api.zhizengzeng.com/v1")
    api_key = getattr(cfg, "api_key", None)

    refiner = StepClusteringRefiner(
        api_url=api_url,
        api_key=api_key,
        vector_method="embedding_api",
        embedding_model=embedding_model,
        reduce_dim=False,          # 本模块只负责向量化，不降维
        batch_size=batch_size if batch_size is not None else 20,
    )

    print(f"\n[向量化] 使用模型: {embedding_model}，批量大小: {refiner.batch_size}")
    print(f"[向量化] 仅使用上下文增强后的 step 文本（不注入题干前缀）")

    print(f"\n[向量化] 直接对 {n_valid} 个 embedding_text 批量调用 API...")
    all_vectors = refiner.get_text_embeddings_batch(
        embedding_texts,
        batch_size=refiner.batch_size,
    )

    # 过滤 API 调用失败返回 None 的项
    valid_vectors = []
    valid_indices = []
    for i, vec in enumerate(all_vectors):
        if vec is not None and len(vec) > 0:
            valid_vectors.append(vec)
            valid_indices.append(step_indices[i])
        else:
            print(f"  [警告] step_index={step_indices[i]} 的 embedding 调用失败，已跳过")

    if not valid_vectors:
        print(f"\n[错误] 所有 embedding 调用均失败，无法生成向量")
        print(f"{{'='*80}}")
        return np.empty((0, 0)), []

    # 统一维度（处理偶发的维度不一致）
    dims = [len(v) for v in valid_vectors]
    from collections import Counter
    dim_counts = Counter(dims)
    target_dim = dim_counts.most_common(1)[0][0]

    unified_vectors = []
    for vec in valid_vectors:
        if len(vec) != target_dim:
            if len(vec) < target_dim:
                vec = np.pad(vec, (0, target_dim - len(vec)), "constant", constant_values=0)
            else:
                vec = vec[:target_dim]
        unified_vectors.append(vec)

    step_vectors = np.array(unified_vectors, dtype=np.float32)
    step_indices = valid_indices

    print(
        f"\n[向量化] 完成: {step_vectors.shape[0]} 个步骤 -> "
        f"{step_vectors.shape[1]} 维向量 (模型: {embedding_model})"
    )
    print(f"{{'='*80}}")
    return step_vectors, step_indices


# ============================================================
# 5. 调试辅助：预览构造后的 embedding_text
# ============================================================

def inspect_context_embedding_texts(
    all_steps: List[Dict[str, Any]],
    max_agents: int = 2,
    max_steps: int = 3,
) -> None:
    """
    打印少量构造后的 embedding_text，方便调试。

    不参与主流程，仅用于人工检查上下文构造是否符合预期。

    Args:
        all_steps: 已调用 build_context_step_texts 后的 step 列表。
        max_agents: 最多展示几个 agent 的 step。
        max_steps: 每个 agent 最多展示几个 step。
    """
    print(f"\n{{'='*70}}")
    print("[调试] 上下文增强 embedding_text 预览")
    print(f"{{'='*70}}")

    agent_groups: Dict[int, List[Dict[str, Any]]] = {}
    for step in all_steps:
        aid = step.get("agent_id")
        if aid is None:
            continue
        if aid not in agent_groups:
            agent_groups[aid] = []
        agent_groups[aid].append(step)

    shown_agents = 0
    for aid in sorted(agent_groups.keys()):
        if shown_agents >= max_agents:
            break
        shown_agents += 1
        steps = sorted(agent_groups[aid], key=lambda s: s.get("step_number", 0))
        print(f"\n--- Agent {aid} (共 {len(steps)} 个 step) ---")
        for i, step in enumerate(steps):
            if i >= max_steps:
                print(f"  ... (还有 {len(steps) - max_steps} 个 step 未展示)")
                break
            emb = step.get("embedding_text", "")
            preview = emb[:300] + "..." if len(emb) > 300 else emb
            print(f"\n  Step {step.get('step_number', '?')}:")
            for line in preview.split("\n"):
                print(f"    {line}")

    print(f"\n{{'='*70}}")


# ============================================================
# 集成说明（供 wzy_multi_agent_debate_expand.py 使用）
# ============================================================
"""
【集成方式】

在 wzy_multi_agent_debate_expand.py 中，选择以下任一方式引入本模块：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
方式一（推荐）：直接替换 expand_run_embedding 函数定义
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在 wzy_multi_agent_debate_expand.py 顶部添加 import：

    from wzy_context_step_vectorization import expand_run_embedding_contextual

然后将原有的 expand_run_embedding 函数体替换为：

    def expand_run_embedding(all_steps, cfg):
        return expand_run_embedding_contextual(all_steps, cfg)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
方式二：在 run_expand_pipeline 中局部替换调用点
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在 run_expand_pipeline 函数中，找到：

    step_vectors, step_indices = expand_run_embedding(all_steps, cfg)

替换为：

    from wzy_context_step_vectorization import expand_run_embedding_contextual
    step_vectors, step_indices = expand_run_embedding_contextual(all_steps, cfg)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【注意事项】

1. 本模块不修改 all_steps 的原始顺序和字段，仅在有效 step 上新增 "embedding_text" 字段。
2. 降维（PCA/UMAP）和聚类（KMeans/HDBSCAN）仍由原有 clustering / exchange 模块负责，
   本模块只负责生成 step_vectors 和 step_indices。
3. 如果某 step content 为空，会被跳过，不会出现在返回的 step_vectors / step_indices 中。
4. 调试时可在 run_expand_pipeline 中调用 inspect_context_embedding_texts(all_steps) 预览结果。
"""
