"""
Step 聚类脚本

功能：
1. 调用 wzy_multi_agent_debate_expand 获取向量化后的 step 数据
2. 对高维向量进行 PCA 降维（4096 -> 128 维）
3. 使用固定 k（遍历 KMEANS_K_MIN ~ KMEANS_K_MAX）分别跑完整 KMeans 流程，
   以及 DBSCAN 聚类，验证不同聚类数量对 step 标签修改结果的影响
4. 根据每个聚类中正确 step 的比例（majority_threshold=0.6）对类簇打标签：
   - correct_ratio >= 0.6 -> 该类簇标记为正确
   - wrong_ratio >= 0.6 -> 该类簇标记为错误
   - 否则 -> 无明确多数，不标记
5. 根据聚类标签更新 step 标签：聚类标签为正确时，将该聚类内所有 step 置为正确；为错误时保持原标签
6. 打印每个 k 的详细结果，以及最终汇总对比表

公共工具函数（供 exchange.py 导入使用）：
- extract_answer_from_text  : 从单条回复中提取答案（严格参照 expand.py 实现）
- is_correct_answer         : 判断预测答案是否与参考答案一致（严格参照 expand.py 实现）
- get_majority_answer_from_contexts  : 从 expand 阶段 agent_contexts[2] 多数投票（参照 expand.py）
- get_majority_answer_from_latest    : 从任意轮次 context[-1] 多数投票（供多轮 exchange 使用）
"""

import sys
import asyncio
import copy
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

# 答案处理依赖（与 expand.py 完全一致）
from common.math_equivalence import strip_string
from eval_all_round import (
    parse_answer, solve_math_problems, parse_math_anser, parse_YN, most_frequent,
    parse_answer_fallback,
)

# 聚类与降维依赖
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 配置
MAJORITY_THRESHOLD = 0.6   # 聚类中多数投票的阈值：正确/错误占比 >= 60% 时生效
TARGET_DIM = 128           # PCA 降维目标维度
DBSCAN_EPS = 0.9           # DBSCAN 的 eps 参数
DBSCAN_MIN_SAMPLES = 2     # DBSCAN 的 min_samples 参数
KMEANS_K_MIN = 3           # 固定 k 实验的起始值（含）
KMEANS_K_MAX = 10          # 固定 k 实验的终止值（含）

# IS_MATH: math_500_id 等数学数据集设为 True；BBH 选项格式任务设为 False
IS_MATH = True


# ============================================================
# 公共工具函数（严格参照 wzy_multi_agent_debate_expand.py 实现）
# ============================================================

def extract_answer_from_text(text: str, is_math: bool = IS_MATH):
    """从单个 agent 的回复中提取答案。

    与 expand.py 的 extract_answer_from_text 完全一致：
    is_math=True 三级提取链：
      1. parse_math_anser       → \\boxed{...}（最显式）
      2. parse_answer_fallback  → "The answer is: ..."（模型明确陈述答案）
      3. solve_math_problems    → 括号整数 (-?\\d+)（纯格式猜测，兜底）
    is_math=False：parse_answer → solve_math_problems → parse_YN
    """
    if not text:
        return None
    if is_math:
        pred_answer = parse_math_anser(text)
        if pred_answer is not None:
            return strip_string(pred_answer)
        pred_answer = parse_answer_fallback(text)
        if pred_answer is not None:
            return pred_answer
        pred_answer = solve_math_problems(text)
        if pred_answer is not None:
            return pred_answer
        return None
    else:
        pred_answer = parse_answer(text)
        if pred_answer is None:
            pred_answer = solve_math_problems(text)
        if pred_answer is None:
            pred_answer = parse_YN(text)
        return pred_answer


def is_correct_answer(pred, ref: str, is_math: bool = IS_MATH) -> bool:
    """判断预测答案是否与标准答案一致。

    逻辑完全参照 expand.py 中 is_correct_answer：
      - is_math=True : strip_string(ref) == pred  （pred 已经 strip_string 过）
      - is_math=False: ref == pred
    """
    if pred is None or ref is None:
        return False
    if is_math:
        return strip_string(ref) == pred
    else:
        return ref == pred


def get_majority_answer_from_contexts(agent_contexts: list, is_math: bool = IS_MATH):
    """从各 agent 的 expand 回复（上下文索引 2）多数投票得出参考答案。

    与 eval_all_round.compute_accuracy 列表分支及 expand.get_majority_answer_from_expand 一致：
      extract_answer_from_text → most_frequent

    用于：expand 阶段结束后，从 agent_contexts[2] 计算 majority_answer。
    """
    pred_answers = []
    for ctx in agent_contexts:
        if len(ctx) >= 3 and ctx[2].get("role") == "assistant":
            pred_solution = ctx[2].get("content", "")
            pred_answer = extract_answer_from_text(pred_solution, is_math=is_math)
            if pred_answer is not None:
                pred_answers.append(pred_answer)
    if not pred_answers:
        return None
    return most_frequent(pred_answers)


def get_majority_answer_from_latest(agent_contexts: list, is_math: bool = IS_MATH):
    """从各 agent 最新的 assistant 回复（context[-1]）多数投票得出参考答案。

    与 get_majority_answer_from_contexts 的区别：不硬编码 context[2]，
    而是取 context[-1]，可用于任意轮次 exchange 结束后。
    答案提取与投票逻辑与 expand.py 中 get_majority_answer_from_expand 完全一致。

    用于：每轮 exchange 结束后，从最新 assistant 消息计算新的 majority_answer。
    """
    pred_answers = []
    for ctx in agent_contexts:
        if not ctx:
            continue
        last_msg = ctx[-1]
        if last_msg.get("role") != "assistant":
            continue
        pred_solution = last_msg.get("content", "")
        pred_answer = extract_answer_from_text(pred_solution, is_math=is_math)
        if pred_answer is not None:
            pred_answers.append(pred_answer)
    if not pred_answers:
        return None
    return most_frequent(pred_answers)


def _reduce_dimensions_pca(vectors: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """
    使用 PCA 对向量进行降维

    Args:
        vectors: 原始向量矩阵，形状 (N, dim)
        target_dim: 目标维度

    Returns:
        降维后的向量矩阵
    """
    # 样本数少于 2 时不做降维（PCA 至少需要 2 个样本）
    if vectors.shape[0] < 2:
        return vectors
    #对每列做标准化（减均值、除标准差）
    scaler = StandardScaler()
    # 计算实际可用的 PCA 维数 max_dim
    vectors_scaled = scaler.fit_transform(vectors)
    max_dim = min(target_dim, vectors.shape[0] - 1, vectors.shape[1])
    pca = PCA(n_components=max_dim)
    reduced = pca.fit_transform(vectors_scaled)
    print(f"[降维] PCA: {vectors.shape[1]} 维 -> {max_dim} 维，解释方差比: {pca.explained_variance_ratio_.sum():.2%}")
    return reduced


def _select_k_by_silhouette(vectors: np.ndarray) -> tuple[int, np.ndarray]:
    """
    根据轮廓系数选择 KMeans 的最优聚类数 k，并返回对应的聚类标签（避免重复运行 KMeans）

    轮廓系数范围 [-1, 1]，越接近 1 表示聚类效果越好。
    遍历 k=2 到 k_max，选取平均轮廓系数最高的 k。

    Args:
        vectors: 向量矩阵，形状 (N, dim)

    Returns:
        (最优的聚类数 k, 对应的聚类标签数组)
    """
    n_samples = vectors.shape[0]
    if n_samples < 4:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        return 2, labels
    k_max = min(KMEANS_K_MAX, n_samples - 1)
    best_k = 2
    best_score = -1.0
    best_labels = None
    scores = []
    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        try:
            score = silhouette_score(vectors, labels, metric="euclidean")
        except Exception:
            score = -1.0
        scores.append((k, score))
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels  # 保存当前 k 的标签，避免后续重复运行
    print(f"[轮廓系数] 尝试 k=2..{k_max}, 最优 k={best_k} (轮廓系数={best_score:.4f})")
    for k, s in scores:
        mark = " <- 选中" if k == best_k else ""
        print(f"  k={k}: 轮廓系数={s:.4f}{mark}")
    return best_k, best_labels


def _cluster_kmeans(vectors: np.ndarray) -> np.ndarray:
    """KMeans 聚类（k 由轮廓系数自动选择，直接使用选 k 时的聚类结果，不重复运行）"""
    if vectors.shape[0] < 2:
        return np.array([0] * vectors.shape[0])
    n_clusters, labels = _select_k_by_silhouette(vectors)
    print(f"[KMeans] n_samples={vectors.shape[0]}, n_clusters={n_clusters}, 聚类大小: {dict(Counter(labels))}")
    return labels


def _cluster_kmeans_fixed_k(vectors: np.ndarray, k: int) -> np.ndarray:
    """KMeans 聚类（固定聚类数 k，不使用轮廓系数自动选 k）

    Args:
        vectors: 向量矩阵，形状 (N, dim)
        k: 指定的聚类数量

    Returns:
        聚类标签数组，形状 (N,)
    """
    n_samples = vectors.shape[0]
    if n_samples < 2:
        return np.array([0] * n_samples)
    # k 不能超过样本数，自动收紧
    k_actual = min(k, n_samples)
    kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    print(f"[KMeans k={k_actual}] n_samples={n_samples}, 聚类大小: {dict(Counter(labels))}")
    return labels


def _cluster_dbscan(vectors: np.ndarray, eps: float = 0.9, min_samples: int = 2) -> np.ndarray:
    """DBSCAN 聚类"""
    if vectors.shape[0] < 2:
        return np.array([0] * vectors.shape[0])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(vectors)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"[DBSCAN] n_samples={vectors.shape[0]}, n_clusters={n_clusters}, 噪声点={n_noise}")
    return labels


def _label_clusters_by_majority(
    cluster_labels: np.ndarray,
    step_indices: list,
    all_steps: list,
    majority_threshold: float = 0.6,
    method_name: str = "",
) -> dict:
    """
    根据每个聚类中正确 step 的比例，对类簇打标签

    Args:
        cluster_labels: 聚类标签数组，与 step_vectors 一一对应
        step_indices: 有效 step 在 all_steps 中的索引
        all_steps: 原始 step 列表，每个元素含 is_correct
        majority_threshold: 多数投票阈值
        method_name: 方法名（用于打印）

    Returns:
        {cluster_id: "correct" | "wrong" | "no_majority"}
    """
    # cluster_to_indices：{cluster_id: [该聚类内所有向量的下标列表]}
    # 如：cluster_labels = [0, 1, 0, 2]，得到 {0: [0, 2], 1: [1], 2: [3]}
    cluster_to_indices = {}
    for i, cid in enumerate(cluster_labels):
        if cid not in cluster_to_indices:
            cluster_to_indices[cid] = []
        cluster_to_indices[cid].append(i)

    cluster_labels_result = {}
    for cluster_id, vec_indices in cluster_to_indices.items():
        if cluster_id == -1:
            cluster_labels_result[cluster_id] = "noise"
            continue
        correct_count = 0
        wrong_count = 0
        for vec_i in vec_indices:
            step_idx = step_indices[vec_i]
            is_correct = all_steps[step_idx].get("is_correct")
            if is_correct is True:
                correct_count += 1
            elif is_correct is False:
                wrong_count += 1
        total_labeled = correct_count + wrong_count
        if total_labeled < 2:
            cluster_labels_result[cluster_id] = "no_majority"
            continue
        correct_ratio = correct_count / total_labeled
        wrong_ratio = wrong_count / total_labeled
        if correct_ratio >= majority_threshold:
            cluster_labels_result[cluster_id] = "correct"
        elif wrong_ratio >= majority_threshold:
            cluster_labels_result[cluster_id] = "wrong"
        else:
            cluster_labels_result[cluster_id] = "no_majority"
    return cluster_labels_result


def _print_clustering_results(
    cluster_labels: np.ndarray,
    cluster_label_results: dict,
    step_indices: list,
    all_steps: list,
    method_name: str,
):
    """打印聚类结果"""
    print(f"\n{'='*60}")
    print(f"[{method_name}] 聚类结果")
    print("="*60)
    # 打印每个聚类的标签和数量
    """
    unique_clusters = [0, 1, 2]
    """
    unique_clusters = sorted(set(cluster_labels))
    for cid in unique_clusters:
        if cid == -1:
            count = list(cluster_labels).count(-1)
            print(f"  聚类 -1 (噪声): {count} 个 step")
            continue
        label = cluster_label_results.get(cid, "?")
        # 找出属于当前类簇 cid 的向量（step）下标（向量在 step_vectors 中的下标）
        vec_indices = [i for i in range(len(cluster_labels)) if cluster_labels[i] == cid]
        # 统计属于当前类簇 cid 的 step 中正确和错误的数量
        correct_count = sum(1 for vi in vec_indices if all_steps[step_indices[vi]].get("is_correct") is True)
        wrong_count = sum(1 for vi in vec_indices if all_steps[step_indices[vi]].get("is_correct") is False)
        # 统计属于当前类簇 cid 的 step 总数
        total = len(vec_indices)
        # 计算正确率
        ratio_str = f"{correct_count}/{total} 正确" if total > 0 else "0"
        print(f"  聚类 {cid}: {total} 个 step, {ratio_str}, 类簇标签 -> {label}")
        # 打印每个 step 对应的 agent_id 和 step_number
        step_labels = []
        for vi in vec_indices:
            step_idx = step_indices[vi]
            step = all_steps[step_idx]
            agent_id = step.get("agent_id")
            step_number = step.get("step_number")
            is_correct = step.get("is_correct")
            correct_mark = "对" if is_correct else "错"
            step_labels.append(f"Agent{agent_id}-Step{step_number}[{correct_mark}]")
        # 若 step 较多，只展示前 20 个，后面用 ... 省略
        max_show = 20
        if len(step_labels) <= max_show:
            print(f"    包含: {', '.join(step_labels)}")
        else:
            shown = ", ".join(step_labels[:max_show])
            print(f"    包含: {shown}, ... (共 {len(step_labels)} 个)")


def _apply_cluster_labels_to_steps(
    cluster_labels: np.ndarray,
    cluster_label_results: dict,
    step_indices: list,
    all_steps: list,
) -> tuple[list, int, list]:
    """
    根据聚类标签更新 step 的 is_correct 标签

    规则：
    - 若聚类标签为 "correct"（正确占比 >= 60%）：将该聚类内所有 step 的 is_correct 置为 True
    - 若聚类标签为 "wrong"（错误占比 >= 60%）：保持原标签，不修改
    - 若聚类标签为 "no_majority" 或 "noise"：保持原标签，不修改

    Args:
        cluster_labels: 聚类标签数组，与 step_vectors 一一对应
        cluster_label_results: {cluster_id: "correct" | "wrong" | "no_majority" | "noise"}
        step_indices: 有效 step 在 all_steps 中的索引
        all_steps: step 列表（原始数据，不会被修改）

    Returns:
        (修改后的 all_steps 副本, 被修改的 step 数量, 变更记录列表)
        变更记录列表每项格式：
        {
            "cluster_id": int,        # 所属聚类 id
            "cluster_tag": str,       # 聚类标签（correct/wrong/no_majority/noise）
            "agent_id": int,          # agent 编号
            "step_number": int,       # step 序号
            "original_label": bool,   # 修改前的 is_correct 值
            "new_label": bool,        # 修改后的 is_correct 值
            "changed": bool,          # 是否发生了实际改变
        }
    """
    # 深拷贝，避免修改原始数据影响后续 DBSCAN 的聚类标签计算
    steps_copy = copy.deepcopy(all_steps)
    modified_count = 0
    change_records = []  # 记录每个 step 在本次修改中的变更情况

    for vec_i, cid in enumerate(cluster_labels):
        cluster_tag = cluster_label_results.get(cid, "no_majority")
        step_idx = step_indices[vec_i]
        step = steps_copy[step_idx]
        original_label = step.get("is_correct")

        if cluster_tag == "correct":
            # 聚类标签为 "correct"：将该聚类内所有 step 置为正确
            new_label = True
            changed = (original_label is not True)
            if changed:
                step["is_correct"] = True
                modified_count += 1
        else:
            # 聚类标签为 "wrong"、"no_majority"、"noise"：保持原标签，不修改
            new_label = original_label
            changed = False

        # 记录该 step 的变更信息（无论是否发生改变，均记录以便完整打印）
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


def _print_step_label_changes_by_cluster(
    change_records: list,
    method_name: str,
):
    """
    按聚类维度打印每个 step 的：所属聚类、聚类标签、agent_id、step_number、原始标签 -> 修改后标签

    Args:
        change_records: _apply_cluster_labels_to_steps 返回的变更记录列表
        method_name: 聚类方法名（用于打印标题）
    """
    print(f"\n{'='*60}")
    print(f"[{method_name}] 聚类标签修改详情（按聚类分组）")
    print("="*60)

    # 按 cluster_id 分组记录
    cluster_to_records = {}
    for rec in change_records:
        cid = rec["cluster_id"]
        if cid not in cluster_to_records:
            cluster_to_records[cid] = []
        cluster_to_records[cid].append(rec)

    label_str = {True: "对", False: "错", None: "未定"}

    for cid in sorted(cluster_to_records.keys()):
        records = cluster_to_records[cid]
        cluster_tag = records[0]["cluster_tag"]  # 同一聚类内所有记录的 cluster_tag 相同
        print(f"\n  聚类 {cid}  [聚类标签: {cluster_tag}]")
        for rec in sorted(records, key=lambda r: (r["agent_id"], r["step_number"])):
            orig = label_str.get(rec["original_label"], str(rec["original_label"]))
            new  = label_str.get(rec["new_label"], str(rec["new_label"]))
            # 若标签发生改变，用 "-> " 标注；否则标注 "(不变)"
            if rec["changed"]:
                change_note = f"{orig} -> {new}  [已修改]"
            else:
                change_note = f"{orig}  (不变)"
            print(f"    Agent{rec['agent_id']}-Step{rec['step_number']}: {change_note}")

    # 统计本次修改影响的 step 数
    changed_cnt = sum(1 for r in change_records if r["changed"])
    print(f"\n  汇总: 共 {len(change_records)} 个 step，其中 {changed_cnt} 个被修改")
    print("="*60)


def _print_step_labels_after_modification(
    all_steps: list,
    method_name: str,
    modified_count: int,
):
    """
    打印聚类标签修改后，所有 step 的标签展示

    格式：按 agent 分组，展示 Agent X: Step1[对/错], Step2[对/错], ...
    """
    print(f"\n{'='*60}")
    print(f"[{method_name}] 聚类标签修改后的 Step 标签展示")
    print(f"  共修改了 {modified_count} 个 step 的标签（仅 correct 类簇内的 step 会被置为正确）")
    print("="*60)

    # 按 agent_id 分组
    agent_to_steps = {}
    for step in all_steps:
        agent_id = step.get("agent_id", -1)
        if agent_id not in agent_to_steps:
            agent_to_steps[agent_id] = []
        agent_to_steps[agent_id].append(step)

    for agent_id in sorted(agent_to_steps.keys()):
        # 按 step_number 排序，保证展示顺序一致
        steps = sorted(agent_to_steps[agent_id], key=lambda s: s.get("step_number", 0))
        labels = []
        for s in steps:
            correct_mark = "对" if s.get("is_correct") is True else "错"
            step_num = s.get("step_number", "?")
            labels.append(f"Step{step_num}[{correct_mark}]")
        print(f"  Agent {agent_id}: {', '.join(labels)}")

    # 统计修改后的正确/错误数量
    true_cnt = sum(1 for s in all_steps if s.get("is_correct") is True)
    false_cnt = sum(1 for s in all_steps if s.get("is_correct") is False)
    none_cnt = sum(1 for s in all_steps if s.get("is_correct") is None)
    print(f"\n  汇总: 正确={true_cnt}, 错误={false_cnt}, 未定={none_cnt}, 总计={len(all_steps)}")
    print("="*60 + "\n")


async def run_clustering():
    """主入口：执行 expand + 降维 + 两种聚类"""
    if not SKLEARN_AVAILABLE:
        print("[错误] 需要安装 sklearn: pip install scikit-learn")
        return

    print("\n" + "#" * 70)
    print("# Step 聚类 - 使用 expand 向量化数据")
    print("#" * 70)

    # 1. 获取向量化数据（懒加载 expand，避免模块级副作用影响其他导入方）
    from wzy_multi_agent_debate_expand import main as main_expand
    print("\n[聚类] 正在调用 expand 获取向量化数据...")
    result = await main_expand(return_vectorization_data=True)
    if result is None:
        print("\n[错误] expand 流程未返回向量化数据（可能 API 验证失败或题目未找到）")
        return
    print("[聚类] expand 已返回，开始聚类分析")

    # run_single_question_expand 返回 (step_vectors, step_indices, all_steps, agent_contexts, majority_answer, ground_truth)
    # 聚类脚本只需要前三个，后三个用 _ 忽略
    step_vectors, step_indices, all_steps, _agent_contexts, _majority_answer, _ground_truth = result
    if step_vectors is None or len(step_vectors) < 2:
        n_vec = step_vectors.shape[0] if step_vectors is not None else 0
        n_steps = len(all_steps) if all_steps else 0
        print(f"\n[错误] 向量数量不足，无法进行聚类（需要至少 2 个 step）")
        print(f"  向量数: {n_vec}, all_steps 数: {n_steps}")
        return

    if step_indices is None or len(step_indices) != step_vectors.shape[0]:
        print("\n[错误] step_indices 与 step_vectors 不匹配")
        return

    print(f"\n[聚类] 向量形状: {step_vectors.shape}, step 数: {len(step_indices)}")

    # 2. 降维
    vectors_reduced = _reduce_dimensions_pca(step_vectors, target_dim=TARGET_DIM)

    # 3. L2 归一化（与 step_clustering 一致，便于聚类）
    norms = np.linalg.norm(vectors_reduced, axis=1, keepdims=True)
    vectors_reduced = vectors_reduced / (norms + 1e-12)
    print("[预处理] L2 归一化完成")

    # 4. KMeans 聚类：遍历 k = KMEANS_K_MIN .. KMEANS_K_MAX，每个 k 跑完整流程
    kmeans_summary = []   # 每个 k 的汇总信息，供最后打印对比表

    for k in range(KMEANS_K_MIN, KMEANS_K_MAX + 1):
        method_tag = f"KMeans k={k}"
        print("\n" + "=" * 60)
        print(f"[实验] {method_tag}")
        print("=" * 60)

        # 4-1. 用固定 k 运行 KMeans
        labels_k = _cluster_kmeans_fixed_k(vectors_reduced, k)
        actual_k = len(set(labels_k))   # 实际簇数（受样本数限制可能 < k）

        # 4-2. 对每个簇打正确/错误标签
        cluster_label_results_k = _label_clusters_by_majority(
            labels_k, step_indices, all_steps,
            majority_threshold=MAJORITY_THRESHOLD,
            method_name=method_tag,
        )

        # 4-3. 打印聚类结果（每个簇包含哪些 step）
        _print_clustering_results(
            labels_k, cluster_label_results_k,
            step_indices, all_steps, method_tag,
        )

        # 4-4. 根据簇标签更新 step 的 is_correct
        steps_after_k, n_modified_k, changes_k = _apply_cluster_labels_to_steps(
            labels_k, cluster_label_results_k, step_indices, all_steps
        )

        # 4-5. 打印每个簇内 step 的标签变更详情
        _print_step_label_changes_by_cluster(changes_k, method_tag)

        # 4-6. 打印修改后所有 step 的整体标签视图
        _print_step_labels_after_modification(steps_after_k, method_tag, n_modified_k)

        # 统计各簇标签分布，用于汇总表
        tag_counter = Counter(cluster_label_results_k.values())
        kmeans_summary.append({
            "k": k,
            "actual_k": actual_k,
            "n_correct_clusters": tag_counter.get("correct", 0),
            "n_wrong_clusters": tag_counter.get("wrong", 0),
            "n_no_majority_clusters": tag_counter.get("no_majority", 0),
            "n_modified_steps": n_modified_k,
        })

    # 5. DBSCAN 聚类
    print("\n" + "=" * 60)
    print("[实验] DBSCAN")
    print("=" * 60)
    labels_dbscan = _cluster_dbscan(
        vectors_reduced,
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
    )
    cluster_labels_dbscan = _label_clusters_by_majority(
        labels_dbscan, step_indices, all_steps,
        majority_threshold=MAJORITY_THRESHOLD,
        method_name="DBSCAN",
    )
    _print_clustering_results(
        labels_dbscan, cluster_labels_dbscan,
        step_indices, all_steps, "DBSCAN",
    )
    steps_after_dbscan, n_modified_dbscan, changes_dbscan = _apply_cluster_labels_to_steps(
        labels_dbscan, cluster_labels_dbscan, step_indices, all_steps
    )
    _print_step_label_changes_by_cluster(changes_dbscan, "DBSCAN")
    _print_step_labels_after_modification(
        steps_after_dbscan, "DBSCAN", n_modified_dbscan
    )

    # 6. 汇总对比表
    print("\n" + "#" * 70)
    print("# 汇总对比表（KMeans 固定 k 实验）")
    print("#" * 70)
    print(f"  majority_threshold = {MAJORITY_THRESHOLD}  |  总 step 数 = {len(step_indices)}")
    print(f"  聚类标签规则: correct 类簇 -> 其内所有 step 置为正确; 其余保持原标签\n")
    header = f"  {'方法':<18} {'实际簇数':>6} {'correct簇':>9} {'wrong簇':>8} {'无多数簇':>8} {'修改step数':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in kmeans_summary:
        tag = f"KMeans k={row['k']}"
        print(
            f"  {tag:<18} {row['actual_k']:>6} "
            f"{row['n_correct_clusters']:>9} {row['n_wrong_clusters']:>8} "
            f"{row['n_no_majority_clusters']:>8} {row['n_modified_steps']:>10}"
        )
    # DBSCAN 行
    dbscan_tag_cnt = Counter(cluster_labels_dbscan.values())
    dbscan_actual_k = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    print(
        f"  {'DBSCAN':<18} {dbscan_actual_k:>6} "
        f"{dbscan_tag_cnt.get('correct', 0):>9} {dbscan_tag_cnt.get('wrong', 0):>8} "
        f"{dbscan_tag_cnt.get('no_majority', 0):>8} {n_modified_dbscan:>10}"
    )
    print("#" * 70 + "\n")


if __name__ == "__main__":
    print("\n[入口] 正在运行 wzy_multi_agent_debate_clustering.py（聚类脚本）\n")
    asyncio.run(run_clustering())
