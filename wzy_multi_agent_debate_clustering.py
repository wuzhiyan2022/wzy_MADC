"""
Step 聚类脚本

功能：
1. 调用 wzy_multi_agent_debate_expand 获取向量化后的 step 数据
2. 对高维向量进行 PCA 降维（4096 -> 128 维）
3. 分别使用 KMeans 和 HDBSCAN 进行聚类（KMeans 的 k 由 HDBSCAN 自动探测确定）
4. 根据每个聚类中正确 step 的比例（majority_threshold=0.6）对类簇打标签：
   - correct_ratio >= 0.6 -> 该类簇标记为正确
   - wrong_ratio >= 0.6 -> 该类簇标记为错误
   - 否则 -> 无明确多数，不标记
5. 根据聚类标签更新 step 标签：聚类标签为正确时，将该聚类内所有 step 置为正确；为错误时保持原标签
6. 打印两种聚类方式的结果对比及修改后的 step 标签

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
from typing import Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

# 答案处理依赖（与 expand.py 完全一致）
from common.math_equivalence import strip_string
from eval_all_round import (
    parse_answer, parse_answer_bbh, solve_math_problems, parse_math_anser, parse_YN, most_frequent,
    parse_answer_fallback,
)

# 聚类与降维依赖
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans  # [已注释] DBSCAN 已移除
    # from sklearn.metrics import silhouette_score  # [已注释] 轮廓系数已移除
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# HDBSCAN 后端：sklearn>=1.3 内置，否则 fallback 到独立 hdbscan 包
HDBSCAN_BACKEND = None
_SK_HDBSCAN = None
_hdbscan_pkg = None
try:
    from sklearn.cluster import HDBSCAN as _SK_HDBSCAN  # type: ignore
    HDBSCAN_BACKEND = "sklearn"
except ImportError:
    try:
        import hdbscan as _hdbscan_pkg  # type: ignore
        HDBSCAN_BACKEND = "hdbscan"
    except ImportError:
        HDBSCAN_BACKEND = None

# UMAP 后端：可选依赖；未安装时调用降维会抛 ImportError
try:
    import umap  # type: ignore
    import warnings
    warnings.filterwarnings("ignore", message="n_jobs value .* overridden.*random_state")
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# 配置
MAJORITY_THRESHOLD = 0.8   # 聚类中多数投票的阈值：正确/错误占比 >= 80% 时生效
TARGET_DIM = 15            # PCA 默认目标维（向后兼容；纯 pca 路径请用 resolve_pca_target_dim）
PCA_TARGET_DIM_SMALL = 15  # step 数 < 80 时的 PCA 目标维
PCA_TARGET_DIM_LARGE = 20  # step 数 >= 80 时的 PCA 目标维
PCA_TARGET_DIM_STEP_THRESHOLD = 80
# 「PCA→UMAP」两阶段路径专用：第一阶段 PCA 的中间维（先去噪线性压缩，再交给 UMAP）
PCA_UMAP_INTERMEDIATE_DIM = 30
# [已注释] DBSCAN 聚类方式已移除，改用 HDBSCAN 自动探测
# DBSCAN_EPS = 0.9          # DBSCAN 的 eps 参数
# DBSCAN_MIN_SAMPLES = 2    # DBSCAN 的 min_samples 参数

# [已注释] 固定K值已移除，改用HDBSCAN自动探测
# # KMeans 固定聚类数：按降维方法分别配置，因为不同降维方法揭示的"自然簇数"不同
# # - PCA 是线性投影，不主动揭示簇结构 → 较大的 k 用来均匀切片
# # - UMAP 是非线性保拓扑，会把数据收缩成几个明显的"团" → k 应当 ≈ 团数
# # 对 50~120 step、10 agent 的场景，UMAP 的自然簇数大约 3~6
# KMEANS_K_FIXED_PCA = 7     # PCA 路径的经验值
# KMEANS_K_FIXED_UMAP = 4    # UMAP 路径，匹配 UMAP 的自然簇数
# KMEANS_K_FIXED = KMEANS_K_FIXED_PCA  # 向后兼容旧名

# HDBSCAN 配置：兼容两条降维路径（PCA 不做 post-L2 / UMAP 不 L2）
# - min_cluster_size=3：允许 3 个 step 的少数派错误小簇被正式识别为 wrong，
#   避免被吞为 noise 而失去 bidirectional 修正机会
# - min_samples=2：core point 的最小邻居数，平衡噪声敏感度
# - metric=euclidean：
#     · PCA 路径：euclidean 直接作用在 PCA 坐标上（各主成分尺度由方差决定）
#     · UMAP 路径：UMAP 输出本身是 manifold embedding，euclidean 是其自然度量
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_METRIC = "euclidean"

# UMAP 降维参数：纯 UMAP 视角调优（不为对齐 PCA 妥协）
# - n_components=15 保留更多局部语义结构，适合 50~130 step 的场景
# - n_neighbors=10 ≈ 期望同质簇大小（每条推理路径上 ~10 个相似 step），保留少数派
# - min_dist=0.1 给簇内更宽松的呼吸空间，避免 HDBSCAN 因簇内点过密导致核心距离梯度被压平、
#   簇边界判不准；调大此值可让簇间更分离，缓解过度合并
# - metric=cosine 是 LLM embedding 的本征几何
# - random_state=42 复现性；副作用：n_jobs 会自动变 1，N 小不影响速度
# - pre_l2/post_l2 默认 OFF：cosine metric 已内置标度无关，不必再 L2
UMAP_N_COMPONENTS = 15
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42
UMAP_PRE_L2 = False        # UMAP 之前不做 L2（cosine metric 内置归一化处理）
UMAP_POST_L2 = False       # UMAP 之后不做 L2（保留 UMAP manifold 几何）

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
    is_math=False：parse_answer_bbh → solve_math_problems → parse_YN
      parse_answer_bbh 覆盖 (X)、"The answer is: X"、"Answer: X" 等格式，
      统一返回 (X) 格式与 GT 对齐，与 eval_all_round.compute_accuracy 完全一致。
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
        pred_answer = parse_answer_bbh(text)
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


def resolve_pca_target_dim(n_steps: int) -> int:
    """按 step 数量选择纯 PCA 路径的目标维度。"""
    if n_steps >= PCA_TARGET_DIM_STEP_THRESHOLD:
        return PCA_TARGET_DIM_LARGE
    return PCA_TARGET_DIM_SMALL


def _reduce_dimensions_pca(vectors: np.ndarray, target_dim: int = TARGET_DIM) -> np.ndarray:
    """
    针对 LLM 句向量的 PCA 降维。

    设计要点：
      1. 不使用 StandardScaler——逐维 z-score 会破坏 embedding 的余弦几何，并在小 N 下放大
         "近常数维度"的噪声。PCA 内部已自动 centering，无需额外标准化。
      2. PCA 前先做 L2 归一化——把所有向量投到单位球面，让 PCA 在 LLM embedding 的"母语
         几何"（余弦/角距离）上工作；这样学到的主轴反映方向（语义）方差，而非长度方差。
      3. 维度上限取 min(target_dim, N-1, original_dim)，N 较小时 PCA 自动收缩到 N-1。
      4. PCA 后不做 L2 归一化；下游聚类直接在 PCA 坐标上使用 euclidean 距离。

    Args:
        vectors: 原始向量矩阵，形状 (N, dim)
        target_dim: 目标维度

    Returns:
        降维后的向量矩阵
    """
    if vectors.shape[0] < 2:
        return vectors

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_normed = vectors / (norms + 1e-12)

    max_dim = min(target_dim, vectors.shape[0] - 1, vectors.shape[1])
    pca = PCA(n_components=max_dim, random_state=42)
    reduced = pca.fit_transform(vectors_normed)
    print(
        f"[降维] PCA: {vectors.shape[1]} 维 -> {max_dim} 维，"
        f"解释方差比: {pca.explained_variance_ratio_.sum():.2%}（PCA 前已 L2 归一化）"
    )
    return reduced


def _reduce_dimensions_umap(
    vectors: np.ndarray,
    target_dim: int = UMAP_N_COMPONENTS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    metric: str = UMAP_METRIC,
    random_state: int = UMAP_RANDOM_STATE,
    pre_l2: bool = UMAP_PRE_L2,
) -> np.ndarray:
    """
    针对 LLM 句向量的 UMAP 降维。

    设计要点：
      1. metric=cosine 与 LLM embedding 的本征几何（方向）一致；cosine 公式内已归一化
         分母，所以 pre_l2 默认 OFF（不做归一化），与 UMAP 作者建议一致。
      2. min_dist=0.0：簇内最紧密、簇间最分离，利于下游 KMeans/HDBSCAN/DBSCAN 聚类。
      3. n_neighbors=10：≈ 期望同质簇大小（每条推理路径上 ~10 个相似 step），既能保
         留少数派子簇又不过分碎裂。
      4. 小样本兜底：N<5 直接返回原向量；N<15 自动收缩 n_neighbors 与 n_components。
      5. UMAP 输出本身已是优化过的 manifold embedding，下游不再做 L2 归一化（保留
         UMAP 学到的几何结构）。

    Args:
        vectors: 原始向量矩阵，形状 (N, dim)
        target_dim: 目标维度（实际值会被 N 自动收缩到 min(target_dim, N-2)）
        n_neighbors: UMAP 局部邻域大小（小样本会被自动收缩到 N-1）
        min_dist: 输出空间中点的最小间距
        metric: 原空间距离度量
        random_state: 随机种子（复现性）
        pre_l2: 是否在 UMAP 前做 L2 归一化（默认 False）

    Returns:
        降维后的向量矩阵，形状 (N, n_components_eff)
    """
    if not UMAP_AVAILABLE:
        raise ImportError(
            "UMAP 未安装：请运行 'pip install umap-learn'；"
            "或将 REDUCTION_METHOD 改回 'pca'（不含 UMAP 的路径）"
        )

    n_samples = vectors.shape[0]
    if n_samples < 5:
        print(
            f"[降维][UMAP] 跳过：样本数 {n_samples} < 5，UMAP 拟合不稳定，"
            f"直接返回原向量"
        )
        return vectors

    if pre_l2:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-12)

    # n_components: UMAP 要求 < N-1，对小 N 自动收缩
    n_components_eff = min(target_dim, max(2, n_samples - 2), vectors.shape[1])
    # n_neighbors: 必须 < N，对小 N 自动收缩；过小会让局部图退化
    n_neighbors_eff = min(n_neighbors, max(2, n_samples - 1))

    if n_samples < 15:
        print(
            f"[降维][UMAP][警告] 样本数 {n_samples} 偏小，自动收缩参数："
            f"n_components={n_components_eff}, n_neighbors={n_neighbors_eff}"
        )

    reducer = umap.UMAP(
        n_components=n_components_eff,
        n_neighbors=n_neighbors_eff,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(vectors)
    print(
        f"[降维] UMAP: {vectors.shape[1]} 维 -> {n_components_eff} 维 "
        f"(n_neighbors={n_neighbors_eff}, min_dist={min_dist}, "
        f"metric={metric}, pre_l2={pre_l2})"
    )
    return reduced


def _reduce_dimensions_pca_then_umap(
    vectors: np.ndarray,
    *,
    pca_dim: int = PCA_UMAP_INTERMEDIATE_DIM,
    umap_dim: int = UMAP_N_COMPONENTS,
) -> np.ndarray:
    """
    两阶段降维：L2 + PCA（线性去噪/压缩）→ UMAP（非线性流形嵌入）。

    与单独 PCA / 单独 UMAP 的关系：
      - 第一阶段复用 ``_reduce_dimensions_pca``：已在 PCA 前做 L2，且不使用 StandardScaler，
        保持 LLM 句向量在余弦几何下的语义主轴。
      - 第二阶段将 PCA 坐标作为 UMAP 输入；在高维上直接跑 UMAP 往往更慢且更易受噪声影响，
        先用 PCA 压到 ``pca_dim`` 通常能减轻计算并稳定邻域图。
      - UMAP 的超参数沿用模块级常量（``UMAP_N_NEIGHBORS``、``UMAP_MIN_DIST`` 等），
        ``pre_l2`` 默认 False：metric=cosine 时已对标量长度不敏感；PCA 输出也不必强制再 L2。

    Args:
        vectors: 原始向量矩阵，形状 (N, dim)
        pca_dim: PCA 阶段目标维（实际维数会被收缩为 min(pca_dim, N-1, dim)）
        umap_dim: UMAP 阶段目标维（实际维数见 ``_reduce_dimensions_umap`` 内收缩逻辑）

    Returns:
        形状 (N, umap_eff) 的降维矩阵；依赖 ``umap-learn``。
    """
    # ── 阶段一：与纯 PCA 路径共享同一套 L2→PCA 实现（含小样本维数收缩）
    vecs_pca = _reduce_dimensions_pca(vectors, target_dim=pca_dim)
    # ── 阶段二：在 PCA 坐标上做 UMAP；输入维已降至 vecs_pca.shape[1]，显著快于原始高维
    return _reduce_dimensions_umap(vecs_pca, target_dim=umap_dim)


def _select_k_by_silhouette(vectors: np.ndarray) -> tuple[int, np.ndarray]:
    """
    [已注释] 轮廓系数在小样本下不稳定，改用HDBSCAN自动探测

    根据轮廓系数选择 KMeans 的最优聚类数 k，并返回对应的聚类标签（避免重复运行 KMeans）

    轮廓系数范围 [-1, 1]，越接近 1 表示聚类效果越好。
    遍历 k=2 到 k_max，选取平均轮廓系数最高的 k。

    Args:
        vectors: 向量矩阵，形状 (N, dim)

    Returns:
        (最优的聚类数 k, 对应的聚类标签数组)
    """
    # [已注释] 改用 HDBSCAN 自动探测自然簇数
    raise NotImplementedError("轮廓系数选择K方法已禁用，请使用HDBSCAN自动探测")


def _cluster_kmeans(
    vectors: np.ndarray,
    n_clusters: int = 4,  # 默认值，避免未传参时报错
) -> np.ndarray:
    """KMeans 聚类。

    Args:
        vectors: 待聚类向量矩阵 (N, dim)
        n_clusters: 期望聚类数；由上层通过 HDBSCAN 自动探测确定。
                     exchange.py 中的 _resolve_kmeans_k_with_hdbscan 函数负责计算此值。
    """
    if vectors.shape[0] < 2:
        return np.array([0] * vectors.shape[0])
    k_target = n_clusters
    n_clusters_eff = min(k_target, vectors.shape[0])
    kmeans = KMeans(n_clusters=n_clusters_eff, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    print(
        f"[KMeans] n_samples={vectors.shape[0]}, n_clusters={n_clusters_eff} "
        f"(请求 k={k_target}), 聚类大小: {dict(Counter(labels))}"
    )
    return labels


# [已注释] DBSCAN 已移除，改用 HDBSCAN 自动探测
# def _cluster_dbscan(vectors: np.ndarray, eps: float = 0.9, min_samples: int = 2) -> np.ndarray:
#     """DBSCAN 聚类"""
#     if vectors.shape[0] < 2:
#         return np.array([0] * vectors.shape[0])
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(vectors)
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise = list(labels).count(-1)
#     print(f"[DBSCAN] n_samples={vectors.shape[0]}, n_clusters={n_clusters}, 噪声点={n_noise}")
#     return labels


def _cluster_hdbscan(
    vectors: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
) -> np.ndarray:
    """HDBSCAN 聚类。

    通过分层互达密度自适应发现不同密度的簇，噪声点用 -1 标识。

    后端兼容：优先 sklearn.cluster.HDBSCAN（sklearn>=1.3），
              否则 fallback 到独立 hdbscan 包；二者 API 接近，统一返回 labels 数组。

    Returns:
        cluster_labels: shape (N,)，每个元素是该样本的簇 id；-1 表示噪声。
    """
    if vectors.shape[0] < 2:
        return np.array([0] * vectors.shape[0])

    if HDBSCAN_BACKEND is None:
        raise RuntimeError(
            "HDBSCAN 后端未安装：请升级 scikit-learn>=1.3 或 pip install hdbscan"
        )

    eff_min_cluster_size = max(2, min(min_cluster_size, vectors.shape[0]))
    eff_min_samples = max(1, min(min_samples, vectors.shape[0]))

    if HDBSCAN_BACKEND == "sklearn":
        # copy=True：显式传入，避免 sklearn>=1.7 的 FutureWarning，并防止 fit_predict 原地修改 vectors。
        clusterer = _SK_HDBSCAN(
            min_cluster_size=eff_min_cluster_size,
            min_samples=eff_min_samples,
            metric=metric,
            cluster_selection_method="eom",
            copy=True,
        )
        labels = clusterer.fit_predict(vectors)
    else:
        clusterer = _hdbscan_pkg.HDBSCAN(
            min_cluster_size=eff_min_cluster_size,
            min_samples=eff_min_samples,
            metric=metric,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(vectors)

    labels = np.asarray(labels)
    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(
        f"[HDBSCAN] n_samples={vectors.shape[0]}, n_clusters={n_clusters}, "
        f"噪声点={n_noise}，参数: min_cluster_size={eff_min_cluster_size}, "
        f"min_samples={eff_min_samples}, metric={metric}, backend={HDBSCAN_BACKEND}"
    )
    if vectors.shape[0] > 0 and n_noise / vectors.shape[0] > 0.5:
        print(
            f"[HDBSCAN][警告] 噪声点占比 {n_noise/vectors.shape[0]:.1%} > 50%，"
            f"可能样本规模偏小或维度偏高；可考虑减小 min_cluster_size 或降低 PCA 目标维度"
        )
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


# ============================================================
# 簇过度合并诊断
# ============================================================

# 阈值：超过即视为该项告警
OVERMERGE_LARGEST_RATIO_THRESHOLD = 0.65   # 最大簇占非噪声 step 的比例
OVERMERGE_NO_MAJORITY_RATIO_THRESHOLD = 0.5  # no_majority 簇占非噪声簇的比例
OVERMERGE_MEAN_IMPURITY_THRESHOLD = 0.35   # 各簇 min(c_ratio, w_ratio) 的均值
OVERMERGE_MIN_CLUSTERS_FOR_N = (30, 2)     # 当样本 >= 30 时，有效簇数 <= 2 视为告警


def _diagnose_cluster_overmerge(
    cluster_labels: np.ndarray,
    cluster_label_results: dict,
    step_indices: list,
    all_steps: list,
    method_name: str = "",
) -> dict:
    """计算 4 个互补指标，判断簇是否被过度合并并打印结论。

    指标：
      1. 有效簇数（排除噪声）：过少说明 UMAP 把多条推理路径压成一片
      2. 最大簇占比：单一簇吞掉绝大多数 step → 典型巨型合并簇
      3. no_majority 簇占比：簇内 correct/wrong 各占近一半 → 混合簇直接证据
      4. 平均不纯度（mean impurity）：每簇 min(c_ratio, w_ratio) 的均值，越高越混

    任意 ≥2 项告警 → 建议把 ``UMAP_MIN_DIST`` 调到 0.2 或更大。

    Returns:
        dict 含上述指标与告警标记，方便上层程序化使用。
    """
    arr = np.asarray(cluster_labels)
    n_total = int(arr.shape[0])
    n_noise = int(np.sum(arr == -1))
    n_clustered = n_total - n_noise

    real_cluster_ids = [int(c) for c in set(arr.tolist()) if int(c) != -1]
    n_real_clusters = len(real_cluster_ids)

    # 各簇大小、各簇 correct/wrong 计数
    sizes: list[int] = []
    impurities: list[float] = []
    no_majority_cnt = 0
    for cid in real_cluster_ids:
        vec_idx = [i for i, c in enumerate(arr) if int(c) == cid]
        sizes.append(len(vec_idx))
        c_cnt = sum(1 for i in vec_idx if all_steps[step_indices[i]].get("is_correct") is True)
        w_cnt = sum(1 for i in vec_idx if all_steps[step_indices[i]].get("is_correct") is False)
        labeled = c_cnt + w_cnt
        if labeled >= 2:
            ratio = min(c_cnt, w_cnt) / labeled
            impurities.append(ratio)
        if cluster_label_results.get(cid) == "no_majority":
            no_majority_cnt += 1

    largest_size = max(sizes) if sizes else 0
    largest_ratio = (largest_size / n_clustered) if n_clustered > 0 else 0.0
    no_majority_ratio = (no_majority_cnt / n_real_clusters) if n_real_clusters > 0 else 0.0
    mean_impurity = float(np.mean(impurities)) if impurities else 0.0

    # 4 项告警判定
    n_min, k_min = OVERMERGE_MIN_CLUSTERS_FOR_N
    flag_too_few = (n_total >= n_min) and (n_real_clusters <= k_min)
    flag_largest = largest_ratio > OVERMERGE_LARGEST_RATIO_THRESHOLD
    flag_no_maj = no_majority_ratio > OVERMERGE_NO_MAJORITY_RATIO_THRESHOLD
    flag_impure = mean_impurity > OVERMERGE_MEAN_IMPURITY_THRESHOLD
    n_flags = sum([flag_too_few, flag_largest, flag_no_maj, flag_impure])

    title = method_name or "簇过度合并诊断"
    print(f"\n{'='*60}")
    print(f"[{title}] 过度合并诊断")
    print("="*60)
    print(f"  总 step={n_total}, 噪声={n_noise}, 有效簇内 step={n_clustered}, 有效簇数={n_real_clusters}")
    if sizes:
        print(f"  各簇大小（降序）: {sorted(sizes, reverse=True)}")

    def _mark(ok: bool) -> str:
        return "✗" if ok else "✓"  # ✗=告警，✓=通过

    print(
        f"  [{_mark(flag_too_few)}] 有效簇数 ≤ {k_min}（且 N ≥ {n_min}）: {n_real_clusters}"
    )
    print(
        f"  [{_mark(flag_largest)}] 最大簇占比 > {OVERMERGE_LARGEST_RATIO_THRESHOLD:.0%}: "
        f"{largest_ratio:.1%}（最大簇 {largest_size} / 有效 {n_clustered}）"
    )
    print(
        f"  [{_mark(flag_no_maj)}] no_majority 簇占比 > {OVERMERGE_NO_MAJORITY_RATIO_THRESHOLD:.0%}: "
        f"{no_majority_ratio:.1%}（{no_majority_cnt}/{n_real_clusters}）"
    )
    print(
        f"  [{_mark(flag_impure)}] 平均不纯度 > {OVERMERGE_MEAN_IMPURITY_THRESHOLD:.2f}: "
        f"{mean_impurity:.3f}（每簇 min(c,w)/labeled 的均值）"
    )

    if n_flags >= 2:
        print(
            f"\n  [告警] 命中 {n_flags}/4 项 → 簇可能被过度合并；"
            f"建议调整 UMAP_MIN_DIST 0.1 → 0.2（簇内更宽松、簇间更分离）"
        )
    elif n_flags == 1:
        print(f"\n  [提示] 命中 1/4 项 → 边界情况，可暂不调整，多跑几道题观察")
    else:
        print(f"\n  [OK] 4 项均通过，当前簇结构无明显过度合并迹象")
    print("="*60)

    return {
        "n_total": n_total,
        "n_noise": n_noise,
        "n_clustered": n_clustered,
        "n_real_clusters": n_real_clusters,
        "sizes": sizes,
        "largest_ratio": largest_ratio,
        "no_majority_ratio": no_majority_ratio,
        "mean_impurity": mean_impurity,
        "flag_too_few": flag_too_few,
        "flag_largest": flag_largest,
        "flag_no_majority": flag_no_maj,
        "flag_mean_impurity": flag_impure,
        "n_flags": n_flags,
    }


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
    # 深拷贝，避免修改原始数据
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

    # 2. 降维（目标维按 step 数动态选择；PCA 后不做 L2）
    n_steps = step_vectors.shape[0]
    pca_target_dim = resolve_pca_target_dim(n_steps)
    print(
        f"[预处理] PCA 目标维: {pca_target_dim}（n_steps={n_steps}, "
        f"rule: >={PCA_TARGET_DIM_STEP_THRESHOLD}→{PCA_TARGET_DIM_LARGE} else {PCA_TARGET_DIM_SMALL}）"
    )
    vectors_reduced = _reduce_dimensions_pca(step_vectors, target_dim=pca_target_dim)

    # 3. KMeans 聚类
    print("\n" + "-" * 60)
    print("[聚类1] KMeans")
    print("-" * 60)

    labels_kmeans = _cluster_kmeans(vectors_reduced)
    cluster_labels_kmeans = _label_clusters_by_majority(
        labels_kmeans, step_indices, all_steps,
        majority_threshold=MAJORITY_THRESHOLD,
        method_name="KMeans",
    )
    _print_clustering_results(
        labels_kmeans, cluster_labels_kmeans,
        step_indices, all_steps, "KMeans",
    )
    # 根据聚类标签更新 step：correct 类簇内的 step 置为正确，wrong 类簇保持原标签
    steps_after_kmeans, n_modified_kmeans, changes_kmeans = _apply_cluster_labels_to_steps(
        labels_kmeans, cluster_labels_kmeans, step_indices, all_steps
    )
    # 打印每个聚类内 step 的原始标签 -> 修改后标签对比
    _print_step_label_changes_by_cluster(changes_kmeans, "KMeans")
    # 打印修改后所有 step 的整体标签视图（按 agent 分组）
    _print_step_labels_after_modification(
        steps_after_kmeans, "KMeans", n_modified_kmeans
    )

    # [已注释] DBSCAN 已移除
    # # 5. DBSCAN 聚类
    # print("\n" + "-" * 60)
    # print("[聚类2] DBSCAN")
    # print("-" * 60)
    # labels_dbscan = _cluster_dbscan(
    #     vectors_reduced,
    #     eps=DBSCAN_EPS,
    #     min_samples=DBSCAN_MIN_SAMPLES,
    # )
    # cluster_labels_dbscan = _label_clusters_by_majority(
    #     labels_dbscan, step_indices, all_steps,
    #     majority_threshold=MAJORITY_THRESHOLD,
    #     method_name="DBSCAN",
    # )
    # _print_clustering_results(
    #     labels_dbscan, cluster_labels_dbscan,
    #     step_indices, all_steps, "DBSCAN",
    # )
    # # 根据聚类标签更新 step：correct 类簇内的 step 置为正确，wrong 类簇保持原标签
    # steps_after_dbscan, n_modified_dbscan, changes_dbscan = _apply_cluster_labels_to_steps(
    #     labels_dbscan, cluster_labels_dbscan, step_indices, all_steps
    # )
    # # 打印每个聚类内 step 的原始标签 -> 修改后标签对比
    # _print_step_label_changes_by_cluster(changes_dbscan, "DBSCAN")
    # # 打印修改后所有 step 的整体标签视图（按 agent 分组）
    # _print_step_labels_after_modification(
    #     steps_after_dbscan, "DBSCAN", n_modified_dbscan
    # )

    # 6. 汇总
    print("\n" + "#" * 70)
    print("# 聚类完成")
    print("#" * 70)
    print(f"  majority_threshold: {MAJORITY_THRESHOLD} (正确/错误占比 >= 60% 时生效)")
    print(f"  聚类标签规则: correct 类簇 -> 其内所有 step 置为正确; wrong 类簇 -> 保持原标签")
    print(f"  KMeans: 聚类数={len(set(labels_kmeans)) - (1 if -1 in labels_kmeans else 0)}, 修改 step 数={n_modified_kmeans}")
    # print(f"  DBSCAN: 聚类数={len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)}, 噪声点={list(labels_dbscan).count(-1)}, 修改 step 数={n_modified_dbscan}")  # [已注释]
    print("#" * 70 + "\n")


if __name__ == "__main__":
    print("\n[入口] 正在运行 wzy_multi_agent_debate_clustering.py（聚类脚本）\n")
    asyncio.run(run_clustering())
