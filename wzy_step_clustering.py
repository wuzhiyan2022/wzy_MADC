"""
Step 聚类标注修正模块

功能：
1. 对每个agent的step进行文本向量化（支持embedding API和hidden state两种方法）
2. 对所有向量进行降维（UMAP/PCA）
3. 对向量进行聚类
4. 基于聚类结果修正step标注：
   - 注意：is_correct字段表示"该step所属agent的最终答案是否正确"（通过多数投票判断），
     而不是"该step本身的推理过程是否正确"
   - 如果聚类中多数steps的is_correct=True，将该聚类中is_correct=False的改为True；反之亦然
5. 整合到step级别的真相后置处理中
"""

import numpy as np
import time
import time as time_module
from typing import List, Dict, Optional, Tuple
from collections import Counter
from openai import OpenAI
import os
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

def show_progress_bar(current, total, prefix="进度", suffix="", length=40):
    """
    显示进度条

    Args:
        current: 当前进度
        total: 总数
        prefix: 前缀文本
        suffix: 后缀文本
        length: 进度条长度
    """
    if total <= 0:
        return

    progress = int((current / total) * length)
    bar = "=" * progress + "-" * (length - progress)
    percentage = (current / total * 100)

    print(f"\r{prefix}: [{bar}] {percentage:.1f}% ({current}/{total}) {suffix}",
          end="", flush=True)

    if current >= total:
        print()  # 完成时换行

def log_clustering_stats(steps, corrections, cluster_count, duration):
    """
    记录聚类分析的详细统计信息

    Args:
        steps: 步骤列表
        corrections: 修正数量
        cluster_count: 聚类数量
        duration: 处理耗时(秒)
    """
    print(f"\n[统计] 聚类分析结果:")
    print(f"   * 总步骤数: {len(steps)}")
    print(f"   * 聚类数量: {cluster_count}")
    print(f"   * 标注修正: {corrections} 个步骤")

    if len(steps) > 0:
        correction_rate = corrections / len(steps) * 100
        print(f"   * 修正率: {correction_rate:.1f}%")
        print(f"   * 处理耗时: {duration:.1f} 秒")

        # 分析每个聚类的修正情况
        # 修复：对字典使用 'in' 检查键，而非 hasattr
        if steps and isinstance(steps[0], dict) and 'cluster_id' in steps[0]:
            cluster_stats = {}
            for step in steps:
                cluster_id = step.get('cluster_id', 'unknown')
                was_corrected = step.get('was_corrected', False)

                if cluster_id not in cluster_stats:
                    cluster_stats[cluster_id] = {'total': 0, 'corrected': 0}
                cluster_stats[cluster_id]['total'] += 1
                if was_corrected:
                    cluster_stats[cluster_id]['corrected'] += 1

            print(f"   * 聚类修正详情:")
            for cluster_id, stats in sorted(cluster_stats.items()):
                if stats['total'] > 0:
                    rate = stats['corrected'] / stats['total'] * 100
                    print(f"      聚类{cluster_id}: {stats['total']}步, 修正{stats['corrected']}步 ({rate:.1f}%)")
    else:
        print(f"   * 无步骤数据")

    print(f"   * 聚类分析完成")

# 降维和聚类库
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[警告] sklearn未安装，无法使用PCA和KMeans。请运行: pip install scikit-learn")

try:
    import umap
    import warnings
    warnings.filterwarnings("ignore", message="n_jobs value .* overridden.*random_state")
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    # 不打印警告，因为umap是可选的（可以使用PCA）

# Transformers库（用于hidden_state方法）
try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[警告] transformers未安装，无法使用hidden_state方法。请运行: pip install transformers torch")

# --------------------
# 默认配置（供 perform_online_clustering_with_majority 等便捷函数使用）
# 之前这里缺失会导致 NameError: DEFAULT_* is not defined
# --------------------
DEFAULT_API_URL = "https://api.zhizengzeng.com/v1"
DEFAULT_API_KEY = None
DEFAULT_VECTOR_METHOD = "embedding_api"
DEFAULT_REDUCE_DIM = True
DEFAULT_REDUCTION_METHOD = "pca"
DEFAULT_TARGET_DIM = 128
DEFAULT_CLUSTERING_METHOD = "kmeans"
DEFAULT_MAJORITY_THRESHOLD = 0.6
DEFAULT_MAX_WORKERS = 10  # 优化并行：15个并发API调用




class StepClusteringRefiner:
    """Step聚类标注修正器"""
    
    def __init__(self, 
                 api_url: str = "https://api.zhizengzeng.com/v1",
                 api_key: str = None,
                 embedding_model: str = "qwen3-embedding-8b",  # 默认使用qwen3-embedding-8b，也可以使用其他embedding模型
                 vector_method: str = "embedding_api",  # "embedding_api" 或 "hidden_state"
                 reduce_dim: bool = True,
                 reduction_method: str = "pca",  # "pca" 或 "umap"
                 target_dim: int = 128,
                 clustering_method: str = "kmeans",  # "kmeans" 或 "dbscan"
                 n_clusters: Optional[int] = None,  # None表示自动确定（改进版：小样本固定2-3簇，中等样本sqrt(n)，大样本保守比例）
                 majority_threshold: float = 0.6,  # 聚类中多数投票的阈值
                 cache_dir: str = ".step_clustering_cache",
                 max_workers: int = 3,  # API并行调用数（embedding_api方法，降低以减少限流）
                 batch_size: int = 1,  # 批量embedding的批次大小（降低以避免API失败）
                 # hidden_state方法相关参数
                 hf_model_name: str = None,  # HuggingFace模型名称，如 "meta-llama/Llama-2-7b-hf"
                 hf_model_device: str = "cuda",  # "cuda" 或 "cpu"
                 hf_model_dtype: str = "auto"):  # torch数据类型
        """
        初始化聚类标注修正器
        
        参数:
            api_url: API地址
            api_key: API密钥
            embedding_model: embedding模型名称
            vector_method: 向量化方法，"embedding_api" 或 "hidden_state"
            reduce_dim: 是否降维
            reduction_method: 降维方法，"pca" 或 "umap"
            target_dim: 目标维度
            clustering_method: 聚类方法，"kmeans" 或 "dbscan"
            n_clusters: 聚类数量（None表示自动确定，改进版避免小样本波动：n≤20用2-3簇，n≤100用sqrt(n)，n>100用n//10）
            majority_threshold: 聚类中多数投票的阈值（0.6表示60%以上）
            cache_dir: 缓存目录
            hf_model_name: HuggingFace模型名称（hidden_state方法需要）
            hf_model_device: 模型运行设备
            hf_model_dtype: 模型数据类型
        """
        self.api_url = api_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.vector_method = vector_method
        self.reduce_dim = reduce_dim
        self.reduction_method = reduction_method
        self.target_dim = target_dim
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.majority_threshold = majority_threshold
        self.cache_dir = cache_dir
        self.hf_model_name = hf_model_name
        self.max_workers = max_workers  # API并行调用数
        self.batch_size = batch_size  # 批量embedding的批次大小
        self.hf_model_device = hf_model_device
        self.hf_model_dtype = hf_model_dtype
        
        # 初始化OpenAI客户端（embedding_api方法）
        if self.vector_method == "embedding_api":
            self.client = OpenAI(base_url=api_url, api_key=api_key) if api_key else None
        
        # 初始化HuggingFace模型（hidden_state方法）
        self.hf_model = None
        self.hf_tokenizer = None
        if self.vector_method == "hidden_state":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("hidden_state方法需要安装transformers: pip install transformers torch")
            if not hf_model_name:
                raise ValueError("hidden_state方法需要提供hf_model_name参数")
            self._load_hf_model()
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 保存向量缓存
        self.vector_cache = {}

        # 聚类/降维模型缓存（用于 save_model / 后续复用）
        self.saved_cluster_model = None
        self.saved_cluster_labels = None
        self.saved_scaler = None
        self.saved_reducer = None
        self._last_cluster_model = None
        self._last_reducer = None
        self._last_scaler = None

    def _calculate_clusters_for_samples(self, n_samples: int) -> int:
        """
        计算给定样本数量的聚类数量（用于分析不稳定性）

        Args:
            n_samples: 样本数量

        Returns:
            聚类数量
        """
        if n_samples <= 15:
            # 小样本：使用更稳定的公式，避免跳变
            return max(2, min(3, (n_samples + 2) // 4))
        elif n_samples <= 50:
            # 中等样本：使用平滑的log函数过渡，避免sqrt的跳跃
            import math
            log_clusters = math.log2(n_samples)
            return max(2, min(6, int(log_clusters + 1.5)))
        elif n_samples <= 100:
            # 中等偏大样本：继续平滑增长
            import math
            log_clusters = math.log2(n_samples)
            return max(2, min(8, int(log_clusters + 0.8)))
        else:
            # 大样本：使用更保守的比例
            return max(2, min(10, n_samples // 12))
    
    def _load_hf_model(self):
        """加载HuggingFace模型和tokenizer"""
        print(f"[信息] 加载HuggingFace模型: {self.hf_model_name}...")
        try:
            # 加载tokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            
            # 加载模型，设置output_hidden_states=True以获取hidden states
            self.hf_model = AutoModel.from_pretrained(
                self.hf_model_name,
                output_hidden_states=True,
                torch_dtype=self.hf_model_dtype,
                device_map=self.hf_model_device
            )
            
            # 设置为评估模式
            self.hf_model.eval()
            print(f"[OK] 模型加载完成")
        except Exception as e:
            print(f"[错误] 模型加载失败: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键 - 修复：包含所有影响向量的参数"""
        payload = f"{self.vector_method}|{self.embedding_model}|{self.hf_model_name}|{text}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """从缓存加载向量"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, vector: np.ndarray):
        """保存向量到缓存"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(vector, f)
    
    def get_text_embedding(self, text: str, max_retries: int = 3, retry_delay: float = 1.0) -> np.ndarray:
        """
        方法1：使用embedding API获取文本向量（带重试机制）
        
        参数:
            text: 文本内容
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            
        返回:
            向量数组
        """
        if self.vector_method != "embedding_api":
            raise ValueError("此方法仅适用于embedding_api模式")
        
        # 检查缓存
        cache_key = self._get_cache_key(text)
        cached_vector = self._load_from_cache(cache_key)
        if cached_vector is not None:
            return cached_vector
        
        # 调用API（带重试）
        if not self.client:
            raise ValueError("客户端未初始化，请提供api_key")
        
        # 确定embedding维度
        embedding_dim = 4096 if "qwen3-embedding-8b" in self.embedding_model else 1536
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                
                # 检查响应数据
                if not response or not response.data or len(response.data) == 0:
                    raise ValueError("No embedding data received")
                
                vector = np.array(response.data[0].embedding)
                
                # 检查向量维度
                if len(vector.shape) != 1:
                    raise ValueError(f"Unexpected vector shape: {vector.shape}")
                
                if len(vector) == 0:
                    raise ValueError("Empty embedding vector")
                
                # 保存到缓存
                self._save_to_cache(cache_key, vector)
                return vector
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # 不是最后一次尝试，等待后重试（降低延迟）
                    time.sleep(retry_delay * 0.5)  # 降低延迟时间
                    continue
                else:
                    # 最后一次尝试也失败
                    print(f"[警告] 获取embedding失败（已重试{max_retries}次）: {e}")
                    # 返回None，避免污染聚类
                    return None

        # 如果所有重试都失败，返回None
        print(f"[警告] 获取embedding最终失败: {last_error}")
        return None
    
    def get_text_embeddings_batch(self, texts: List[str], max_retries: int = 3, retry_delay: float = 0.5, batch_size: int = 5) -> List[np.ndarray]:
        """
        批量获取embedding向量（性能优化版本）
        
        参数:
            texts: 文本列表
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            batch_size: 每批处理的文本数量（避免单次请求过大）
            
        返回:
            向量列表（与输入文本顺序对应）
        """
        if self.vector_method != "embedding_api":
            raise ValueError("此方法仅适用于embedding_api模式")
        
        if not self.client:
            raise ValueError("客户端未初始化，请提供api_key")
        
        # 缓存机制：对相同文本只请求一次 embedding，之后复用本地结果
        cached_vectors = {}  # 已从缓存命中的 embedding 向量 {原始idx: vector}
        texts_to_fetch = []  # 缓存未命中、需要调用 API 的文本 [(原始idx, text, cache_key)]
        
        for idx, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_vector = self._load_from_cache(cache_key)
            if cached_vector is not None:
                cached_vectors[idx] = cached_vector
            else:
                texts_to_fetch.append((idx, text, cache_key))
        
        # 如果所有文本都已缓存，直接返回
        if not texts_to_fetch:
            print(f"[信息] 所有 {len(texts)} 个向量都已缓存，直接返回")
            return [cached_vectors[i] for i in range(len(texts))]
        
        # 显示缓存统计
        cache_hit_rate = len(cached_vectors) / len(texts) * 100
        print(f"[信息] 缓存命中: {len(cached_vectors)}/{len(texts)} ({cache_hit_rate:.1f}%)，需要获取: {len(texts_to_fetch)} 个向量")
        
        # 初始化结果容器vectors_dict，并预填已命中的向量
        # 最终对外使用的是 vectors_dict，cached_vectors 是中间量，临时用
        # 命中：idx 从 cached_vectors 复制到 vectors_dict
        # 未命中：vectors_dict[idx] = None
        vectors_dict = {idx: cached_vectors.get(idx) for idx in range(len(texts))}
        
        # 将需要获取的文本分批
        total_batches = (len(texts_to_fetch) + batch_size - 1) // batch_size

        print(f"[信息] 开始批量向量化 {len(texts_to_fetch)} 个步骤...")

        for batch_idx in range(0, len(texts_to_fetch), batch_size):
            batch_items = texts_to_fetch[batch_idx:batch_idx + batch_size]
            batch_texts = [text for _, text, _ in batch_items]
            current_batch = batch_idx // batch_size + 1

            show_progress_bar(current_batch, total_batches, "向量化进度", f"{len(texts_to_fetch)} 个步骤")

            for attempt in range(max_retries):
                try:
                    # 批量调用API（一次请求多个文本）
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=batch_texts
                    )

                    # 检查响应数据
                    if not response or not response.data:
                        raise ValueError("No embedding data received")

                    if len(response.data) != len(batch_texts):
                        raise ValueError(f"返回的向量数量({len(response.data)})与请求数量({len(batch_texts)})不匹配")

                    # 使用 embedding_data.index 映射结果，避免依赖返回顺序
                    # （第三方兼容 API 不保证 response.data 顺序与输入顺序一致）
                    for embedding_data in response.data:
                        i = embedding_data.index  # API 返回的位置标识，对应 batch_items[i]
                        original_idx, text, cache_key = batch_items[i]
                        vector = np.array(embedding_data.embedding)

                        # 检查向量维度
                        if len(vector.shape) != 1:
                            print(f"[警告] 向量 {original_idx} 维度异常: {vector.shape}")
                            vectors_dict[original_idx] = None
                            continue

                        if len(vector) == 0:
                            print(f"[警告] 向量 {original_idx} 为空")
                            vectors_dict[original_idx] = None
                            continue

                        # 保存到缓存，并写回结果字典
                        self._save_to_cache(cache_key, vector)
                        vectors_dict[original_idx] = vector

                    break  # 成功，跳出重试循环

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1) * 0.5
                        print(f"[警告] 批次 {current_batch}/{total_batches} 失败（尝试 {attempt + 1}/{max_retries}），{wait_time:.1f}秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # 最后一次尝试也失败，将这批文本标记为 None
                        print(f"[警告] 批次 {current_batch}/{total_batches} 最终失败（已重试{max_retries}次）: {e}")
                        for original_idx, text, cache_key in batch_items:
                            vectors_dict[original_idx] = None
        
        # 构建结果列表（按原始顺序），未成功获取的位置填 None
        result = []
        for idx in range(len(texts)):
            result.append(vectors_dict.get(idx, None))
        
        return result
    
    def get_hidden_state_vector(self, text: str) -> np.ndarray:
        """
        方法2：使用LLM推理时的hidden_state作为向量
        基于你提供的代码实现：提取最后一层最后一个token的hidden state
        
        参数:
            text: 文本内容
            
        返回:
            向量数组（最后一层最后一个token的hidden state）
        """
        if self.vector_method != "hidden_state":
            raise ValueError("此方法仅适用于hidden_state模式")
        
        if not self.hf_model or not self.hf_tokenizer:
            raise ValueError("模型未初始化，请先调用_load_hf_model()")
        
        # 检查缓存
        cache_key = self._get_cache_key(text)
        cached_vector = self._load_from_cache(cache_key)
        if cached_vector is not None:
            return cached_vector
        
        try:
            # 1. Tokenize输入
            inputs = self.hf_tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.hf_model_device) for k, v in inputs.items()}
            
            # 2. 模型推理（获取hidden_states）
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
            
            # 3. 提取hidden_states
            # hidden_states是一个tuple: (layer0, layer1, ..., layerN)
            hidden_states = outputs.hidden_states
            
            # 4. 取最后一层（最后一层的hidden state）
            last_layer_hidden_states = hidden_states[-1]  # shape: [batch_size, seq_len, hidden_dim]
            
            # 5. 取最后一个token的向量（最后一个token通常包含整个序列的信息）
            last_token_vector = last_layer_hidden_states[0, -1, :]  # shape: [hidden_dim]
            
            # 6. 转换为numpy数组
            vector = last_token_vector.cpu().numpy()
            
            # 保存到缓存
            self._save_to_cache(cache_key, vector)
            return vector
            
        except Exception as e:
            print(f"[警告] 获取hidden_state失败: {e}")
            # 返回零向量作为fallback
            if self.hf_model:
                # 尝试获取模型的hidden_size
                try:
                    hidden_size = self.hf_model.config.hidden_size
                    return np.zeros(hidden_size)
                except:
                    return np.zeros(4096)  # 默认维度
            return np.zeros(4096)
    
    def vectorize_steps(self, steps: List[Dict]) -> Tuple[np.ndarray, List[int]]:
        """
        对所有steps进行向量化（支持批量API调用，性能优化）
        
        参数:
            steps: step列表，每个step包含 'content' 字段
            
        返回:
            (向量矩阵, step索引列表)
        """
        # 准备需要向量化的steps，只保留 content 非空的 step
        tasks = []
        for idx, step in enumerate(steps):
            content = step.get('content', '')
            # idx = 该 step 在 all_steps（即 steps）中的下标
            # step：原始 step 对象
            # content：step 的文本内容
            if content:
                tasks.append((idx, step, content))
        
        if not tasks:
            return np.array([]), []
        
        step_indices = []
        vectors = []
        
        # 使用批量API（embedding_api方法，性能优化）
        if self.vector_method == "embedding_api":
            # texts 是对每个 step 的 content（推理步骤文本内容） 的列表
            texts = [content for _, _, content in tasks]
            
            # 批量调用API（一次性获取所有向量，大幅提升速度）
            try:
                print(f"[信息]开始批量向量化 {len(texts)} 个步骤...")
                import time as time_module
                start_time = time_module.time()

                # 所有非空 step 的向量化表示
                all_vectors = self.get_text_embeddings_batch(texts, batch_size=self.batch_size)
                
                # 将向量与索引对应
                valid_count = 0
                invalid_count = 0
                
                for i, (idx, step, content) in enumerate(tasks):
                    if i < len(all_vectors):
                        vector = all_vectors[i]
                        # 检查向量是否有效（非 None 且非空）
                        if vector is not None and len(vector) > 0:
                            # 向量矩阵：(N, target_dim) 的 float32 数组
                            vectors.append(vector)
                            # step_indices 是一个索引映射
                            # 向量矩阵里第 i 行对应原始 steps 列表中的第几个 step
                            step_indices.append(idx)
                            valid_count += 1
                        else:
                            invalid_count += 1
                    else:
                        invalid_count += 1
                
                elapsed_time = time_module.time() - start_time
                
                # 检查是否有任何有效的向量
                if len(vectors) == 0:
                    # 所有向量都失败了
                    print(f"\n[错误] 批量向量化完全失败: {len(tasks)} 个步骤全部失败，耗时 {elapsed_time:.1f}秒")
                    print(f"[错误] 失败统计: 总步骤={len(tasks)}, 无效={invalid_count}")
                    print(f"[错误] 原因: 批量API调用失败或返回的向量全部无效")
                    print(f"[错误] 建议:")
                    print(f"[错误]   1. 检查API配置（API_URL, API_KEY, embedding_model）")
                    print(f"[错误]   2. 检查网络连接")
                    print(f"[错误]   3. 查看上面的错误信息，可能有API限流或服务异常")
                    print(f"[错误]   4. 尝试减少 batch_size 或使用并行模式")
                    return np.array([]), []
                
                # 部分成功的情况
                success_rate = len(vectors) / len(tasks) * 100

                print(f"\n[信息] 批量向量化完成: {len(vectors)}/{len(tasks)} 成功 ({success_rate:.1f}%)，耗时 {elapsed_time:.1f}秒")
                
                if invalid_count > 0:
                    print(f"[警告] {invalid_count} 个步骤向量化失败（API调用失败或返回无效向量）")
                
                # 如果成功率太低（少于50%），给出警告并询问是否继续
                if success_rate < 50:
                    print(f"[警告] 向量化成功率过低 ({success_rate:.1f}%)，聚类结果可能不准确")
                    print(f"[警告] 建议: 检查API配置或网络连接，成功率应至少 > 50%")
                
            except Exception as e:
                print(f"[警告] 批量API调用异常，回退到并行模式: {e}")
                import traceback
                traceback.print_exc()
                # 回退到并行模式
                return self._vectorize_steps_parallel(tasks)
        
        # 串行处理（hidden_state方法）
        # 用 LLM 模型前向推理 时最后一层、最后一个 token 的 hidden state 作为向量
        elif self.vector_method == "hidden_state":
            for idx, step, content in tasks:
                try:
                    vector = self.get_hidden_state_vector(content)
                    if vector is not None:
                        vectors.append(vector)
                        step_indices.append(idx)
                except Exception as e:
                    print(f"[警告] 步骤 {idx} 向量化失败: {e}")
                    continue
        else:
            raise ValueError(f"未知的向量化方法: {self.vector_method}")
        
        # 如果没有成功向量化任何步骤，返回空数组
        if not vectors:
            print(f"[错误] 向量化完全失败，无法继续聚类分析")
            return np.array([]), []
        
        # 检查所有向量的维度是否一致，如果不一致则统一
        if vectors:
            # 找到最常见的维度（通常是成功的API调用返回的维度）
            dims = [len(v) for v in vectors]
            dim_counts = Counter(dims)
            target_dim = dim_counts.most_common(1)[0][0]
            
            # 统一所有向量的维度
            dimension_mismatch_count = 0
            for i, vec in enumerate(vectors):
                vec_dim = len(vec)
                if vec_dim != target_dim:
                    dimension_mismatch_count += 1
                    if dimension_mismatch_count <= 5:  # 只显示前5个详细警告，避免输出过多
                        print(f"[警告] 向量 {i} 维度不一致: {vec_dim} vs {target_dim}，调整为 {target_dim}")
                        if vec_dim < target_dim:
                            print(f"      [注意] 使用零填充，可能引入噪声")
                        else:
                            print(f"      [注意] 使用截断，可能丢失信息")
                    if vec_dim < target_dim:
                        # 如果维度小，用零填充
                        vec = np.pad(vec, (0, target_dim - vec_dim), 'constant', constant_values=0)
                    else:
                        # 如果维度大，截断
                        vec = vec[:target_dim]
                    vectors[i] = vec
            
            if dimension_mismatch_count > 5:
                print(f"[警告] 还有 {dimension_mismatch_count - 5} 个向量维度不一致（已统一处理）")
            elif dimension_mismatch_count > 0:
                print(f"[警告] 共 {dimension_mismatch_count} 个向量维度不一致（已统一处理）")
        
        try:
            return np.array(vectors, dtype=np.float32), step_indices
        except ValueError as e:
            print(f"[错误] 无法创建向量数组: {e}")
            print(f"[错误] 向量数量: {len(vectors)}")
            if vectors:
                dims = [len(v) for v in vectors]
                print(f"[错误] 向量维度: {dims[:10]}...")  # 只显示前10个
            # 如果还是失败，返回空数组
            return np.array([]), []
    
    def _vectorize_steps_parallel(self, tasks: List[Tuple]) -> Tuple[np.ndarray, List[int]]:
        """
        并行向量化（回退方法，当批量API失败时使用）
        
        参数:
            tasks: (idx, step, content) 元组列表
            
        返回:
            (向量矩阵, step索引列表)
        """
        vectors = [None] * len(tasks)
        step_indices = []
        
        def vectorize_single(task):
            idx, step, content = task
            try:
                vector = self.get_text_embedding(content)
                return idx, vector, None
            except Exception as e:
                return idx, None, e
        
        # 使用线程池并行调用API
        success_count = 0
        fail_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(vectorize_single, task): task for task in tasks}
            
            for future in as_completed(futures):
                idx, vector, error = future.result()
                if error:
                    fail_count += 1
                    if fail_count <= 5:  # 只显示前5个错误
                        print(f"[警告] 步骤 {idx} 向量化失败: {error}")
                    elif fail_count == 6:
                        print(f"[警告] ... (更多错误，不再显示)")
                    continue
                if vector is not None:
                    success_count += 1
                    vectors[len(step_indices)] = vector
                    step_indices.append(idx)
        
        # 显示统计信息
        if fail_count > 0:
            print(f"[信息] 并行向量化统计: 成功 {success_count}, 失败 {fail_count} (失败率: {fail_count/(success_count+fail_count)*100:.1f}%)")
        
        # 过滤掉None值
        valid_vectors = [v for v in vectors if v is not None]
        if not valid_vectors:
            return np.array([]), []
        
        # 检查所有向量的维度是否一致，如果不一致则统一
        if valid_vectors:
            # 找到最常见的维度（通常是成功的API调用返回的维度）
            dims = [len(v) for v in valid_vectors]
            dim_counts = Counter(dims)
            target_dim = dim_counts.most_common(1)[0][0]
            
            # 统一所有向量的维度
            for i, vec in enumerate(valid_vectors):
                vec_dim = len(vec)
                if vec_dim != target_dim:
                    print(f"[警告] 向量 {i} 维度不一致: {vec_dim} vs {target_dim}，调整为 {target_dim}")
                    if vec_dim < target_dim:
                        # 如果维度小，用零填充
                        vec = np.pad(vec, (0, target_dim - vec_dim), 'constant', constant_values=0)
                    else:
                        # 如果维度大，截断
                        vec = vec[:target_dim]
                    valid_vectors[i] = vec
        
        try:
            return np.array(valid_vectors, dtype=np.float32), step_indices
        except ValueError as e:
            print(f"[错误] 无法创建向量数组: {e}")
            print(f"[错误] 向量数量: {len(valid_vectors)}")
            if valid_vectors:
                dims = [len(v) for v in valid_vectors]
                print(f"[错误] 向量维度: {dims[:10]}...")  # 只显示前10个
            # 如果还是失败，返回空数组
            return np.array([]), []
        
        # 注意：以下重复代码已删除（原代码此处有重复块）
    
    def reduce_dimensions(self, vectors: np.ndarray) -> np.ndarray:
        """
        降维处理
        
        参数:
            vectors: 原始向量矩阵
            
        返回:
            降维后的向量矩阵
        """
        if not self.reduce_dim:
            return vectors
        
        if vectors.shape[0] < 2:
            return vectors  # 样本太少，无法降维
        
        # 标准化
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)
        # 缓存本轮scaler，便于后续复用/排查
        self._last_scaler = scaler
        
        if self.reduction_method == "pca":
            if not SKLEARN_AVAILABLE:
                raise ImportError("需要安装sklearn: pip install scikit-learn")
            
            # PCA降维
            max_dim = min(self.target_dim, vectors.shape[0] - 1, vectors.shape[1])
            pca = PCA(n_components=max_dim)
            reduced_vectors = pca.fit_transform(vectors_scaled)
            # 缓存本轮reducer
            self._last_reducer = pca
            
            print(f"[信息] PCA降维: {vectors.shape[1]} -> {max_dim} 维，解释方差比: {pca.explained_variance_ratio_.sum():.2%}")
            return reduced_vectors
        
        elif self.reduction_method == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError("需要安装umap: pip install umap-learn")
            
            # UMAP降维 - 修复：使用cosine距离更适合语义向量
            reducer = umap.UMAP(n_components=self.target_dim, random_state=42, metric="cosine")
            reduced_vectors = reducer.fit_transform(vectors_scaled)
            # 缓存本轮reducer
            self._last_reducer = reducer
            
            print(f"[信息] UMAP降维: {vectors.shape[1]} -> {self.target_dim} 维")
            return reduced_vectors
        
        else:
            raise ValueError(f"未知的降维方法: {self.reduction_method}")
    
    def cluster_steps(self, vectors: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
        """
        对向量进行聚类
        
        参数:
            vectors: 向量矩阵
            
        返回:
            (聚类标签数组, 聚类器对象)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("需要安装sklearn: pip install scikit-learn")
        
        # 清空上一轮缓存，避免误用
        self._last_cluster_model = None

        if vectors.shape[0] < 2:
            return np.array([0] * vectors.shape[0]), None  # 只有一个样本，返回单个聚类
        
        if self.clustering_method == "kmeans":
            # 自动确定聚类数量 - 改进的稳定方法，避免小样本时的剧烈波动
            if self.n_clusters is None:
                n_samples = vectors.shape[0]
                if n_samples <= 15:
                    # 小样本：使用更稳定的公式，避免跳变
                    # 使用 n//4 而不是 n//5，在小范围内变化更平滑
                    n_clusters = max(2, min(3, (n_samples + 2) // 4))  # 更保守的上限
                elif n_samples <= 50:
                    # 中等样本：使用平滑的log函数过渡，避免sqrt的跳跃
                    import math
                    # 从log2开始，平滑增长到log(n)/log(2)≈4左右
                    log_clusters = math.log2(n_samples)
                    n_clusters = max(2, min(6, int(log_clusters + 1.5)))  # +1.5提供平滑过渡
                elif n_samples <= 100:
                    # 中等偏大样本：继续平滑增长
                    import math
                    log_clusters = math.log2(n_samples)
                    n_clusters = max(2, min(8, int(log_clusters + 0.8)))  # 调整偏移量保持连续
                else:
                    # 大样本：使用更保守的比例，避免过度聚类
                    n_clusters = max(2, min(10, n_samples // 12))  # 从//10改为//12，更保守

                formula_result = n_clusters
            else:
                n_clusters = min(self.n_clusters, vectors.shape[0])
                formula_result = None
            
            # 调试打印：KMeans聚类参数
            if formula_result is not None:
                n_samples = vectors.shape[0]
                if n_samples <= 15:
                    formula_desc = f"stable_small: max(2, min(3, ({n_samples}+2)//4))"
                elif n_samples <= 50:
                    formula_desc = f"smooth_medium: max(2, min(6, log2({n_samples})+1.5))"
                elif n_samples <= 100:
                    formula_desc = f"smooth_large: max(2, min(8, log2({n_samples})+0.8))"
                else:
                    formula_desc = f"conservative_huge: max(2, min(10, {n_samples}//12))"
                print(f"[调试][KMEANS] n_samples={vectors.shape[0]} n_clusters={n_clusters} (公式: {formula_desc})")

                # 分析不稳定性：显示相邻样本大小的聚类数量变化
                if n_samples >= 2:
                    prev_n_samples = n_samples - 1
                    prev_clusters = self._calculate_clusters_for_samples(prev_n_samples)
                    if prev_clusters != n_clusters:
                        jump_ratio = n_clusters / prev_clusters if prev_clusters > 0 else float('inf')
                        print(f"[警告][聚类不稳定] 聚类数量跳变: n_samples {prev_n_samples} -> {n_samples}, clusters {prev_clusters} -> {n_clusters} ({jump_ratio:.1f}x)")
                        print(f"[警告][聚类不稳定] 这可能导致不同seed/agent数的聚类结果不一致")
            else:
                print(f"[调试][KMEANS] n_samples={vectors.shape[0]} n_clusters={n_clusters} (用户指定)")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            # 缓存本轮聚类器
            self._last_cluster_model = kmeans
            
            # 调试打印：cluster sizes
            cluster_sizes = Counter(labels)
            print(f"[调试][KMEANS] 聚类大小: {dict(cluster_sizes)}")
            
            print(f"[信息] KMeans聚类: {vectors.shape[0]} 个steps -> {n_clusters} 个聚类")
            return labels, kmeans
        
        elif self.clustering_method == "dbscan":
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            labels = dbscan.fit_predict(vectors)
            # 缓存本轮聚类器
            self._last_cluster_model = dbscan
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"[信息] DBSCAN聚类: {vectors.shape[0]} 个steps -> {n_clusters} 个聚类，{n_noise} 个噪声点")
            return labels, dbscan
        
        else:
            raise ValueError(f"未知的聚类方法: {self.clustering_method}")
    
    def refine_annotations_by_clustering_with_constraint(self,
                                        steps: List[Dict],
                                        step_annotations: Dict,
                                        question_id: str,
                                        round: Optional[int] = None,
                                        allow_false_to_true_only: bool = True,
                                        save_model: bool = False,
                                        reuse_saved_model: bool = False) -> Dict:
        """
        基于聚类结果修正step标注（支持约束条件）

        参数:
            allow_false_to_true_only: True=仅允许false转true，False=双向修正
            save_model: 是否保存聚类模型供后续使用
            reuse_saved_model: 是否尝试复用实例中已保存的聚类模型/降维器（不满足条件则回退为重新聚类）
        """
        if len(steps) < 2:
            return step_annotations  # 步骤太少，无法聚类

        # 调试打印：约束参数
        print(f"[调试][约束] allow_false_to_true_only={allow_false_to_true_only} save_model={save_model} reuse_saved_model={reuse_saved_model} majority_threshold={self.majority_threshold}")
        # 用户验证点：函数入口打印配置（更直观）
        print("[调试] allow_false_to_true_only =", allow_false_to_true_only)

        # 1. 向量化
        print(f"[信息] [聚类] 聚类分析开始: {len(steps)} 个步骤")
        import time as time_module
        total_start_time = time_module.time()

        vectors, step_indices = self.vectorize_steps(steps)

        # 检查向量化结果
        if vectors is None or vectors.shape[0] == 0:
            print("[错误] 向量化完全失败，无法继续聚类分析")
            # 向量化失败时，至少要将agent级别的标注保存到step_annotations中
            return self._build_annotations_from_steps(steps, step_annotations, question_id, round)

        if vectors.shape[0] < 2:
            print(f"[警告] 向量化后的steps太少 ({vectors.shape[0]} < 2)，跳过聚类修正")
            # 向量化失败时，至少要将agent级别的标注保存到step_annotations中
            return self._build_annotations_from_steps(steps, step_annotations, question_id, round)

        # 检查是否所有向量都是零向量
        try:
            if vectors.shape[0] > 0 and np.all(np.all(vectors == 0, axis=1)):
                print("[错误] 所有向量都是零向量，说明批量API调用失败，无法继续聚类分析")
                print("[错误] 建议：检查API配置、网络连接或尝试使用并行模式")
                # 向量化失败时，至少要将agent级别的标注保存到step_annotations中
                return self._build_annotations_from_steps(steps, step_annotations, question_id, round)
        except Exception as e:
            print(f"[错误] 向量格式检查失败: {e}，无法继续聚类分析")
            # 向量化失败时，至少要将agent级别的标注保存到step_annotations中
            return self._build_annotations_from_steps(steps, step_annotations, question_id, round)

        # 2. 降维 + 聚类（支持复用已保存模型）
        reused_model = False
        vectors_for_cluster = None
        cluster_labels = None
        cluster_model = None

        if reuse_saved_model and self.saved_cluster_model is not None:
            try:
                vectors_for_cluster = vectors
                if self.reduce_dim:
                    # 复用模式需要可 transform 的 scaler/reducer（PCA 支持 transform）
                    if self.saved_scaler is not None and self.saved_reducer is not None and hasattr(self.saved_reducer, "transform"):
                        dim_before = vectors_for_cluster.shape[1]
                        vectors_scaled = self.saved_scaler.transform(vectors_for_cluster)
                        vectors_for_cluster = self.saved_reducer.transform(vectors_scaled)
                        print(f"[调试][复用模型] 向量变换: {dim_before} -> {vectors_for_cluster.shape[1]} 使用已保存 scaler+reducer")
                    else:
                        print("[调试][复用模型][跳过] 缺少已保存 scaler/reducer 或 reducer 无 transform；将重新拟合降维+聚类")
                        vectors_for_cluster = None

                # 向量预处理（L2 normalize）- 在降维之后，聚类之前
                if vectors_for_cluster is not None:
                    print(f"[信息] 向量预处理 (L2 归一化) 复用模式...")
                    norms = np.linalg.norm(vectors_for_cluster, axis=1, keepdims=True)
                    vectors_for_cluster = vectors_for_cluster / (norms + 1e-12)  # 避免除零
                    print(f"[信息] 向量预处理完成（复用）")

                if vectors_for_cluster is not None and hasattr(self.saved_cluster_model, "predict"):
                    cluster_labels = self.saved_cluster_model.predict(vectors_for_cluster)
                    cluster_model = self.saved_cluster_model
                    reused_model = True
                    print(f"[调试][复用模型] 使用已保存聚类模型: {type(cluster_model).__name__}")
                    print(f"[调试][复用模型] 预测 {len(cluster_labels)} 个标签，{len(set(cluster_labels))} 个唯一聚类")
                else:
                    print(f"[调试][复用模型][跳过] vectors_for_cluster 为 None 或模型无 predict 方法")
                    print(f"[调试][复用模型][跳过] vectors_for_cluster: {vectors_for_cluster is not None}")
                    print(f"[调试][复用模型][跳过] has predict: {hasattr(self.saved_cluster_model, 'predict') if self.saved_cluster_model else False}")
            except Exception as e:
                print(f"[调试][复用模型][失败] {e} ；将重新拟合降维+聚类")
                import traceback
                traceback.print_exc()
                reused_model = False
                vectors_for_cluster = None
                cluster_labels = None
                cluster_model = None

        if not reused_model:
            # 3. 降维（如果需要）
            if self.reduce_dim:
                print(f"[信息] 降维处理 ({self.reduction_method})...")
                dim_before = vectors.shape[1]
                vectors = self.reduce_dimensions(vectors)
                dim_after = vectors.shape[1]
                print(f"[信息] 降维完成: {dim_before}维 -> {dim_after}维")

            # 4. 向量预处理（L2 normalize）- 在降维之后，聚类之前
            print(f"[信息] 向量预处理 (L2 normalize)...")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-12)  # 避免除零
            print(f"[信息] 向量预处理完成")

            # 5. 聚类
            print(f"[信息] [聚类] 聚类处理 ({self.clustering_method})...")
            cluster_labels, cluster_model = self.cluster_steps(vectors)
        else:
            # 复用模式下，vectors_for_cluster 已经是降维后的特征（若启用降维）
            vectors = vectors_for_cluster

        unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_points = list(cluster_labels).count(-1) if -1 in cluster_labels else 0
        print(f"[信息] [完成] 聚类完成: {unique_clusters} 个聚类，{noise_points} 个噪声点")

        # 5. 基于聚类结果修正标注（支持约束）
        print(f"[信息] [修正] 基于聚类结果修正标注...")
        refined_annotations = copy.deepcopy(step_annotations)  # 使用深拷贝避免修改原始数据
        # 关键修复：先把当前 steps 的基础标注写入（即使"无修正"，step_annotations 也不应全是 None）
        # 这能避免后续 step-level exchange 读取标注时大量 str-path=None。
        refined_annotations = self._build_annotations_from_steps(steps, refined_annotations, question_id, round)

        # 组织聚类结果
        cluster_annotations = {}  # {cluster_id: [step_indices]}
        for cluster_id in set(cluster_labels):
            cluster_annotations[cluster_id] = []

        for i, cluster_id in enumerate(cluster_labels):
            original_step_idx = step_indices[i]
            cluster_annotations[cluster_id].append(original_step_idx)

        # 对每个聚类进行多数投票（支持约束条件）
        correction_count = 0
        question_id_str = str(question_id)

        if question_id_str not in refined_annotations:
            refined_annotations[question_id_str] = {}

        for cluster_id, step_indices_in_cluster in cluster_annotations.items():
            if cluster_id == -1:  # DBSCAN的噪声点，跳过
                continue

            if len(step_indices_in_cluster) < 2:
                continue  # 聚类太小，跳过

            # 统计聚类中的标注分布
            correct_count = 0
            wrong_count = 0
            none_count = 0
            correct_steps = []  # 记录正确的步骤标识
            wrong_steps = []    # 记录错误的步骤标识

            for step_idx in step_indices_in_cluster:
                step = steps[step_idx]
                agent_id = step.get('agent_id')
                step_number = step.get('step_number')
                is_correct = step.get('is_correct')

                # 生成步骤标识
                step_label = f"{chr(65 + agent_id) if agent_id is not None else '?'}{step_number if step_number is not None else '?'}"

                if is_correct is True:
                    correct_count += 1
                    correct_steps.append(step_label)
                elif is_correct is False:
                    wrong_count += 1
                    wrong_steps.append(step_label)
                else:
                    none_count += 1

            total_labeled = correct_count + wrong_count
            cluster_size = len(step_indices_in_cluster)

            # 计算ratio（用于调试和阈值判断）
            correct_ratio = correct_count / total_labeled if total_labeled > 0 else 0
            wrong_ratio = wrong_count / total_labeled if total_labeled > 0 else 0

            # 应用约束条件进行修正
            if total_labeled >= 2:  # 至少需要2个标注才能进行多数投票
                # 调试打印：cluster决策的ratio/threshold/rule_used
                print(f"[调试][聚类] cid={cluster_id} 正确={correct_count} 错误={wrong_count} 总数={total_labeled} correct_ratio={correct_ratio:.3f} wrong_ratio={wrong_ratio:.3f} threshold={self.majority_threshold} allow_false_to_true_only={allow_false_to_true_only}")

                # 计算多数投票结果（按 ratio 阈值，而不是仅按 count）
                rule_used = None
                majority_label = None
                if correct_ratio >= self.majority_threshold:
                    majority_label = True
                    rule_used = "correct_majority_by_ratio"
                elif wrong_ratio >= self.majority_threshold:
                    majority_label = False
                    rule_used = "wrong_majority_by_ratio"
                else:
                    rule_used = "skipped (no_majority_by_ratio)"
                    print(f"[调试][决策] cid={cluster_id} decision=跳过 rule_used='{rule_used}'")
                    continue

                if majority_label is True:
                    target_value = True
                    steps_to_correct = wrong_steps
                    if allow_false_to_true_only:
                        rule_used = "correct_majority_by_ratio (allow_false_to_true_only=True)"
                    else:
                        rule_used = "correct_majority_by_ratio (bidirectional)"
                else:
                    # majority_label is False
                    if allow_false_to_true_only:
                        rule_used = "skipped (wrong_majority_by_ratio but allow_false_to_true_only=True)"
                        print(f"[调试][决策] cid={cluster_id} decision=跳过 rule_used='{rule_used}'")
                        continue
                    target_value = False
                    steps_to_correct = correct_steps
                    rule_used = "wrong_majority_by_ratio (bidirectional)"

                if not steps_to_correct:
                    # 没有任何需要改写的 step（例如该cluster里本来就全True或全False）
                    print(f"[调试][决策] cid={cluster_id} decision=无操作 target_value={target_value} rule_used='{rule_used}'")
                    continue

                print(f"[调试][决策] cid={cluster_id} decision=应用 target_value={target_value} rule_used='{rule_used}' steps_to_correct={len(steps_to_correct)}")

                # 应用修正
                for step_idx in step_indices_in_cluster:
                    step = steps[step_idx]
                    agent_id = step.get('agent_id')
                    step_number = step.get('step_number')
                    current_round = step.get('round', round)

                    if agent_id is not None and step_number is not None:
                        step_label = f"{chr(65 + agent_id)}{step_number}"

                        # 检查是否需要修正
                        if step_label in steps_to_correct:
                            current_is_correct = step.get('is_correct')
                            
                            # 检查约束违反
                            old = current_is_correct
                            new = target_value
                            if old is True and new is False:
                                print(
                                    "[调试][违反约束] True->False 被约束阻止！",
                                    "qid=", question_id_str,
                                    "agent=", agent_id,
                                    "round=", current_round,
                                    "step=", step_number,
                                    "cid=", cluster_id,
                                    "rule=", rule_used
                                )
                                continue  # 阻止违反约束的修正
                            elif allow_false_to_true_only and old is False and new is True:
                                print(f"[调试][约束] 允许 False->True")
                            
                            # 更新标注 - 统一使用字符串key确保一致性
                            agent_id_str = str(agent_id)
                            round_str = str(current_round)
                            step_str = str(step_number)
                            
                            if agent_id_str not in refined_annotations[question_id_str]:
                                refined_annotations[question_id_str][agent_id_str] = {}

                            if round_str not in refined_annotations[question_id_str][agent_id_str]:
                                refined_annotations[question_id_str][agent_id_str][round_str] = {}

                            # 验证A：写入时的key类型诊断
                            print("[调试][写入] qid=", question_id_str,
                            "agent_id(类型)=", agent_id, type(agent_id),
                            "round(类型)=", current_round, type(current_round),
                            "step(类型)=", step_number, type(step_number),
                            "->", target_value)

                            refined_annotations[question_id_str][agent_id_str][round_str][step_str] = target_value
                            
                            # 验证写入是否成功
                            stored_value = refined_annotations.get(question_id_str, {}).get(agent_id_str, {}).get(round_str, {}).get(step_str, None)
                            if stored_value != target_value:
                                print(f"[警告] 写入验证失败: 期望{target_value}, 实际{stored_value}")
                            correction_count += 1

                            print(f"  [修正] {step_label}: {current_is_correct} → {target_value} (聚类 {cluster_id})")

        total_time = time_module.time() - total_start_time

        # 输出详细的聚类统计信息
        log_clustering_stats(steps, correction_count, len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0), total_time)

        print(f"[信息] 聚类修正完成: 修正了 {correction_count} 个步骤标注，耗时 {total_time:.2f}秒")

        # 修复：将 save_model 逻辑移到函数末尾，避免提前返回导致 refined_annotations 未定义
        if save_model:
            # 验证 cluster_model 是有效的聚类模型
            try:
                from sklearn.cluster import KMeans, DBSCAN
                if cluster_model is not None and isinstance(cluster_model, (KMeans, DBSCAN)):
                    # 保存聚类模型到实例变量中
                    # 直接保存从 cluster_steps() 返回的聚类器对象（KMeans/DBSCAN）
                    self.saved_cluster_model = cluster_model
                    self.saved_cluster_labels = cluster_labels
                    self.saved_scaler = getattr(self, "_last_scaler", None)
                    self.saved_reducer = getattr(self, "_last_reducer", None)

                    print(f"[信息] [保存] 聚类模型已保存: {type(cluster_model).__name__}")
                    print(f"[调试][保存模型] 模型有 {getattr(cluster_model, 'n_clusters', 'unknown')} 个聚类")
                    if hasattr(cluster_model, 'cluster_centers_'):
                        print(f"[调试][保存模型] 模型有 {len(cluster_model.cluster_centers_)} 个聚类中心")
                else:
                    print(f"[警告] [保存] 无法保存聚类模型: cluster_model 不是有效的 KMeans/DBSCAN 对象")
                    print(f"[警告] cluster_model type: {type(cluster_model)}, value: {cluster_model}")
                    # 不保存无效的模型，但不阻止返回结果
                    self.saved_cluster_model = None
                    self.saved_cluster_labels = None
                    self.saved_scaler = None
                    self.saved_reducer = None

            except ImportError:
                print(f"[警告] [保存] sklearn 未安装，无法验证聚类模型")
                self.saved_cluster_model = None
                self.saved_cluster_labels = None
                self.saved_scaler = None
                self.saved_reducer = None

            # 额外检查 scaler 和 reducer
            if self.saved_scaler is not None:
                print(f"[调试][保存模型] 已保存 scaler: {type(self.saved_scaler).__name__}")
            else:
                print(f"[调试][保存模型] 未保存 scaler")

            if self.saved_reducer is not None:
                print(f"[调试][保存模型] 已保存 reducer: {type(self.saved_reducer).__name__}")
            else:
                print(f"[调试][保存模型] 未保存 reducer")

        return refined_annotations

    def _build_annotations_from_steps(self, steps: List[Dict], step_annotations: Dict, question_id: str, round: Optional[int] = None) -> Dict:
        """
        从steps中构建基础标注（向量化失败时的回退方案）
        
        将steps中的is_correct标注保存到step_annotations字典中
        """
        question_id_str = str(question_id)
        
        # 确保question_id存在（保留已有的majority_answer等）
        if question_id_str not in step_annotations:
            step_annotations[question_id_str] = {}
        
        # 从steps中提取标注
        for step in steps:
            agent_id = step.get('agent_id')
            step_number = step.get('step_number')
            is_correct = step.get('is_correct')
            current_round = step.get('round', round)
            
            if agent_id is not None and step_number is not None and is_correct is not None:
                agent_id_str = str(agent_id)
                round_str = str(current_round) if current_round is not None else str(round) if round is not None else "0"
                step_number_str = str(step_number)
                
                if agent_id_str not in step_annotations[question_id_str]:
                    step_annotations[question_id_str][agent_id_str] = {}
                
                if round_str not in step_annotations[question_id_str][agent_id_str]:
                    step_annotations[question_id_str][agent_id_str][round_str] = {}
                
                step_annotations[question_id_str][agent_id_str][round_str][step_number_str] = is_correct
        
        return step_annotations

    def refine_annotations_by_clustering(self,
                                        steps: List[Dict],
                                        step_annotations: Dict,
                                        question_id: str,
                                        round: Optional[int] = None) -> Dict:
        """
        基于聚类结果修正step标注
        
        参数:
            steps: step列表，包含 'agent_id', 'step_number', 'round', 'content', 'is_correct' 等字段
                  - is_correct字段表示：该step所属agent的最终答案是否正确（通过多数投票判断）
                  - 不是：该step本身的推理过程是否正确
                  如果steps中包含'round'字段，则支持跨round的标注修正；否则使用参数中的round
            step_annotations: 原始标注字典 {question_id: {agent_id: {round: {step_number: is_correct}}}}
            question_id: 问题ID
            round: 轮次（可选，如果steps中包含round字段则忽略此参数）
            
        返回:
            修正后的标注字典
        """
        if len(steps) < 2:
            return step_annotations  # 步骤太少，无法聚类
        
        # 1. 向量化
        print(f"\n[信息] 聚类分析开始: {len(steps)} 个步骤")
        import time as time_module
        total_start_time = time_module.time()
        
        vectors, step_indices = self.vectorize_steps(steps)
        
        # 检查向量化结果
        if vectors is None or vectors.shape[0] == 0:
            print("[错误] 向量化完全失败，无法继续聚类分析")
            # 向量化失败时，至少要将agent级别的标注保存到step_annotations中
            return self._build_annotations_from_steps(steps, step_annotations, question_id, round)
        
        if vectors.shape[0] < 2:
            print(f"[警告] 向量化后的steps太少 ({vectors.shape[0]} < 2)，跳过聚类修正")
            # 向量化失败时，至少要将agent级别的标注保存到step_annotations中
            return self._build_annotations_from_steps(steps, step_annotations, question_id, round)
        
        # 检查是否所有向量都是零向量（可能表示批量API失败）
        # 注意：需要处理空数组的情况
        try:
            if vectors.shape[0] > 0 and np.all(np.all(vectors == 0, axis=1)):
                print("[错误] 所有向量都是零向量，说明批量API调用失败，无法继续聚类分析")
                print("[错误] 建议：检查API配置、网络连接或尝试使用并行模式")
                # 向量化失败时，至少要将agent级别的标注保存到step_annotations中
                return self._build_annotations_from_steps(steps, step_annotations, question_id, round)
        except Exception as e:
            # 如果检查失败，说明向量格式有问题，也应该返回
            print(f"[错误] 向量格式检查失败: {e}，无法继续聚类分析")
            # 向量化失败时，至少要将agent级别的标注保存到step_annotations中
            return self._build_annotations_from_steps(steps, step_annotations, question_id, round)
        
        # 2. 向量预处理（L2 normalize）
        print(f"[信息] 向量预处理 (L2 normalize)...")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-12)  # 避免除零
        print(f"[信息] 向量预处理完成")

        # 3. 降维（如果需要）
        if self.reduce_dim:
            print(f"[信息] 降维处理 ({self.reduction_method})...")
            dim_before = vectors.shape[1]
            vectors = self.reduce_dimensions(vectors)
            dim_after = vectors.shape[1]
            print(f"[信息] 降维完成: {dim_before}维 -> {dim_after}维")
        
        # 3. 聚类
        print(f"[信息] 聚类处理 ({self.clustering_method})...")
        cluster_labels, cluster_model = self.cluster_steps(vectors)
        unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_points = list(cluster_labels).count(-1) if -1 in cluster_labels else 0
        print(f"[信息] 聚类完成: {unique_clusters} 个聚类，{noise_points} 个噪声点")
        
        # 4. 基于聚类结果修正标注
        print(f"[信息] 🎯 基于聚类结果修正标注...")
        refined_annotations = copy.deepcopy(step_annotations)  # 使用深拷贝避免修改原始数据
        
        # 组织聚类结果
        cluster_annotations = {}  # {cluster_id: [step_indices]}
        for cluster_id in set(cluster_labels):
            cluster_annotations[cluster_id] = []
        
        for i, cluster_id in enumerate(cluster_labels):
            original_step_idx = step_indices[i]
            cluster_annotations[cluster_id].append(original_step_idx)
        
        # 对每个聚类进行多数投票
        correction_count = 0
        question_id_str = str(question_id)
        
        if question_id_str not in refined_annotations:
            refined_annotations[question_id_str] = {}
        
        for cluster_id, step_indices_in_cluster in cluster_annotations.items():
            if cluster_id == -1:  # DBSCAN的噪声点，跳过
                continue
            
            if len(step_indices_in_cluster) < 2:
                continue  # 聚类太小，跳过
            
            # 统计聚类中的标注分布
            correct_count = 0
            wrong_count = 0
            none_count = 0
            correct_steps = []  # 记录正确的步骤标识（如C31, C32）
            wrong_steps = []    # 记录错误的步骤标识（如C33, C34, B23）
            
            for step_idx in step_indices_in_cluster:
                step = steps[step_idx]
                agent_id = step.get('agent_id')
                step_number = step.get('step_number')
                is_correct = step.get('is_correct')
                
                # 生成步骤标识（如A11, B21, C33）
                step_label = f"{chr(65 + agent_id) if agent_id is not None else '?'}{step_number if step_number is not None else '?'}"
                
                if is_correct is True:
                    correct_count += 1
                    correct_steps.append(step_label)
                elif is_correct is False:
                    wrong_count += 1
                    wrong_steps.append(step_label)
                else:
                    none_count += 1
            
            total_labeled = correct_count + wrong_count
            cluster_size = len(step_indices_in_cluster)
            
            # 打印聚类块的详细信息（对应图片中的分析思路）
            correct_steps_str = ', '.join(correct_steps[:10]) + ('...' if len(correct_steps) > 10 else '') if correct_steps else '无'
            wrong_steps_str = ', '.join(wrong_steps[:10]) + ('...' if len(wrong_steps) > 10 else '') if wrong_steps else '无'
            print(f"  [聚类块 {cluster_id}] 块大小: {cluster_size}, 对的有: {correct_count}个 ({correct_steps_str}), "
                  f"错的有: {wrong_count}个 ({wrong_steps_str})")
            
            if total_labeled == 0:
                continue  # 没有标注，跳过
            
            # 多数投票
            correct_ratio = correct_count / total_labeled if total_labeled > 0 else 0
            wrong_ratio = wrong_count / total_labeled if total_labeled > 0 else 0

            # 调试打印：cluster决策的ratio/threshold/rule_used
            if correct_ratio >= self.majority_threshold:
                rule_used = "correct_majority"
                majority_label = True
            elif wrong_ratio >= self.majority_threshold:
                rule_used = "wrong_majority"
                majority_label = False
            else:
                rule_used = "no_majority"
                majority_label = None

            # 修正标注
            # 注意：这里的"正确/错误"指的是step所属agent的最终答案是否正确，不是step本身的正确性
            if majority_label is None:
                # 没有明确的多数，不修正
                continue
            
            # 应用修正
            for step_idx in step_indices_in_cluster:
                step = steps[step_idx]
                agent_id = step.get('agent_id')
                step_number = step.get('step_number')
                current_is_correct = step.get('is_correct')
                # 支持从step中获取round（用于跨round聚类）
                step_round = step.get('round', round)
                
                if agent_id is None or step_number is None or step_round is None:
                    continue
                
                # 只修正有标注的步骤（把错误的改成正确的，或反之）
                if current_is_correct is not None and current_is_correct != majority_label:
                    # 更新refined_annotations - 统一使用字符串key
                    agent_id_str = str(agent_id)
                    step_round_str = str(step_round)
                    step_number_str = str(step_number)
                    
                    if agent_id_str not in refined_annotations[question_id_str]:
                        refined_annotations[question_id_str][agent_id_str] = {}
                    if step_round_str not in refined_annotations[question_id_str][agent_id_str]:
                        refined_annotations[question_id_str][agent_id_str][step_round_str] = {}
                    
                    refined_annotations[question_id_str][agent_id_str][step_round_str][step_number_str] = majority_label
                    correction_count += 1
        
        # 计算总耗时
        total_elapsed_time = time_module.time() - total_start_time
        
        print(f"[信息] 标注修正完成: 共修正 {correction_count} 个步骤")
        print(f"[信息] ⏱️  总耗时: {total_elapsed_time:.1f}秒 ({total_elapsed_time/60:.1f}分钟)")
        print(f"[信息] 📈 平均速度: {len(steps)/total_elapsed_time:.1f} 步骤/秒")
        
        return refined_annotations


def refine_step_annotations_with_clustering(
    steps: List[Dict],
    step_annotations: Dict,
    question_id: str,
    round: Optional[int] = None,
    api_url: str = "https://api.zhizengzeng.com/v1",
    api_key: str = None,
    **kwargs
) -> Dict:
    """
    便捷函数：使用聚类修正step标注
    
    参数:
        steps: step列表（如果包含'round'字段，则支持跨round聚类）
        step_annotations: 原始标注字典
        question_id: 问题ID
        round: 轮次（可选，如果steps中包含round字段则忽略此参数）
        api_url: API地址
        api_key: API密钥
        **kwargs: 其他参数传递给StepClusteringRefiner
        
    返回:
        修正后的标注字典
    """
    refiner = StepClusteringRefiner(
        api_url=api_url,
        api_key=api_key,
        **kwargs
    )
    
    return refiner.refine_annotations_by_clustering(
        steps=steps,
        step_annotations=step_annotations,
        question_id=question_id,
        round=round
    )


def perform_online_clustering_with_majority(
    steps, majority_answer, question_id, round, api_url=None, api_key=None,
    save_model=False, **kwargs
):
    """
    基于多数答案进行在线聚类标注（第一轮）

    1. 基于多数答案进行agent级别标注
    2. 聚类分析，基于聚类结果修正标注（仅允许false转true）
    3. 可选择保存聚类模型供第二轮使用
    """
    # 初始化聚类器
    refiner = StepClusteringRefiner(
        api_url=api_url or DEFAULT_API_URL,
        api_key=api_key or DEFAULT_API_KEY,
        vector_method=DEFAULT_VECTOR_METHOD,
        reduce_dim=DEFAULT_REDUCE_DIM,
        reduction_method=DEFAULT_REDUCTION_METHOD,
        target_dim=DEFAULT_TARGET_DIM,
        clustering_method=DEFAULT_CLUSTERING_METHOD,
        majority_threshold=DEFAULT_MAJORITY_THRESHOLD,
        max_workers=DEFAULT_MAX_WORKERS,
        **kwargs
    )

    # 调试打印：验证不同 round 是否复用同一 refiner 实例
    print("[调试] round=", round, "使用 refiner id:", id(refiner))

    # 1. 基于多数答案进行agent级别标注
    step_annotations = {}
    question_id_str = str(question_id)

    # 为每个agent的所有steps设置初始标注
    agent_correctness = {}  # agent_id -> is_correct
    for step in steps:
        agent_id = step.get('agent_id')
        if agent_id not in agent_correctness:
            # 检查该agent的答案是否等于多数答案
            # 注意：这里需要从step中获取agent的答案，但当前step中可能不包含答案信息
            # 警告：用户需要确保step中包含agent_answer字段，或从外部传入agent_answers字典
            agent_answer = step.get('agent_answer')  # 尝试从step获取
            if agent_answer is None:
                print(f"[警告] 无法获取agent {agent_id} 的答案，假设为错误")
                agent_correctness[agent_id] = False
            else:
                agent_correctness[agent_id] = (agent_answer == majority_answer)

    # 设置step级别的标注
    if question_id_str not in step_annotations:
        step_annotations[question_id_str] = {}

    for step in steps:
        agent_id = step.get('agent_id')
        step_number = step.get('step_number')
        current_round = step.get('round', round)

        # 关键：统一将所有 key 转为字符串，避免后续策略中出现 int/str 混用（如 r.isdigit()）
        if agent_id is None or step_number is None:
            continue
        agent_id_str = str(agent_id)
        round_str = str(current_round if current_round is not None else (round if round is not None else 0))
        step_number_str = str(step_number)

        if agent_id_str not in step_annotations[question_id_str]:
            step_annotations[question_id_str][agent_id_str] = {}

        if round_str not in step_annotations[question_id_str][agent_id_str]:
            step_annotations[question_id_str][agent_id_str][round_str] = {}

        # agent级别标注：如果agent答案正确，所有steps都标记为正确
        is_correct = agent_correctness.get(agent_id, False)
        step_annotations[question_id_str][agent_id_str][round_str][step_number_str] = is_correct
        step['is_correct'] = is_correct

        # 调试打印：WRITE操作的key类型
        print(f"[调试] 写入: {question_id_str}[{agent_id_str}][{round_str}][{step_number_str}] = {is_correct} (类型: str/str/str/str)")

    # 2. 聚类分析（仅允许false转true）
    if len(steps) >= 2:
        refined_annotations = refiner.refine_annotations_by_clustering_with_constraint(
            steps=steps,
            step_annotations=step_annotations,
            question_id=question_id,
            round=round,
            allow_false_to_true_only=True,  # 仅允许false转true
            save_model=save_model
        )
        return refined_annotations

    return step_annotations

def update_clustering_with_round1_results(
    question_id, round1_majority, api_url=None, api_key=None, **kwargs
):
    """
    复用第一轮聚类模型，基于第一轮结果更新标注（第二轮）

    1. 加载第一轮的聚类模型
    2. 基于第一轮的多数答案重新判断agent正确性
    3. 更新标注（仅允许false转true）
    """
    # TODO: 实现复用聚类模型的逻辑
    # 这里需要：
    # 1. 加载第一轮保存的聚类模型
    # 2. 使用第一轮的结果重新标注
    # 3. 应用聚类修正（仅允许false转true）

    # 暂时返回空字典，需要完整实现
    return {}


def analyze_cluster_stability():
    """
    分析聚类数量自动确定的稳定性问题（修复：统一函数名并修正打印内容）
    """
    print("=" * 80)
    print("聚类数量稳定性分析")
    print("=" * 80)

    # 测试不同样本大小的聚类数量
    test_samples = list(range(2, 51))  # 测试2-50个样本

    print("\n当前聚类数量计算规则:")
    print("n <= 15: max(2, min(3, (n+2)//4))")
    print("n <= 50: max(2, min(6, log2(n)+1.5))")
    print("n <= 100: max(2, min(8, log2(n)+0.8))")
    print("n > 100: max(2, min(10, n//12))")

    print("\n样本大小 -> 聚类数量:")

    refiner = StepClusteringRefiner()  # 创建实例来使用方法
    prev_clusters = None
    jump_points = []

    for n in test_samples:
        clusters = refiner._calculate_clusters_for_samples(n)
        marker = ""
        if prev_clusters is not None and clusters != prev_clusters:
            marker = " ← JUMP!"
            jump_points.append((n-1, n, prev_clusters, clusters))
        # 修复：原来错误地打印了"6"，现在打印正确的样本数和聚类数
        print(f"  {n:3d} -> {clusters}{marker}")
        prev_clusters = clusters

    print("\n跳变点分析:")
    if jump_points:
        for prev_n, curr_n, prev_c, curr_c in jump_points:
            ratio = curr_c / prev_c if prev_c > 0 else float('inf')
            print(f"  {prev_n} → {curr_n}: {prev_c} → {curr_c} (变化倍数: {ratio:.1f}x)")
    else:
        print("  无跳变点！聚类数量变化平滑")

    print("\n稳定性改进:")
    print("1. 使用(n+2)//4替代n//5，减少小样本跳变")
    print("2. 使用log2(n)替代sqrt(n)，提供更平滑的过渡")
    print("3. 调整范围边界，从n<=20改为n<=15，减少过渡阶段跳变")
    print("4. 大样本使用n//12替代n//10，更保守减少过度聚类")

    print("\n预期效果:")
    print("- 小样本波动减少：不同agent数导致的样本大小变化不再引起大的聚类数量跳变")
    print("- 过渡平滑：从固定值到log函数的过渡更加自然")
    print("- flip行为稳定：聚类结果的一致性提高，减少随机性")


