"""
诊断脚本：排查 exchange1 "需要至少 2 个 step 向量，当前 0" 问题

以 question_id=418 为例，逐步追溯 expand → flatten → embedding 全链路，
打印每一关键节点的状态，精确定位向量数为 0 的原因。

排查思路：
  1. 从 expand 缓存加载 418 题的 agent 回复
  2. 调用 extract_steps_from_response 检查每个 agent 的 step 提取结果
  3. 调用 expand_flatten_all_steps 汇总 all_steps，检查数量
  4. 调用 expand_run_embedding，捕获内部日志，观察是否进入向量化
  5. 单独模拟 vectorize_steps 流程，检查：
     - 缓存命中情况（.step_clustering_cache/）
     - Embedding API 是否可以成功调用
     - 若 API 失败，显示具体错误信息
"""

import json
import sys
import os
import numpy as np

# ── 路径与配置 ──────────────────────────────────────────────────────────────
EXPAND_CACHE_PATH = (
    r"D:\AAAI2026_MADC_wzy\qwen2.5-7b-instruct\results\debate_zy\math_500_id"
    r"\debate_zy_qwen2.5-7b-instruct_10_1_expand_agent_com0_False.json"
)
TARGET_QID = "431"

# ── 导入项目模块 ─────────────────────────────────────────────────────────────
sys.path.insert(0, r"D:\AAAI2026_MADC_wzy")

from wzy_multi_agent_debate_expand import (
    default_expand_config,
    extract_steps_from_response,
    extract_answer_from_text,
    is_correct_answer,
    expand_compute_majority_and_agent_results,
    expand_flatten_all_steps,
    expand_run_embedding,
    get_expand_cache_entry,
)
from wzy_step_clustering import StepClusteringRefiner


def _sep(title="", width=70, char="─"):
    if title:
        print(f"\n{'─'*4} {title} {'─'*(width - len(title) - 6)}")
    else:
        print(char * width)


# ════════════════════════════════════════════════════════════════════════════
# Step 1: 从缓存加载 418 题数据
# ════════════════════════════════════════════════════════════════════════════
_sep("Step 1: 加载 418 题的 expand 缓存", char="═")

with open(EXPAND_CACHE_PATH, "r", encoding="utf-8") as f:
    cache_data = json.load(f)

entry = None
matched_key = None
for k, v in cache_data.items():
    if isinstance(v, list) and len(v) >= 3 and str(v[2]) == TARGET_QID:
        entry = v
        matched_key = k
        break

if entry is None:
    print(f"[错误] 未在缓存文件中找到 question_id={TARGET_QID} 的记录，脚本终止。")
    sys.exit(1)

agent_contexts = entry[0]
ground_truth   = entry[1]
question_id    = entry[2]
question       = matched_key          # 缓存以题干文本作为 key

print(f"  question_id : {question_id}")
print(f"  ground_truth: {ground_truth!r}")
print(f"  agent 数量  : {len(agent_contexts)}")
print(f"  题干前60字符: {question[:60]!r}...")


# ════════════════════════════════════════════════════════════════════════════
# Step 2: 检查每个 agent 的 assistant 回复与 step 提取结果
# ════════════════════════════════════════════════════════════════════════════
_sep("Step 2: agent 回复与 step 提取检查", char="═")

cfg = default_expand_config()
cfg.num_agents = len(agent_contexts)

empty_response_agents = []
zero_step_agents = []

for i, ctx in enumerate(agent_contexts):
    if len(ctx) >= 3 and ctx[2].get("role") == "assistant":
        resp = ctx[2].get("content", "")
    else:
        resp = ""

    steps = extract_steps_from_response(resp)
    answer = extract_answer_from_text(resp, is_math=cfg.is_math)

    status = "OK" if steps else "NO_STEPS"
    if not resp:
        status = "EMPTY_RESPONSE"
        empty_response_agents.append(i)
    if not steps:
        zero_step_agents.append(i)

    print(
        f"  Agent {i:>2}: 回复={len(resp):>5}字符  steps={len(steps):>2}个  "
        f"answer={answer!r:<12}  [{status}]"
    )

print()
if empty_response_agents:
    print(f"  [警告] 以下 agent 回复为空: {empty_response_agents}")
else:
    print("  [OK] 所有 agent 均有回复内容")
if zero_step_agents:
    print(f"  [警告] 以下 agent 未提取到 step: {zero_step_agents}")
else:
    print("  [OK] 所有 agent 均成功提取到 step")


# ════════════════════════════════════════════════════════════════════════════
# Step 3: 计算多数票 & 展平 all_steps
# ════════════════════════════════════════════════════════════════════════════
_sep("Step 3: 多数票 & all_steps 汇总", char="═")

majority_answer, agent_results = expand_compute_majority_and_agent_results(
    agent_contexts, cfg
)
print(f"  majority_answer: {majority_answer!r}")
print(f"  ground_truth   : {ground_truth!r}")
print(
    f"  是否与 GT 一致 : "
    f"{is_correct_answer(majority_answer, ground_truth, is_math=cfg.is_math)}"
)

all_steps = expand_flatten_all_steps(agent_results)
total_steps = len(all_steps)
correct_steps = sum(1 for s in all_steps if s.get("is_correct") is True)
wrong_steps   = sum(1 for s in all_steps if s.get("is_correct") is False)

print(f"\n  all_steps 总数  : {total_steps}")
print(f"    is_correct=True : {correct_steps}")
print(f"    is_correct=False: {wrong_steps}")

if total_steps < 2:
    print(f"\n  [致命] all_steps 数量={total_steps} < 2，"
          f"expand_run_embedding 将直接跳过向量化，返回 (None, None)")
    print("  → 根因: expand 阶段 agent 回复为空，未提取到足够的 step")
    sys.exit(0)
else:
    print(f"\n  [OK] all_steps 数量={total_steps} >= 2，可以进入向量化")


# ════════════════════════════════════════════════════════════════════════════
# Step 4: 检查 step_clustering_cache 命中情况
# ════════════════════════════════════════════════════════════════════════════
_sep("Step 4: 向量缓存命中情况检查", char="═")

# 与 expand_run_embedding 一致：仅使用各 step 的 content，不拼接题干
texts = [s.get("content", "") for s in all_steps if s.get("content", "")]
print(f"  待向量化文本数: {len(texts)}")

# 用与 expand_run_embedding 完全相同的参数构建 refiner，仅做缓存查询
refiner = StepClusteringRefiner(
    api_url=cfg.api_url,
    api_key=cfg.api_key,
    vector_method="embedding_api",
    embedding_model=cfg.embedding_model,
    reduce_dim=False,
    batch_size=20,
)

cache_hits = 0
cache_misses = 0
miss_indices = []

for idx, text in enumerate(texts):
    if not text:
        continue
    key = refiner._get_cache_key(text)
    vec = refiner._load_from_cache(key)
    if vec is not None:
        cache_hits += 1
    else:
        cache_misses += 1
        miss_indices.append(idx)

print(f"  缓存命中: {cache_hits}/{len(texts)}")
print(f"  缓存缺失: {cache_misses}/{len(texts)}")

if cache_misses == 0:
    print("  [OK] 所有向量已缓存，向量化不依赖 API 调用")
    print("  -> 若仍出现 '向量数为 0'，说明缓存读取逻辑本身有问题（pkl 损坏或维度异常）")
else:
    print(f"  [信息] 以下 {len(miss_indices)} 个 step 索引需调用 API：")
    for mi in miss_indices[:10]:
        print(f"    [{mi}] {texts[mi][:60]!r}...")
    if len(miss_indices) > 10:
        print(f"    ...（共 {len(miss_indices)} 个，仅显示前 10）")


# ════════════════════════════════════════════════════════════════════════════
# Step 5: 测试 Embedding API 是否可用（仅用第 1 个未命中的 step）
# ════════════════════════════════════════════════════════════════════════════
_sep("Step 5: Embedding API 连通性测试", char="═")

if cache_misses == 0:
    print("  [跳过] 所有向量均已缓存，无需调用 API")
else:
    test_text = texts[miss_indices[0]] if miss_indices else texts[0]
    print(f"  测试文本前80字符: {test_text[:80]!r}...")
    print(f"  Embedding 模型: {cfg.embedding_model}")
    print(f"  API URL: {cfg.api_url}")

    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.api_url, api_key=cfg.api_key)
        response = client.embeddings.create(
            model=cfg.embedding_model,
            input=[test_text],
        )
        if response and response.data:
            vec = np.array(response.data[0].embedding)
            print(f"  [OK] API 调用成功！向量维度: {vec.shape[0]}")
        else:
            print("  [错误] API 返回空数据（response.data 为空）")
    except Exception as e:
        print(f"  [错误] API 调用失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════════
# Step 6: 直接调用 expand_run_embedding，观察最终返回值
# ════════════════════════════════════════════════════════════════════════════
_sep("Step 6: 调用 expand_run_embedding，检查返回值", char="═")

step_vectors, step_indices = expand_run_embedding(all_steps, cfg)

print()
_sep("Step 6 诊断结论", char="═")
if step_vectors is None:
    n = 0
    print(f"  [结论] step_vectors = None（向量化被跳过或发生异常）")
    if total_steps < 2:
        print(f"  → 根因: all_steps 数量={total_steps} < 2，向量化入口直接返回")
    else:
        print(f"  → 根因: 向量化过程中抛出了未被内层捕获的异常")
elif len(step_vectors) == 0:
    n = 0
    print(f"  [结论] step_vectors 是空数组（向量化执行了但全部失败）")
    if cache_misses > 0:
        print(f"  → 根因: {cache_misses} 个 step 缓存缺失，且 Embedding API 调用失败")
    else:
        print(f"  → 根因: 缓存全命中但读取向量后维度检查失败，或其他内部错误")
else:
    n = len(step_vectors)
    print(f"  [结论] step_vectors 形状: {step_vectors.shape}，"
          f"step_indices 长度: {len(step_indices)}")
    if n < 2:
        print(f"  [警告] 向量数 {n} < 2，exchange1 仍会被跳过")
        print(f"  → 根因: 大部分 step 缓存缺失且 API 调用失败（仅 {n} 个成功）")
    else:
        print(f"  [OK] 向量数 {n} >= 2，可以正常进入 exchange1")
        print(f"  → 当前运行无问题；若批量测试时出错，可能是某次 API 超时导致")

print()
_sep("诊断完成", char="═")
