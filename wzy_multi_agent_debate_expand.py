"""
单题多 Agent 辩论测试脚本（参照 debate_bbh_qwen3b.py expand 逻辑重写）

功能：
1. 从数据集中选择一道题（固定或随机）
2. 若干 agent 分别独立推理（expand 策略，优先读取已有 debate_zy 缓存结果）
3. 少数服从多数确定"正确答案"
4. 标记每个 agent 的最终答案是否正确
5. 根据最终答案标记每个 agent 的推理步骤是否正确
6. 对推理步骤进行向量化（Embedding API，供后续聚类使用）
7. 将结果以格式化 JSON 保存至 {MODEL_NAME}/results/debate_zy/{task_name}/

核心流程由 run_expand_pipeline()与各 expand_* 步骤函数实现，供本文件与 wzy_multi_agent_debate_math.py 共用。

可配置项：
- NUM_AGENTS: agent 数量
- FIXED_QUESTION_ID: 设为 None 随机选题，设为整数如 61 则使用固定题目
- task_name: 数据集名称
"""

import sys
import io
import asyncio
import time
import re
import json
import random
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI

from common.utils import read_txt, read_json
from common.math_equivalence import strip_string
from eval_all_round import (
    parse_answer, solve_math_problems, parse_math_anser, parse_YN, most_frequent,
    parse_answer_fallback,
)
from wzy_multi_agent_debate_clustering import get_majority_answer_from_latest


def _init_stdio_utf8() -> None:
    """Windows 下控制台常为 GBK，需先设 UTF-8 代码页再绑定 stdout/stderr，避免中文乱码。"""
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

# ========== 可编辑配置 ==========
NUM_AGENTS = 10
# 题目选择：设为 None 则随机选择；设为整数如 61 则使用固定题目
FIXED_QUESTION_ID = 159
# 向量化验证：设为 True 时在向量化完成后执行验证逻辑
VERIFY_VECTORIZATION = True
# 调试：设为 True 时（且 VERIFY_VECTORIZATION 为 True）将向量保存到文件
SAVE_DEBUG_FILES = True
# 数据集与 prompt 配置
task_name = "math_500_id"
agent_com_name = "agent_com0"
is_hard = False
# IS_MATH: math_500_id 等数学数据集设为 True；BBH 选项格式任务设为 False
IS_MATH = True
# ================================

API_URL = "https://api.zhizengzeng.com/v1"
API_KEY = "sk-zk28544f5e4fdc6ce482ee6ae603f8af06469f20a6a4d4b6"
# glm-4-flashx gpt-4o-mini
MODEL_NAME = "qwen-turbo"
MODEL_TAG = "qwen-turbo"
# MODEL_NAME = "qwen2.5-7b-instruct"
# MODEL_TAG = "qwen2.5-7b-instruct"

client = OpenAI(base_url=API_URL, api_key=API_KEY)
async_client = AsyncOpenAI(base_url=API_URL, api_key=API_KEY)


@dataclass
class ExpandConfig:
    """expand 流水线参数；供 expand 脚本与 wzy_multi_agent_debate_math 共用。"""
    num_agents: int
    task_name: str
    agent_com_name: str
    is_hard: bool
    is_math: bool
    model_name: str
    model_tag: str
    api_url: str
    api_key: str
    fixed_question_id: Optional[int]
    expand_concurrent_limit: int = 5
    embedding_model: str = "qwen3-embedding-8b"


def default_expand_config() -> ExpandConfig:
    return ExpandConfig(
        num_agents=NUM_AGENTS,
        task_name=task_name,
        agent_com_name=agent_com_name,
        is_hard=is_hard,
        is_math=IS_MATH,
        model_name=MODEL_NAME,
        model_tag=MODEL_TAG,
        api_url=API_URL,
        api_key=API_KEY,
        fixed_question_id=FIXED_QUESTION_ID,
    )


# ---------- 工具函数（完全参照 debate_bbh_qwen3b.py） ----------

def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


MAX_RETRIES = 5

def generate_answer(answer_context, retry_count: int = 0):
    try:
        response = client.chat.completions.create(
            model=MODEL_TAG, messages=answer_context, max_tokens=4096, n=1
        )
        completion = json.loads(response.json())
    except Exception as e:
        if retry_count >= MAX_RETRIES:
            print(f"[错误] API 调用失败已达最大重试次数 ({MAX_RETRIES})，放弃重试")
            print(f"       最后错误: {e}")
            raise
        print(f"[重试] API 调用失败 (尝试 {retry_count + 1}/{MAX_RETRIES}): {e}")
        time.sleep(20)
        return generate_answer(answer_context, retry_count + 1)
    return completion


async def agenerate_answer(answer_context, retry_count: int = 0):
    try:
        response = await async_client.chat.completions.create(
            model=MODEL_TAG, messages=answer_context, max_tokens=4096, n=1
        )
        completion = json.loads(response.json())
    except Exception as e:
        if retry_count >= MAX_RETRIES:
            print(f"[错误] API 调用失败已达最大重试次数 ({MAX_RETRIES})，放弃重试")
            print(f"       最后错误: {e}")
            raise
        print(f"[重试] API 调用失败 (尝试 {retry_count + 1}/{MAX_RETRIES}): {e}")
        time.sleep(20)
        return generate_answer(answer_context, retry_count + 1)
    return completion


# ---------- 答案提取与步骤提取（完全参照 eval_all_round.compute_accuracy） ----------

def extract_answer_from_text(text: str, is_math: bool = False):
    """从单个 agent 的回复中提取答案。

    is_math=True 三级提取链：
      1. parse_math_anser       → \\boxed{...}（最显式）
      2. parse_answer_fallback  → "The answer is: ..."（模型明确陈述答案）
      3. solve_math_problems    → 括号整数 (-?\\d+)（兜底方法）
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


def get_majority_answer_from_expand(agent_contexts: list, is_math: bool = False):
    """从各 agent 的 expand 回复（上下文索引 2）多数投票得出参考答案。
    与 eval_all_round.compute_accuracy(..., is_math=...) 在列表分支一致：
      逐条用与 _extract_math_answer /非数学链相同的 extract_answer_from_text → most_frequent
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


def is_correct_answer(pred, ref: str, is_math: bool = False) -> bool:
    """判断预测答案是否与多数投票答案一致。
    逻辑完全参照 compute_accuracy 中的比较方式：
      - is_math=True : strip_string(ref) == pred  （pred 已经 strip_string 过）
      - is_math=False: ref == pred
    """
    if pred is None or ref is None:
        return False
    if is_math:
        return strip_string(ref) == pred
    else:
        return ref == pred


def extract_steps_from_response(response: str) -> list:
    """从模型回复中解析出「分步推理」列表。

    期望模型按 ``Step 1``、``Step 2`` … 书写；用 ``"Step "`` 切分后逐段解析编号与正文。
    返回 ``list[dict]``，每项形如 ``{"step_number": int, "content": str}``，供后续
    ``expand_flatten_all_steps``、聚类/向量化等使用。

    行为要点：
    - 含 ``Step `` 时：每段尽量用正则抽出 ``Step N:`` / ``Step N.`` 后的内容；否则顺延编号。
    - 全文无任何 ``Step `` 时：整段回复视为单一步骤 ``step_number=1``（兜底）。
    """
    steps = []
    if not response:
        return steps

    # 按字面 "Step " 切开：第一段是 Step 1 之前的前缀（常为空或开场白），从第二段起才是各步正文
    raw_steps = response.split("Step ")[1:]
    for raw in raw_steps:
        # 匹配 "Step 1: ..." 或 "Step 1. ..." 格式
        m = re.match(r'^(\d+)[:\.\s]+(.*)', raw, re.DOTALL)
        if m:
            step_num = int(m.group(1))
            content = m.group(2).strip()
        else:
            # 格式不规范时，步骤编号就自动按当前列表长度 +1 顺延，内容直接取原始文本
            step_num = len(steps) + 1
            content = raw.strip()
        steps.append({
            "step_number": step_num,
            "content": content,
        })

    # 兜底：若回复中不含任何 "Step " 关键字，将整个回复作为一个步骤保存
    if not steps and response.strip():
        steps.append({
            "step_number": 1,
            "content": response.strip(),
        })
    return steps


# ---------- 安全字符串打印 ----------

def _safe_str(s, max_len=80):
    """将字符串转为可安全打印的格式，避免 Windows 控制台编码问题。max_len 为 None 时不截断"""
    if s is None:
        return ""
    s = str(s)
    if max_len is not None and len(s) > max_len:
        s = s[:max_len] + "..."
    try:
        s.encode("gbk")
        return s
    except UnicodeEncodeError:
        return s.encode("ascii", "replace").decode("ascii")


# ---------- 向量化验证相关函数 ----------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _verify_vectorization_basic(step_vectors, step_indices, all_steps) -> bool:
    """基础验证：检查向量形状、非零性、索引对应关系"""
    print("\n" + "=" * 60)
    print("[向量化验证1] 基础检查")
    print("=" * 60)

    passed = True
    if step_vectors is None or len(step_vectors) == 0:
        print("  [失败] step_vectors 为空或 None")
        return False
    if step_indices is None or len(step_indices) == 0:
        print("  [失败] step_indices 为空或 None")
        return False

    print(f"  [通过] step_vectors 形状: {step_vectors.shape}")
    print(f"  [通过] step_indices 长度: {len(step_indices)}")

    if len(step_indices) != step_vectors.shape[0]:
        print("  [失败] step_indices 长度与 step_vectors 行数不一致")
        passed = False
    else:
        print("  [通过] step_indices 与 step_vectors 行数一致")

    nonzero_count = np.sum(np.any(step_vectors != 0, axis=1))
    total_count = step_vectors.shape[0]
    nonzero_ratio = nonzero_count / total_count * 100 if total_count > 0 else 0
    print(f"  [信息] 非零向量: {nonzero_count}/{total_count} ({nonzero_ratio:.1f}%)")

    if nonzero_count == 0:
        print("  [失败] 所有向量均为零向量，向量化可能失败")
        passed = False
    elif nonzero_ratio < 50:
        print("  [警告] 非零向量比例较低，部分 step 可能向量化失败")
    else:
        print("  [通过] 非零向量比例正常")

    invalid_indices = [i for i in step_indices if i < 0 or i >= len(all_steps)]
    if invalid_indices:
        print(f"  [失败] step_indices 中存在无效索引: {invalid_indices[:5]}...")
        passed = False
    else:
        print("  [通过] step_indices 均在 all_steps 有效范围内")

    print("\n  [示例] 向量与 step 的对应关系（前 3 个）:")
    for i in range(min(3, len(step_indices), len(step_vectors))):
        idx = step_indices[i]
        content = all_steps[idx].get("content", "")
        content_preview = content[:60] + "..." if len(content) > 60 else content
        print(f"    向量[{i}] -> all_steps[{idx}]: {content_preview!r}")

    print("\n  [结论] 基础验证通过" if passed else "\n  [结论] 基础验证未通过")
    return passed


def _verify_vectorization_similarity(step_vectors, step_indices, all_steps) -> bool:
    """语义合理性验证：相同内容的 step 应具有高余弦相似度"""
    print("\n" + "=" * 60)
    print("[向量化验证2] 语义相似度检查")
    print("=" * 60)

    if step_vectors is None or len(step_vectors) < 2:
        print("  [跳过] 向量数量不足 2，无法进行相似度验证")
        return True

    passed = True
    pairs_to_check = [(0, 1)]
    if len(step_vectors) >= 4:
        pairs_to_check.append((2, 3))

    print("  向量对余弦相似度:")
    for i, j in pairs_to_check:
        sim = _cosine_similarity(step_vectors[i], step_vectors[j])
        content_i = all_steps[step_indices[i]].get("content", "")[:40]
        content_j = all_steps[step_indices[j]].get("content", "")[:40]
        print(f"    向量[{i}] vs 向量[{j}]: {sim:.4f}")
        print(f"      step[{i}] 内容预览: {content_i!r}...")
        print(f"      step[{j}] 内容预览: {content_j!r}...")
        if np.isnan(sim) or np.isinf(sim):
            print(f"  [失败] 相似度异常: {sim}")
            passed = False

    content_to_idx = {}
    duplicate_found = False
    for vec_i, step_idx in enumerate(step_indices):
        content = all_steps[step_idx].get("content", "")
        if content in content_to_idx:
            other_vec_i = content_to_idx[content]
            sim = _cosine_similarity(step_vectors[vec_i], step_vectors[other_vec_i])
            print(f"\n  [信息] 发现重复内容 step: 向量[{vec_i}] 与 向量[{other_vec_i}] 内容相同")
            print(f"    余弦相似度: {sim:.4f} (应接近 1.0)")
            if sim < 0.99:
                print("  [警告] 相同内容的向量相似度偏低，可能存在问题")
            duplicate_found = True
        else:
            content_to_idx[content] = vec_i

    if not duplicate_found:
        print("\n  [信息] 未发现内容完全相同的 step（正常情况）")

    print("\n  [结论] 语义相似度验证通过" if passed else "\n  [结论] 语义相似度验证存在异常")
    return passed


def _verify_vectorization_cache_hint():
    """缓存验证提示"""
    print("\n" + "=" * 60)
    print("[向量化验证3] 缓存验证说明")
    print("=" * 60)
    print("  缓存验证需手动进行：")
    print("  1. 第一次运行，观察「需要获取: X 个向量」")
    print("  2. 不修改题目再次运行")
    print("  3. 第二次应看到「缓存命中: X/X (100%)」或「所有 X 个向量都已缓存」")
    print("=" * 60)


def _save_vectors_for_debug(step_vectors, step_indices, all_steps, output_prefix="step_vectors_debug"):
    """可选：将向量和元数据保存到文件，供离线分析"""
    if step_vectors is None or len(step_vectors) == 0:
        print("\n[调试] 无可保存的向量")
        return
    vectors_path = f"{output_prefix}_vectors.npy"
    meta_path = f"{output_prefix}_meta.txt"
    np.save(vectors_path, step_vectors)
    print(f"\n[调试] 已保存向量到: {vectors_path} (shape: {step_vectors.shape})")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("index\tstep_index\tcontent_preview\n")
        for i, idx in enumerate(step_indices):
            content = all_steps[idx].get("content", "")[:80].replace("\n", " ")
            f.write(f"{i}\t{idx}\t{content}\n")
    print(f"[调试] 已保存元数据到: {meta_path}")


def _run_vectorization_verification(step_vectors, step_indices, all_steps, save_debug_files: bool = False):
    """执行全部向量化验证逻辑"""
    print("\n" + "#" * 70)
    print("# 向量化验证 - 开始")
    print("#" * 70)

    basic_ok = _verify_vectorization_basic(step_vectors, step_indices, all_steps)
    similarity_ok = _verify_vectorization_similarity(step_vectors, step_indices, all_steps)
    _verify_vectorization_cache_hint()

    if save_debug_files and step_vectors is not None:
        _save_vectors_for_debug(step_vectors, step_indices, all_steps)

    print("\n" + "#" * 70)
    print("# 向量化验证 - 汇总")
    print("#" * 70)
    print(f"  基础验证:    {'通过' if basic_ok else '未通过'}")
    print(f"  相似度验证:  {'通过' if similarity_ok else '未通过/跳过'}")
    print("  缓存验证:    请按上述说明手动验证")
    if basic_ok and similarity_ok:
        print("\n  [结论] 向量化验证整体通过")
    else:
        print("\n  [结论] 向量化验证存在问题，请检查上述失败项")
    print("#" * 70 + "\n")


# ---------- expand 流水线（拆分自原 main，供 expand / math 共用） ----------

def expand_load_question(
    cfg: ExpandConfig,
    question_item: Optional[dict],
) -> Optional[Tuple[str, str, str]]:
    """返回 (question, ground_truth, question_id)；题目缺失时返回 None。"""
    if question_item is not None:
        q = question_item["input"]
        gt = question_item["target"]
        qid = str(question_item.get("question_id", "?"))
        return q, gt, qid

    task_file = f"{cfg.model_name}/data/{cfg.task_name}.json"
    with open(task_file, "r", encoding="utf-8") as f:
        data = json.load(f)["examples"]

    if cfg.fixed_question_id is not None:
        print(f"[配置] 使用固定题目 question_id={cfg.fixed_question_id}")
        item = next(
            (d for d in data if str(d.get("question_id")) == str(cfg.fixed_question_id)),
            None,
        )
        if item is None:
            print(f"[错误] 未找到 question_id={cfg.fixed_question_id} 的题目")
            return None
    else:
        item = random.choice(data)

    return item["input"], item["target"], str(item.get("question_id", "?"))


def expand_print_question_header(question: str, ground_truth: str, question_id: str) -> None:
    print("\n" + "=" * 80)
    print(f"[题目] ID: {question_id}")
    print(f"[题目] 标准答案: {ground_truth}")
    print(f"[题目] 内容:\n{question[:200]}{'...' if len(question) > 200 else ''}")
    print("=" * 80)


def expand_build_agent_contexts(
    question: str,
    cfg: ExpandConfig,
) -> Tuple[str, List[List[Dict[str, Any]]]]:
    fewshot_ost_config = read_json("prompt/fewshot_ost_config.json")
    fewshot_ost_prompt = read_txt("prompt/fewshot_ost_prompt.txt")
    fewshot_content = fewshot_ost_config["prompt_template"].format(
        examples=fewshot_ost_prompt,
        instruction=question,
    )
    with open(f"prompt/{cfg.agent_com_name}.json", "r", encoding="utf-8") as f:
        system_prompts = json.load(f)["agents"]
    agent_contexts = [
        [
            {"role": "system", "content": system_prompts[agent_idx % len(system_prompts)]["system"]},
            {"role": "user", "content": fewshot_content},
        ]
        for agent_idx in range(cfg.num_agents)
    ]
    return fewshot_content, agent_contexts


def get_cache_path(cfg: ExpandConfig, action_name: str = "expand") -> str:
    """通用缓存路径：action_name 可为 'expand'、'exchange1' 等。"""
    return (
        f"{cfg.model_name}/results/debate_zy/{cfg.task_name}/"
        f"debate_zy_{cfg.model_name}_{cfg.num_agents}_1_{action_name}_{cfg.agent_com_name}_{cfg.is_hard}.json"
    )


def expand_get_cache_path(cfg: ExpandConfig) -> str:
    return get_cache_path(cfg, "expand")


def get_expand_cache_entry(
    cached_results: Dict[str, Any],
    question_id: str,
    question: Optional[str] = None,
) -> Optional[List[Any]]:
    """
    从 debate_zy 缓存 dict 中取出本题记录 ``[agent_contexts, ground_truth, question_id]``。

    查找顺序：
    1. 顶层 key 为 ``str(question_id)``（若未来改为 id 作 key）
    2. 顶层 key 为完整题干 ``question``（旧格式）
    3. 遍历 value，匹配 ``value[2] == str(question_id)``
    """
    if not isinstance(cached_results, dict):
        return None
    qid = str(question_id)
    if qid in cached_results:
        v = cached_results[qid]
        if isinstance(v, list) and len(v) >= 3:
            return v
    if question is not None and question in cached_results:
        v = cached_results[question]
        if isinstance(v, list) and len(v) >= 3:
            return v
    for v in cached_results.values():
        if isinstance(v, list) and len(v) >= 3 and str(v[2]) == qid:
            return v
    return None


async def expand_run_inference(
    agent_contexts: List[List[Dict[str, Any]]],
    question: str,
    cache_path: str,
    cfg: ExpandConfig,
    save_cache: bool,
    question_id: str,
) -> None:
    print(f"\n[执行] 正在执行 expand：{cfg.num_agents} 个 agent 独立推理...")
    print(f"[缓存] 路径: {cache_path}")

    cached_blob: Optional[Dict[str, Any]] = None
    if save_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_blob = json.load(f)
        except Exception as e:
            print(f"  [缓存] 预读失败: {e}")

    async def _expand_one(agent_idx: int):
        agent_context = agent_contexts[agent_idx]
        if save_cache and cached_blob is not None:
            try:
                entry = get_expand_cache_entry(cached_blob, question_id, question)
                if entry is not None:
                    cxt = entry[0][agent_idx][2]["content"]
                    completion = {"choices": [{"message": {"content": cxt}}]}
                    print(f"  [缓存命中] Agent {agent_idx}")
                    return agent_idx, completion, None
            except Exception as e:
                print(f"  [缓存读取失败] Agent {agent_idx}: {e}，将重新调用 API")

        try:
            completion = await agenerate_answer(agent_context)
            return agent_idx, completion, None
        except Exception as e:
            return agent_idx, None, e

    lim = cfg.expand_concurrent_limit
    for batch_start in range(0, cfg.num_agents, lim):
        batch_end = min(batch_start + lim, cfg.num_agents)
        tasks = [_expand_one(i) for i in range(batch_start, batch_end)]
        batch_results = await asyncio.gather(*tasks)
        for agent_idx, completion, err in batch_results:
            if err is not None:
                print(f"  [失败] Agent {agent_idx} expand 调用失败: {err}")
            elif completion is not None:
                assistant_message = construct_assistant_message(completion)
                agent_contexts[agent_idx].append(assistant_message)
        if batch_end < cfg.num_agents:
            await asyncio.sleep(1)


def expand_print_step2_responses(agent_contexts: List[List[Dict[str, Any]]]) -> None:
    print(f"\n{'═'*80}")
    print("  [Expand Step 2] 各 Agent 独立推理结果")
    print(f"{'═'*80}")
    for agent_idx, context in enumerate(agent_contexts):
        if len(context) >= 3 and context[2].get("role") == "assistant":
            response_text = context[2].get("content", "")
            print(f"\n  ┌─ Agent {agent_idx} (回复长度: {len(response_text)} 字符) {'─'*40}")
            for line in response_text.split("\n"):
                print(f"  │  {line}")
            print(f"  └{'─'*70}")
        else:
            print(f"\n  ┌─ Agent {agent_idx} {'─'*55}")
            print(f"  │  (未获取到 assistant 回复)")
            print(f"  └{'─'*70}")
    print(f"{'═'*80}")


def expand_save_cache_if_enabled(
    save_cache: bool,
    question: str,
    agent_contexts: List[List[Dict[str, Any]]],
    ground_truth: str,
    question_id: str,
    cache_path: str,
    cfg: ExpandConfig,
) -> None:
    """将本题结果写入 cache_path；若文件已存在则读入后按 question 合并，再原子替换写入。"""
    if save_cache:
        save_dir = f"{cfg.model_name}/results/debate_zy/{cfg.task_name}"
        os.makedirs(save_dir, exist_ok=True)

        merged: Dict[str, Any] = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    merged = loaded
                else:
                    print(
                        f"\n[缓存合并] 现有文件顶层非 JSON 对象，将仅写入本题: {cache_path}"
                    )
            except json.JSONDecodeError as e:
                print(f"\n[缓存合并] 解析失败 ({e})，将覆盖为仅含本题: {cache_path}")
            except OSError as e:
                print(f"\n[缓存合并] 读取失败 ({e})，将覆盖为仅含本题: {cache_path}")

        merged[question] = [agent_contexts, ground_truth, question_id]

        tmp_path = f"{cache_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, cache_path)
        except OSError:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        print(
            f"\n[保存] 已合并写入: {cache_path}（共 {len(merged)} 题，本题 key 已更新）"
        )
    else:
        print(f"\n[跳过] 批量模式下不写入单题缓存（由外层统一存档）")


def expand_compute_majority_and_agent_results(
    agent_contexts: List[List[Dict[str, Any]]],
    cfg: ExpandConfig,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    majority_answer = get_majority_answer_from_expand(agent_contexts, is_math=cfg.is_math)
    agent_results: List[Dict[str, Any]] = []
    for agent_idx, agent_context in enumerate(agent_contexts):
        expand_response = ""
        if len(agent_context) >= 3 and agent_context[2].get("role") == "assistant":
            expand_response = agent_context[2].get("content", "")

        agent_answer = extract_answer_from_text(expand_response, is_math=cfg.is_math)
        steps = extract_steps_from_response(expand_response)
        final_answer_correct = (
            is_correct_answer(agent_answer, majority_answer, is_math=cfg.is_math) if agent_answer else False
        )
        step_correctness = [final_answer_correct] * len(steps) if steps else []

        agent_results.append({
            "agent_idx": agent_idx,
            "answer": agent_answer,
            "final_answer_correct": final_answer_correct,
            "steps": steps,
            "step_correctness": step_correctness,
            "raw_response": expand_response,
        })
    return majority_answer, agent_results


def expand_compute_majority_and_agent_results_from_latest(
    agent_contexts: List[List[Dict[str, Any]]],
    cfg: ExpandConfig,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """与 expand_compute_majority_and_agent_results 相同，但从各 agent 最后一条 assistant 取推理。

    多数票用 get_majority_answer_from_latest；每步 is_correct 仍由「该 agent 最终答案是否与多数票一致」继承。
    供 exchange1 之后构造 exchange2 的 all_steps / 向量化输入。
    """
    majority_answer = get_majority_answer_from_latest(agent_contexts, is_math=cfg.is_math)
    agent_results: List[Dict[str, Any]] = []
    for agent_idx, agent_context in enumerate(agent_contexts):
        latest_response = ""
        if agent_context and agent_context[-1].get("role") == "assistant":
            latest_response = agent_context[-1].get("content", "")

        agent_answer = extract_answer_from_text(latest_response, is_math=cfg.is_math)
        steps = extract_steps_from_response(latest_response)
        final_answer_correct = (
            is_correct_answer(agent_answer, majority_answer, is_math=cfg.is_math) if agent_answer else False
        )
        step_correctness = [final_answer_correct] * len(steps) if steps else []

        agent_results.append({
            "agent_idx": agent_idx,
            "answer": agent_answer,
            "final_answer_correct": final_answer_correct,
            "steps": steps,
            "step_correctness": step_correctness,
            "raw_response": latest_response,
        })
    return majority_answer, agent_results


def expand_print_step3_majority(
    agent_contexts: List[List[Dict[str, Any]]],
    cfg: ExpandConfig,
    majority_answer: Optional[str],
) -> None:
    print(f"\n{'═'*80}")
    print("  [Expand Step 3] 答案提取与多数投票")
    print(f"{'═'*80}")
    _extracted_answers_for_print = []
    for _aidx, _ctx in enumerate(agent_contexts):
        if len(_ctx) >= 3 and _ctx[2].get("role") == "assistant":
            _ans = extract_answer_from_text(_ctx[2].get("content", ""), is_math=cfg.is_math)
        else:
            _ans = None
        _extracted_answers_for_print.append(_ans)
        print(f"    Agent {_aidx}: 提取答案 = {_ans}")
    print(f"\n  {'─'*70}")
    print(f"    答案列表: {_extracted_answers_for_print}")
    print(f"    majority_answer (多数投票结果) = {majority_answer}")
    if majority_answer is None:
        print("    [警告] 所有 agent 均未提取到有效答案，majority_answer 为 None")
    print(f"{'═'*80}")


def expand_print_step4_agent_vs_majority(
    agent_results: List[Dict[str, Any]],
    majority_answer: Optional[str],
    num_agents: int,
) -> None:
    print(f"\n{'═'*80}")
    print("  [Expand Step 4] 各 Agent 答案与 majority_answer 对比")
    print(f"{'═'*80}")
    for ar in agent_results:
        _idx = ar["agent_idx"]
        _ans = ar["answer"] or "(未提取到)"
        _mark = "[正确]" if ar["final_answer_correct"] else "[错误]"
        print(f"    Agent {_idx}: 答案={_ans}, 与 majority_answer={majority_answer} 对比 → {_mark}")
    _correct_cnt = sum(1 for ar in agent_results if ar["final_answer_correct"])
    print(f"\n    与 majority_answer 一致的 agent 数: {_correct_cnt}/{num_agents}")
    print(f"{'═'*80}")


def expand_print_step5_steps(agent_results: List[Dict[str, Any]]) -> None:
    print(f"\n{'═'*80}")
    print("  [Expand Step 5] Step 提取与标签继承")
    print(f"{'═'*80}")
    print(f"  [标签继承规则]")
    print(f"    若 agent 最终答案与 majority_answer 一致 → 该 agent 所有 step 标记为 True (正确)")
    print(f"    若 agent 最终答案与 majority_answer 不一致 → 该 agent 所有 step 标记为 False (错误)")
    print(f"    (注意: 标签不是逐步验证的，而是从 agent 级别结果统一继承)")
    for ar in agent_results:
        _idx = ar["agent_idx"]
        _label = "True" if ar["final_answer_correct"] else "False"
        _n_steps = len(ar["steps"])
        _mark = "正确" if ar["final_answer_correct"] else "错误"
        print(f"\n  ┌─ Agent {_idx} (答案{_mark}, {_n_steps} 个 step, 全部标记为 {_label}) {'─'*20}")
        if ar["steps"]:
            for i, step in enumerate(ar["steps"]):
                _step_label = ar["step_correctness"][i] if i < len(ar["step_correctness"]) else False
                _label_str = "[True]" if _step_label else "[False]"
                _content = _safe_str(step["content"], max_len=None)
                print(f"  \u2502  Step {step['step_number']} {_label_str}:")
                for _line in _content.split("\n"):
                    print(f"  \u2502    {_line}")
        else:
            _raw = _safe_str(ar["raw_response"], max_len=200)
            print(f"  \u2502  (未提取到结构化步骤, 原始回复预览: {_raw})")
        _corner = "\u2514"
        _h = "\u2500"
        print(f"  {_corner}{_h * 70}")
    print(f"{'═'*80}")


def expand_flatten_all_steps(agent_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_steps: List[Dict[str, Any]] = []
    for ar in agent_results:
        for i, step in enumerate(ar["steps"]):
            step_correct = ar["step_correctness"][i] if i < len(ar["step_correctness"]) else False
            all_steps.append({
                "agent_id": ar["agent_idx"],
                "step_number": step["step_number"],
                "content": step.get("content", ""),
                "is_correct": step_correct,
            })
    return all_steps


def expand_print_step6_all_steps(all_steps: List[Dict[str, Any]], num_agents: int) -> None:
    print(f"\n{'═'*80}")
    print("  [Expand Step 6] All Steps 汇总 (展平后)")
    print(f"{'═'*80}")
    _n_correct_steps = sum(1 for s in all_steps if s.get("is_correct") is True)
    _n_wrong_steps = sum(1 for s in all_steps if s.get("is_correct") is False)
    print(f"    总 step 数: {len(all_steps)} (来自 {num_agents} 个 agent)")
    print(f"    标记为 correct (True):  {_n_correct_steps}")
    print(f"    标记为 incorrect (False): {_n_wrong_steps}")
    _agent_step_counts: Dict[Any, int] = {}
    for s in all_steps:
        _aid = s.get("agent_id")
        _agent_step_counts[_aid] = _agent_step_counts.get(_aid, 0) + 1
    print(f"    各 agent 贡献 step 数: {dict(sorted(_agent_step_counts.items()))}")
    print(f"{'═'*80}")

# 修改前embedding的方式
# def expand_run_embedding(
#     all_steps: List[Dict[str, Any]],
#     cfg: ExpandConfig,
# ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
#     """对展平后的 step 列表做 Embedding API 向量化。

#     每道题内各 agent 的 step 均针对同一题干，故仅对 step 正文向量化，不拼接题干前缀，
#     以降低 token 与缓存 key 歧义（缓存键仅依赖 step 文本本身）。
#     """
#     print(f"\n{'═'*80}")
#     print("  [Expand Step 7] 步骤向量化 (Embedding)")
#     print(f"{'═'*80}")

#     step_vectors = None
#     step_indices = None
#     if len(all_steps) >= 2:
#         try:
#             from wzy_step_clustering import StepClusteringRefiner

#             refiner = StepClusteringRefiner(
#                 api_url=cfg.api_url,
#                 api_key=cfg.api_key,
#                 vector_method="embedding_api",
#                 embedding_model=cfg.embedding_model,
#                 reduce_dim=False,
#                 batch_size=20,
#             )
#             print("\n[向量化] 仅使用各 step 的 content 文本（不注入题干前缀）")
#             step_vectors, step_indices = refiner.vectorize_steps(all_steps)
#             if step_vectors is not None and step_vectors.shape[0] > 0:
#                 print(
#                     f"\n[向量化] 完成: {step_vectors.shape[0]} 个步骤 -> "
#                     f"{step_vectors.shape[1]} 维向量 (模型: {cfg.embedding_model})"
#                 )
#             else:
#                 print("\n[向量化] 失败或无可向量化步骤")
#         except Exception as e:
#             print(f"\n[向量化] 异常: {e}")
#             import traceback
#             traceback.print_exc()
#     else:
#         print(f"\n[向量化] 跳过: 步骤数 {len(all_steps)} < 2，无法进行向量化")
#     print(f"{'═'*80}")
#     return step_vectors, step_indices

# 修改后embedding的方式
from wzy_context_step_vectorization import expand_run_embedding_contextual
def expand_run_embedding(
    all_steps: List[Dict[str, Any]],
    cfg: ExpandConfig,
) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    return expand_run_embedding_contextual(all_steps, cfg)


def try_load_full_expand_cache(
    cfg: ExpandConfig,
    question_id: str,
    question: Optional[str] = None,
) -> Optional[List[List[Dict[str, Any]]]]:
    """
    尝试一次性整题加载 expand 阶段的 agent_contexts 缓存。

    与 expand_run_inference 中"逐 agent 命中"的细粒度逻辑不同，本函数从整题视角检查：
    缓存文件是否存在 → 是否含本题记录 → agent 数量与每个 agent 的 assistant 回复是否
    都齐备。任一条件不满足即返回 None，让上层退回完整 expand 流程。

    Args:
        cfg: ExpandConfig 实例
        question_id: 题目 id（字符串）
        question: 题干文本（用于按 question 作 key 的旧格式缓存兼容）

    Returns:
        命中：返回 agent_contexts（每个 ctx 至少 system/user/assistant 三条消息）；
        未命中：返回 None。
    """
    cache_path = expand_get_cache_path(cfg)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_blob = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [快速通道][缓存读取失败] {e}")
        return None

    entry = get_expand_cache_entry(cached_blob, question_id, question)
    if entry is None:
        return None

    agent_contexts = entry[0]
    if not isinstance(agent_contexts, list):
        return None

    if len(agent_contexts) != cfg.num_agents:
        print(
            f"  [快速通道][校验失败] 缓存中 agent 数 {len(agent_contexts)} != "
            f"配置 num_agents={cfg.num_agents}"
        )
        return None

    for aidx, ctx in enumerate(agent_contexts):
        if not isinstance(ctx, list) or len(ctx) < 3:
            print(f"  [快速通道][校验失败] Agent {aidx} 上下文消息数 < 3")
            return None
        if ctx[2].get("role") != "assistant":
            print(f"  [快速通道][校验失败] Agent {aidx} 的 ctx[2] 非 assistant")
            return None
        if not (ctx[2].get("content") or "").strip():
            print(f"  [快速通道][校验失败] Agent {aidx} 的 assistant 回复为空")
            return None

    return agent_contexts


async def run_expand_pipeline_from_cache(
    cfg: ExpandConfig,
    question_item: Optional[dict] = None,
) -> Optional[Tuple]:
    """expand 阶段「缓存快速通道」：仅依赖磁盘缓存重建 expand_pack，不调用任何模型 API。

    流程：
      1. 加载题目（若 question_item 为 None 则按 cfg.fixed_question_id 选）
      2. 调 try_load_full_expand_cache 整题查找缓存 agent_contexts；
         - 未命中或不完整 → 返回 None，由上层决定是否退回完整 pipeline
         - 命中 → 在缓存基础上跑后续无 API 的"轻量"流程（多数票/step 切分/向量化）
      3. 复用 expand_compute_majority_and_agent_results / expand_flatten_all_steps /
         expand_run_embedding（embedding 已有 hash 缓存，几乎零成本）

    Returns:
        命中：与 run_expand_pipeline(return_vectorization_data=True) 完全相同的 8 元组：
              (step_vectors, step_indices, all_steps, agent_contexts,
               majority_answer, ground_truth, question, question_id)
        未命中或题目加载失败：None
    """
    loaded = expand_load_question(cfg, question_item)
    if loaded is None:
        return None
    question, ground_truth, question_id = loaded

    agent_contexts = try_load_full_expand_cache(cfg, question_id, question)
    if agent_contexts is None:
        return None

    print(
        f"\n[expand 快速通道] 缓存命中：question_id={question_id}，已加载 "
        f"{len(agent_contexts)} 个 agent_contexts，跳过推理 API"
    )

    expand_print_question_header(question, ground_truth, question_id)
    expand_print_step2_responses(agent_contexts)

    majority_answer, agent_results = expand_compute_majority_and_agent_results(
        agent_contexts, cfg
    )
    expand_print_step3_majority(agent_contexts, cfg, majority_answer)
    expand_print_step4_agent_vs_majority(agent_results, majority_answer, cfg.num_agents)
    expand_print_step5_steps(agent_results)

    all_steps = expand_flatten_all_steps(agent_results)
    expand_print_step6_all_steps(all_steps, cfg.num_agents)
    step_vectors, step_indices = expand_run_embedding(all_steps, cfg)

    return (
        step_vectors,
        step_indices,
        all_steps,
        agent_contexts,
        majority_answer,
        ground_truth,
        question,
        question_id,
    )


async def run_expand_pipeline(
    cfg: ExpandConfig,
    question_item: Optional[dict] = None,
    save_cache: bool = True,
    return_vectorization_data: bool = False,
    verify_vectorization: bool = False,
    save_debug_files: bool = False,
) -> Optional[Tuple]:
    """执行单题 expand 全流程；与原先 main() 行为一致。

    Returns:
        return_vectorization_data 为 True 时：
        (step_vectors, step_indices, all_steps, agent_contexts,
         majority_answer, ground_truth, question, question_id)
        题目加载失败时返回 None。
    """
    loaded = expand_load_question(cfg, question_item)
    if loaded is None:
        return None
    question, ground_truth, question_id = loaded

    expand_print_question_header(question, ground_truth, question_id)
    fewshot_content, agent_contexts = expand_build_agent_contexts(question, cfg)

    if agent_contexts:
        sample_idx = 0
        sample_context = agent_contexts[sample_idx]
        print("=" * 80)
        print(f"[agent_contexts 示例] 共 {len(agent_contexts)} 个 agent，展示 agent[{sample_idx}]：")
        print("-" * 80)
        for msg_idx, message in enumerate(sample_context):
            role = message.get("role", "")
            content = message.get("content", "")
            print(f"--- message[{msg_idx}] | role: {role} ---")
            print(content)
            print()
        print("=" * 80)
    cache_path = expand_get_cache_path(cfg)
    await expand_run_inference(agent_contexts, question, cache_path, cfg, save_cache, question_id)
    expand_print_step2_responses(agent_contexts)
    expand_save_cache_if_enabled(
        save_cache, question, agent_contexts, ground_truth, question_id, cache_path, cfg
    )

    majority_answer, agent_results = expand_compute_majority_and_agent_results(agent_contexts, cfg)
    expand_print_step3_majority(agent_contexts, cfg, majority_answer)
    expand_print_step4_agent_vs_majority(agent_results, majority_answer, cfg.num_agents)
    expand_print_step5_steps(agent_results)

    all_steps = expand_flatten_all_steps(agent_results)
    expand_print_step6_all_steps(all_steps, cfg.num_agents)
    step_vectors, step_indices = expand_run_embedding(all_steps, cfg)

    # if verify_vectorization and step_vectors is not None:
    #     _run_vectorization_verification(step_vectors, step_indices, all_steps, save_debug_files)


    if return_vectorization_data:
        return (step_vectors, step_indices, all_steps, agent_contexts,
                majority_answer, ground_truth, question, question_id)
    return None


# ---------- 主流程（薄封装） ----------

async def main(
    return_vectorization_data: bool = True,
    verify_vectorization: bool = False,
    save_debug_files: bool = True,
    question_item: dict = None,
    save_cache: bool = True,
    cfg: Optional[ExpandConfig] = None,
):
    """执行单题 expand 流程（完全参照 debate_bbh_qwen3b.py expand action 逻辑）

    流程：
      1. 加载 fewshot 配置 + 题目
      2. 初始化 agent_contexts（system + fewshot_content）
      3. Expand：优先读取 debate_zy 缓存（save_cache=True 时），无缓存则并发调用 API
      4. 将结果以格式化 JSON 保存至 debate_zy 目录（save_cache=True 时）
      5. 多数投票 → 提取答案 → 标记步骤正确性 → 向量化

    Args:
        return_vectorization_data: 若为 True，返回 (step_vectors, step_indices, all_steps,
                                    agent_contexts, majority_answer, ground_truth, question, question_id)
        verify_vectorization: 若为 True，向量化完成后执行验证逻辑
        save_debug_files: 若为 True 且 verify_vectorization 为 True，将向量保存到文件
        question_item: 若非 None，直接使用该题目字典（含 input/target/question_id），跳过文件读取
        save_cache: 是否启用单题缓存的读写。
        cfg: 若为 None，使用模块内 default_expand_config()。
    """
    if cfg is None:
        cfg = default_expand_config()
    return await run_expand_pipeline(
        cfg=cfg,
        question_item=question_item,
        save_cache=save_cache,
        return_vectorization_data=return_vectorization_data,
        verify_vectorization=verify_vectorization,
        save_debug_files=save_debug_files,
    )


async def run_single_question_expand(
    return_vectorization_data: bool = True,
    question_item: dict = None,
    cfg: Optional[ExpandConfig] = None,
):
    """供 exchange.py 调用的 expand 入口，参数签名与 exchange 调用侧保持一致。

    批量模式下由 math_500.py 统一管理存档，因此传入 save_cache=False，
    跳过单题缓存读写，避免与批量存档重复。

    Returns:
        与 main(return_vectorization_data=True) 相同：
        (step_vectors, step_indices, all_steps, agent_contexts,
         majority_answer, ground_truth, question, question_id)
        若流程中途失败则返回 None。
    """
    if cfg is None:
        cfg = default_expand_config()
    return await run_expand_pipeline(
        cfg=cfg,
        question_item=question_item,
        save_cache=False,
        return_vectorization_data=return_vectorization_data,
        verify_vectorization=False,
        save_debug_files=False,
    )


if __name__ == "__main__":
    asyncio.run(main(
        verify_vectorization=VERIFY_VECTORIZATION,
        save_debug_files=SAVE_DEBUG_FILES,
    ))
