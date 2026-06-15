"""
patch_expand_cache.py

功能：为 expand 缓存中 agent 上下文不完整的条目（缺少 assistant 回复）
      单独调用 API 补全，并原子写回缓存文件。

使用前确认下方配置区的参数，然后直接运行：
    python patch_expand_cache.py
"""

import json
import os
import time

from openai import OpenAI

# ============================================================
# 配置区
# ============================================================
CACHE_PATH = (
    r"qwen3-8b\results\debate_zy\math_500_id"
    r"\debate_zy_qwen3-8b_10_1_expand_agent_com0_False.json"
)

API_URL   = "https://api.zhizengzeng.com/v1"
API_KEY   = "sk-zk28544f5e4fdc6ce482ee6ae603f8af06469f20a6a4d4b6"
MODEL_TAG = "qwen3-8b"
MAX_TOKENS = 8192
MAX_RETRIES = 5
# ============================================================


client = OpenAI(base_url=API_URL, api_key=API_KEY)


def call_api(context: list, retry: int = 0) -> str:
    """调用推理 API，返回 assistant 回复文本。"""
    try:
        response = client.chat.completions.create(
            model=MODEL_TAG,
            messages=context,
            max_tokens=MAX_TOKENS,
            # max_completion_tokens=30000,  # gpt-5-* 等新模型（按需启用）
            n=1,
        )
        return response.choices[0].message.content
    except Exception as e:
        if retry >= MAX_RETRIES:
            raise RuntimeError(f"API 调用失败已达最大重试次数 ({MAX_RETRIES})") from e
        print(f"  [重试 {retry + 1}/{MAX_RETRIES}] 错误: {e}")
        time.sleep(20)
        return call_api(context, retry + 1)


def atomic_save(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def find_incomplete_agents(agent_contexts: list) -> list[int]:
    """返回上下文不完整（缺少 assistant 回复）的 agent 索引列表。"""
    bad = []
    for aidx, ctx in enumerate(agent_contexts):
        if not isinstance(ctx, list) or len(ctx) < 3:
            bad.append(aidx)
        elif ctx[2].get("role") != "assistant":
            bad.append(aidx)
        elif not (ctx[2].get("content") or "").strip():
            bad.append(aidx)
    return bad


def main():
    print(f"[加载] {CACHE_PATH}")
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache: dict = json.load(f)
    print(f"[加载] 共 {len(cache)} 道题")

    patched_total = 0

    for key, entry in cache.items():
        if not isinstance(entry, list) or len(entry) < 3:
            continue
        agent_contexts = entry[0]
        qid = entry[2]

        bad_agents = find_incomplete_agents(agent_contexts)
        if not bad_agents:
            continue

        print(f"\n[发现] question_id={qid}  不完整 agent 索引: {bad_agents}")
        print(f"  题目前 80 字: {key[:80]}")

        for aidx in bad_agents:
            ctx = agent_contexts[aidx]
            # 取出 [system, user] 部分调 API（截断到前 2 条，防止残留异常消息）
            input_ctx = ctx[:2]
            print(f"  [补全] agent_{aidx}：调用 API（输入 {len(input_ctx)} 条消息）...")
            try:
                content = call_api(input_ctx)
                # 写入 agent_contexts
                agent_contexts[aidx] = input_ctx + [
                    {"role": "assistant", "content": content}
                ]
                print(f"  [完成] agent_{aidx}：已获得 assistant 回复（{len(content)} 字符）")
                patched_total += 1
            except Exception as e:
                print(f"  [跳过] agent_{aidx} 补全失败: {e}")

        # 验证修复结果
        still_bad = find_incomplete_agents(agent_contexts)
        if still_bad:
            print(f"  [警告] question_id={qid} 仍有不完整 agent: {still_bad}，本题暂不写回")
        else:
            entry[0] = agent_contexts
            print(f"  [写回] question_id={qid} 所有 agent 已完整，写回缓存")

    if patched_total > 0:
        print(f"\n[保存] 共补全 {patched_total} 个 agent，正在原子写回...")
        atomic_save(CACHE_PATH, cache)
        print(f"[保存] 完成：{CACHE_PATH}")
    else:
        print("\n[完成] 未发现需要补全的条目，缓存无需修改。")

    # 最终校验
    print("\n[校验] 重新扫描缓存...")
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache2 = json.load(f)
    bad_count = 0
    for entry in cache2.values():
        if isinstance(entry, list) and len(entry) >= 3:
            if find_incomplete_agents(entry[0]):
                bad_count += 1
    if bad_count == 0:
        print("[校验] 所有题目缓存完整 ✓")
    else:
        print(f"[校验] 仍有 {bad_count} 道题不完整，请检查上方日志")


if __name__ == "__main__":
    main()
