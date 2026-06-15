"""
delete_key.py

功能：清除指定题目在 checkpoint.json 和各阶段结果 JSON 中的旧记录，
      使这些题目可以在下次运行时被重新处理并写入最新结果。

使用方式：
  1. 在下方"用户配置区"填写需要重跑的 question_id 列表及相关参数
  2. 直接运行：python delete_key.py
"""

import json
import os

# ============================================================
# 用户配置区（每次使用前修改这里）
# ============================================================

# 需要重跑的题目 ID 列表（字符串或整数均可，脚本内统一转为字符串处理）
QUESTION_IDS = [
    # 在此填写需要重跑的 question_id，例如：
    # "3", "15", "42",
    "97", "137", "246", "286",
]

# 模型名称（对应结果目录和 checkpoint 的根目录）
MODEL_NAME = "qwen3-8b"

# 任务名称（对应结果目录中的子文件夹）
TASK_NAME = "math_500_id"

# 参与 debate 的 agent 数量
NUM_AGENTS = 10

# agent 通信方式标识
AGENT_COM_NAME = "agent_com0"

# 是否只跑困难题（对应文件名中的 is_hard 字段）
IS_HARD = False

# 需要清除的 action 阶段列表（留空则跳过结果 JSON 的清理，只清 checkpoint）
ACTION_NAMES = [
    "expand",
    "exchange1",
    "exchange2",
    "exchange_bidirectional_1",
    "exchange_bidirectional_2",
]

# ============================================================
# 以下内容无需修改
# ============================================================


def _result_json_path(action: str) -> str:
    """按照与 get_cache_path 完全一致的规则生成结果 JSON 路径。"""
    return (
        f"{MODEL_NAME}/results/debate_zy/{TASK_NAME}/"
        f"debate_zy_{MODEL_NAME}_{NUM_AGENTS}_1_{action}_{AGENT_COM_NAME}_{IS_HARD}.json"
    )


def _atomic_save(path: str, data: dict) -> None:
    """原子替换写入，格式与 expand_save_cache_if_enabled 完全一致。"""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def delete_from_checkpoint(qids: list[str]) -> int:
    """从 checkpoint.json 中删除指定 question_id，返回实际删除数量。"""
    ck_path = f"{MODEL_NAME}/checkpoint.json"

    if not os.path.exists(ck_path):
        print(f"[checkpoint] 文件不存在，跳过: {ck_path}")
        return 0

    try:
        with open(ck_path, "r", encoding="utf-8") as f:
            data: dict = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[checkpoint] 读取失败 ({e})，跳过")
        return 0

    deleted = 0
    for qid in qids:
        if qid in data:
            del data[qid]
            deleted += 1
            print(f"  [checkpoint] 删除 question_id={qid} ✓")
        else:
            print(f"  [checkpoint] question_id={qid} 不存在，跳过")

    if deleted > 0:
        _atomic_save(ck_path, data)
        print(f"[checkpoint] 已保存，共删除 {deleted} 条")
    else:
        print(f"[checkpoint] 无需改动")

    return deleted


def delete_from_result_json(action: str, qids: list[str]) -> int:
    """
    从指定 action 的结果 JSON 中删除目标题目的 key，返回实际删除数量。

    结果 JSON 的 key 为完整题干文本，value 为 [agent_contexts, ground_truth, question_id]。
    通过匹配 value[2]（即 question_id）来定位需要删除的 key。
    同时兼容以 question_id 字符串直接作为 key 的格式。
    """
    path = _result_json_path(action)

    if not os.path.exists(path):
        print(f"  [{action}] 文件不存在，跳过: {path}")
        return 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            data: dict = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [{action}] 读取失败 ({e})，跳过，不修改文件")
        return 0

    keys_to_delete: list[str] = []
    for qid in qids:
        found_keys = []
        # 格式一：question_id 直接作为 key
        if qid in data:
            found_keys.append(qid)
        # 格式二：完整题干文本作为 key，value[2] 为 question_id
        for k, v in data.items():
            if k == qid:
                continue  # 已在格式一中处理
            if isinstance(v, list) and len(v) >= 3 and str(v[2]) == qid:
                found_keys.append(k)

        if not found_keys:
            print(f"  [{action}] question_id={qid} 未找到，跳过")
        elif len(found_keys) > 1:
            print(f"  [{action}] question_id={qid} 发现多个 key（异常），全部删除: {[k[:30] + '...' if len(k) > 30 else k for k in found_keys]}")
            keys_to_delete.extend(found_keys)
        else:
            key_preview = found_keys[0] if len(found_keys[0]) <= 40 else found_keys[0][:37] + "..."
            print(f"  [{action}] 删除 question_id={qid}，key=\"{key_preview}\" ✓")
            keys_to_delete.extend(found_keys)

    if not keys_to_delete:
        print(f"  [{action}] 无需改动")
        return 0

    for k in keys_to_delete:
        del data[k]

    _atomic_save(path, data)
    print(f"  [{action}] 已保存，共删除 {len(keys_to_delete)} 条，剩余 {len(data)} 题")
    return len(keys_to_delete)


def main():
    if not QUESTION_IDS:
        print("[错误] QUESTION_IDS 为空，请先在配置区填写需要重跑的题目 ID")
        return

    # 统一转为字符串，与 checkpoint 和结果 JSON 中的存储格式一致
    qids = [str(q) for q in QUESTION_IDS]

    print("=" * 60)
    print(f"  待清除题目: {qids}")
    print(f"  MODEL_NAME: {MODEL_NAME} | TASK_NAME: {TASK_NAME}")
    print("=" * 60)

    # 第一步：清除 checkpoint 记录
    print(f"\n[Step 1] 清除 checkpoint 记录")
    ck_deleted = delete_from_checkpoint(qids)

    # 第二步：清除各阶段结果 JSON 中的旧 key
    total_json_deleted = 0
    if ACTION_NAMES:
        print(f"\n[Step 2] 清除结果 JSON 中的旧 key（共 {len(ACTION_NAMES)} 个阶段）")
        for action in ACTION_NAMES:
            print(f"\n  -- {action} --")
            total_json_deleted += delete_from_result_json(action, qids)
    else:
        print(f"\n[Step 2] ACTION_NAMES 为空，跳过结果 JSON 清理")

    print("\n" + "=" * 60)
    print(f"  [完成] checkpoint 删除 {ck_deleted} 条，结果 JSON 共删除 {total_json_deleted} 条")
    print(f"  上述题目现在可以重新运行并写入最新结果")
    print("=" * 60)


if __name__ == "__main__":
    main()
