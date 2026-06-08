"""
根据 question_id 读取 debate 结果 JSON：
  - 按阶段展示：先 expand（全体 agent），再 exchange1（全体 agent），以此类推；
  - 磁盘上单行压缩的 JSON 可在「本题原始条目」一节用 indent=2 缩进打印。

用法：在下方「可配置项」里改 QUESTION_ID、DEBATE_JSON_PATH（或 DEBATE_DIR + DEBATE_JSON_FILE），然后运行:
  python test.py

查看指定题目：COUNT_FILE / COUNT_DIR / COMPARE_JSON_FILES 设为 None，填写 DEBATE_JSON_PATH 与 QUESTION_ID。
COMPARE_JSON_FILES：对比两个 JSON 的 question_id（优先级最高）。
COUNT_FILE：仅统计单个 JSON 的题目数。
COUNT_DIR：统计目录下所有 JSON（COMPARE_JSON_FILES 与 COUNT_FILE 为 None 时生效）。

QUESTION_ID 可填单个题号（如 1）或列表（如 [1, 5, 10]），一次打印多题时按列表顺序依次输出。

默认结果目录为 qwen2.5-7b-instruct/results/debate（与 debate_bbh 一致）。
若读取 wzy 管线产物，文件在 results/debate_zy/... 下：请在下方设置
DEBATE_DIR = "qwen2.5-7b-instruct/results/debate_zy"，或让 DEBATE_JSON_FILE 使用
以 debate_zy 开头的文件名（未手动设 DEBATE_DIR 时会自动选用 debate_zy 根目录）。
题库默认：qwen2.5-7b-instruct/data/math_500_id.json。JSON 在子目录 math_500_id 下时，
只要文件名正确，会在该结果根目录下递归查找。
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

# 本脚本所在目录 = 项目根（与 qwen2.5-7b-instruct 等文件夹同级）
_REPO_ROOT = Path(__file__).resolve().parent


def _as_path(p: str | Path | None, default_under_repo: Path) -> Path:
    """
    - None / 空字符串：使用 default_under_repo（相对仓库根）
    - str / Path：若为相对路径，则相对 _REPO_ROOT 解析；绝对路径则原样 resolve
    """
    if p is None or p == "":
        return default_under_repo.resolve()
    path = Path(p)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _rel_display(base: Path, p: Path) -> str:
    """打印用相对路径；若 p 不在 base 下（盘符/解析差异），则退回绝对路径字符串。"""
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(p.resolve())


# ---------- 可配置项（改这里即可）----------

# ── 模式 A：查看指定题目（填 DEBATE_JSON_PATH + QUESTION_ID，下面两项统计置 None）──
DEBATE_JSON_PATH: str | Path | None = (
    "gpt-5-mini/results/debate_zy/math_500_id/"
    "debate_zy_gpt-5-mini_10_1_exchange2_agent_com0_False.json"
)
QUESTION_ID: int | str | list[int | str] = [1,2]  # 单题填 1；多题填 [1, 5, 10]

# 题库 JSON（用于根据 question_id 匹配结果文件中的键）；与结果文件同模型目录时写 gpt-5-mini/...
MATH_ID_JSON: str | Path | None = "gpt-5-mini/data/math_500_id.json"

COUNT_ONLY = False  # True：加载文件后只打印题目总数，不打印各 agent 内容

# ── 模式 B：对比两个 JSON 的 question_id（填 2 个文件名或相对/绝对路径）──
COMPARE_JSON_DIR: str | Path | None = "gpt-5-mini/results/debate/math_500_id"
COMPARE_JSON_FILES: list[str | Path] | None = None
# 置为 None 时会自动扫描 COMPARE_JSON_DIR 目录下的所有 JSON 文件；
# 若需手动指定，改回类似：
# COMPARE_JSON_FILES = [
#     "debate_gpt-5-mini_10_3_expand_exchangeI41_exchangeI41_agent_com0_False.json",
#     "debate_gpt-5-mini_10_3_expand_exchangeI61_exchangeI61_agent_com0_False.json",
# ]
# 对比时是否打印完整 id 列表（题多时可改为 False，仅看数量与差集）
COMPARE_SHOW_ID_LISTS: bool = True

# ── 模式 C：仅统计题目数 ──
COUNT_FILE: str | Path | None = None
COUNT_DIR: str | Path | None = None
SHOW_QUESTION_IDS: bool = True

# ── 模式 D：未设 DEBATE_JSON_PATH 时，用目录 + 文件名定位结果 JSON ──
DEBATE_JSON_FILE: str | None = "debate_zy_gpt-5-mini_10_1_exchange2_agent_com0_False.json"
DEBATE_DIR: str | Path | None = "gpt-5-mini/results/debate_zy/math_500_id"

# True：在按 agent 打印之后，再输出本题在 JSON 中的整条记录（json缩进，易读）
PRETTY_PRINT_QUESTION_ENTRY = False

# True：额外把整个 debate 文件用缩进打印（题很多时会非常长，一般保持 False）
PRETTY_PRINT_ENTIRE_FILE = False


def _configure_stdio_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def build_debate_dict_key(user_input: str) -> str:
    """与 debate_bbh_qwen3b.parse_question_answer 中 question_content 一致。"""
    return (
        f"Can you answer the following question as accurately as possible? {user_input} \n"
        " Explain your answer.Make sure putting the answer in the form (X) at the end of your response."
    )


def resolve_debate_question_key(debate_data: dict, meta: dict) -> tuple[str | None, str]:
    """
    在 debate JSON 顶层 dict 中解析本题对应的键。

    - debate_bbh 产物：键为 build_debate_dict_key(input)（带 Can you answer... 前缀）。
    - debate_zy / wzy expand 缓存：键为数据集 ``input`` 原文（见 expand_save_cache_if_enabled 的 merged[question]）。
    - 少数情况：键可能为 str(question_id)。
    """
    inp = meta.get("input")
    qid = meta.get("question_id")
    candidates: list[tuple[str, str]] = []
    if inp is not None and str(inp) != "":
        candidates.append((build_debate_dict_key(str(inp)), "debate_bbh 包装键"))
        candidates.append((str(inp), "数据集 input 原文（debate_zy）"))
    if qid is not None:
        candidates.append((str(qid), "question_id 字符串键"))
    for key, label in candidates:
        if key in debate_data:
            return key, label
    return None, ""


def normalize_question_ids(question_ids: int | str | list[int | str]) -> list[str]:
    """将 QUESTION_ID 规范化为字符串列表，支持单值或列表。"""
    if isinstance(question_ids, (list, tuple)):
        if not question_ids:
            raise ValueError("QUESTION_ID 列表不能为空")
        return [str(qid) for qid in question_ids]
    return [str(question_ids)]


def load_example_by_question_id(math_json_path: Path, question_id: str) -> dict | None:
    with math_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    qid = str(question_id)
    for ex in data.get("examples", []):
        if str(ex.get("question_id", "")) == qid:
            return ex
    return None


def unwrap_debate_value(raw):
    """
    debate_bbh_qwen3b 每题写入: (agent_contexts, answer, question_id)
    JSON 中变为 [ agent_contexts, answer, question_id ]。

    若 raw 已是「agent 列表」（每项为该 agent 的消息 dict 列表），则原样返回第一个元素为整段 raw。
    """
    if not isinstance(raw, list) or len(raw) != 3:
        return raw, None, None
    a0, a1, a2 = raw
    if not isinstance(a0, list):
        return raw, None, None
    if a0:
        first = a0[0]
        # 三元组格式：第一项的每个元素是「消息列表」list[dict]
        if isinstance(first, list) and first and isinstance(first[0], dict):
            return a0, a1, a2
        # 否则视为恰好 3 个 agent，每项消息列表以 dict 开头
        return raw, None, None
    # 空 agent 列表但文件仍可能是 ( [], gt, qid )
    if not isinstance(a1, list):
        return a0, a1, a2
    return raw, None, None


def _print_block(title: str, subtitle: str, content: str, line_prefix: str = "  |  ") -> None:
    """打印单个 agent 在某阶段的回复块，超长行自动软折行以保证可读性。"""
    head = f"  --- {title} | {subtitle}"
    print("\n" + head + " " + "-" * max(0, 72 - len(head)))
    text = content if content else "(空)"
    # 可用于实际文字的列宽（去掉行前缀后剩余）
    wrap_width = 100
    for line in text.split("\n"):
        if not line:
            # 空行原样保留（段落间距）
            print(line_prefix)
        elif len(line) <= wrap_width:
            print(f"{line_prefix}{line}")
        else:
            # 超长行按 wrap_width 软折行，续行对齐到前缀
            wrapped = textwrap.wrap(line, width=wrap_width,
                                    subsequent_indent=" " * len(line_prefix))
            for wl in wrapped:
                print(f"{line_prefix}{wl}")
    print("  " + "-" * 74)


def _stage_name_from_assistant_index(assistant_turn: int) -> str:
    """第 1 条 assistant = expand；第 2 条 = exchange1；第 3 条 = exchange2 …"""
    if assistant_turn <= 0:
        return "unknown"
    if assistant_turn == 1:
        return "expand"
    return f"exchange{assistant_turn - 1}"


def _count_assistant_messages(agent_ctx: list) -> int:
    if not isinstance(agent_ctx, list):
        return 0
    return sum(
        1 for m in agent_ctx if isinstance(m, dict) and m.get("role") == "assistant"
    )


def _get_nth_assistant_content(agent_ctx: list, n: int) -> str | None:
    """返回该 agent 上下文中第 n 条 assistant 的 content（n 从 1 起）；没有则 None。"""
    if not isinstance(agent_ctx, list) or n < 1:
        return None
    seen = 0
    for msg in agent_ctx:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        seen += 1
        if seen == n:
            return msg.get("content", "")
    return None


def print_agents_for_question(debate_data: dict, question_key: str | None, meta: dict) -> None:
    if not question_key or question_key not in debate_data:
        print(f"[错误] JSON 中找不到该题对应的键（question_id={meta.get('question_id')}）。")
        print(" 题目 input 前 120 字符预览:")
        preview = meta.get("input", "")[:120].replace("\n", " ")
        print(f"  {preview!r}")
        print(
            "  说明: debate_zy 使用 input 原文作键；debate_bbh 使用「Can you answer...」包装键。"
            " 若本题不在该 JSON 内，也会匹配失败。"
        )
        keys = list(debate_data.keys())
        if keys:
            k0 = keys[0]
            tail = "..." if len(k0) > 100 else ""
            print(f"本文件共 {len(keys)} 题；示例键前 100 字符: {k0[:100]!r}{tail}")
        return

    raw_entry = debate_data[question_key]
    agents_payload, file_answer, file_qid = unwrap_debate_value(raw_entry)
    if not isinstance(agents_payload, list):
        print(f"[错误] 该题对应的数据类型为 {type(agents_payload)}，期望 list[agent_context].")
        return

    qid = meta.get("question_id", "?")
    gt = meta.get("target", "?")
    inp = meta.get("input", "")

    sep = "=" * 80
    print(sep)
    print(f"  question_id = {qid}")
    print(f"  ground_truth = {gt}")
    if file_answer is not None or file_qid is not None:
        print(f"  (JSON 内附带) file_answer = {file_answer!r}  file_question_id = {file_qid!r}")
    print(f"  题目 input（前 200 字符）:")
    for line in inp[:200].split("\n"):
        print(f"    {line}")
    if len(inp) > 200:
        print("    ...")
    print(sep)

    max_turns = 0
    for agent_ctx in agents_payload:
        max_turns = max(max_turns, _count_assistant_messages(agent_ctx))

    if max_turns == 0:
        print("  (所有 agent 均无 assistant 回复)")
        return

    for turn in range(1, max_turns + 1):
        stage = _stage_name_from_assistant_index(turn)
        banner = f" 阶段: {stage}（全 agent 第 {turn} 条 assistant） "
        line = banner.center(80, "═")
        print(f"\n{'═' * 80}\n{line}\n{'═' * 80}")

        for agent_idx, agent_ctx in enumerate(agents_payload):
            if not isinstance(agent_ctx, list):
                print(f"\n  [Agent {agent_idx}] (非列表上下文，跳过: {type(agent_ctx)})")
                continue
            content = _get_nth_assistant_content(agent_ctx, turn)
            if content is None:
                _print_block(
                    f"Agent {agent_idx}",
                    f"{stage}（无本条回复）",
                    "(本 agent 在此阶段无 assistant 消息)",
                )
                continue
            _print_block(f"Agent {agent_idx}", stage, content)


def _all_json_under(debate_dir: Path) -> list[Path]:
    """递归收集 debate_dir 下所有 .json（含子目录如 math_500_id）。"""
    if not debate_dir.is_dir():
        return []
    return sorted(debate_dir.rglob("*.json"))


def pick_debate_file(debate_dir: Path, file_arg: str | None) -> Path:
    if not debate_dir.is_dir():
        raise FileNotFoundError(
            f"debate 目录不存在或不是文件夹: {debate_dir}\n"
            f"  请确认路径，或在 test.py 里设置 DEBATE_DIR 为相对仓库根的路径，例如:\n"
            f'  DEBATE_DIR = "qwen2.5-7b-instruct/results/debate"'
        )

    all_json = _all_json_under(debate_dir)
    if not all_json:
        raise FileNotFoundError(
            f"目录下没有 .json（含子目录）: {debate_dir}\n"
            f"  仓库根目录为: {_REPO_ROOT}"
        )

    if file_arg:
        # 1) 相对 DEBATE_DIR 的直接路径
        direct = debate_dir / file_arg
        if direct.is_file():
            return direct.resolve()
        # 2) 仅文件名：在整棵子树中找第一个同名文件
        name_only = Path(file_arg).name
        matches = [p for p in all_json if p.name == name_only]
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            print(f"[提示] 找到多个同名 JSON，请改用相对路径或唯一文件名。候选:")
            for p in matches:
                print(f"  - {_rel_display(debate_dir, p)}")
            raise SystemExit(1)
        raise FileNotFoundError(
            f"在 {debate_dir} 下找不到文件: {file_arg!r}（已搜索子目录）"
        )

    if len(all_json) > 1:
        print(f"[提示] 目录树中共有 {len(all_json)} 个 JSON，请在 test.py 顶部设置 DEBATE_JSON_FILE，例如:")
        for f in all_json[:30]:
            print(f"  - {_rel_display(debate_dir, f)}")
        if len(all_json) > 30:
            print(f"  ... 另有 {len(all_json) - 30} 个文件未列出")
        raise SystemExit(1)

    return all_json[0].resolve()


def resolve_debate_json_path() -> Path:
    """
    解析要读取的结果 JSON 路径。
    优先级：DEBATE_JSON_PATH（完整相对/绝对路径）> DEBATE_DIR + DEBATE_JSON_FILE。
    """
    if DEBATE_JSON_PATH is not None and DEBATE_JSON_PATH != "":
        p = _as_path(DEBATE_JSON_PATH, _REPO_ROOT)
        if not p.is_file():
            raise FileNotFoundError(f"DEBATE_JSON_PATH 指向的文件不存在: {p}")
        return p.resolve()
    debate_dir = _as_path(DEBATE_DIR, _default_debate_root())
    return pick_debate_file(debate_dir, DEBATE_JSON_FILE)


def dump_json_readable(obj, title: str) -> None:
    """将任意可 JSON 序列化对象缩进打印到 stdout（单文件一行也能看清结构）。"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        print(f"  (无法序列化为 JSON: {e})")
        print(repr(obj))
        return
    for line in text.splitlines():
        print(line)
    print("=" * 80)


def _default_debate_root() -> Path:
    """未指定 DEBATE_DIR 时：debate_zy 文件名 → debate_zy 目录，否则 → debate 目录。"""
    base = _REPO_ROOT / "qwen2.5-7b-instruct" / "results"
    if DEBATE_JSON_FILE and str(DEBATE_JSON_FILE).strip().startswith("debate_zy"):
        return base / "debate_zy"
    return base / "debate"


def _extract_question_ids(data: dict) -> list[str]:
    """
    从 debate / debate_zy 结果 JSON 中提取所有 question_id，按数值排序返回。

    支持两种格式：
      - debate_zy：value = [agent_contexts, gt, question_id]，取 value[2]
      - debate_bbh：value 结构相同，取 value[2]
      - 兜底：若 value[2] 不可用，则用顶层 key 本身作为 id
    """
    ids: list[str] = []
    for key, value in data.items():
        qid = None
        if isinstance(value, list) and len(value) >= 3:
            qid = str(value[2])
        if qid is None:
            qid = str(key) if not isinstance(key, str) or len(key) <= 20 else None
        if qid is not None:
            ids.append(qid)
    # 尽量按数值排序，无法转换的按字符串排序
    try:
        ids.sort(key=lambda x: int(x))
    except ValueError:
        ids.sort()
    return ids


def count_all_jsons_in_dir(dir_path: Path, show_question_ids: bool = False) -> None:
    """统计指定目录下所有 .json 文件各含多少道题（顶层 key 数量），按文件名排序输出。

    show_question_ids=True 时，额外列出每个文件中的所有 question_id。
    """
    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        print(f"[统计] 目录下未找到任何 .json 文件: {dir_path}")
        return

    print(f"\n[目录统计] {dir_path}")
    print(f"{'─' * 70}")
    max_name_len = max(len(p.name) for p in json_files)
    total_row = []
    for p in json_files:
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                print(f"  {p.name:<{max_name_len}}  N/A（非 dict）")
                total_row.append((p.name, "N/A"))
                continue
            count = len(data)
        except json.JSONDecodeError as e:
            print(f"  {p.name:<{max_name_len}}  解析失败: {e}")
            total_row.append((p.name, f"解析失败"))
            continue
        except OSError as e:
            print(f"  {p.name:<{max_name_len}}  读取失败: {e}")
            total_row.append((p.name, f"读取失败"))
            continue

        print(f"\n  {'─' * 66}")
        print(f"  文件: {p.name}")
        print(f"  题目数: {count} 道")

        if show_question_ids:
            qids = _extract_question_ids(data)
            if qids:
                # 每行打印 20 个 id，避免单行过长
                chunk = 20
                print(f"  question_id 列表（共 {len(qids)} 个，按数值升序）:")
                for i in range(0, len(qids), chunk):
                    line = ", ".join(qids[i: i + chunk])
                    print(f"    {line}")
            else:
                print(f"  （无法提取 question_id）")

        total_row.append((p.name, count))

    print(f"\n  {'─' * 66}")
    numeric_counts = [c for _, c in total_row if isinstance(c, int)]
    if numeric_counts:
        print(f"  共 {len(json_files)} 个文件，题目数范围: {min(numeric_counts)} ~ {max(numeric_counts)}")
    print(f"{'─' * 70}")


def _resolve_compare_json_path(dir_path: Path, file_spec: str | Path) -> Path:
    """将文件名或路径解析为绝对路径；相对路径依次尝试 COMPARE_JSON_DIR、仓库根。"""
    p = Path(file_spec)
    if p.is_file():
        return p.resolve()
    for base in (dir_path, _REPO_ROOT):
        cand = (base / p).resolve()
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"找不到对比用 JSON: {file_spec!r}（已尝试目录 {dir_path} 与仓库根）"
    )


def _print_question_id_list(title: str, qids: list[str], chunk: int = 20) -> None:
    print(f"  {title}（共 {len(qids)} 个）:")
    if not qids:
        print("    (无)")
        return
    for i in range(0, len(qids), chunk):
        print(f"    {', '.join(qids[i: i + chunk])}")


def compare_two_json_question_ids(
    file_a: Path,
    file_b: Path,
    *,
    show_lists: bool = True,
) -> None:
    """加载两个 debate 结果 JSON，对比各自包含的 question_id。"""
    def load_ids(path: Path) -> list[str]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"{path.name} 顶层不是 dict")
        return _extract_question_ids(data)

    ids_a = load_ids(file_a)
    ids_b = load_ids(file_b)
    set_a = set(ids_a)
    set_b = set(ids_b)
    only_a = sorted(set_a - set_b, key=lambda x: (len(x), x))
    only_b = sorted(set_b - set_a, key=lambda x: (len(x), x))
    both = sorted(set_a & set_b, key=lambda x: (len(x), x))

    try:
        only_a.sort(key=int)
        only_b.sort(key=int)
        both.sort(key=int)
    except ValueError:
        pass

    print(f"\n[双文件对比] question_id")
    print(f"{'─' * 70}")
    print(f"  文件 A: {_rel_display(_REPO_ROOT, file_a)}")
    print(f"         题目数 {len(ids_a)}")
    print(f"  文件 B: {_rel_display(_REPO_ROOT, file_b)}")
    print(f"         题目数 {len(ids_b)}")
    print(f"  两文件共有: {len(both)} 道")
    print(f"  仅在 A 中: {len(only_a)} 道")
    print(f"  仅在 B 中: {len(only_b)} 道")

    if show_lists:
        print(f"\n  {'─' * 66}")
        _print_question_id_list("文件 A 全部 question_id", ids_a)
        print(f"\n  {'─' * 66}")
        _print_question_id_list("文件 B 全部 question_id", ids_b)
        if only_a:
            print(f"\n  {'─' * 66}")
            _print_question_id_list("仅在 A、不在 B", only_a)
        if only_b:
            print(f"\n  {'─' * 66}")
            _print_question_id_list("仅在 B、不在 A", only_b)
    else:
        if only_a:
            print(f"  仅在 A: {', '.join(only_a)}")
        if only_b:
            print(f"  仅在 B: {', '.join(only_b)}")

    print(f"{'─' * 70}")


def count_single_json_file(file_path: Path, show_question_ids: bool = True) -> None:
    """统计指定 JSON 文件中的题目数量，并可列出全部 question_id。"""
    if not file_path.exists():
        print(f"[错误] 文件不存在: {file_path}")
        return

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[错误] JSON 解析失败: {e}")
        return
    except OSError as e:
        print(f"[错误] 读取失败: {e}")
        return

    if not isinstance(data, dict):
        print(f"[错误] 顶层不是 dict，无法统计题目数")
        return

    count = len(data)
    print(f"\n[单文件统计] {file_path.resolve()}")
    print(f"{'─' * 70}")
    print(f"  题目数: {count} 道")

    if show_question_ids:
        qids = _extract_question_ids(data)
        if qids:
            chunk = 20
            print(f"  question_id 列表（共 {len(qids)} 个，按数值升序）:")
            for i in range(0, len(qids), chunk):
                line = ", ".join(qids[i: i + chunk])
                print(f"    {line}")
        else:
            print("  （无法从 value[2] 提取 question_id，顶层 key 数量即题目数）")
    print(f"{'─' * 70}")


def main() -> None:
    _configure_stdio_utf8()

    # 双文件 question_id 对比（优先级最高）
    if COMPARE_JSON_FILES is not None and len(COMPARE_JSON_FILES) > 0:
        if len(COMPARE_JSON_FILES) != 2:
            print("[错误] COMPARE_JSON_FILES 须恰好包含 2 个 JSON 路径或文件名")
            raise SystemExit(1)
        compare_dir = _as_path(
            COMPARE_JSON_DIR,
            _REPO_ROOT / "gpt-5-mini" / "results" / "debate" / "math_500_id",
        )
        if not compare_dir.is_dir():
            print(f"[错误] COMPARE_JSON_DIR 不存在: {compare_dir}")
            raise SystemExit(1)
        path_a = _resolve_compare_json_path(compare_dir, COMPARE_JSON_FILES[0])
        path_b = _resolve_compare_json_path(compare_dir, COMPARE_JSON_FILES[1])
        compare_two_json_question_ids(
            path_a,
            path_b,
            show_lists=COMPARE_SHOW_ID_LISTS,
        )
        return

    # 自动扫描目录模式：COMPARE_JSON_FILES 为 None 且 COMPARE_JSON_DIR 已设置
    if COMPARE_JSON_FILES is None and COMPARE_JSON_DIR is not None and COMPARE_JSON_DIR != "":
        compare_dir = _as_path(
            COMPARE_JSON_DIR,
            _REPO_ROOT / "gpt-5-mini" / "results" / "debate" / "math_500_id",
        )
        if not compare_dir.is_dir():
            print(f"[错误] COMPARE_JSON_DIR 不存在: {compare_dir}")
            raise SystemExit(1)
        json_files = sorted(compare_dir.glob("*.json"))
        if not json_files:
            print(f"\n[提示] 目录中没有 JSON 文件: {compare_dir}")
            raise SystemExit(0)
        print(f"\n[目录扫描] {compare_dir}")
        print(f"  共发现 {len(json_files)} 个 JSON 文件")
        if len(json_files) == 2:
            # 恰好两个文件：直接对比
            compare_two_json_question_ids(
                json_files[0],
                json_files[1],
                show_lists=COMPARE_SHOW_ID_LISTS,
            )
        else:
            # 其他数量：逐文件统计各自的 question_id
            for jf in json_files:
                count_single_json_file(jf, show_question_ids=SHOW_QUESTION_IDS)
        return

    # 单文件统计模式
    if COUNT_FILE is not None and COUNT_FILE != "":
        count_file = _as_path(COUNT_FILE, _REPO_ROOT)
        count_single_json_file(count_file, show_question_ids=SHOW_QUESTION_IDS)
        return

    # 目录批量统计模式
    if COUNT_DIR is not None and COUNT_DIR != "":
        count_dir = _as_path(COUNT_DIR, _REPO_ROOT)
        if not count_dir.exists():
            print(f"[错误] COUNT_DIR 路径不存在: {count_dir}")
        else:
            count_all_jsons_in_dir(count_dir, show_question_ids=SHOW_QUESTION_IDS)
        return

    debate_path = resolve_debate_json_path()

    print(f"[读取] {debate_path}")
    with debate_path.open("r", encoding="utf-8") as f:
        debate_data = json.load(f)

    total_keys = len(debate_data)
    print(f"\n[统计] 该文件共包含 {total_keys} 道题目")
    print(f"[统计] 文件路径: {debate_path.resolve()}")

    if COUNT_ONLY:
        return

    math_json = _as_path(
        MATH_ID_JSON,
        _REPO_ROOT / "qwen2.5-7b-instruct" / "data" / "math_500_id.json",
    )

    question_ids = normalize_question_ids(QUESTION_ID)
    print(f"\n[计划] 将依次打印 {len(question_ids)} 道题: {', '.join(question_ids)}")

    for idx, qid in enumerate(question_ids, start=1):
        if len(question_ids) > 1:
            print("\n" + "#" * 80)
            print(f"# 题目进度 {idx}/{len(question_ids)}  question_id={qid}")
            print("#" * 80)

        ex = load_example_by_question_id(math_json, qid)
        if ex is None:
            print(f"[错误] 在 {math_json} 中未找到 question_id={qid!r}")
            continue

        question_key, key_kind = resolve_debate_question_key(debate_data, ex)
        if question_key:
            print(f"[匹配键] {key_kind}")

        print_agents_for_question(debate_data, question_key, ex)

        if PRETTY_PRINT_QUESTION_ENTRY and question_key and question_key in debate_data:
            dump_json_readable(
                debate_data[question_key],
                f"本题在 JSON 中的原始条目（缩进显示，question_id={qid}）",
            )

    if PRETTY_PRINT_ENTIRE_FILE:
        dump_json_readable(debate_data, "完整 debate 文件内容（缩进显示）")


if __name__ == "__main__":
    main()
