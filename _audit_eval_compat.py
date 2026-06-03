"""Audit all debate_zy JSON files for eval_bbh IndexError risks."""
import json
from pathlib import Path

BASE = Path("qwen2.5-7b-instruct/results/debate_zy/math_500_id")


def eval_rounds_to_check(first_agent_len: int) -> list[int]:
    """Same rounds eval_bbh actually uses."""
    return [r for r in range(first_agent_len) if r % 2 == 0 and r != 0]


def audit_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {"error": "top-level not dict"}

    questions = list(data.keys())
    if not questions:
        return {"error": "empty file"}

    first = data[questions[0]]
    if not isinstance(first, list) or len(first) < 1:
        return {"error": "bad first entry structure"}

    agent_contexts = first[0]
    if not agent_contexts or not isinstance(agent_contexts[0], list):
        return {"error": "bad agent_contexts"}

    ref_len = len(agent_contexts[0])
    rounds_checked = eval_rounds_to_check(ref_len)

    qids_in_file = []
    failures = []  # (qid, agent_idx, round, actual_len, roles)
    uneven_questions = []  # qid with different lens across agents

    for qkey, entry in data.items():
        if not isinstance(entry, list) or len(entry) < 3:
            failures.append((qkey[:20], -1, -1, 0, ["BAD_ENTRY"]))
            continue
        qid = str(entry[2])
        qids_in_file.append(int(qid) if qid.isdigit() else qid)
        agents = entry[0]
        lens = [len(a) if isinstance(a, list) else 0 for a in agents]
        if min(lens) != max(lens):
            uneven_questions.append((qid, lens))

        for aidx, ctx in enumerate(agents):
            if not isinstance(ctx, list):
                failures.append((qid, aidx, -1, 0, ["NOT_LIST"]))
                continue
            for r in rounds_checked:
                if r >= len(ctx):
                    failures.append((qid, aidx, r, len(ctx), [m.get("role") for m in ctx]))

    try:
        qids_sorted = sorted(qids_in_file, key=lambda x: int(x) if str(x).isdigit() else str(x))
    except Exception:
        qids_sorted = sorted(str(x) for x in qids_in_file)

    numeric_qids = [int(x) for x in qids_in_file if str(x).isdigit()]
    missing_1_500 = [i for i in range(1, 501) if i not in numeric_qids] if numeric_qids else []

    return {
        "n_questions": len(data),
        "ref_agent_msg_len": ref_len,
        "rounds_evaluated": rounds_checked,
        "failures": failures,
        "uneven": uneven_questions,
        "qid_min": min(numeric_qids) if numeric_qids else None,
        "qid_max": max(numeric_qids) if numeric_qids else None,
        "missing_1_500_count": len(missing_1_500),
        "missing_1_500_sample": missing_1_500[:15],
    }


def main():
    files = sorted(BASE.glob("*.json"))
    print(f"Auditing {len(files)} files under {BASE.resolve()}\n")

    all_problem_qids = {}  # qid -> list of (filename, details)

    for path in files:
        print("=" * 72)
        print(path.name)
        print("=" * 72)
        try:
            r = audit_file(path)
        except Exception as e:
            print(f"  FATAL: {e}")
            continue

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        print(f"  题目数: {r['n_questions']}")
        print(f"  第一题 agent 消息数: {r['ref_agent_msg_len']}")
        print(f"  eval 会访问的 round 下标: {r['rounds_evaluated']}")
        if r["qid_min"] is not None:
            print(f"  question_id 范围: {r['qid_min']} ~ {r['qid_max']}")
            print(f"  相对 1-500 缺失: {r['missing_1_500_count']} 题")

        if r["failures"]:
            print(f"\n  [会触发 IndexError] 共 {len(r['failures'])} 条 agent×round 问题:")
            by_qid = {}
            for qid, aidx, rnd, ln, roles in r["failures"]:
                by_qid.setdefault(qid, []).append((aidx, rnd, ln, roles))
            for qid in sorted(by_qid.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                items = by_qid[qid]
                print(f"    question_id={qid}:")
                for aidx, rnd, ln, roles in items[:10]:
                    print(f"      agent={aidx}, round={rnd}, len={ln}, roles={roles}")
                if len(items) > 10:
                    print(f"      ... 另有 {len(items)-10} 个 agent")
                all_problem_qids.setdefault(str(qid), []).append(path.name)
        else:
            print("\n  [结构] 无 IndexError 风险（在 eval 使用的 round 下标上）")

        if r["uneven"] and not r["failures"]:
            print(f"\n  [注意] {len(r['uneven'])} 题 agent 消息数不一致但未触及 eval round:")
            for qid, lens in r["uneven"][:5]:
                print(f"    qid={qid}: {lens}")

        print()

    print("=" * 72)
    print("汇总：会导致 eval 崩溃的 question_id（按文件）")
    print("=" * 72)
    for qid, files_list in sorted(all_problem_qids.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        print(f"  question_id={qid}: {', '.join(files_list)}")


if __name__ == "__main__":
    main()
