import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import eval_all_round as ev


def latest_assistant_round(context: List[dict]) -> int:
    for idx in range(len(context) - 1, -1, -1):
        if context[idx].get("role") == "assistant":
            return idx
    return -1


def extract_round_answers(record: list, round_idx: Optional[int]) -> Tuple[Optional[bool], Optional[str], List[Optional[str]]]:
    responses, gt, _qid = record
    pred_solutions = []
    for ctx in responses:
        if round_idx is None:
            ridx = latest_assistant_round(ctx)
        else:
            ridx = round_idx
        if ridx < 0 or ridx >= len(ctx):
            pred_solutions.append("")
        else:
            pred_solutions.append(ctx[ridx].get("content", ""))
    accurate, majority = ev.compute_accuracy(gt, pred_solutions, is_math=True)
    agent_answers = ev.extract_agent_answers_from_solutions(pred_solutions, is_math=True)
    return accurate, majority, agent_answers


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_by_qid(data: dict) -> Dict[str, Tuple[str, list]]:
    out = {}
    for question, record in data.items():
        if isinstance(record, list) and len(record) >= 3:
            out[str(record[2])] = (question, record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-turbo")
    parser.add_argument("--task", default="math_500_id")
    parser.add_argument("--exchange-i41", default=None)
    parser.add_argument("--exchange1", default=None)
    parser.add_argument("--exchange-i41-round", type=int, default=None)
    parser.add_argument("--exchange1-round", type=int, default=None)
    parser.add_argument("--out", default="analysis_outputs/model_label_diagnostics/math_i41_correct_exchange1_wrong.json")
    parser.add_argument("--max-print", type=int, default=30)
    args = parser.parse_args()

    model = args.model
    task = args.task
    i41_path = Path(args.exchange_i41 or f"{model}/results/debate/{task}/debate_{model}_10_3_expand_exchangeI41_exchangeI41_agent_com0_False.json")
    ex1_path = Path(args.exchange1 or f"{model}/results/debate_zy/{task}/debate_zy_{model}_10_1_exchange1_agent_com0_False.json")

    i41_data = get_by_qid(load_json(i41_path))
    ex1_data = get_by_qid(load_json(ex1_path))
    common_qids = sorted(set(i41_data) & set(ex1_data), key=lambda x: int(x) if x.isdigit() else x)

    candidates = []
    summary = Counter()
    for qid in common_qids:
        question_i41, rec_i41 = i41_data[qid]
        question_ex1, rec_ex1 = ex1_data[qid]
        i41_ok, i41_majority, i41_agents = extract_round_answers(rec_i41, args.exchange_i41_round)
        ex1_ok, ex1_majority, ex1_agents = extract_round_answers(rec_ex1, args.exchange1_round)
        summary[(bool(i41_ok), bool(ex1_ok))] += 1
        if i41_ok == 1 and ex1_ok == 0:
            candidates.append(
                {
                    "qid": qid,
                    "question": question_ex1,
                    "gt": rec_ex1[1],
                    "exchangeI41_majority": i41_majority,
                    "exchange1_majority": ex1_majority,
                    "exchangeI41_agent_answers": i41_agents,
                    "exchange1_agent_answers": ex1_agents,
                }
            )

    output = {
        "model": model,
        "task": task,
        "exchangeI41_path": str(i41_path),
        "exchange1_path": str(ex1_path),
        "exchangeI41_round": args.exchange_i41_round,
        "exchange1_round": args.exchange1_round,
        "n_common": len(common_qids),
        "summary": {f"i41_{k[0]}__ex1_{k[1]}": v for k, v in sorted(summary.items())},
        "n_candidates": len(candidates),
        "candidates": candidates,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: output[k] for k in ["model", "task", "n_common", "summary", "n_candidates"]}, ensure_ascii=False, indent=2))
    print(f"[DONE] wrote {out_path}")
    for row in candidates[: args.max_print]:
        print(
            f"qid={row['qid']} gt={row['gt']!r} "
            f"exchangeI41={row['exchangeI41_majority']!r} exchange1={row['exchange1_majority']!r}"
        )


if __name__ == "__main__":
    main()
