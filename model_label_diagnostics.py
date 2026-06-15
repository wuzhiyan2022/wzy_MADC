import argparse
import contextlib
import hashlib
import io
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_MODELS = [
    "qwen-turbo",
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "gpt-5-mini",
]
DEFAULT_TASKS = [
    "math_500_id",
    "geometric_shapes_id",
    "logical_deduction_seven_objects_id",
]
DEFAULT_OUT_DIR = Path("analysis_outputs/model_label_diagnostics")
_EXP_MODULE = None


def get_exp():
    global _EXP_MODULE
    if _EXP_MODULE is None:
        import wzy_multi_agent_debate_expand as exp_module

        _EXP_MODULE = exp_module
    return _EXP_MODULE


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def result_expand_path(model: str, task: str) -> Path:
    return Path(model) / "results" / "debate_zy" / task / f"debate_zy_{model}_10_1_expand_agent_com0_False.json"


def load_data_item_map(model: str, task: str) -> Dict[str, dict]:
    path = Path(model) / "data" / f"{task}.json"
    if not path.exists():
        return {}
    data = json.load(path.open("r", encoding="utf-8"))
    return {str(x.get("question_id")): x for x in data.get("examples", [])}


def make_expand_cfg(model: str, task: str) -> Any:
    exp = get_exp()
    cfg = exp.default_expand_config()
    cfg.model_name = model
    cfg.model_tag = model
    cfg.task_name = task
    cfg.is_math = task == "math_500_id"
    return cfg


def latest_assistant(ctx: list) -> str:
    for msg in reversed(ctx or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""


def answer_from_text(text: str, is_math: bool) -> Optional[str]:
    exp = get_exp()
    return exp.extract_answer_from_text(text, is_math=is_math)


def is_correct(pred: Optional[str], ref: str, is_math: bool) -> bool:
    exp = get_exp()
    return bool(pred is not None and exp.is_correct_answer(pred, ref, is_math=is_math))


def most_frequent(values: List[str]) -> Optional[str]:
    if not values:
        return None
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1
    return max(values, key=lambda v: counts[v])


def build_steps_for_question(
    *,
    model: str,
    task: str,
    question_text: str,
    question_id: str,
    responses: list,
    gt: str,
    is_math: bool,
) -> Tuple[List[dict], Optional[str], List[dict]]:
    exp = get_exp()
    agent_answers = []
    agent_rows = []
    for agent_id, ctx in enumerate(responses):
        response_text = latest_assistant(ctx)
        ans = answer_from_text(response_text, is_math=is_math)
        if ans is not None:
            agent_answers.append(ans)
        agent_rows.append(
            {
                "agent_id": agent_id,
                "answer": ans,
                "response_text": response_text,
                "steps": exp.extract_steps_from_response(response_text),
            }
        )

    majority_answer = most_frequent(agent_answers)

    all_steps = []
    for ar in agent_rows:
        inherited = is_correct(ar["answer"], majority_answer, is_math) if majority_answer else False
        gt_final_correct = is_correct(ar["answer"], gt, is_math)
        for step in ar["steps"]:
            all_steps.append(
                {
                    "model": model,
                    "task": task,
                    "question_id": question_id,
                    "question": question_text,
                    "ground_truth": gt,
                    "majority_answer": majority_answer,
                    "agent_id": ar["agent_id"],
                    "agent_answer": ar["answer"],
                    "agent_answer_matches_majority": inherited,
                    "agent_answer_matches_gt": gt_final_correct,
                    "step_number": step.get("step_number"),
                    "content": step.get("content", ""),
                    "is_correct": inherited,
                    "inherited_label": inherited,
                    "cluster_id": None,
                    "cluster_tag": None,
                    "cluster_label": inherited,
                    "cluster_label_changed": False,
                }
            )

    return all_steps, majority_answer, agent_rows


def apply_cluster_labels(all_steps: List[dict], cfg: Any) -> List[dict]:
    if len(all_steps) < 2:
        return all_steps
    exp = get_exp()
    from wzy_multi_agent_debate_clustering import (
        MAJORITY_THRESHOLD,
        _apply_cluster_labels_to_steps,
        _cluster_hdbscan,
        _label_clusters_by_majority,
        _reduce_dimensions_pca,
        resolve_pca_target_dim,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        step_vectors, step_indices = exp.expand_run_embedding(all_steps, cfg)
    if step_vectors is None or len(step_vectors) < 2:
        return all_steps

    target_dim = resolve_pca_target_dim(step_vectors.shape[0])
    with contextlib.redirect_stdout(io.StringIO()):
        vectors_reduced = _reduce_dimensions_pca(step_vectors, target_dim=target_dim)
        cluster_labels = _cluster_hdbscan(vectors_reduced)
        cluster_tags = _label_clusters_by_majority(
            cluster_labels,
            step_indices,
            all_steps,
            majority_threshold=MAJORITY_THRESHOLD,
        )
        steps_modified, _n_modified, change_records = _apply_cluster_labels_to_steps(
            cluster_labels,
            cluster_tags,
            step_indices,
            all_steps,
        )

    change_by_key = {
        (rec.get("agent_id"), rec.get("step_number")): rec for rec in change_records
    }
    for vec_i, cid in enumerate(cluster_labels):
        step_idx = step_indices[vec_i]
        step = steps_modified[step_idx]
        step["cluster_id"] = int(cid)
        step["cluster_tag"] = cluster_tags.get(int(cid), "no_majority")
        step["cluster_label"] = step.get("is_correct")
        rec = change_by_key.get((step.get("agent_id"), step.get("step_number")))
        step["cluster_label_changed"] = bool(rec and rec.get("changed"))
    return steps_modified


def sample_questions(rows: Dict[str, list], max_questions: int, seed: int) -> List[Tuple[str, Any]]:
    keys = sorted(rows.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    if max_questions is None or max_questions <= 0 or len(keys) <= max_questions:
        return [(k, rows[k]) for k in keys]
    rng = random.Random(seed)
    selected = set(rng.sample(keys, max_questions))
    return [(k, rows[k]) for k in keys if k in selected]


def load_qid_filter(path_text: Optional[str]) -> Optional[set]:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.load(path.open("r", encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("candidates"), list):
            return {str(row["qid"]) for row in data["candidates"] if "qid" in row}
        if isinstance(data, dict) and isinstance(data.get("qids"), list):
            return {str(x) for x in data["qids"]}
        if isinstance(data, list):
            qids = set()
            for row in data:
                if isinstance(row, dict) and "qid" in row:
                    qids.add(str(row["qid"]))
                else:
                    qids.add(str(row))
            return qids
    qids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            qids.add(line)
    return qids


def select_questions(
    rows: Dict[str, list],
    max_questions: int,
    seed: int,
    qid_filter: Optional[set],
) -> List[Tuple[str, Any]]:
    if qid_filter is None:
        return sample_questions(rows, max_questions, seed)
    selected = []
    for key, record in rows.items():
        if isinstance(record, list) and len(record) >= 3 and str(record[2]) in qid_filter:
            selected.append((key, record))
    selected.sort(
        key=lambda x: int(str(x[1][2])) if str(x[1][2]).isdigit() else str(x[1][2])
    )
    if max_questions and max_questions > 0:
        selected = selected[:max_questions]
    return selected


def prepare(args: argparse.Namespace) -> None:
    exp = get_exp()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    items_path = out_dir / "step_label_items.jsonl"
    questions_path = out_dir / "questions.jsonl"
    if args.overwrite:
        items_path.unlink(missing_ok=True)
        questions_path.unlink(missing_ok=True)

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    qid_filter = load_qid_filter(args.qid_file)
    if qid_filter is not None:
        log(f"[PREPARE] qid_filter={len(qid_filter)} from {args.qid_file}")
    total_steps = 0
    total_questions = 0

    for model in models:
        for task in tasks:
            path = result_expand_path(model, task)
            if not path.exists():
                log(f"[SKIP] missing expand result: {path}")
                continue
            is_math = task == "math_500_id"
            data = json.load(path.open("r", encoding="utf-8"))
            qmap = load_data_item_map(model, task)

            cfg = make_expand_cfg(model, task)

            log(f"[PREPARE] {model} / {task}: loaded {len(data)} questions")
            selected_questions = select_questions(
                data, args.max_questions_per_combo, args.seed, qid_filter
            )
            log(f"[PREPARE] {model} / {task}: selected {len(selected_questions)} questions")
            for qid, record in selected_questions:
                if not isinstance(record, list) or len(record) < 3:
                    continue
                responses, gt, record_qid = record[0], record[1], str(record[2])
                question_text = qmap.get(record_qid, {}).get("input", qid)
                all_steps, majority_answer, agent_rows = build_steps_for_question(
                    model=model,
                    task=task,
                    question_text=question_text,
                    question_id=record_qid,
                    responses=responses,
                    gt=gt,
                    is_math=is_math,
                )
                if args.with_cluster:
                    all_steps = apply_cluster_labels(all_steps, cfg)

                question_row = {
                    "model": model,
                    "task": task,
                    "question_id": record_qid,
                    "question": question_text,
                    "ground_truth": gt,
                    "majority_answer": majority_answer,
                    "n_agents": len(agent_rows),
                    "n_steps": len(all_steps),
                    "with_cluster": bool(args.with_cluster),
                }
                append_jsonl(questions_path, question_row)
                for step in all_steps:
                    step["step_uid"] = f"{model}|{task}|{record_qid}|{step['agent_id']}|{step['step_number']}"
                    append_jsonl(items_path, step)
                    total_steps += 1
                total_questions += 1
                log(f"  qid={record_qid}: steps={len(all_steps)}, majority={majority_answer!r}")

    log(f"[DONE] wrote {total_questions} questions and {total_steps} steps")
    log(f"       {items_path}")
    log(f"       {questions_path}")


def group_steps_for_verify(items_path: Path, max_agents_per_question: int) -> Dict[Tuple[str, str, str, int], List[dict]]:
    groups = defaultdict(list)
    for row in iter_jsonl(items_path):
        aid = int(row["agent_id"])
        if aid >= max_agents_per_question:
            continue
        key = (row["model"], row["task"], str(row["question_id"]), aid)
        groups[key].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: int(r.get("step_number") or 0))
    return groups


def label_cache_key(group: List[dict], verifier_model: str) -> str:
    payload = {
        "verifier_model": verifier_model,
        "model": group[0]["model"],
        "task": group[0]["task"],
        "question_id": group[0]["question_id"],
        "agent_id": group[0]["agent_id"],
        "steps": [(x.get("step_number"), x.get("content", "")) for x in group],
    }
    return hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def select_verify_keys(
    groups: Dict[Tuple[str, str, str, int], List[dict]],
    done: set,
    verifier_model: str,
    max_chains: int,
    selection: str,
    seed: int,
) -> List[Tuple[str, str, str, int]]:
    pending = []
    for key in sorted(groups.keys()):
        if label_cache_key(groups[key], verifier_model) not in done:
            pending.append(key)
    if selection == "sequential":
        return pending[:max_chains] if max_chains and max_chains > 0 else pending

    rng = random.Random(seed)
    by_combo = defaultdict(list)
    for key in pending:
        model, task, _qid, _aid = key
        by_combo[(model, task)].append(key)
    for keys in by_combo.values():
        rng.shuffle(keys)

    selected = []
    combos = sorted(by_combo.keys())
    while combos and (not max_chains or max_chains <= 0 or len(selected) < max_chains):
        next_combos = []
        for combo in combos:
            keys = by_combo[combo]
            if keys and (not max_chains or max_chains <= 0 or len(selected) < max_chains):
                selected.append(keys.pop())
            if keys:
                next_combos.append(combo)
        combos = next_combos
    return selected


def focus_group_keys_from_items(items_path: Path, changed_only: bool) -> set:
    keys = set()
    for row in iter_jsonl(items_path):
        if changed_only and not row.get("cluster_label_changed"):
            continue
        keys.add((row["model"], row["task"], str(row["question_id"]), int(row["agent_id"])))
    return keys


def verifier_prompt(group: List[dict]) -> str:
    first = group[0]
    steps_text = "\n".join(
        f"Step {row['step_number']}: {row.get('content', '')}" for row in group
    )
    return f"""You are auditing chain-of-thought steps for a reasoning benchmark.

Task:
Decide whether each step is logically valid and useful for solving the original problem.
Use the ground truth answer only as a verification reference, not as a shortcut.

Labels:
- correct: the step is logically valid and relevant.
- incorrect: the step contains a false claim, invalid operation, or moves toward a wrong answer.
- irrelevant: the step is mostly boilerplate or does not materially affect the solution.
- uncertain: cannot judge from the available context.

Return strict JSON only:
{{
  "final_answer_correct": true/false,
  "steps": [
    {{"step_number": 1, "label": "correct|incorrect|irrelevant|uncertain", "reason": "short reason"}}
  ]
}}

Original problem:
{first['question']}

Ground truth answer:
{first['ground_truth']}

Agent final answer:
{first.get('agent_answer')}

Agent reasoning steps:
{steps_text}
"""


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found")


def verify(args: argparse.Namespace) -> None:
    from openai import OpenAI

    exp = get_exp()
    out_dir = Path(args.out_dir)
    items_path = out_dir / "step_label_items.jsonl"
    labels_path = out_dir / "verifier_labels.jsonl"
    if not items_path.exists():
        raise FileNotFoundError(f"Missing {items_path}. Run prepare first.")
    if args.overwrite:
        labels_path.unlink(missing_ok=True)

    done = {row.get("label_cache_key") for row in iter_jsonl(labels_path)}
    groups = group_steps_for_verify(items_path, args.max_agents_per_question)
    if args.focus_items_file:
        focus_keys = focus_group_keys_from_items(Path(args.focus_items_file), args.changed_only)
        groups = {key: value for key, value in groups.items() if key in focus_keys}
        log(
            f"[VERIFY] focus_keys={len(focus_keys)}, matched_groups={len(groups)}, "
            f"changed_only={args.changed_only}"
        )
    keys = select_verify_keys(
        groups,
        done,
        args.verifier_model,
        args.max_chains,
        args.selection,
        args.seed,
    )

    api_url = args.api_url or exp.API_URL
    api_key = args.api_key or exp.API_KEY
    client = OpenAI(base_url=api_url, api_key=api_key)
    combo_counts = defaultdict(int)
    for model, task, _qid, _aid in keys:
        combo_counts[f"{model}/{task}"] += 1
    log(
        f"[VERIFY] pending_chains={len(keys)}, already_done={len(done)}, "
        f"selection={args.selection}, verifier_model={args.verifier_model}"
    )
    log(f"[VERIFY] selected combos: {json.dumps(dict(sorted(combo_counts.items())), ensure_ascii=False)}")

    for idx, key in enumerate(keys, start=1):
        group = groups[key]
        cache_key = label_cache_key(group, args.verifier_model)
        model, task, qid, aid = key
        log(f"[VERIFY] {idx}/{len(keys)} {model}/{task}/qid={qid}/agent={aid}/steps={len(group)}")
        prompt = verifier_prompt(group)
        last_error = None
        for attempt in range(1, args.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=args.verifier_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=args.max_tokens,
                )
                content = response.choices[0].message.content or ""
                parsed = extract_json_object(content)
                row = {
                    "label_cache_key": cache_key,
                    "verifier_model": args.verifier_model,
                    "source_model": model,
                    "task": task,
                    "question_id": qid,
                    "agent_id": aid,
                    "raw_response": content,
                    "parsed": parsed,
                }
                append_jsonl(labels_path, row)
                break
            except Exception as exc:
                last_error = exc
                log(f"  attempt {attempt} failed: {exc}")
                time.sleep(args.retry_delay)
        else:
            append_jsonl(
                labels_path,
                {
                    "label_cache_key": cache_key,
                    "verifier_model": args.verifier_model,
                    "source_model": model,
                    "task": task,
                    "question_id": qid,
                    "agent_id": aid,
                    "error": str(last_error),
                },
            )

    log(f"[DONE] labels written to {labels_path}")


def group_items_by_question(items_path: Path) -> Dict[Tuple[str, str, str], List[dict]]:
    groups = defaultdict(list)
    for row in iter_jsonl(items_path):
        key = (row["model"], row["task"], str(row["question_id"]))
        groups[key].append(row)
    for key in groups:
        groups[key].sort(
            key=lambda r: (int(r.get("agent_id") or 0), int(r.get("step_number") or 0))
        )
    return groups


def verifier_question_keys(labels_path: Path) -> set:
    keys = set()
    for row in iter_jsonl(labels_path):
        if row.get("error"):
            continue
        keys.add((row["source_model"], row["task"], str(row["question_id"])))
    return keys


def cluster(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    items_path = Path(args.items_file) if args.items_file else out_dir / "step_label_items.jsonl"
    labels_path = out_dir / "verifier_labels.jsonl"
    output_path = (
        Path(args.output_items)
        if args.output_items
        else out_dir / "step_label_items_clustered.jsonl"
    )
    if not items_path.exists():
        raise FileNotFoundError(f"Missing {items_path}. Run prepare first.")
    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
        else:
            raise FileExistsError(f"{output_path} exists. Use --overwrite to replace it.")

    log(f"[CLUSTER] loading items from {items_path}")
    groups = group_items_by_question(items_path)
    keys = sorted(groups.keys())
    if not args.all_questions:
        labeled_keys = verifier_question_keys(labels_path)
        keys = [key for key in keys if key in labeled_keys]
    if args.max_questions and args.max_questions > 0:
        keys = keys[: args.max_questions]

    log(f"[CLUSTER] questions={len(keys)}, output={output_path}")
    total_steps = 0
    total_changed = 0
    for idx, key in enumerate(keys, start=1):
        model, task, qid = key
        steps = groups[key]
        cfg = make_expand_cfg(model, task)
        log(f"[CLUSTER] {idx}/{len(keys)} {model}/{task}/qid={qid}/steps={len(steps)}")
        before = {
            (row.get("agent_id"), row.get("step_number")): bool(row.get("cluster_label"))
            for row in steps
        }
        try:
            clustered_steps = apply_cluster_labels(steps, cfg)
        except Exception as exc:
            log(f"  cluster failed: {exc}")
            clustered_steps = []
            for step in steps:
                failed_step = dict(step)
                failed_step["cluster_error"] = str(exc)
                clustered_steps.append(failed_step)
        for step in clustered_steps:
            after = bool(step.get("cluster_label"))
            total_changed += int(
                before.get((step.get("agent_id"), step.get("step_number"))) != after
            )
            append_jsonl(output_path, step)
            total_steps += 1
        log(f"  wrote {len(clustered_steps)} clustered steps")

    log(f"[DONE] clustered steps={total_steps}, changed={total_changed}")
    log(f"       {output_path}")


def verifier_binary(label: str) -> Optional[bool]:
    if label == "correct":
        return True
    if label == "incorrect":
        return False
    return None


def report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    items_path = Path(args.items_file) if args.items_file else out_dir / "step_label_items.jsonl"
    labels_path = out_dir / "verifier_labels.jsonl"
    report_path = Path(args.report_file) if args.report_file else out_dir / "diagnostic_report.json"
    log(f"[REPORT] loading items from {items_path}")
    items = {row["step_uid"]: row for row in iter_jsonl(items_path)}
    log(f"[REPORT] loaded {len(items)} step items")

    verifier_by_uid = {}
    label_rows = 0
    for row in iter_jsonl(labels_path):
        label_rows += 1
        if row.get("error"):
            continue
        parsed = row.get("parsed") or {}
        for step in parsed.get("steps", []):
            uid = f"{row['source_model']}|{row['task']}|{row['question_id']}|{row['agent_id']}|{step.get('step_number')}"
            verifier_by_uid[uid] = {
                "label": step.get("label"),
                "reason": step.get("reason"),
                "verifier_model": row.get("verifier_model"),
            }
    log(f"[REPORT] loaded {label_rows} verifier rows, {len(verifier_by_uid)} labeled steps")

    stats = defaultdict(lambda: defaultdict(int))
    examples = []
    for uid, vrow in verifier_by_uid.items():
        item = items.get(uid)
        if not item:
            continue
        vb = verifier_binary(vrow.get("label"))
        if vb is None:
            stats[(item["model"], item["task"])]["verifier_skipped"] += 1
            continue
        inherited = bool(item.get("inherited_label"))
        cluster = bool(item.get("cluster_label"))
        key = (item["model"], item["task"])
        stats[key]["n"] += 1
        stats[key]["inherited_match"] += int(inherited == vb)
        stats[key]["cluster_match"] += int(cluster == vb)
        stats[key]["cluster_changed"] += int(bool(item.get("cluster_label_changed")))
        if inherited != vb or cluster != vb:
            examples.append(
                {
                    "step_uid": uid,
                    "model": item["model"],
                    "task": item["task"],
                    "question_id": item["question_id"],
                    "agent_id": item["agent_id"],
                    "step_number": item["step_number"],
                    "verifier_label": vrow.get("label"),
                    "inherited_label": inherited,
                    "cluster_label": cluster,
                    "cluster_tag": item.get("cluster_tag"),
                    "content": item.get("content", "")[:500],
                    "reason": vrow.get("reason"),
                }
            )

    summary = {}
    for key, s in sorted(stats.items()):
        n = s["n"]
        summary_key = f"{key[0]}/{key[1]}"
        summary[summary_key] = {
            "n_labeled_steps": n,
            "inherited_match_rate": (s["inherited_match"] / n if n else None),
            "cluster_match_rate": (s["cluster_match"] / n if n else None),
            "cluster_changed_steps": s["cluster_changed"],
            "verifier_skipped": s["verifier_skipped"],
        }

    output = {
        "summary": summary,
        "mismatch_examples": examples[: args.max_examples],
    }
    dump_json(report_path, output)
    log(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    log(f"[DONE] report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--max-questions-per-combo", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--qid-file", default=None)
    p.add_argument("--with-cluster", action="store_true")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=prepare)

    p = sub.add_parser("verify")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--api-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--verifier-model", default="gpt-5-mini")
    p.add_argument("--max-agents-per-question", type=int, default=10)
    p.add_argument("--max-chains", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=3.0)
    p.add_argument("--selection", choices=["stratified", "sequential"], default="stratified")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--focus-items-file", default=None)
    p.add_argument("--changed-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=verify)

    p = sub.add_parser("cluster")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--items-file", default=None)
    p.add_argument("--output-items", default=None)
    p.add_argument("--max-questions", type=int, default=0)
    p.add_argument("--all-questions", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cluster)

    p = sub.add_parser("report")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--items-file", default=None)
    p.add_argument("--report-file", default=None)
    p.add_argument("--max-examples", type=int, default=50)
    p.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
