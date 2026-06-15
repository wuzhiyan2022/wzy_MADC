import argparse
import asyncio
import contextlib
import io
import json
import time

import wzy_multi_agent_debate_expand as exp
import wzy_multi_agent_debate_exchange_promptclean as ex


DEFAULT_QIDS = [
    "10",
    "61",
    "90",
    "127",
    "148",
    "293",
    "296",
    "356",
    "372",
    "467",
    "492",
    "499",
]


def log(message: str) -> None:
    print(message, flush=True)


async def run_one(item: dict, cfg, args) -> dict:
    qid = str(item.get("question_id"))
    log("")
    log("=" * 80)
    log(f"[QID {qid}] start")

    captured = io.StringIO()
    t0 = time.time()
    log(f"[QID {qid}] [1/4] loading expand cache")
    with contextlib.redirect_stdout(captured):
        pack = await exp.run_expand_pipeline_from_cache(cfg=cfg, question_item=item)
    log(f"[QID {qid}] [1/4] cache loaded in {time.time() - t0:.1f}s")

    if pack is None:
        log(f"[QID {qid}] [ERROR] expand cache miss; skip")
        return {"qid": qid, "error": "expand_cache_miss", "ok1": False, "ok2": False}

    step_vectors, step_indices, all_steps, agent_contexts, expand_majority, gt, question, question_id = pack
    expand_ok = exp.is_correct_answer(expand_majority, gt, is_math=cfg.is_math) if expand_majority else False
    log(f"[QID {qid}] [2/4] expand majority={expand_majority!r}, gt={gt!r}, ok={expand_ok}")
    log(f"[QID {qid}] [2/4] vectors={getattr(step_vectors, 'shape', None)}, steps={len(all_steps)}, agents={len(agent_contexts)}")

    log(f"[QID {qid}] [3/4] running promptclean exchange1")
    t1 = time.time()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        out1 = await ex.run_exchange1_from_expand_outputs(
            step_vectors,
            step_indices,
            all_steps,
            agent_contexts,
            cfg,
            use_method=args.cluster,
            round_num=1,
            reduction_method=args.reduction,
        )
    maj1 = out1.get("majority_answer")
    ok1 = exp.is_correct_answer(maj1, gt, is_math=cfg.is_math) if maj1 else False
    log(f"[QID {qid}] [4/4] exchange1 done in {time.time() - t1:.1f}s; majority={maj1!r}; ok={ok1}; gt={gt!r}")

    result = {"qid": qid, "gt": gt, "expand": expand_majority, "exchange1": maj1, "ok1": ok1}

    if args.exchange2:
        log(f"[QID {qid}] [extra] running promptclean exchange2")
        t2 = time.time()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            out2 = await ex.run_exchange2_from_exchange1_outputs(
                agent_contexts,
                cfg,
                use_method=args.cluster,
                round_num=2,
                reduction_method=args.reduction,
            )
        maj2 = out2.get("majority_answer")
        ok2 = exp.is_correct_answer(maj2, gt, is_math=cfg.is_math) if maj2 else False
        result.update({"exchange2": maj2, "ok2": ok2})
        log(f"[QID {qid}] [extra] exchange2 done in {time.time() - t2:.1f}s; majority={maj2!r}; ok={ok2}; gt={gt!r}")

    return result


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-turbo")
    parser.add_argument("--task", default="math_500_id")
    parser.add_argument("--qids", default=",".join(DEFAULT_QIDS))
    parser.add_argument("--reduction", default="pca")
    parser.add_argument("--cluster", default="hdbscan")
    parser.add_argument("--exchange2", action="store_true")
    args = parser.parse_args()

    qids = [x.strip() for x in args.qids.split(",") if x.strip()]

    exp.MODEL_NAME = args.model
    exp.MODEL_TAG = args.model
    ex.MODEL_TAG = args.model

    cfg = exp.default_expand_config()
    cfg.model_name = args.model
    cfg.model_tag = args.model
    cfg.task_name = args.task
    cfg.is_math = args.task == "math_500_id"
    cfg.fixed_question_id = None

    with open(f"{args.model}/data/{args.task}.json", encoding="utf-8") as f:
        all_items = json.load(f)["examples"]
    by_qid = {str(x.get("question_id")): x for x in all_items}
    items = [by_qid[qid] for qid in qids if qid in by_qid]

    log(f"[START] promptclean batch: model={args.model}, task={args.task}, qids={','.join(qids)}")
    log(f"[CONFIG] reduction={args.reduction}, cluster={args.cluster}, exchange2={args.exchange2}")

    results = []
    t0 = time.time()
    for idx, item in enumerate(items, start=1):
        log(f"[PROGRESS] {idx}/{len(items)}")
        results.append(await run_one(item, cfg, args))

    ok1 = sum(1 for row in results if row.get("ok1"))
    ok2 = sum(1 for row in results if row.get("ok2"))
    skipped = sum(1 for row in results if row.get("error"))

    log("")
    log("=" * 80)
    log("[SUMMARY]")
    log(f"questions attempted: {len(results)} / requested {len(qids)}")
    log(f"skipped: {skipped}")
    log(f"exchange1 correct: {ok1}/{len(results)}")
    if args.exchange2:
        log(f"exchange2 correct: {ok2}/{len(results)}")
    log("per-question:")
    for row in results:
        if row.get("error"):
            log(f"  qid={row['qid']}: ERROR {row['error']}")
            continue
        line = f"  qid={row['qid']}: exchange1={row.get('exchange1')!r}, ok1={row.get('ok1')}"
        if args.exchange2:
            line += f", exchange2={row.get('exchange2')!r}, ok2={row.get('ok2')}"
        log(line)
    log(f"elapsed: {time.time() - t0:.1f}s")
    log("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
