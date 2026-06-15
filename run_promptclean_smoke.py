import argparse
import asyncio
import contextlib
import io
import json
import time

import wzy_multi_agent_debate_expand as exp
import wzy_multi_agent_debate_exchange_promptclean as ex


def log(message: str) -> None:
    print(message, flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-turbo")
    parser.add_argument("--task", default="math_500_id")
    parser.add_argument("--qid", default="10")
    parser.add_argument("--exchange2", action="store_true")
    args = parser.parse_args()

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
        items = json.load(f)["examples"]
    item = next(x for x in items if str(x.get("question_id")) == str(args.qid))

    log(f"[START] promptclean smoke: model={args.model}, task={args.task}, qid={args.qid}")
    log("[1/5] Loading expand cache...")
    t0 = time.time()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        pack = await exp.run_expand_pipeline_from_cache(cfg=cfg, question_item=item)
    log(f"[1/5] Expand cache load finished in {time.time() - t0:.1f}s")

    if pack is None:
        log("[ERROR] Expand cache miss. Stop here to avoid replaying expand API calls.")
        return

    step_vectors, step_indices, all_steps, agent_contexts, expand_majority, gt, question, question_id = pack
    expand_ok = exp.is_correct_answer(expand_majority, gt, is_math=cfg.is_math) if expand_majority else False
    log(f"[2/5] Expand majority={expand_majority!r}, gt={gt!r}, ok={expand_ok}")
    log(f"[2/5] step_vectors={getattr(step_vectors, 'shape', None)}, step_count={len(all_steps)}, agents={len(agent_contexts)}")

    log("[3/5] Running promptclean exchange1: reduction=pca, cluster=hdbscan")
    t1 = time.time()
    out1 = await ex.run_exchange1_from_expand_outputs(
        step_vectors,
        step_indices,
        all_steps,
        agent_contexts,
        cfg,
        use_method="hdbscan",
        round_num=1,
        reduction_method="pca",
    )
    maj1 = out1.get("majority_answer")
    ok1 = exp.is_correct_answer(maj1, gt, is_math=cfg.is_math) if maj1 else False
    log(f"[4/5] exchange1 done in {time.time() - t1:.1f}s; majority={maj1!r}; ok={ok1}; gt={gt!r}")

    if not args.exchange2:
        log("[5/5] Finished after exchange1.")
        return

    log("[5/5] Running promptclean exchange2...")
    t2 = time.time()
    out2 = await ex.run_exchange2_from_exchange1_outputs(
        agent_contexts,
        cfg,
        use_method="hdbscan",
        round_num=2,
        reduction_method="pca",
    )
    maj2 = out2.get("majority_answer")
    ok2 = exp.is_correct_answer(maj2, gt, is_math=cfg.is_math) if maj2 else False
    log(f"[5/5] exchange2 done in {time.time() - t2:.1f}s; majority={maj2!r}; ok={ok2}; gt={gt!r}")


if __name__ == "__main__":
    asyncio.run(main())
