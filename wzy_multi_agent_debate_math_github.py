"""
数学多 Agent 辩论编排入口：按动作列表依次执行（如 expand → exchange1）。

- expand：独立推理、多数票、step 标签、向量化（run_expand_pipeline）。
- exchange1 / exchange2 / exchange_bidirectional_*：见各分支说明。
- 支持单题或批量：批量时在每题每个 action 结束后更新累计正确率并打印。

expand 结束后会 deepcopy 一份 expand_pack 快照；exchange_bidirectional_1/2 只从快照解包，
可与 exchange1/2 同 pipeline 串行，便于在相同 expand 起点上对比两种 exchange。
"""

import os

# 必须在 numpy / sklearn / MKL 相关模块 import 之前设置，
# 防止 Windows + MKL 环境下 KMeans 小数据集时触发 OMP 线程数 > chunk 数的内存泄漏警告。
os.environ.setdefault("OMP_NUM_THREADS", "1")

import contextlib
import copy
import io
import json
import sys


def _init_stdio_utf8() -> None:
    """Windows 控制台默认 GBK，仅 reconfigure 仍可能乱码；先切 UTF-8 代码页再绑定 stdio。"""
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

import asyncio
from typing import Any, Dict, List, Optional

from wzy_multi_agent_debate_expand import (
    VERIFY_VECTORIZATION,
    SAVE_DEBUG_FILES,
    default_expand_config,
    expand_save_cache_if_enabled,
    get_cache_path,
    is_correct_answer,
    run_expand_pipeline,
    run_expand_pipeline_from_cache,
)
from wzy_multi_agent_debate_exchange import (
    run_exchange1_from_expand_outputs,
    run_exchange2_from_exchange1_outputs,
    run_exchange_bidirectional_1_from_expand_outputs,
    run_exchange_bidirectional_2_from_bidirectional_1_outputs,
)

# 与 wzy_multi_agent_debate_expand 模块默认一致；可在此覆盖字段
CONFIG = default_expand_config()

# 依次执行的动作名（示例）：
#   ["expand", "exchange1", "exchange2"]
#   ["expand", "exchange_bidirectional_1", "exchange_bidirectional_2"]
#   ["expand", "exchange1", "exchange2"]
#   亦可合并：bidirectional 使用 expand 快照，与 exchange1/2 互不污染 agent_contexts
ACTION_PIPELINE: List[str] = ["expand", "exchange1", "exchange2","exchange_bidirectional_1", "exchange_bidirectional_2"]

# ---------- 降维 + 聚类配置 ----------
# 推荐组合（两个开关独立，可任意组合）：
#   1) REDUCTION_METHOD="pca",  EXCHANGE1_CLUSTER_METHOD="kmeans"   → 基线（原行为，KMeans k=7）
#   2) REDUCTION_METHOD="umap", EXCHANGE1_CLUSTER_METHOD="hdbscan"  → 学术经典 UMAP+HDBSCAN（主推）
#   3) REDUCTION_METHOD="umap", EXCHANGE1_CLUSTER_METHOD="kmeans"   → UMAP+KMeans 实验组（k 自动用 4）
# 降维方法："pca" | "umap"
REDUCTION_METHOD: str = "pca"
# 聚类方法："kmeans" | "dbscan" | "hdbscan"
EXCHANGE1_CLUSTER_METHOD = "hdbscan"

# ---------- expand 缓存快速通道 ----------
# True（推荐）：执行 expand 前先尝试整题加载缓存的 agent_contexts；
#               命中 → 跳过推理 API；未命中 → 打印提示并正常调 API。
#               仍会调用 embedding API 重建向量（embedding 已有 hash 缓存，几乎零成本）。
# False：跳过快速通道，每题都直接走完整 expand_run_pipeline（用于强制重新生成回复）。
EXPAND_USE_CACHE: bool = True

# ---------- 批量配置 ----------
# True：按 BATCH_QUESTION_IDS / BATCH_MAX_QUESTIONS 从数据集取多题依次跑
# False：单题（沿用 CONFIG.fixed_question_id 为 None 时随机一题，否则固定 id）
RUN_BATCH: bool = True
# 非 None 时只跑这些 question_id（与数据集中类型一致即可，内部转 str 匹配）
BATCH_QUESTION_IDS: Optional[List[Any]] = None
# BATCH_QUESTION_IDS: Optional[List[Any]] = None
# 全表模式下最多跑多少题（None 表示不截断）；在 BATCH_QUESTION_IDS 为 None 时生效
BATCH_MAX_QUESTIONS: Optional[int] = None

# ---------- 日志详细度 ----------
# 0 = 静默（仅关键摘要 + 累计正确率），1 = 全量日志（调试单题）
# 批量时建议 0；单题调试时设 1
BATCH_VERBOSE: int = 0

# True：即使 BATCH_VERBOSE=0，仍在进入 exchange2 前打印 agent_contexts 摘要（便于批量时核对）
PRINT_DEBUG_AGENT_CONTEXTS_BEFORE_EXCHANGE2: bool = False

# ---------- 断点续跑配置 ----------
# True：启用断点续跑，从 checkpoint.json 读取已完成题目并跳过
# False：不启用，每次全量运行
ENABLE_CHECKPOINT: bool = True

import os as _os

class CheckpointManager:
    """管理断点续跑的 checkpoint 文件读写。"""

    def __init__(self, checkpoint_path: str):
        self.path = checkpoint_path
        self.data: Dict[str, bool] = self._load()

    def _load(self) -> Dict[str, bool]:
        if _os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def is_completed(self, question_id: str) -> bool:
        return bool(self.data.get(str(question_id), False))

    def mark_completed(self, question_id: str) -> None:
        qid = str(question_id)
        if not self.data.get(qid):
            self.data[qid] = True
            self.save()
            print(f"[断点续跑] question_id={qid} 已完成，已记录")

    def get_completed_count(self) -> int:
        return sum(1 for v in self.data.values() if v)


@contextlib.contextmanager
def _suppress_stdout():
    """临时将 sys.stdout 重定向到 devnull，屏蔽所有 print 输出。"""
    old = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def _quiet() -> bool:
    """当前是否处于静默模式。"""
    return BATCH_VERBOSE < 1


def _print_agent_contexts_before_exchange2(
    agent_contexts: List[Any],
    question_id: Any,
    *,
    max_preview_chars: int = 160,
) -> None:
    """进入 exchange2 前打印各 agent 对话结构，便于确认已含 exchange1 追加的 user + assistant。

    正常时：expand 后 3 条（system → user → assistant）；exchange1 后再 +2 条（user → assistant），
    共至少 5 条；角色链末尾应为 ``... assistant -> user -> assistant``。
    """
    print("\n" + "─" * 72)
    print(
        f"[调试·exchange2 入口] question_id={question_id}\n"
        f"  说明：expand_pack 中 agent_contexts 与 exchange1 共用同一可变列表，"
        f"此处应为 **已追加 exchange1** 后的内容。"
    )
    print("─" * 72)
    for aidx, ctx in enumerate(agent_contexts):
        if not ctx:
            print(f"\n  Agent {aidx}: （空列表）")
            continue
        roles = " → ".join(str(m.get("role", "?")) for m in ctx)
        n_msg = len(ctx)
        print(f"\n  ┌─ Agent {aidx}  消息数={n_msg}  角色链: {roles}")
        for mi, msg in enumerate(ctx):
            role = msg.get("role", "?")
            raw = msg.get("content") or ""
            vis = raw.replace("\n", " ↵ ")
            if len(vis) > max_preview_chars:
                vis = vis[:max_preview_chars] + f" …（共 {len(raw)} 字符，已截断）"
            print(f"  │  [{mi:>2}] {role:<12} {vis!r}")
        print(f"  └{'─' * 66}")
    print("─" * 72 + "\n")


_STAGES = (
    "expand",
    "exchange1",
    "exchange2",
    "exchange_bidirectional_1",
    "exchange_bidirectional_2",
)


def _init_stage_stats() -> Dict[str, Dict[str, Any]]:
    return {a: {"attempted": 0, "correct": 0, "correct_ids": []} for a in _STAGES}


def _bump_stage(
    stats: Dict[str, Dict[str, Any]],
    action_label: str,
    match_gt: bool,
    question_id: str,
) -> None:
    if action_label not in stats:
        return
    stats[action_label]["attempted"] += 1
    if match_gt:
        stats[action_label]["correct"] += 1
        ids: List[str] = stats[action_label]["correct_ids"]
        ids.append(str(question_id))


def _print_cumulative_stage(action_label: str, stats: Dict[str, Dict[str, Any]],
                             _p: Any = None) -> None:
    if _p is None:
        _p = print
    s = stats.get(action_label)
    if not s:
        return
    att, cor = s["attempted"], s["correct"]
    pct = (cor / att * 100.0) if att else 0.0
    _p(
        f"\n[正确率·{action_label}·累计] {cor}/{att} = {pct:.2f}% "
        f"（多数票与 GT 一致题数 / 已进入该阶段的题数）"
    )
    ids = s.get("correct_ids") or []
    if ids:
        _p(f"  正确题 question_id: {', '.join(ids)}")
    else:
        _p("  正确题 question_id: (无)")


def _print_batch_summary(stats: Dict[str, Dict[str, Any]]) -> None:
    _p = lambda msg: print(msg, file=sys.stderr, flush=True)
    _p("\n" + "=" * 72)
    _p("[汇总 · 各阶段累计正确率]")
    for a in ACTION_PIPELINE:
        if a in stats:
            _print_cumulative_stage(a, stats, _p)
    _p("=" * 72)


def _print_question_stage_results(summary: Dict[str, Any]) -> None:
    """流水线全部 action 跑完后，按 ACTION_PIPELINE 顺序打印各阶段与 GT 是否一致。"""
    _p = lambda msg: print(msg, file=sys.stderr, flush=True)
    qid = summary.get("question_id", "?")
    _p(f"\n[本题各阶段结果] id={qid}")
    for action in ACTION_PIPELINE:
        entry = summary.get(action)
        if entry is None:
            _p(f"  {action}: skipped")
            continue
        if isinstance(entry, dict) and "match_gt" in entry:
            label = "correct" if entry["match_gt"] else "incorrect"
            _p(f"  {action}: {label}")
        else:
            _p(f"  {action}: skipped")


def load_batch_question_items(cfg: Any) -> List[Dict[str, Any]]:
    """从 {model}/data/{task}.json 读取 examples，过滤方式与 expand_load_question 中 is_hard 一致。"""
    path = f"{cfg.model_name}/data/{cfg.task_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["examples"]

    if cfg.is_hard:
        hard_id = {str(i) for i in range(1, 101)}
        data = [d for d in data if str(d.get("question_id")) in hard_id]

    if BATCH_QUESTION_IDS is not None:
        want = {str(x) for x in BATCH_QUESTION_IDS}
        data = [d for d in data if str(d.get("question_id")) in want]

    if BATCH_MAX_QUESTIONS is not None:
        data = data[: int(BATCH_MAX_QUESTIONS)]

    return data


async def run_question_pipeline(
    question_item: Optional[Dict[str, Any]],
    stats: Dict[str, Dict[str, Any]],
    checkpoint: Optional["CheckpointManager"] = None,
) -> Optional[Dict[str, Any]]:
    """
    对单道题执行 ACTION_PIPELINE；更新 stats中各阶段 attempted/correct，并打印本题摘要与累计正确率。
    question_item 为 None 时与原先单题行为一致（随机或 fixed_question_id）。
    """
    summary: Dict[str, Any] = {
        "actions": list(ACTION_PIPELINE),
        "expand": None,
        "exchange1": None,
        "exchange2": None,
        "exchange_bidirectional_1": None,
        "exchange_bidirectional_2": None,
        "ground_truth": None,
        "question_id": str(question_item.get("question_id", "?")) if question_item else None,
    }

    if question_item is not None and question_item.get("question_id") is not None:
        run_qid = str(question_item["question_id"])
    else:
        run_qid = "?"

    expand_pack = None
    snapshot_expand_pack: Any = None

    for action in ACTION_PIPELINE:
        if action == "expand":
            if not _quiet():
                print("\n" + "#" * 72)
                print("# 动作: expand")
                print("#" * 72)

            expand_pack = None

            # 1) 缓存优先：开启时先尝试整题加载，命中则跳过推理 API
            if EXPAND_USE_CACHE:
                if _quiet():
                    with _suppress_stdout():
                        expand_pack = await run_expand_pipeline_from_cache(
                            cfg=CONFIG,
                            question_item=question_item,
                        )
                else:
                    expand_pack = await run_expand_pipeline_from_cache(
                        cfg=CONFIG,
                        question_item=question_item,
                    )
                if expand_pack is None:
                    miss_qid = (
                        str(question_item.get("question_id", "?"))
                        if question_item is not None
                        else (
                            str(CONFIG.fixed_question_id)
                            if CONFIG.fixed_question_id is not None
                            else "?"
                        )
                    )
                    print(
                        f"[expand 缓存] 未命中：question_id={miss_qid}（缓存缺失或不完整），"
                        f"将正常调用推理 API"
                    )
                elif not _quiet():
                    print(f"[expand 缓存] 命中后直接进入下游 exchange，跳过推理 API 调用")

            # 2) 缓存未命中（或 EXPAND_USE_CACHE=False）→ 走完整 expand pipeline 调 API
            if expand_pack is None:
                if _quiet():
                    with _suppress_stdout():
                        expand_pack = await run_expand_pipeline(
                            cfg=CONFIG,
                            question_item=question_item,
                            save_cache=True,
                            return_vectorization_data=True,
                            verify_vectorization=VERIFY_VECTORIZATION,
                            save_debug_files=SAVE_DEBUG_FILES,
                        )
                else:
                    expand_pack = await run_expand_pipeline(
                        cfg=CONFIG,
                        question_item=question_item,
                        save_cache=True,
                        return_vectorization_data=True,
                        verify_vectorization=VERIFY_VECTORIZATION,
                        save_debug_files=SAVE_DEBUG_FILES,
                    )

            if expand_pack is None:
                print("[错误] expand 未返回数据，本题后续动作跳过")
                _bump_stage(stats, "expand", False, run_qid)
                _print_cumulative_stage("expand", stats)
                summary["error"] = "expand_failed"
                return summary

            (
                _sv,
                _si,
                _steps,
                _ctx,
                expand_majority,
                ground_truth,
                _question,
                _question_id,
            ) = expand_pack
            snapshot_expand_pack = copy.deepcopy(expand_pack)
            run_qid = str(_question_id)
            summary["ground_truth"] = ground_truth
            summary["question_id"] = run_qid
            match_gt = (
                is_correct_answer(expand_majority, ground_truth, is_math=CONFIG.is_math)
                if expand_majority and ground_truth
                else False
            )
            summary["expand"] = {
                "majority_answer": expand_majority,
                "match_gt": match_gt,
            }
            print(
                f"\n[摘要·expand] id={_question_id} majority={expand_majority!r}  "
                f"与 GT 一致: {match_gt}  GT={ground_truth!r}"
            )
            _bump_stage(stats, "expand", match_gt, run_qid)
            _print_cumulative_stage("expand", stats)

        elif action == "exchange1":
            if expand_pack is None:
                print("[错误] exchange1 需要先有 expand 结果，本题跳过")
                return summary
            step_vectors, step_indices, all_steps, agent_contexts, _, ground_truth, question, question_id = expand_pack
            if step_vectors is None or len(step_vectors) < 2:
                n = step_vectors.shape[0] if step_vectors is not None else 0
                print(f"[错误] exchange1 需要至少 2 个 step 向量，当前 {n}，本题跳过 exchange")
                summary["error"] = "exchange1_skip_few_steps"
                return summary

            if not _quiet():
                print("\n" + "#" * 72)
                print("# 动作: exchange1（聚类 + 排序 + exchange prompt + 新回复）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    out = await run_exchange1_from_expand_outputs(
                        step_vectors,
                        step_indices,
                        all_steps,
                        agent_contexts,
                        use_method=EXCHANGE1_CLUSTER_METHOD,
                        round_num=1,
                        reduction_method=REDUCTION_METHOD,
                    )
            else:
                out = await run_exchange1_from_expand_outputs(
                    step_vectors,
                    step_indices,
                    all_steps,
                    agent_contexts,
                    use_method=EXCHANGE1_CLUSTER_METHOD,
                    round_num=1,
                    reduction_method=REDUCTION_METHOD,
                )
            exchange1_maj = out.get("majority_answer")
            match_gt = (
                is_correct_answer(exchange1_maj, ground_truth, is_math=CONFIG.is_math)
                if exchange1_maj and ground_truth
                else False
            )
            summary["exchange1"] = {
                "majority_answer": exchange1_maj,
                "match_gt": match_gt,
            }
            print(
                f"\n[摘要·exchange1] id={question_id} majority={exchange1_maj!r}  "
                f"与 GT 一致: {match_gt}  GT={ground_truth!r}"
            )
            _bump_stage(stats, "exchange1", match_gt, str(question_id))
            _print_cumulative_stage("exchange1", stats)

            exchange1_cache_path = get_cache_path(CONFIG, "exchange1")
            if _quiet():
                with _suppress_stdout():
                    expand_save_cache_if_enabled(
                        save_cache=True,
                        question=question,
                        agent_contexts=agent_contexts,
                        ground_truth=ground_truth,
                        question_id=question_id,
                        cache_path=exchange1_cache_path,
                        cfg=CONFIG,
                    )
            else:
                expand_save_cache_if_enabled(
                    save_cache=True,
                    question=question,
                    agent_contexts=agent_contexts,
                    ground_truth=ground_truth,
                    question_id=question_id,
                    cache_path=exchange1_cache_path,
                    cfg=CONFIG,
                )

        elif action == "exchange2":
            if summary["exchange1"] is None:
                print("[错误] exchange2 依赖 exchange1，本题跳过")
                return summary
            if expand_pack is None:
                print("[错误] 缺少 expand 数据包，本题跳过 exchange2")
                return summary
            _sv2, _si2, _st2, agent_contexts, _, ground_truth, question, question_id = expand_pack

            if not _quiet() or PRINT_DEBUG_AGENT_CONTEXTS_BEFORE_EXCHANGE2:
                _print_agent_contexts_before_exchange2(agent_contexts, question_id)

            if not _quiet():
                print("\n" + "#" * 72)
                print("# 动作: exchange2（基于 exchange1 最新回复重建向量 + 聚类 exchange）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    out2 = await run_exchange2_from_exchange1_outputs(
                        agent_contexts,
                        CONFIG,
                        use_method=EXCHANGE1_CLUSTER_METHOD,
                        round_num=2,
                        reduction_method=REDUCTION_METHOD,
                    )
            else:
                out2 = await run_exchange2_from_exchange1_outputs(
                    agent_contexts,
                    CONFIG,
                    use_method=EXCHANGE1_CLUSTER_METHOD,
                    round_num=2,
                    reduction_method=REDUCTION_METHOD,
                )
            exchange2_maj = out2.get("majority_answer")
            match_gt = (
                is_correct_answer(exchange2_maj, ground_truth, is_math=CONFIG.is_math)
                if exchange2_maj and ground_truth
                else False
            )
            summary["exchange2"] = {
                "majority_answer": exchange2_maj,
                "match_gt": match_gt,
            }
            print(
                f"\n[摘要·exchange2] id={question_id} majority={exchange2_maj!r}  "
                f"与 GT 一致: {match_gt}  GT={ground_truth!r}"
            )
            _bump_stage(stats, "exchange2", match_gt, str(question_id))
            _print_cumulative_stage("exchange2", stats)

            exchange2_cache_path = get_cache_path(CONFIG, "exchange2")
            if _quiet():
                with _suppress_stdout():
                    expand_save_cache_if_enabled(
                        save_cache=True,
                        question=question,
                        agent_contexts=agent_contexts,
                        ground_truth=ground_truth,
                        question_id=question_id,
                        cache_path=exchange2_cache_path,
                        cfg=CONFIG,
                    )
            else:
                expand_save_cache_if_enabled(
                    save_cache=True,
                    question=question,
                    agent_contexts=agent_contexts,
                    ground_truth=ground_truth,
                    question_id=question_id,
                    cache_path=exchange2_cache_path,
                    cfg=CONFIG,
                )

        elif action == "exchange_bidirectional_1":
            if snapshot_expand_pack is None:
                print("[错误] exchange_bidirectional_1 需要先有 expand 快照，本题跳过")
                return summary
            step_vectors, step_indices, all_steps, agent_contexts, _, ground_truth, question, question_id = (
                snapshot_expand_pack
            )
            if step_vectors is None or len(step_vectors) < 2:
                n = step_vectors.shape[0] if step_vectors is not None else 0
                print(f"[错误] exchange_bidirectional_1 需要至少 2 个 step 向量，当前 {n}，本题跳过")
                summary["error"] = "exchange_bidirectional_1_skip_few_steps"
                return summary

            if not _quiet():
                print("\n" + "#" * 72)
                print("# 动作: exchange_bidirectional_1（双向标签修正 + 聚类 exchange）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    out_bd1 = await run_exchange_bidirectional_1_from_expand_outputs(
                        step_vectors,
                        step_indices,
                        all_steps,
                        agent_contexts,
                        use_method=EXCHANGE1_CLUSTER_METHOD,
                        round_num=1,
                        reduction_method=REDUCTION_METHOD,
                    )
            else:
                out_bd1 = await run_exchange_bidirectional_1_from_expand_outputs(
                    step_vectors,
                    step_indices,
                    all_steps,
                    agent_contexts,
                    use_method=EXCHANGE1_CLUSTER_METHOD,
                    round_num=1,
                    reduction_method=REDUCTION_METHOD,
                )
            bd1_maj = out_bd1.get("majority_answer")
            match_gt = (
                is_correct_answer(bd1_maj, ground_truth, is_math=CONFIG.is_math)
                if bd1_maj and ground_truth
                else False
            )
            summary["exchange_bidirectional_1"] = {
                "majority_answer": bd1_maj,
                "match_gt": match_gt,
            }
            print(
                f"\n[摘要·exchange_bidirectional_1] id={question_id} majority={bd1_maj!r}  "
                f"与 GT 一致: {match_gt}  GT={ground_truth!r}"
            )
            _bump_stage(stats, "exchange_bidirectional_1", match_gt, str(question_id))
            _print_cumulative_stage("exchange_bidirectional_1", stats)

            bd1_cache_path = get_cache_path(CONFIG, "exchange_bidirectional_1")
            if _quiet():
                with _suppress_stdout():
                    expand_save_cache_if_enabled(
                        save_cache=True,
                        question=question,
                        agent_contexts=agent_contexts,
                        ground_truth=ground_truth,
                        question_id=question_id,
                        cache_path=bd1_cache_path,
                        cfg=CONFIG,
                    )
            else:
                expand_save_cache_if_enabled(
                    save_cache=True,
                    question=question,
                    agent_contexts=agent_contexts,
                    ground_truth=ground_truth,
                    question_id=question_id,
                    cache_path=bd1_cache_path,
                    cfg=CONFIG,
                )

        elif action == "exchange_bidirectional_2":
            if summary["exchange_bidirectional_1"] is None:
                print("[错误] exchange_bidirectional_2 依赖 exchange_bidirectional_1，本题跳过")
                return summary
            if snapshot_expand_pack is None:
                print("[错误] 缺少 expand 快照，本题跳过 exchange_bidirectional_2")
                return summary
            _svb, _sib, _stb, agent_contexts, _, ground_truth, question, question_id = snapshot_expand_pack

            if not _quiet():
                print("\n" + "#" * 72)
                print("# 动作: exchange_bidirectional_2（双向标签修正，基于 bidirectional_1 后回复）")
                print("#" * 72)

            if _quiet():
                with _suppress_stdout():
                    out_bd2 = await run_exchange_bidirectional_2_from_bidirectional_1_outputs(
                        agent_contexts,
                        CONFIG,
                        use_method=EXCHANGE1_CLUSTER_METHOD,
                        round_num=2,
                        reduction_method=REDUCTION_METHOD,
                    )
            else:
                out_bd2 = await run_exchange_bidirectional_2_from_bidirectional_1_outputs(
                    agent_contexts,
                    CONFIG,
                    use_method=EXCHANGE1_CLUSTER_METHOD,
                    round_num=2,
                    reduction_method=REDUCTION_METHOD,
                )
            bd2_maj = out_bd2.get("majority_answer")
            match_gt = (
                is_correct_answer(bd2_maj, ground_truth, is_math=CONFIG.is_math)
                if bd2_maj and ground_truth
                else False
            )
            summary["exchange_bidirectional_2"] = {
                "majority_answer": bd2_maj,
                "match_gt": match_gt,
            }
            print(
                f"\n[摘要·exchange_bidirectional_2] id={question_id} majority={bd2_maj!r}  "
                f"与 GT 一致: {match_gt}  GT={ground_truth!r}"
            )
            _bump_stage(stats, "exchange_bidirectional_2", match_gt, str(question_id))
            _print_cumulative_stage("exchange_bidirectional_2", stats)

            bd2_cache_path = get_cache_path(CONFIG, "exchange_bidirectional_2")
            if _quiet():
                with _suppress_stdout():
                    expand_save_cache_if_enabled(
                        save_cache=True,
                        question=question,
                        agent_contexts=agent_contexts,
                        ground_truth=ground_truth,
                        question_id=question_id,
                        cache_path=bd2_cache_path,
                        cfg=CONFIG,
                    )
            else:
                expand_save_cache_if_enabled(
                    save_cache=True,
                    question=question,
                    agent_contexts=agent_contexts,
                    ground_truth=ground_truth,
                    question_id=question_id,
                    cache_path=bd2_cache_path,
                    cfg=CONFIG,
                )
        else:
            print(f"[警告] 未知动作 {action!r}，已跳过")

    _print_question_stage_results(summary)
    _p = lambda msg: print(msg, file=sys.stderr, flush=True)
    if _quiet():
        _p(f"[本题流水线结束] id={summary.get('question_id')}")
    else:
        _p("\n" + "=" * 72)
        _p(f"[本题流水线结束] {summary}")
        _p("=" * 72)

    if checkpoint is not None and question_item is not None:
        qid = str(question_item.get("question_id", "?"))
        checkpoint.mark_completed(qid)

    return summary


async def main_math() -> Optional[Dict[str, Any]]:
    """单题或批量：返回 summary 或包含 batch 列表与 stats 的字典。"""
    stats = _init_stage_stats()

    if RUN_BATCH:
        items = load_batch_question_items(CONFIG)
        if not items:
            print("[错误] 批量列表为空，请检查 BATCH_QUESTION_IDS / 数据路径 / is_hard 过滤")
            return None

        checkpoint: Optional[CheckpointManager] = None
        if ENABLE_CHECKPOINT: # 断点重跑机制
            ck_path = f"{CONFIG.model_name}/checkpoint.json"
            checkpoint = CheckpointManager(ck_path)
            total_raw = len(items)
            items = [item for item in items
                     if not checkpoint.is_completed(str(item.get("question_id", "?")))]
            skipped = total_raw - len(items)

        if not items:
            print("[断点续跑] 所有题目已完成，无需运行", file=sys.stderr, flush=True)
            return None

        print(f"\n[批量] 共 {len(items)} 道题，ACTION_PIPELINE={ACTION_PIPELINE}", file=sys.stderr, flush=True)
        summaries: List[Optional[Dict[str, Any]]] = []
        for idx, item in enumerate(items):
            qid = item.get("question_id", "?")
            print(f"\n{'#' * 72}\n# 批量进度 {idx + 1}/{len(items)}  question_id={qid}\n{'#' * 72}", file=sys.stderr, flush=True)
            summ = await run_question_pipeline(item, stats, checkpoint=checkpoint)
            summaries.append(summ)
        _print_batch_summary(stats)
        return {
            "batch": True,
            "n_questions": len(items),
            "stats": stats,
            "summaries": summaries,
        }

    summ = await run_question_pipeline(None, stats)
    _print_batch_summary(stats)
    return summ


if __name__ == "__main__":
    result = asyncio.run(main_math())
