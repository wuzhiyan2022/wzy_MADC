"""Inspect debate_zy expand JSON for eval_all_round compatibility."""
import json
from pathlib import Path

path = Path(
    "qwen2.5-7b-instruct/results/debate_zy/math_500_id/"
    "debate_zy_qwen2.5-7b-instruct_10_1_expand_agent_com0_False.json"
)

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print(f"total top-level keys: {len(data)}")

# question_id list
qids = []
for k, v in data.items():
    if isinstance(v, list) and len(v) >= 3:
        qids.append(int(v[2]))
    else:
        qids.append(None)
qids_valid = sorted(x for x in qids if x is not None)
print(f"question_id count (from value[2]): {len(qids_valid)}")
print(f"question_id range: {min(qids_valid)} ~ {max(qids_valid)}")
missing = [i for i in range(1, 501) if i not in qids_valid]
print(f"missing from 1-500: {len(missing)} ids")
if missing:
    print(f"  first 30 missing: {missing[:30]}")
    if len(missing) > 30:
        print(f"  ... ({len(missing)} total)")

# First question structure (what eval uses for round upper bound)
first_key = list(data.keys())[0]
first_entry = data[first_key]
agent_contexts = first_entry[0]
print(f"\nfirst question key preview: {first_key[:60]}...")
print(f"  question_id: {first_entry[2]}")
print(f"  num agents: {len(agent_contexts)}")
print(f"  agent[0] msg count: {len(agent_contexts[0])}")
print(f"  agent[0] roles: {[m.get('role') for m in agent_contexts[0]]}")

# eval_bbh round logic: skip round%2==1 and round==0, so only round=2 if len=3
max_round_from_first = len(agent_contexts[0])
print(f"\neval_bbh will use round range: 0..{max_round_from_first-1}")
print(f"  rounds actually evaluated (skip odd, skip 0): {[r for r in range(max_round_from_first) if r%2==0 and r!=0]}")

# Find agents missing assistant (len < 3) or len mismatch vs first question
bad_assistant = []
len_mismatch = []
ref_len = len(agent_contexts[0])
for k, entry in data.items():
    if not isinstance(entry, list) or len(entry) < 3:
        continue
    qid = entry[2]
    for aidx, ctx in enumerate(entry[0]):
        if len(ctx) < 3:
            bad_assistant.append((qid, aidx, len(ctx), [m.get("role") for m in ctx]))
        elif len(ctx) != ref_len:
            len_mismatch.append((qid, aidx, len(ctx), ref_len))

print(f"\nagents with msg count < 3 (would fail at round=2): {len(bad_assistant)}")
for row in bad_assistant[:25]:
    print(f"  qid={row[0]}, agent={row[1]}, len={row[2]}, roles={row[3]}")
if len(bad_assistant) > 25:
    print(f"  ... {len(bad_assistant)} total")

print(f"\nagents with len != first question ({ref_len}): {len(len_mismatch)}")
for row in len_mismatch[:15]:
    print(f"  qid={row[0]}, agent={row[1]}, len={row[2]} (expected {row[3]})")
if len(len_mismatch) > 15:
    print(f"  ... {len(len_mismatch)} total")

# Per-question min/max agent msg lengths
per_q_issue = []
for k, entry in data.items():
    if not isinstance(entry, list) or len(entry) < 3:
        per_q_issue.append((entry[2] if len(entry)>=3 else "?", "bad_entry"))
        continue
    qid = entry[2]
    lens = [len(c) for c in entry[0]]
    if min(lens) != max(lens) or min(lens) < 3:
        per_q_issue.append((qid, lens))

print(f"\nquestions with uneven or short agent contexts: {len(per_q_issue)}")
for row in per_q_issue[:20]:
    print(f"  qid={row[0]}: {row[1]}")
if len(per_q_issue) > 20:
    print(f"  ... {len(per_q_issue)} total")
