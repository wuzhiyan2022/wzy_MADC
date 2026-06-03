import json

path = (
    "qwen-turbo/results/debate_zy/math_500_id/"
    "debate_zy_qwen-turbo_10_1_expand_agent_com0_False.json"
)
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

first_q = list(data.keys())[0]
first_entry = data[first_q]
agent_contexts_0 = first_entry[0]
print(f"total questions: {len(data)}")
print(f"first question agent count: {len(agent_contexts_0)}")
print(f"first question agent[0] msg count: {len(agent_contexts_0[0])}")
print(f"first question agent[0] roles: {[m['role'] for m in agent_contexts_0[0]]}")

# find questions with agents missing the assistant reply (msg count < 3)
bad = []
for q, entry in data.items():
    agent_contexts = entry[0]
    qid = entry[2]
    for aidx, ctx in enumerate(agent_contexts):
        if len(ctx) < 3:
            bad.append((str(qid), aidx, len(ctx)))

if bad:
    print(f"\nproblematic agents (missing assistant reply): {len(bad)} total")
    shown = bad[:40]
    for qid, aidx, msglen in shown:
        print(f"  question_id={qid}, agent={aidx}, msg_count={msglen}")
    if len(bad) > 40:
        print(f"  ... {len(bad)} total")
    unique_qids = sorted(set(x[0] for x in bad), key=lambda x: int(x))
    print(f"\ndistinct question_ids affected ({len(unique_qids)}): {unique_qids}")
else:
    print("\nAll entries look structurally correct (each agent has >= 3 messages)")
    print("Checking if round index mismatch is caused by first-question length vs others...")
    first_len = len(agent_contexts_0[0])
    mismatch = []
    for q, entry in data.items():
        agent_contexts = entry[0]
        qid = entry[2]
        for aidx, ctx in enumerate(agent_contexts):
            if len(ctx) != first_len:
                mismatch.append((str(qid), aidx, len(ctx)))
    if mismatch:
        print(f"  length mismatch vs first question ({first_len}): {len(mismatch)} agents")
        for qid, aidx, msglen in mismatch[:20]:
            print(f"    question_id={qid}, agent={aidx}, msg_count={msglen}")
    else:
        print(f"  No length mismatch found. All agents have {first_len} messages.")
