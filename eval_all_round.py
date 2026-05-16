import json
import os
import openai
import numpy as np
import time
import re
from openai import OpenAI, AsyncOpenAI
from common.utils import read_txt, read_json
from common.math_equivalence import strip_string

# API configuration - please set your own API endpoint and key
API_URL = "https://api.zhizengzeng.com/v1"
API_KEY = "sk-zk2825bae2adf40f5eb42183b44b3e0630e69c2098d7527d"
MODEL_NAME = "glm-4-flashx"
MODEL_TAG = "glm-4-flashx"

# eval_bbh：每个阶段是否打印每道题的 majority_answer（与 compute_accuracy 返回的 pred_answer 一致）
PRINT_MAJORITY_PER_QUESTION = True
# 是否在 majority_answer 下再打印各 agent 的提取答案（与 pred_solutions 顺序一致，题多时会很长）
PRINT_PER_AGENT_ANSWERS = True

client = OpenAI(base_url=API_URL, api_key=API_KEY)
async_client = AsyncOpenAI(base_url=API_URL, api_key=API_KEY)

class Meta:
    def __init__(self, question, gt, rounds):
        self.question = question
        self.gt = gt
        self.rounds = rounds

    def __str__(self):
        return f"gt: {self.gt}\nrounds: {self.rounds}"

    def __repr__(self):
        return f"gt: {self.gt}\nrounds: {self.rounds}"

class Round:
    def __init__(self, pred_answers, most_answer, is_correct):
        self.pred_answers = pred_answers
        self.most_answer = most_answer
        self.is_correct = is_correct

    def __str__(self):
        return f"pre: {self.pred_answers}: {self.most_answer}: {self.is_correct}"

    def __repr__(self):
        return f"pre: {self.pred_answers}: {self.most_answer}: {self.is_correct}"

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets

def parse_yes_no(string):
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None

# 用于从模型输出的文本中提取数学题的数字答案。
# 用正则 r'\(([-]?\d+)\)' 找出文本中所有形如 (-?\d+) 的数字
# 其中 [-]? 表示数字前可能有负号
# \d+ 表示数字
# 用 re.findall 找出所有匹配的数字
# 若没有匹配则返回 None
# 返回最后一个匹配的数字
# 若没有匹配则返回 None
def solve_math_problems(input_str):
    pattern = r'\(([-]?\d+)\)'

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_YN(input_str):
    pattern = r'(\(Yes\)|\(No\))'
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is not None:
        solution = solution.replace("(", "")
        solution = solution.replace(")", "")
        if solution == "YES":
            solution = "Yes"
        if solution == "NO":
            solution = "No"

    return solution

# parse_answer 用于从模型生成的文本回答中提取选择题答案选项
def parse_answer(input_str):
    # 用正则 r'(\([A-Z]\))' 找出文本中所有形如 (X) 的选项
    # 其中 [A-Z] 表示大写字母 A 到 Z
    pattern = r'(\([A-Z]\))'
    matches = re.findall(pattern, input_str)
    # 若没有匹配则返回 None
    solution = None
    # 从后往前遍历匹配结果（matches[::-1]），取最后出现的那个选项作为最终答案
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    return solution

def parse_math_anser(input_str):
    # 正则匹配 \boxed{...}，可以正确匹配如下的内容
    # \boxed{42} → 42
    # \boxed{\frac{1}{2}} → \frac{1}{2}
    # \boxed{x^{2} + y^{2}} → x^{2} + y^{2}
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"

    matches = re.findall(pattern, input_str)

    solution = None
    #从后往前取最后一个有效答案，如果答案为空，则返回 None
    for match_str in matches[::-1]:
        solution = match_str
        if solution == "":
            return None
        if solution[-1] == ".":
            solution = solution[:-1]
        if solution[0] == "$" and solution[-1] == "$":
            solution = solution[1:-1]
        solution = strip_string(solution).strip()
        solution = solution.replace("\(", "")
        solution = solution.replace("\)", "")

        if solution:
            break

    return solution

def parse_answer_fallback(input_str: str):
    """
    兜底答案提取：覆盖 \boxed 和括号整数均提取失败的情况。

    按优先级依次尝试三种格式，取最后一处匹配并经 strip_string 规范化后返回：
      1. "The answer is: \(\frac{2}{27}\)"  — LaTeX 行内公式 \(...\)
      2. "The answer is: $\frac{16}{27}$"  — $ ... $ 行内公式
      3. "The answer is: 1"                — 纯数字 / 简单分数（如 -3/4）

    中英文冒号（：/:）均可识别。
    """
    if not input_str:
        return None

    # 1. \( ... \) 格式
    m = re.findall(
        r'[Tt]he\s+answer\s+is\s*[：:]\s*\\\((.+?)\\\)',
        input_str,
    )
    if m:
        ans = strip_string(m[-1].strip())
        if ans:
            return ans

    # 2. $ ... $ 格式
    m = re.findall(
        r'[Tt]he\s+answer\s+is\s*[：:]\s*\$(.+?)\$',
        input_str,
    )
    if m:
        ans = strip_string(m[-1].strip())
        if ans:
            return ans

    # 3. 纯数字 / 简单分数（-?\d+ 或 -?\d+/\d+）
    m = re.findall(
        r'[Tt]he\s+answer\s+is\s*[：:]\s*([-]?\d+(?:[./]\d+)?)',
        input_str,
    )
    if m:
        ans = strip_string(m[-1].strip())
        if ans:
            return ans

    return None


def _extract_math_answer(pred_solution: str):
    """
    数学题三级答案提取：
      1. parse_math_anser       → \\boxed{...}（最显式）
      2. parse_answer_fallback  → "The answer is: ..."（模型明确陈述答案）
      3. solve_math_problems    → 括号整数 (-?\\d+)（纯格式猜测，兜底）
    返回经 strip_string 规范化的字符串，或 None。
    三级均失败时打印回复预览，便于人工排查。
    """
    ans = parse_math_anser(pred_solution)
    if ans is not None:
        return strip_string(ans)
    ans = parse_answer_fallback(pred_solution)
    if ans is not None:
        return ans
    ans = solve_math_problems(pred_solution)
    if ans is not None:
        return ans
    return None


def compute_accuracy(gt, pred_solutions, log=False, idx=0, is_math=False):
    if is_math:
        if type(pred_solutions) == list:
            pred_answers = []

            for pred_solution in pred_solutions:
                pred_answer = _extract_math_answer(pred_solution)
                if pred_answer is not None:
                    pred_answers.append(pred_answer)
            if pred_answers == []:
                return 0, None
            pred_answer = most_frequent(pred_answers)
        else:
            pred_answer = _extract_math_answer(pred_solutions)
        equal_res = "no"
        if strip_string(gt) == pred_answer:
            equal_res = "yes"
        if equal_res.lower() == "yes":
            return 1, pred_answer
        else:
            return 0, pred_answer
    else:
        if type(pred_solutions) == list:
            pred_answers = []

            for pred_solution in pred_solutions:
                pred_answer = parse_answer(pred_solution)
                if pred_answer is None:
                    pred_answer = solve_math_problems(pred_solution)
                if pred_answer is None:
                    pred_answer = parse_YN(pred_solution)
                if pred_answer is not None:
                    pred_answers.append(pred_answer)
            if pred_answers == []:
                return 0, None
            pred_answer = most_frequent(pred_answers)
        else:
            pred_answer = parse_answer(pred_solutions)
            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solutions)
        if gt == pred_answer:
            return 1, pred_answer
        else:
            return 0, pred_answer


def extract_agent_answers_from_solutions(pred_solutions: list, is_math: bool) -> list:
    """
    与 compute_accuracy 中对每条 pred_solution 的单 agent 提取规则一致，
    返回与 pred_solutions 等长的列表，解析失败对应位置为 None。
    """
    out = []
    for pred_solution in pred_solutions:
        if is_math:
            out.append(_extract_math_answer(pred_solution))
        else:
            pred_answer = parse_answer(pred_solution)
            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solution)
            if pred_answer is None:
                pred_answer = parse_YN(pred_solution)
            out.append(pred_answer)
    return out


def compare_equal(str1, str2):
    fewshot_ost_prompt = read_txt("prompt/math_equal_prompt.txt")
    fewshot_content = fewshot_ost_prompt.format(
        expression1=str1,
        expression2=str2,
    )

    try:
        response = client.chat.completions.create(model=MODEL_TAG, messages=[{"role": "user", "content": fewshot_content}], max_tokens=4096, n=1)
        completion = json.loads(response.json())
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        print("retrying due to an error......")
        print(e)
        time.sleep(20)
        return compare_equal(str1, str2)["choices"][0]["message"]["content"]

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

def eval_bbh(file_name, is_math=False):
    response_dict = json.load(open(result_path + "/" + file_name + ".json", "r", encoding="utf-8"))
    questions = list(response_dict.keys())
    total = len(questions)
    stage_idx = 0
    stage_names = ["Expand", "Exchange1", "Exchange2", "Exchange3", "Exchange4", "Exchange5"]
    stage_summaries = []
    for round in range(len(response_dict[questions[0]][0][0])):
        if round % 2 == 1:
            continue
        if round == 0:
            continue
        correct_cnt = 0
        correct_question_ids = []
        unparseable = []
        wrong_questions = []
        per_question_majority = []
        idx = 0
        for question in questions:
            responses, gt, question_idx = response_dict[question]
            pred_solutions = []
            for response in responses:
                pred_solution = response[round]['content']
                pred_solutions.append(pred_solution)
            accurate, pred_answer = compute_accuracy(gt, pred_solutions, idx == 47 and round == 4, is_math=is_math)
            agent_extracted = extract_agent_answers_from_solutions(pred_solutions, is_math)
            per_question_majority.append({
                "question_id": question_idx,
                "majority_answer": pred_answer,
                "agent_answers": agent_extracted,
            })
            if accurate is not None:
                correct_cnt += int(accurate)
                if int(accurate) == 1:
                    correct_question_ids.append(question_idx)
                if int(accurate) == 0:
                    wrong_questions.append({
                        "question_id": question_idx,
                        "gt": gt,
                        "pred_answer": pred_answer,
                    })
                if pred_answer is None:
                    unparseable.append({
                        "question_id": question_idx,
                        "gt": gt,
                    })
            else:
                print(f"Warning: Failed to compute accuracy for question {question_idx}")
                print(f"Ground truth: {gt}")
            idx += 1
        name = stage_names[stage_idx] if stage_idx < len(stage_names) else f"Exchange{stage_idx}"
        pct = correct_cnt / total * 100
        print(f"  {name:10s} 阶段正确率 : {correct_cnt}/{total}  ({pct:.1f}%)")
        if PRINT_MAJORITY_PER_QUESTION and per_question_majority:
            print(f"    -> 各题 majority_answer（本阶段多数票解析结果，共 {len(per_question_majority)} 道）:")
            for row in per_question_majority:
                qid = row["question_id"]
                maj = row["majority_answer"]
                maj_disp = "(无法解析)" if maj is None else repr(maj)
                print(f"       question_id={qid}  majority_answer={maj_disp}")
                if PRINT_PER_AGENT_ANSWERS:
                    agents = row.get("agent_answers") or []
                    for ai, ans in enumerate(agents):
                        ad = "(无法解析)" if ans is None else repr(ans)
                        print(f"         agent {ai}: {ad}")
        if wrong_questions:
            print(f"    -> 错题（共 {len(wrong_questions)} 道）question_id / GT / 多数票预测:")
            for w in wrong_questions:
                qid = w.get("question_id")
                gt_s = w.get("gt")
                pred = w.get("pred_answer")
                gt_short = gt_s if gt_s is None or len(str(gt_s)) <= 60 else str(gt_s)[:57] + "..."
                pred_s = "(无法解析多数答案)" if pred is None else repr(pred)
                print(f"       question_id={qid}  gt={gt_short!r}  pred={pred_s}")
        if unparseable:
            print(
                f"    -> 上述错题中，{len(unparseable)} 道因各 agent 均无有效解析答案导致多数票为空"
            )
        stage_summaries.append(
            {
                "name": name,
                "correct_cnt": correct_cnt,
                "total": total,
                "correct_question_ids": list(correct_question_ids),
            }
        )
        stage_idx += 1

    print(f"\n{'-'*60}")
    print("  【汇总】各阶段正确率与正确题目的 question_id")
    print(f"{'-'*60}")
    for s in stage_summaries:
        t = s["total"]
        c = s["correct_cnt"]
        pct = (c / t * 100) if t else 0.0
        print(f"  {s['name']:10s} 正确率: {c}/{t}  ({pct:.1f}%)")
        qids = s["correct_question_ids"]
        print(f"             正确 question_id（共 {len(qids)} 个）:")
        # 每行约 30 个 id，避免单行过长难以阅读
        chunk = 30
        for i in range(0, len(qids), chunk):
            line = ", ".join(str(x) for x in qids[i : i + chunk])
            print(f"             {line}")
    print(f"{'-'*60}\n")

def eval_single(file_name, is_math=False):
    response_dict = json.load(open(result_path + "/" + file_name + ".json", "r", encoding="utf-8"))
    questions = list(response_dict.keys())
    accs = []

    for round in range(len(response_dict[questions[0]][0][0])):
        errcnt = 0
        corrcnt = 0
        if round % 2 == 1:
            continue
        if is_math and round == 0:
            continue
        accuracies = []
        results = []
        idx = 0
        for question in questions:
            responses, gt, question_idx = response_dict[question]
            pred_solutions = []
            pred_solution = responses[0][round]['content']
            pred_solutions.append(pred_solution)
            accurate, pred_answer = compute_accuracy(gt, pred_solutions, idx == 47 and round == 4, is_math=is_math)
            if accurate == 0:
                errcnt += 1
            else:
                corrcnt += 1
            if accurate is not None:
                accuracies.append(float(accurate))
                results.append({
                    "question_id": question,
                    "gt": gt,
                    "pred_answer": pred_answer,
                    "round": round
                })
            else:
                # Handle case where accuracy computation failed
                print(f"Warning: Failed to compute accuracy for question {question_idx}")
                print(f"Ground truth: {gt}")
            idx += 1
        print(f"Round{round},accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
        accs.append(np.mean(accuracies))
        print(f"err_cnt:{errcnt},corrcnt:{corrcnt}")
        break
    print(accs)

# extract_bbh：读取结果JSON
#     ↓
# 对每个问题、每个偶数轮次
#     ├── 收集所有智能体的预测答案
#     ├── 用多数投票选出最终答案 (most_frequent)
#     ├── 判断是否正确 (is_correct)
#     ├── 计算该轮的信息熵 (entropy)
#     └── 累加对数似然 (log_likelihood)
#     ↓
# 输出多维统计指标
def extract_bbh(file_name, is_math=False):
    hard_id = []
    metas = []
    response_dict = json.load(open(result_path + "/" + file_name + ".json", "r", encoding="utf-8"))
    questions = list(response_dict.keys())
    results = {
        "questions_count": len(questions),
        "response_structure": [
            len(response_dict[questions[0]]),
            len(response_dict[questions[0]][0]),
            len(response_dict[questions[0]][0][0])
        ],
        "details": []
    }

    agent_count = len(response_dict[questions[0]][0])
    round_count = len(response_dict[questions[0]][0][0])
    logs = []
    idx = 0
    log_likelihood = 0
    entropy = 0
    for question in questions:
        question_detail = {
            "question": question,
            "gt": None,
            "rounds": []
        }
        response, gt, question_idx = response_dict[question]
        if is_math:
            gt = strip_string(gt)
        question_detail["gt"] = gt
        log_txt = f"question: {question}\n{gt}\n"
        all_correct = []
        rounds = []
        for round in range(1, round_count):
            if round % 2 == 1:
                continue
            if round == 0:
                continue

            round_detail = {
                "round": round,
                "pred_answers": [],
                "most_answer": None,
                "is_correct": None
            }

            pred_solutions = []
            for agent in range(agent_count):
                pred_solutions.append(response[agent][round]['content'])

            pred_answers = []

            if is_math:
                for pred_solution in pred_solutions:
                    pred_answer = parse_math_anser(pred_solution)
                    if pred_answer is not None:
                        pred_answers.append(strip_string(pred_answer))
            else:
                for pred_solution in pred_solutions:
                    pred_answer = parse_answer(pred_solution)

                    if pred_answer is None:
                        pred_answer = solve_math_problems(pred_solution)

                    if pred_answer is None:
                        pred_answer = parse_YN(pred_solution)

                    if pred_answer is not None:
                        pred_answers.append(pred_answer)
            if pred_answers == []:
                pred_answers = ["(None)"] * agent_count
            most_answer = most_frequent(pred_answers)
            is_correct = gt == most_answer

            round_detail["pred_answers"] = pred_answers
            round_detail["most_answer"] = most_answer
            round_detail["is_correct"] = is_correct
            all_correct.append(is_correct)
            log_txt += f"{question_idx}"
            log_txt += f"{pred_answers}"
            log_txt += f"{most_answer}"
            log_txt += f"{is_correct}\n"

            question_detail["rounds"].append(round_detail)
            round = Round(pred_answers, most_answer, is_correct)
            rounds.append(round)

            single_entropy = 0
            correct_prob = pred_answers.count(gt) / len(pred_answers)
            if correct_prob > 0:
                log_likelihood += np.log2(correct_prob)
            for answer in set(pred_answers):
                prob = pred_answers.count(answer) / len(pred_answers)
                single_entropy -= prob * np.log2(prob)
            entropy += single_entropy

        results["details"].append(question_detail)
        logs.append(log_txt)
        meta = Meta(question, gt, rounds)
        metas.append(meta)
        idx += 1
    entropy /= len(metas)
    print(f"Log Likelihood: {log_likelihood}")
    print(f"Entropy: {entropy}")

    count = 0
    for meta in metas:
        if meta.rounds[0].is_correct == False and meta.rounds[1].is_correct == True:
            count += 1
    print(f"correct2incorrect:{count}")

    count = 0
    for meta in metas:
        if meta.rounds[0].is_correct:
            count += 1
    print(f"first round correcct:{count}")

    acc = [0.0 for i in range(500 if is_math else 250)]
    count = 0
    for idx in range(len(metas)):
        meta = metas[idx]
        if meta.rounds[1].is_correct:
            acc[idx] = 1.0
            count += 1
    print(f"second round correct:{count}")

    count = 0
    for meta in metas:
        if meta.rounds[0].is_correct == True and meta.rounds[1].is_correct == False:
            count += 1
    print(f"Number of transitions from correct to incorrect: {count}")

    count = 0
    for meta in metas:
        if meta.rounds[0].pred_answers.count(meta.gt) / len(meta.rounds[0].pred_answers) <= meta.rounds[1].pred_answers.count(meta.gt) / len(meta.rounds[1].pred_answers):
            count += 1
    print(f"Number of cases where first round accuracy is less than or equal to second round accuracy: {count}, Total: {len(metas)}, Proportion: {count/len(metas)}")

    count = 0
    for meta in metas:
        if meta.rounds[0].pred_answers.count(meta.gt) / len(meta.rounds[0].pred_answers) < meta.rounds[1].pred_answers.count(meta.gt) / len(meta.rounds[1].pred_answers):
            count += 1
    print(f"Number of cases where first round accuracy is less than second round accuracy: {count}, Total: {len(metas)}, Proportion: {count/len(metas)}")

    count = 0
    for meta in metas:
        if meta.rounds[0].pred_answers.count(meta.gt) / len(meta.rounds[0].pred_answers) > meta.rounds[1].pred_answers.count(meta.gt) / len(meta.rounds[1].pred_answers):
            count += 1
    print(f"Number of cases where first round accuracy is greater than second round accuracy: {count}, Total: {len(metas)}, Proportion: {count/len(metas)}")

    for r in range(2):
        print(f"Distribution of accuracy in round {r}")
        count = [0 for i in range(11)]
        for meta in metas:
            count[int(meta.rounds[r].pred_answers.count(meta.gt) / len(meta.rounds[r].pred_answers) * 10)] += 1
        print(count)

    with open(f"extract_detail_{file_name}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # list_of_tasks = ["geometric_shapes_id", "logical_deduction_seven_objects_id", "math_500_id"]

    # list_of_actions = [["expand"], ["expand", "exchange"], ["expand", "exchange", "exchange"], ["expand", "exchange", "exchange", "exchange"], ["expand", "exchange", "exchange", "exchange", "exchange"], ["expand", "exchange", "exchange", "exchange", "exchange", "exchange"]]
    # types = ["exchange", "exchangeI4", "exchangeI6"]

    # MODEL_NAME = "gpt-4o-mini"
    # model_names = ["gpt-4o-mini", "qwen2.5-7b-instruct", "qwen2.5-3b-instruct", "glm-4-flashx", "glm-4-flash", "qwen-turbo", "qwen-plus"]

    task_name = "math_500_id"
    result_path = f"{MODEL_NAME}/results/debate_zy/{task_name}"

    # file_names = [
    #    # "debate_zy_qwen2.5-7b-instruct_10_1_expand_agent_com0_False",
    #     # "debate_zy_qwen2.5-7b-instruct_10_1_exchange1_agent_com0_False",
    #     # "debate_zy_qwen2.5-7b-instruct_10_1_exchange2_agent_com0_False",
    # ]
    file_names = [
        # "debate_zy_glm-4-flashx_10_1_expand_agent_com0_False",
        # "debate_zy_glm-4-flashx_10_1_exchange1_agent_com0_False",
        # "debate_zy_glm-4-flashx_10_1_exchange2_agent_com0_False",
        # "debate_zy_glm-4-flashx_10_1_exchange_bidirectional_1_agent_com0_False",
        "debate_zy_glm-4-flashx_10_1_exchange_bidirectional_2_agent_com0_False",
    ]
    for file_name in file_names:
        print(f"\n{'='*60}")
        print(f"  {file_name}")
        print(f"{'='*60}")
        eval_bbh(file_name, is_math=True)
