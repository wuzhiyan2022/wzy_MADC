from glob import glob
from collections import Counter
import json
import time
import random
import openai
from openai import OpenAI,AsyncOpenAI
from tqdm import tqdm
import asyncio
from typing import Dict, Optional, Set
from common.utils import read_txt, read_json
import os
from eval_all_round import (
    parse_answer, solve_math_problems, parse_yes_no,
    parse_math_anser, parse_answer_fallback, _extract_math_answer,
)
from common.math_equivalence import strip_string
from wzy_multi_agent_debate_expand import get_expand_cache_entry



# API configuration - please set your own API endpoint and key
API_URL = "https://api.zhizengzeng.com/v1"
API_KEY = "sk-zk28544f5e4fdc6ce482ee6ae603f8af06469f20a6a4d4b6"
MODEL_NAME = "qwen3-8b"
MODEL_TAG = "qwen3-8b"
# MODEL_NAME = "qwen-turbo"
# MODEL_TAG = "qwen-turbo"
# qwen3-8b 经当前 API 网关时 max_tokens 合法范围为 [1, 8192]；gpt-5-* 等可改用下方注释的 max_completion_tokens
MAX_TOKENS = 8192
client = OpenAI(base_url=API_URL,
                       api_key=API_KEY,
                       )
async_client = AsyncOpenAI(base_url=API_URL,
                       api_key=API_KEY,
                       )

# ---------- 断点续跑配置 ----------
# True：启用断点续跑，从 checkpoint 读取已完成题目并跳过
# False：不启用，每次全量运行（仍会逐题增量写入结果文件）
ENABLE_CHECKPOINT: bool = True


class CheckpointManager:
    """管理断点续跑的 checkpoint 文件读写。"""

    def __init__(self, checkpoint_path: str):
        self.path = checkpoint_path
        self.data: Dict[str, bool] = self._load()

    def _load(self) -> Dict[str, bool]:
        if os.path.exists(self.path):
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


def _get_run_file_stem(
    model_name: str,
    agents: int,
    rounds: int,
    actions: list,
    agent_com_name: str,
    is_hard: bool,
) -> str:
    return "debate_{}_{}_{}_{}_{}_{}".format(
        model_name, agents, rounds, "_".join(actions), agent_com_name, is_hard
    )


def _get_result_path(
    model_name: str,
    task_name: str,
    agents: int,
    rounds: int,
    actions: list,
    agent_com_name: str,
    is_hard: bool,
) -> str:
    stem = _get_run_file_stem(model_name, agents, rounds, actions, agent_com_name, is_hard)
    return os.path.join(model_name, "results", "debate", task_name, stem + ".json")


def _get_checkpoint_path(
    model_name: str,
    task_name: str,
    agents: int,
    rounds: int,
    actions: list,
    agent_com_name: str,
    is_hard: bool,
) -> str:
    stem = _get_run_file_stem(model_name, agents, rounds, actions, agent_com_name, is_hard)
    return os.path.join(model_name, "results", "debate", task_name, ".checkpoint_" + stem + ".json")


def _atomic_save_json(path: str, data: dict) -> None:
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _load_existing_results(result_path: str) -> dict:
    if not os.path.exists(result_path):
        return {}
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            return loaded
        print(f"[断点续跑] 结果文件顶层非 JSON 对象，将视为空: {result_path}")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[断点续跑] 读取结果文件失败 ({e})，将视为空: {result_path}")
    return {}


def _extract_qids_from_results(result_dict: dict) -> Set[str]:
    qids: Set[str] = set()
    for v in result_dict.values():
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            qids.add(str(v[2]))
    return qids


def _get_completed_qids(
    checkpoint: Optional[CheckpointManager],
    result_dict: dict,
) -> Set[str]:
    completed: Set[str] = set()
    if checkpoint is not None:
        for qid, done in checkpoint.data.items():
            if done:
                completed.add(str(qid))
    completed.update(_extract_qids_from_results(result_dict))
    return completed


def _sync_checkpoint_from_results(
    checkpoint: CheckpointManager,
    result_dict: dict,
) -> None:
    changed = False
    for qid in _extract_qids_from_results(result_dict):
        if not checkpoint.is_completed(qid):
            checkpoint.data[qid] = True
            changed = True
    if changed:
        checkpoint.save()
        print(f"[断点续跑] 已从结果文件同步 {checkpoint.get_completed_count()} 条 checkpoint 记录")


def _save_one_result(
    result_path: str,
    question: str,
    agent_contexts,
    answer,
    question_id,
) -> None:
    merged = _load_existing_results(result_path)
    merged[question] = [agent_contexts, answer, question_id]
    _atomic_save_json(result_path, merged)


def construct_exchange_message(agents, question, round):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[2*round]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    
    with open(f"{MODEL_NAME}/data/debate/prefix_string_bbh_test_pure.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}


def construct_exchangeN_message(agents, question, round):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    random.shuffle(agents)
    for agent in agents:
        agent_response = agent[2*round]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    
    with open(f"{MODEL_NAME}/data/debate/prefix_string_bbh_test_pure.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}

def construct_exchangeI_message(agents, question, round):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
    
    answer_counts = Counter(pred_answers)
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: answer_counts[pair[0]], reverse=True)]

    for pred_solution in pred_solutions:
    
        response = "\n\n One agent solution: ```{}```".format(pred_solution)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    
    with open(f"{MODEL_NAME}/data/debate/prefix_string_bbh_test_pure.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}

def construct_exchangeI6_message(agents, question, round):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
    
    answer_counts = Counter(pred_answers)
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: answer_counts[pair[0]], reverse=False)]

    for pred_solution in pred_solutions:
    
        response = "\n\n One agent solution: ```{}```".format(pred_solution)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    
    return {"role": "user", "content": prefix_string}


def construct_exchangeO4I6_message(agents, question, round):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
    
    answer_counts = Counter(pred_answers)
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: answer_counts[pair[0]], reverse=False)]


    chain_list = []
    for i in range(len(pred_solutions)):
        steps = pred_solutions[i].split("Step ")[1:] 
        steps = [f"Step {step.strip()}" for step in steps]

        steps = steps[:-1]
        chain_list.append(steps)

    prefix_string = "These are the final key steps in the solution to the problem from other agents: "
    for chain in chain_list:
        agent_response = "\n\n".join(chain)
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    
    return {"role": "user", "content": prefix_string}


def construct_exchangeO4I61_message(agents, question, round):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    idx = 0
    for pred_solution in pred_solutions:
        idx+=1
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)

        if pred_answer is None:
            pred_answers.append("(None)"+str(idx))
    answer_counts = Counter(pred_answers)
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: answer_counts[pair[0]], reverse=False)]

    chain_list = []
    for i in range(len(pred_solutions)):
        steps = pred_solutions[i].split("Step ")[1:] 
        steps = [f"Step {step.strip()}" for step in steps]

        steps = steps[:-1]
        chain_list.append(steps)

    prefix_string = "These are the final key steps in the solution to the problem from other agents: "
    for chain in chain_list:
        agent_response = "\n\n".join(chain)
        response = "\n\n One agent solution: ```{}```".format(agent_response)


        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    
    return {"role": "user", "content": prefix_string}

def construct_exchangeI61_message(agents, question, round):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    idx = 0
    for pred_solution in pred_solutions:
        idx += 1
        # 直接复用 eval_all_round._extract_math_answer：boxed → fallback → 括号整数兜底
        # （兜底仅当全文无 "The answer is" 标记时启用），与评测端完全一致
        pred_answer = _extract_math_answer(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
        else:
            pred_answers.append("(None)" + str(idx))

    answer_counts = Counter(pred_answers)
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: answer_counts[pair[0]], reverse=False)]

    for pred_solution in pred_solutions:
    
        response = "\n\n One agent solution: ```{}```".format(pred_solution)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + (
        "\n\n Using the reasoning from other agents as additional advice, "
        "can you give an updated answer? "
        "Some of the other agents' reasoning steps may be incorrect, so please examine both your own solution and the other agents' reasoning step by step before deciding whether to use them."
        #"Examine your solution and that other agents step by step. "
        "Please structure your updated reasoning step by step in the format: "
        "Step 1: ... Step 2: ... and so on. "
        "Put your answer in the form (X) at the end of your response."
    )

    return {"role": "user", "content": prefix_string}

def construct_exchangeI7_message(agents, question, round,gt):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)

        if pred_answer is None:
            pred_answers.append("(None)")
    
    origin_ans = pred_answers
    answer_counts = Counter(pred_answers)

    ans_preds = zip(pred_answers, pred_solutions)
    ans_preds = [ans_pred for ans_pred in ans_preds if answer_counts[ans_pred[0]]>1]
    
    temp_preds = [pred for _,pred in ans_preds ]
    pred_answers = []
    for pred_solution in temp_preds:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)

        if pred_answer is None:
            pred_answers.append("(None)")
    delete_ans = pred_answers


    pred_solutions = [x for _, x in sorted(ans_preds, key=lambda pair: answer_counts[pair[0]], reverse=False)]

    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)

        if pred_answer is None:
            pred_answers.append("(None)")
    if(str(origin_ans)!=str(delete_ans)):
        print(f"\noriginal:{str(origin_ans)}")
        print(f"delete: {str(delete_ans)}")
        print(f"final:  {str(pred_answers)}\n")
        print(f"gt: {gt}")


    for pred_solution in pred_solutions:
    
        response = "\n\n One agent solution: ```{}```".format(pred_solution)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    
    return {"role": "user", "content": prefix_string}


def construct_exchangeI3_message(agents, question, round,gt,original_question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
    
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: pair[0]==gt, reverse=True)]

    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)
        if pred_answer is None:
            pred_answer = parse_yes_no(pred_solution)
        if pred_answer is not None:
            pred_answers.append(pred_answer)

    for pred_solution in pred_solutions:
        response = "\n\n One agent solution: ```{}```".format(pred_solution)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + f"""\n\n  The original question is {original_question}. Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    
    with open(f"{MODEL_NAME}/data/debate/prefix_string_bbh_test_pure.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}

def construct_exchangeI4_message(agents, question, round,gt,original_question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
    
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: pair[0]==gt, reverse=False)]

    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)
        if pred_answer is None:
            pred_answer = parse_yes_no(pred_solution)
        if pred_answer is not None:
            pred_answers.append(pred_answer)

    for pred_solution in pred_solutions:
        response = "\n\n One agent solution: ```{}```".format(pred_solution)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + f"""\n\nUsing the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    return {"role": "user", "content": prefix_string}

def construct_exchangeI41_message(agents, question, round,gt,original_question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    idx = 0
    for pred_solution in pred_solutions:
        idx += 1
        # 直接复用 eval_all_round._extract_math_answer：boxed → fallback → 括号整数兜底
        # （兜底仅当全文无 "The answer is" 标记时启用），与评测端完全一致
        pred_answer = _extract_math_answer(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
        else:
            pred_answers.append("(None)" + str(idx))

    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: pair[0]==gt, reverse=False)]

    for pred_solution in pred_solutions:
        response = "\n\n One agent solution:\n{}".format(pred_solution)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + (
        "\n\nUsing the reasoning from other agents as additional advice, "
        "can you give an updated answer?"
        "Some of the other agents' reasoning steps may be incorrect, so please examine both your own solution and the other agents' reasoning step by step before deciding whether to use them."
       # "Examine your solution and that other agents step by step."
        "Please structure your updated reasoning step by step in the format: "
        "Step 1: ... Step 2: ... and so on. "
        "Put your answer in the form (X) at the end of your response."
    )
    return {"role": "user", "content": prefix_string}


def construct_exchangeI5_message(agents, question, round,gt,original_question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
    
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: pair[0]==gt, reverse=False)]

    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)
        if pred_answer is None:
            pred_answer = parse_yes_no(pred_solution)
        if pred_answer is not None:
            pred_answers.append(pred_answer)

    for pred_solution in pred_solutions:
        response = "\n\n One agent solution: ```{}```".format(pred_solution)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + f"""\n\n  The original question is {original_question}. Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    
    return {"role": "user", "content": prefix_string}

def construct_exchangeI2_message(agents, question, round,gt):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
    
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: pair[0]==gt, reverse=True)]

    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)
        if pred_answer is None:
            pred_answer = parse_yes_no(pred_solution)
        if pred_answer is not None:
            pred_answers.append(pred_answer)

    for pred_solution in pred_solutions:
        response = "\n\n One agent solution: ```{}```".format(pred_solution)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    
    with open(f"{MODEL_NAME}/data/debate/prefix_string_bbh_test_pure.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}

def construct_exchangeI1_message(agents, question, round,original_question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = f"The original question is: {original_question}.These are the solutions to the problem from other agents: "

    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
    
    answer_counts = Counter(pred_answers)
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: answer_counts[pair[0]], reverse=True)]

    for pred_solution in pred_solutions:
        response = "\n\n One agent solution: ```{}```".format(pred_solution)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    
    with open(f"{MODEL_NAME}/data/debate/prefix_string_bbh_test_pure.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}

def construct_exchangeG_message(agents, question, round):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    cots = [agent[2*round]["content"] for agent in agents]
    chain_list = []
    for i in range(len(cots)):
        steps = cots[i].split("Step ")[1:]
        steps = [f"Step {step.strip()}" for step in steps]
        chain_list.append(steps)

    with open(f"{MODEL_NAME}/data/debate/chain_list_bbh_test.json", "w") as f:
        json.dump(chain_list, f)

    max_len = max([len(chain) for chain in chain_list])
    prefix_string = "These are the steps to the problem from other agents: "

    for idx in range(max_len):
        for i in range(len(chain_list)):
            if idx < len(chain_list[i]):
                prefix_string = prefix_string + chain_list[i][idx]+"\n\n"

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    
    with open(f"{MODEL_NAME}/data/debate/prefix_string_bbh_test.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}

def construct_exchangeJ_message(agents, question, round):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    cots = [agent[2*round]["content"] for agent in agents]
    chain_list = []
    for i in range(len(cots)):
        steps = cots[i].split("Step ")[1:]
        chain_list.append(steps)

    with open(f"{MODEL_NAME}/data/debate/{task_name}/steps.json", "w") as f:
        json.dump(chain_list, f)

    max_len = max([len(chain) for chain in chain_list])
    prefix_string = "These are the steps to the problem from other agents: "

    for idx in range(max_len):
        for i in range(len(chain_list)):
            if idx < len(chain_list[i]):
                prefix_string = prefix_string + chain_list[i][idx]+"\n\n"

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    
    with open(f"{MODEL_NAME}/data/debate/{task_name}/prefix_string_bbh_test.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}

def construct_exchangeO_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}
    random.shuffle(agents)
    cots = [agent[2*round]["content"] for agent in agents]
    chain_list = []
    for i in range(len(cots)):
        steps = cots[i].split("Step ")[1:] 
        steps = [f"Step {step.strip()}" for step in steps]
        
        steps = steps[int(len(steps) * 0.2):]
        chain_list.append(steps)

    
    with open(f"{MODEL_NAME}/data/debate/{task_name}/steps_O.json", "w") as f:
        json.dump(chain_list, f)


    prefix_string = "These are the final key steps in the solution to the problem from other agents: "
    for chain in chain_list:
        agent_response = "\n\n".join(chain)
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
     
    with open(f"{MODEL_NAME}/data/debate/{task_name}/prefix_string_bbh_test.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}

def construct_exchangeO1_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}
    
    cots = [agent[2*round]["content"] for agent in agents]
    chain_list = []
    for i in range(len(cots)):
        steps = cots[i].split("Step ")[1:] 
        steps = [f"Step {step.strip()}" for step in steps]
        
        steps = steps[int(len(steps) * 0.4):]
        chain_list.append(steps)

    
    with open(f"{MODEL_NAME}/data/debate/{task_name}/steps_O1.json", "w") as f:
        json.dump(chain_list, f)


    prefix_string = "These are the final key steps in the solution to the problem from other agents: "
    for chain in chain_list:
        agent_response = "\n\n".join(chain)
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
     
    with open(f"{MODEL_NAME}/data/debate/{task_name}/prefix_string_bbh_test.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}
def construct_exchangeO2_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}
    
    cots = [agent[2*round]["content"] for agent in agents]
    chain_list = []
    for i in range(len(cots)):
        steps = cots[i].split("Step ")[1:] 
        steps = [f"Step {step.strip()}" for step in steps]
        
        steps = steps[int(len(steps) * 0.6):]
        chain_list.append(steps)

    
    with open(f"{MODEL_NAME}/data/debate/{task_name}/steps_O2.json", "w") as f:
        json.dump(chain_list, f)


    prefix_string = "These are the final key steps in the solution to the problem from other agents: "
    for chain in chain_list:
        agent_response = "\n\n".join(chain)
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
     
    with open(f"{MODEL_NAME}/data/debate/{task_name}/prefix_string_bbh_test.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}

def construct_exchangeO3_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}
    
    cots = [agent[2*round]["content"] for agent in agents]
    chain_list = []
    for i in range(len(cots)):
        steps = cots[i].split("Step ")[1:] 
        steps = [f"Step {step.strip()}" for step in steps]
        
        steps = steps[int(len(steps) * 0.8):]
        chain_list.append(steps)

    
    with open(f"{MODEL_NAME}/data/debate/{task_name}/steps_O3.json", "w") as f:
        json.dump(chain_list, f)


    prefix_string = "These are the final key steps in the solution to the problem from other agents: "
    for chain in chain_list:
        agent_response = "\n\n".join(chain)
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
     
    with open(f"{MODEL_NAME}/data/debate/{task_name}/prefix_string_bbh_test.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}

def construct_exchangeO4_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}
    
    cots = [agent[2*round]["content"] for agent in agents]
    chain_list = []
    for i in range(len(cots)):
        steps = cots[i].split("Step ")[1:] 
        steps = [f"Step {step.strip()}" for step in steps]

        steps = steps[:-1]
        chain_list.append(steps)

    
    with open(f"{MODEL_NAME}/data/debate/{task_name}/steps_O2.json", "w") as f:
        json.dump(chain_list, f)


    prefix_string = "These are the final key steps in the solution to the problem from other agents: "
    for chain in chain_list:
        agent_response = "\n\n".join(chain)
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
     
    with open(f"{MODEL_NAME}/data/debate/{task_name}/prefix_string_bbh_test.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}
def construct_exchangeH_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    cots = [agent[2*round]["content"] for agent in agents]

    
    
    
    
    
    
    
    pred_answers = []
    for pred_solution in cots:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)


    
    
    answer_counts = Counter(pred_answers)
    cots = [x for _, x in sorted(zip(pred_answers, cots), key=lambda pair: answer_counts[pair[0]], reverse=True)]
    
    
    

    
    

    
    
    

    chain_list = []
    for i in range(len(cots)):
        steps = cots[i].split("Step ")[1:]  
        steps = [f"Step {step.strip()}" for step in steps]  
        chain_list.append(steps)
    
    with open(f"{MODEL_NAME}/data/debate/chain_list_bbh_test.json", "w") as f:
        json.dump(chain_list, f)

    max_len = max([len(chain) for chain in chain_list])
    prefix_string = "These are the solution to the problem from other agents in steps order: "

    for idx in range(max_len):
        prefix_string = prefix_string + f"Step{idx+1} of other agents:\n"
        for i in range(len(chain_list)):
            if idx < len(chain_list[i]):
        
                prefix_string = prefix_string + chain_list[i][idx]+"\n\n"

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Pay close attention to the earlier steps in the reasoning process. Examine your solution and the steps of other agents one by one.Put your answer in the form (X) at the end of your response. {question}""".format(question=question)
    
    with open(f"{MODEL_NAME}/data/debate/prefix_string_bbh_test.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}


def construct_exchangeK_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    cots = [agent[2*round]["content"] for agent in agents]
    chain_list = []
    for i in range(len(cots)):
        steps = cots[i].split("Step ")[1:]  
        steps = [f"Step {step.strip()}" for step in steps]  
        chain_list.append(steps)
    
    with open(f"{MODEL_NAME}/data/debate/chain_list_bbh_test.json", "w") as f:
        json.dump(chain_list, f)

    max_len = max([len(chain) for chain in chain_list])
    prefix_string = "These are the solution to the problem from other agents in steps order: "

    for idx in range(max_len):
        prefix_string = prefix_string + f"Step{idx+1} of other agents:\n"
        for i in range(len(chain_list)):
            if idx < len(chain_list[i]):
        
                prefix_string = prefix_string + chain_list[i][idx]+"\n\n"

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Pay close attention to the earlier steps in the reasoning process. Examine your solution and the steps of other agents one by one.Put your answer in the form (X) at the end of your response."""
    
    with open(f"{MODEL_NAME}/data/debate/prefix_string_bbh_test.json", "w") as f:
        json.dump(prefix_string, f)
    return {"role": "user", "content": prefix_string}


def construct_exchangeL_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "
    for agent in agents:
        agent_response = agent[2*round]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Pay close attention to the earlier steps in the reasoning process. Examine your solution and the steps of other agents one by one.Put your answer in the form (X) at the end of your response. {question}""".format(question=question)
    return {"role": "user", "content": prefix_string}


def construct_exchangeM_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}



    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
    
    
    answer_counts = Counter(pred_answers)
    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions), key=lambda pair: answer_counts[pair[0]], reverse=True)]

    prefix_string = "These are the solutions to the problem from other agents: "
    for pred_solution in pred_solutions:
        response = "\n\n One agent solution: ```{}```".format(pred_solution)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Pay close attention to the earlier steps in the reasoning process. Examine your solution and the steps of other agents one by one.Put your answer in the form (X) at the end of your response. {question}""".format(question=question)
    return {"role": "user", "content": prefix_string}

def construct_exchangeA_message(agent_context, instruction, idx):
    prefix_string = "Here are a list of opinions from different agents: "
    options = ""
    for agent in agent_context:
        agent_response = agent[-1]["content"]
        options += "\n\n One agent response: ```{}```".format(agent_response)


    prefix_string = prefix_string + options + "\n\n Write a summary of the different opinions from each of the individual agent."

    message = [{"role": "user", "content": prefix_string}]

    try:
        print("before summary")
        response = client.chat.completions.create(
            model=MODEL_TAG,
            messages=message,
            max_tokens=MAX_TOKENS,
            # max_completion_tokens=30000,  # gpt-5-* 等新模型（按需启用）
            n=1,
        )
        print("afore summary")
        response_data = json.loads(response.json())
        content = response_data['choices'][0]['message']['content']
        
    except:
        print("retrying ChatGPT due to an error......")
        time.sleep(5)
        return construct_exchangeA_message(agent_context, instruction, idx)

    prefix_string = f"Here is a summary of responses from other agents: {content}"
    prefix_string = prefix_string + "\n\n Use this summarization carefully as additional advice, can you provide an updated answer? " + instruction+". Make sure put your answer in the form (X) at the end of your response."
    return {"role": "user", "content": prefix_string}




def construct_exchangeD_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[2*round]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response
    percent = 100-25*round
    prefix_string = prefix_string + f"""\n\n Using the reasoning from other agents as additional advice.You can consider this suggestion to a {percent}% extent. can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_exchangeE_message(agents, question, round):
    
    
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[2*round]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response
    percent = 80-25*round
    prefix_string = prefix_string + f"""\n\n Using the reasoning from other agents as additional advice.You can consider this suggestion to a {percent}% extent. can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}



def construct_verify_message():
    return {"role":"user","content":"Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

def construct_verifyA_message():
    return {"role":"user","content":"Please come up with a question based on the topic, answer it, and check if it conflicts with existing answer. If it does, update your answer. Put your final answer in the form (X) at the end of your response."}

def construct_verifyB_message():
    return {"role":"user","content":"Please verify if there are any errors by applying your answer to the question. Check if your answer needs updating.Regardless of whether you update it, make sure put your final answer in the form (X) at the end of your response."}

def construct_verifyC_message():
    return {"role":"user","content":"Please verify if there are any errors in your Premise Information and Reasoning Analysis. Check if your answer needs updating.Regardless of whether you update it, make sure put your final answer in the form (X) at the end of your response."}



def construct_expandA_message(question,agent_contexts,idx):
    
    
    if len(agent_contexts) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent_context in agent_contexts:
        agent_response = agent_context[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give a different way to solve this question?. Put your answer in the form (X) at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}

def construct_expandB_message(question):
    return {"role":"user","content":f""" Can you answer the following question as accurately as possible? {question}.
            Please provide your approach to solving this problem in the following format:

Premise Information: (Extract objectively from the problem without additional interpretation)
Reasoning Analysis: (Analyze step by step based on the premise information)
Basis for Reasoning Analysis: (Provide the basis for the reasoning analysis above)
Exceptions: (Describe under what conditions the reasoning might not hold)
Conclusion: (The final reasoning conclusion)
        Make sure put your answer in the form (X) at the end of your response."""}


def construct_expandC_message(question):
    return {"role":"user","content":f"""Firstly,Can you answer the following question as accurately as possible? {question}.
            Secondly,Please provide your approach to solving this problem in the following format:

Premise Information: (Extract objectively from the problem without additional interpretation)
Reasoning Analysis: (Analyze step by step based on the premise information)
Basis for Reasoning Analysis: (Provide the basis for the reasoning analysis above)
Exceptions: (Describe under what conditions the reasoning might not hold)
Conclusion: (The final reasoning conclusion)
        Make sure put your answer in the form (X) at the end of your response."""}

def construct_expandD_message(question):
    return {"role":"user","content":f""" Can you answer the following question as accurately as possible? {question}. Let's think step by step.Make sure put your answer in the form (X) at the end of your response."""}

def construct_expandE_message(question,tmpl):
    tmpl = tmpl.replace("<QUESTION>",question)
    return {"role":"user","content":f"""{tmpl}.Make sure put your answer in the form (X) at the end of your response."""}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def generate_answer(answer_context):
    try:
        response = client.chat.completions.create(
            model=MODEL_TAG,
            messages=answer_context,
            max_tokens=MAX_TOKENS,
            # max_completion_tokens=30000,  # gpt-5-* 等新模型（按需启用）
            n=1,
        )
        completion=response.model_dump()
        
        
        
        
    except Exception as e:
        print("retrying due to an error......")
        print(e)
        time.sleep(20)
        return generate_answer(answer_context) 

    return completion

async def agenerate_answer(answer_context):
    try:
        
        response = await async_client.chat.completions.create(
            model=MODEL_TAG,
            messages=answer_context,
            max_tokens=MAX_TOKENS,
            # max_completion_tokens=30000,  # gpt-5-* 等新模型（按需启用）
            n=1,
        )
        completion=response.model_dump()
        
    except Exception as e:
        print("retrying due to an error......")
        print(e)
        time.sleep(20)
        return generate_answer(answer_context) 

    return completion

def parse_question_answer(tasks, ix):
    question = tasks[ix]['input']
   

    question_content = f"Can you answer the following question as accurately as possible? {question} \n Explain your answer.Make sure putting the answer in the form (X) at the end of your response."
    answer = tasks[ix]['target']
    original_question = question
    question_id = tasks[ix]['question_id']
    return question_content, answer,question_id,original_question


async def main(agents,rounds,actions): 
    task_file = f"{MODEL_NAME}/data/{task_name}"
    agent_com_name = "agent_com0"
    user_config = "u1"
    is_hard = False

    with open(task_file+".json", "r", encoding="utf-8") as f:
        data = json.load(f)['examples']

    with open("prompt/"+agent_com_name+".json", "r", encoding="utf-8") as f:
        system_prompts= json.load(f)['agents']

    with open("prompt/"+f"prompt_{user_config}"+".txt", "r", encoding="utf-8") as f:
        user_prompt_tmp = f.read()

    hard_id = [str(i) for i in range(1, 101)]
    if is_hard:
        data = [d for d in data if d['question_id'] in hard_id]
        eval_cnt = len(hard_id)
    else:
        eval_cnt = 500
    fewshot_ost_config = read_json("prompt/fewshot_ost_config.json")
    fewshot_ost_prompt = read_txt("prompt/fewshot_ost_prompt.txt")
    # debate_zy_qwen2.5-7b-instruct_10_1_expand_agent_com0_False.json
    expand_cache_path = r"qwen3-8b\results\debate_zy\math_500_id\debate_zy_qwen3-8b_10_1_expand_agent_com0_False.json"
    with open(expand_cache_path, "r", encoding="utf-8") as f:
        expand_cache = json.load(f)

    out_dir = os.path.join(MODEL_NAME, "results", "debate", task_name)
    os.makedirs(out_dir, exist_ok=True)
    result_path = _get_result_path(
        MODEL_NAME, task_name, agents, rounds, actions, agent_com_name, is_hard
    )
    checkpoint_path = _get_checkpoint_path(
        MODEL_NAME, task_name, agents, rounds, actions, agent_com_name, is_hard
    )

    result_dict = _load_existing_results(result_path)
    checkpoint: Optional[CheckpointManager] = None
    if ENABLE_CHECKPOINT:
        checkpoint = CheckpointManager(checkpoint_path)
        _sync_checkpoint_from_results(checkpoint, result_dict)

    completed_qids = _get_completed_qids(checkpoint, result_dict) if ENABLE_CHECKPOINT else set()
    pending_indices = [
        i for i in range(eval_cnt)
        if str(data[i]["question_id"]) not in completed_qids
    ]

    if ENABLE_CHECKPOINT:
        print(f"\n[断点续跑] 结果文件: {result_path}")
        print(f"[断点续跑] checkpoint: {checkpoint_path}")
        print(f"[断点续跑] 已完成 {len(completed_qids)}/{eval_cnt} 题，本次待跑 {len(pending_indices)} 题")
        if not pending_indices:
            print("[断点续跑] 所有题目已完成，无需运行")
            return

    async def infer_one(data,i):

        question, answer,question_id,original_question = parse_question_answer(data,i)
        fewshot_content = fewshot_ost_config["prompt_template"].format(
                examples=fewshot_ost_prompt,
                instruction=question,
            ) 
        
        agent_contexts = [[{"role": "system","content":system_prompts[agent_idx]["system"]},{"role": "user", "content": fewshot_content}] for agent_idx in range(agents)]
        async def agen_one_round(agent_contexts, agent_context,agent_idx,question, action,round):
          
            if action =="expandA":
                if agent_idx!=0:
                    agent_contexts_before = agent_contexts[:i]
                    message = construct_expandA_message(question,agent_contexts_before, 2 * round + 2) 
                    agent_context[ 2 * round +1]=message
                
            elif action=="expandB":
                message = construct_expandB_message(question)
                agent_context[ 2 * round +1]=message
            elif action=="expandD":
                message = construct_expandD_message(question)
                agent_context[ 2 * round +1]=message
            elif action=="expandE":
                message = construct_expandE_message(question,user_prompt_tmp)
                agent_context[ 2 * round +1]=message
            elif action == "expand":
                _entry = get_expand_cache_entry(expand_cache, question_id, original_question)
                if _entry is None:
                    raise KeyError(
                        f"expand 缓存中未找到 question_id={question_id!r}（已尝试 id 匹配与题干 key）"
                    )
                cxt = _entry[0][agent_idx][2]["content"]
                completion = {"choices": [{"message": {"content": cxt}}]}
                # if agent_idx == 0 and question_id == "1":
                #     print("\n[expand_cache 示例]")
                #     print(f"  question_id : {question_id}")
                #     print(f"  agent_idx   : {agent_idx}")
                #     print(f"  original_question: {original_question[:80]}...")
                #     print(f"  completion content (前200字):\n{cxt[:2000]}")
                #     print("[expand_cache 示例结束]\n")
                #     os._exit(0)
                return completion
              
            elif action == "exchange":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchange_message(agent_contexts_other, question, round)
                agent_context.append(message)
            elif action == "exchangeA":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeA_message(agent_contexts_other, question, round)
                agent_context.append(message)    
            elif action == "exchangeB":
                agent_contexts_other = [agent_contexts[(agent_idx-1) % len(agent_contexts)], agent_contexts[(agent_idx+1) % len(agent_contexts)]]
                message = construct_exchange_message(agent_contexts_other, question, round)
                agent_context.append(message)    
            elif action == "exchangeC":
                agent_contexts_other = random.sample(agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:], len(agent_contexts) // 2)
                message = construct_exchange_message(agent_contexts_other, question, round)
                agent_context.append(message)         
            elif action == "exchangeD":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:],
                message = construct_exchangeD_message(agent_contexts_other, question, round)
                agent_context.append(message)           
            elif action == "exchangeE":
                agent_contexts_other = random.sample(agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:], len(agent_contexts) // 2)
                message = construct_exchangeE_message(agent_contexts_other, question, round)
                agent_context.append(message)
            elif action == "exchangeG":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeG_message(agent_contexts_other, question, round)
                agent_context.append(message)              
            elif action == "exchangeH":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeH_message(agent_contexts_other, question, round)
                agent_context.append(message)       
            elif action == "exchangeJ":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeJ_message(agent_contexts_other, question, round)
                agent_context.append(message)     
            elif action == "exchangeI":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeI_message(agent_contexts_other, question, round)
                agent_context.append(message)  
            elif action == "exchangeI6":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeI6_message(agent_contexts_other, question, round)
                agent_context.append(message)  
            elif action == "exchangeI61":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeI61_message(agent_contexts_other, question, round)
                print("message:",message)
                agent_context.append(message)  
            elif action == "exchangeI7":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeI7_message(agent_contexts_other, question, round,gt=answer)
                agent_context.append(message)  
            elif action == "exchangeI41":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeI41_message(agent_contexts_other, question, round,answer,original_question)
                agent_context.append(message) 
            elif action == "exchangeI1":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeI1_message(agent_contexts_other, question, round,original_question)
                agent_context.append(message)  
            elif action == "exchangeI2":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeI2_message(agent_contexts_other, question, round,answer)
                agent_context.append(message)  
            elif action == "exchangeI3":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeI3_message(agent_contexts_other, question, round,answer,original_question)
                agent_context.append(message)  
            elif action == "exchangeI4":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeI4_message(agent_contexts_other, question, round,answer,original_question)
                agent_context.append(message)  
            elif action == "exchangeI5":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeI5_message(agent_contexts_other, question, round,answer,original_question)
                agent_context.append(message)  
            elif action == "exchangeL":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeL_message(agent_contexts_other, question, round)
                agent_context.append(message)
            elif action == "exchangeM":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeM_message(agent_contexts_other, question, round)
                agent_context.append(message)  
            elif action == "exchangeN":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeN_message(agent_contexts_other, question, round)
                agent_context.append(message)  
            elif action == "exchangeK":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeK_message(agent_contexts_other, question, round)
                agent_context.append(message)  
            elif action == "exchangeO":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeO_message(agent_contexts_other, question, round)
                agent_context.append(message)  
            elif action == "exchangeO1":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeO1_message(agent_contexts_other, question, round)
                agent_context.append(message) 
            elif action == "exchangeO2":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeO2_message(agent_contexts_other, question, round)
                agent_context.append(message) 
            elif action == "exchangeO3":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeO3_message(agent_contexts_other, question, round)
                agent_context.append(message)   
            elif action == "exchangeO4":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeO4_message(agent_contexts_other, question, round)
                agent_context.append(message)     
            elif action == "exchangeO4I6":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeO4I6_message(agent_contexts_other, question, round)
                agent_context.append(message)     
            elif action == "exchangeO4I61":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchangeO4I61_message(agent_contexts_other, question, round)
                agent_context.append(message)                          
            elif action == "verify":
                message = construct_verify_message()
                agent_context.append(message)
            elif action == "verifyA":
                message = construct_verifyA_message()
                agent_context.append(message)
            elif action == "verifyB":
                message = construct_verifyB_message()
                agent_context.append(message)
            elif action == "verifyC":
                message = construct_verifyC_message()
                agent_context.append(message)
            elif action == "update":
                agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                message = construct_exchange_message(agent_contexts_other, question, round)
                agent_context.append(message)
                
            completion = await agenerate_answer(agent_context)

            return completion

        for round in range(rounds):
            
            action = actions[round]
            if action=="exchangeJ":
                cots = [agent[2*round]["content"] for agent in agent_contexts]
            
                return
            if action!="expandA":
                atasks = [agen_one_round(agent_contexts, agent_context,i,question, actions[round],round) for i, agent_context in enumerate(agent_contexts)]
                results = await asyncio.gather(*atasks)
                
                for i, completion in enumerate(results):
                    assistant_message = construct_assistant_message(completion)
                    agent_contexts[i].append(assistant_message)
                    
            else:
                for agent_idx,agent_context in enumerate(agent_contexts):
                    completion = await agen_one_round(agent_contexts, agent_context,agent_idx,question, actions[round],round)
                    assistant_message = construct_assistant_message(completion)
                    agent_contexts[i].append(assistant_message)

        _save_one_result(result_path, question, agent_contexts, answer, question_id)
        if checkpoint is not None:
            checkpoint.mark_completed(str(question_id))

    batch = 2
    for batch_start in tqdm(range(0, len(pending_indices), batch)):
        batch_indices = pending_indices[batch_start : batch_start + batch]
        atasks = [infer_one(data, i) for i in batch_indices]
        await asyncio.gather(*atasks)

    final_count = len(_load_existing_results(result_path))
    print(f"\n[完成] 结果已保存至 {result_path}（共 {final_count} 题）")
    
if __name__ == "__main__":
    
    #all actions
    # exchangeH: step wise2 + most first I
    # exchangeI: most first
    # exchangeI6: most last
    # exchangeI61: most last 
    # exchangeI2: gt first
    # exchangeI3: gt first+question_end
    # exchangeI4: gt last
    # exchangeI41: gt last 
    # exchangeI5: gt last+question_end
    # exchangeI1: q_start+most first
    # exchangeK: step wise2
    # exchagneL: step wise prompt+ solution order
    # exchagneM: step wise prompt+ solution order+most first
    # exchagneN: random shuffle order

    list_of_tasks = ["math_500_id"]
    list_of_actions = [["expand","exchangeI61","exchangeI61"],["expand","exchangeI41","exchangeI41"]]

    # for agent in agents: D
    #     # for round in rounds:
    #         print("[current agent]:", agent)
    #         print("[current round]:", round)
    for task_name in list_of_tasks:
        for actions in list_of_actions:
            asyncio.run(main(10, len(actions), actions))


