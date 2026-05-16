import json
import os
import time
import random
import openai
from openai import OpenAI,AsyncOpenAI
from tqdm import tqdm
import asyncio
from common.utils import read_txt, read_json
from eval_all_round import parse_answer

# API configuration - please set your own API endpoint and key
API_URL = "YOUR_API_URL_HERE"
API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "qwen2.5-7b-instruct"
MODEL_TAG = "Qwen/Qwen2.5-7B-Instruct"
client = OpenAI(base_url=API_URL,
                       api_key=API_KEY,
                       )
async_client = AsyncOpenAI(base_url=API_URL,
                       api_key=API_KEY,
                       )





fewshot_ost_config = read_json("prompt/fewshot_ost_config.json")
fewshot_ost_prompt = read_txt("prompt/fewshot_ost_prompt.txt")


io_steps = []



def cot(user_question):
    question_content = f"Can you answer the following question as accurately as possible? {user_question} \n Explain your answer.Make sure putting the final answer in the form (X) at the end of your response. Let's think step by step."
    content = question_content
    
    
    
    
    messages = [{"role": "user", "content": content}]
    
    
    response = client.chat.completions.create(model=MODEL_TAG, messages=messages, n=1)
    completion=json.loads(response.json())
    return completion



def exchange_messages(user_question,key):
    pure_cots = read_json(f"{MODEL_NAME}/data/cots_{key}.json")["cots"]
    
    
    cots = [pure_cots[4],pure_cots[6],pure_cots[11],pure_cots[12]]
    prefix_string = "These are the solutions to the problem from other agents: "
    for i in range(4):
        agent_response = cots[i]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(user_question)
    question_content = f"Can you answer the following question as accurately as possible? {user_question} \n Explain your answer.Make sure putting the answer in the form (X) at the end of your response."
  
    q = fewshot_ost_config["prompt_template"].format(
                    examples=fewshot_ost_prompt,
                    instruction=question_content)
    a = pure_cots[0]
    messages =  [{"role":"user","content":q},{"role":"assistant","content":a},{"role": "user", "content": prefix_string}]
    
    try:
        response = client.chat.completions.create(model=MODEL_TAG, messages=messages,  n=1)
        completion=json.loads(response.json())
        messages.append({"role": "assistant", "content": completion['choices'][0]['message']['content']})
        
        with open(f'{MODEL_NAME}/results/exchange_messages_{key}.json', 'w') as f:
            json.dump({"messages": messages}, f, indent=4)
    except Exception as e:
        print("retrying due to an error......")
        print(e)
        time.sleep(20)
        return exchange_messages(user_question,key)

    return completion



def exchange_messages_gt_other(qid,gt_ans,other_ans,center_ans,max_solution,solution_cnt):
    solutions = read_json(f'{MODEL_NAME}/results/case_study_id/solution_{qid}.json')
    gt_solutions= solutions[gt_ans]
    other_solutions = solutions[other_ans]
    question = solutions["question"]
    first_round_solution = solutions[center_ans][0]

    
    prefix_string = "These are the solutions to the problem from other agents: "
    for i in range(0,min(solution_cnt,max_solution//2)):
        
        agent_response = gt_solutions[i]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    for i in range(0,max(0,solution_cnt-max_solution//2)):
        
        agent_response = other_solutions[i]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    question_content = f"Can you answer the following question as accurately as possible? {question} \n Explain your answer.Make sure putting the answer in the form (X) at the end of your response."
  
    q = fewshot_ost_config["prompt_template"].format(
                    examples=fewshot_ost_prompt,
                    instruction=question_content)
    a = first_round_solution
    messages =  [{"role":"user","content":q},{"role":"assistant","content":a},{"role": "user", "content": prefix_string}]
    
    try:
        response = client.chat.completions.create(model=MODEL_TAG, messages=messages,  n=1)
        completion=json.loads(response.json())
        messages.append({"role": "assistant", "content": completion['choices'][0]['message']['content']})
        
        
        
    except Exception as e:
        print("retrying due to an error......")
        print(e)
        time.sleep(20)
        return exchange_messages_gt_other(qid,gt_ans,other_ans,center_ans,max_solution,solution_cnt)

    return completion


def exchange_messages_other_gt(qid,gt_ans,other_ans,center_ans,max_solution,solution_cnt):
    solutions = read_json(f'{MODEL_NAME}/results/case_study_id/solution_{qid}.json')
    gt_solutions= solutions[gt_ans]
    other_solutions = solutions[other_ans]
    question = solutions["question"]
    first_round_solution = solutions[center_ans][0]

    
    prefix_string = "These are the solutions to the problem from other agents: "
    for i in range(0,min(solution_cnt,max_solution//2)):
        
        agent_response = other_solutions[i]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    for i in range(0,max(0,solution_cnt-max_solution//2)):
        
        agent_response = gt_solutions[i]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    question_content = f"Can you answer the following question as accurately as possible? {question} \n Explain your answer.Make sure putting the answer in the form (X) at the end of your response."
  
    q = fewshot_ost_config["prompt_template"].format(
                    examples=fewshot_ost_prompt,
                    instruction=question_content)
    a = first_round_solution
    messages =  [{"role":"user","content":q},{"role":"assistant","content":a},{"role": "user", "content": prefix_string}]
    
    try:
        response = client.chat.completions.create(model=MODEL_TAG, messages=messages,  n=1)
        completion=json.loads(response.json())
        messages.append({"role": "assistant", "content": completion['choices'][0]['message']['content']})
        
        
        
    except Exception as e:
        print("retrying due to an error......")
        print(e)
        time.sleep(20)
        return exchange_messages_other_gt(qid,gt_ans,other_ans,center_ans,max_solution,solution_cnt)

    return completion

def exchange_messages_gt_other_alt(qid,gt_ans,other_ans,center_ans,max_solution,solution_cnt):
    solutions = read_json(f'{MODEL_NAME}/results/case_study_id/solution_{qid}.json')
    gt_solutions= solutions[gt_ans]
    other_solutions = solutions[other_ans]
    question = solutions["question"]
    first_round_solution = solutions[center_ans][0]

    
    prefix_string = "These are the solutions to the problem from other agents: "
    for i in range(min(solution_cnt,max_solution)):
        
        if i % 2 ==0:
            idx = i//2
            agent_response = gt_solutions[idx]
        else:
            idx = (i+1)//2
            agent_response = other_solutions[idx]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    question_content = f"Can you answer the following question as accurately as possible? {question} \n Explain your answer.Make sure putting the answer in the form (X) at the end of your response."
  
    q = fewshot_ost_config["prompt_template"].format(
                    examples=fewshot_ost_prompt,
                    instruction=question_content)
    a = first_round_solution
    messages =  [{"role":"user","content":q},{"role":"assistant","content":a},{"role": "user", "content": prefix_string}]
    
    try:
        response = client.chat.completions.create(model=MODEL_TAG, messages=messages,  n=1)
        completion=json.loads(response.json())
        messages.append({"role": "assistant", "content": completion['choices'][0]['message']['content']})
        
        
        
    except Exception as e:
        print("retrying due to an error......")
        print(e)
        time.sleep(20)
        return exchange_messages_gt_other_alt(qid,gt_ans,other_ans,center_ans,max_solution,solution_cnt)

    return completion

def parse_question_answer(tasks, ix):
    question = tasks[ix]['input']
   

    question_content = f"Can you answer the following question as accurately as possible? {question} \n Explain your answer.Make sure putting the answer in the form (X) at the end of your response."
    answer = tasks[ix]['target']
    original_question = question
    question_id = tasks[ix]['question_id']
    return question_content, answer,question_id,original_question




def cot_one_case(qid,pure_question,gt_ans):


   
    import concurrent.futures

    solution_list = []
    ans_list = []



    def get_solution(_):
        completion = cot(pure_question)
        return completion['choices'][0]['message']['content']

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        solution_list = list(tqdm(executor.map(get_solution, range(400)), total=400))

    ans_list = [parse_answer(solution) for solution in solution_list]
    print(f"len(ans_list): {len(ans_list)},len(solution_list): {len(solution_list)}")
    
    solution_dict = {}
    solution_dict["qid"] = qid
    solution_dict["question"] = pure_question
    solution_dict["gt_ans"] = gt_ans
    for i in range(len(ans_list)):
        if ans_list[i] in solution_dict:
            solution_dict[ans_list[i]].append(solution_list[i])
        else:
            solution_dict[ans_list[i]] = [solution_list[i]]

    
    

    if not os.path.exists(f'{MODEL_NAME}/results/case_study_id'):
        os.makedirs(f'{MODEL_NAME}/results/case_study_id')
    with open(f'{MODEL_NAME}/results/case_study_id/solution_{qid}.json', 'w') as f:
        json.dump(solution_dict, f, indent=4)

    ans_dict = {}
    for ans in ans_list:
        if ans in ans_dict:
            ans_dict[ans] += 1
        else:
            ans_dict[ans] = 1
    
    ans_acc = {}
    for ans in ans_dict:
        ans_acc[ans] = ans_dict[ans]/400
    ans_acc = dict(sorted(ans_acc.items(), key=lambda item: item[1],reverse=True))
    print(f"Question {qid}:", ans_acc)
    


while True:

    task_file = f"{MODEL_NAME}/data/case_study_id"
    with open(task_file+".json", "r") as f:
        data = json.load(f)['examples']

    user_input = input().strip()
    if not user_input:
        print("Input cannot be empty. Please try again.")
        continue
 
    if user_input=="1":
       pass
        
    elif user_input=="2":
        pass
        
    elif user_input=="3":
        for i in tqdm(range(250)):
            question, answer,question_id,original_question = parse_question_answer(data,i)
            cot_one_case(question_id,original_question,answer)
    elif user_input=="4":
        for i in range(2,80):
            with open(f"{MODEL_NAME}/results/case_study_id/solution_{i}.json", 'r') as f:
                dict = json.load(f)
                
                str_fre = ""
                count = 0
                for key in dict:
                    if key != "qid" and key != "question" and key != "gt_ans":
                        frequency = len(dict[key]) / 400
                        str_fre += f"{key}：{frequency} "
                        if frequency > 0.3:
                            count += 1
                if count >= 2:
                    print(f"Question {i}, Correct Answer: {dict['gt_ans']}")
                    print(str_fre, "\n")
    
    elif user_input=="5":
        logs = []
        max_solution = 50
        for i in range(5):
            qid = data[i]['question_id']
            gt_ans = data[i]['target']
            other_ans = data[i]['other']
            center_ans = data[i]['zero']
            for cnt in range(1,max_solution+1):
                results = []
                import concurrent.futures

                def get_result(_):
                    return exchange_messages_gt_other(qid, gt_ans, other_ans, center_ans, max_solution, cnt)['choices'][0]['message']['content']
                with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                    results = list(tqdm(executor.map(get_result, range(50)), total=50))
                ans_count = {}
                for result in results:
                    ans = parse_answer(result)
                    if ans in ans_count:
                        ans_count[ans] += 1
                    else:
                        ans_count[ans] = 1
                gt_acc = 0.0
                if gt_ans in ans_count:
                    gt_acc= ans_count[gt_ans]/50
            
                logs.append({"ans_count":ans_count,"gt_acc":gt_acc})
                print(f"results for qid: {qid}, solution count {cnt}, acc:{gt_acc}, {ans_count}")
        for obj in logs:
            print(obj["gt_acc"])
        
        with open(f'{MODEL_NAME}/results/{qid}_gt_other_{max_solution}_results.json', 'w') as f:
            json.dump(logs, f, indent=4)
    elif user_input=="6":
        logs = []
        max_solution = 50
        for i in range(5):
            qid = data[i]['question_id']
            gt_ans = data[i]['target']
            other_ans = data[i]['other']
            center_ans = data[i]['zero']
            for cnt in range(1,max_solution+1):
                results = []
                import concurrent.futures

                def get_result(_):
                    return exchange_messages_other_gt(qid, gt_ans, other_ans, center_ans, max_solution, cnt)['choices'][0]['message']['content']
                with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                    results = list(tqdm(executor.map(get_result, range(50)), total=50))
                ans_count = {}
                for result in results:
                    ans = parse_answer(result)
                    if ans in ans_count:
                        ans_count[ans] += 1
                    else:
                        ans_count[ans] = 1
                gt_acc = 0.0
                if gt_ans in ans_count:
                    gt_acc= ans_count[gt_ans]/50
            
                logs.append({"ans_count":ans_count,"gt_acc":gt_acc})
                print(f"results for qid: {qid}, solution count {cnt}, acc:{gt_acc}, {ans_count}")
        for obj in logs:
            print(obj["gt_acc"])
        with open(f'{MODEL_NAME}/results/{qid}_other_gt_{max_solution}_results.json', 'w') as f:
            json.dump(logs, f, indent=4)
    elif user_input=="7":
        logs = []
        max_solution = 50
        for i in range(5):
            qid = data[i]['question_id']
            gt_ans = data[i]['target']
            other_ans = data[i]['other']
            center_ans = data[i]['zero']
            for cnt in range(1,max_solution+1):
                results = []
                import concurrent.futures

                def get_result(_):
                    return exchange_messages_gt_other_alt(qid, gt_ans, other_ans, center_ans, max_solution, cnt)['choices'][0]['message']['content']
                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                    results = list(tqdm(executor.map(get_result, range(50)), total=50))
                ans_count = {}
                for result in results:
                    ans = parse_answer(result)
                    if ans in ans_count:
                        ans_count[ans] += 1
                    else:
                        ans_count[ans] = 1
                gt_acc = 0.0
                if gt_ans in ans_count:
                    gt_acc= ans_count[gt_ans]/50
            
                logs.append({"ans_count":ans_count,"gt_acc":gt_acc})
                print(f"results for qid: {qid}, solution count {cnt}, acc:{gt_acc}, {ans_count}")
        for obj in logs:
            print(obj["gt_acc"])
        with open(f'{MODEL_NAME}/results/{qid}_gt_other_alt_{max_solution}_results.json', 'w') as f:
            json.dump(logs, f, indent=4)
    elif user_input=="8":
        
        logs = read_json(f'{MODEL_NAME}/results/94_gt_other_50_results.json')
        
        acc_list = []
        for obj in logs:
            acc_list.append(obj["gt_acc"])
        print(f"len = {len(acc_list)}")
        print(f"gt_other:{acc_list}")    
        logs = read_json(f'{MODEL_NAME}/results/94_other_gt_50_results.json')
        
        acc_list = []
        for obj in logs:
            acc_list.append(obj["gt_acc"])
        print(f"other_gt:{acc_list}")    
        logs = read_json(f'{MODEL_NAME}/results/94_gt_other_alt50_results.json')
        
        acc_list = []
        for obj in logs:
            acc_list.append(obj["gt_acc"])
        print(f"gt_other_alt:{acc_list}")    
    elif user_input=="9":
        
        qids = ["49","54","66","112","94"]
        logs = read_json(f'{MODEL_NAME}/results/94_gt_other_50_results.json')
        for qid in qids:
            for i in range(5):
                chunk = logs[i*50:(i+1)*50]
                with open(f'{MODEL_NAME}/results/{qid}_gt_other_50_results.json', 'w') as f:
                    json.dump(chunk, f, indent=4)
