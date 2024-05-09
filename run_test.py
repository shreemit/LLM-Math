import os
from dotenv import load_dotenv

import pandas as pd 
from tqdm import tqdm
import time
from openai import OpenAI
import os
import google.generativeai as genai
import re

TOOLFORMER_CALCULATOR_PROMPT_1Shot = """
Your task is to add calls to a
Calculator API to a piece of text.
The calls should help you get
information required to complete the
text. You can call the API by writing
"[Calculator(expression)]" where
"expression" is the expression to be
computed. Here are some examples of API
calls:
Input: The number in the next term is 18
+ 12 x 3 = 54.
Output: The number in the next term is
18 + 12 x 3 = [Calculator(18 + 12 * 3)]
54.
"""

class LLM:
    def __init__(self, open_ai_api_key = None, google_api_key = None):
        self.api_key = open_ai_api_key
        if open_ai_api_key is not None:
            self.client = OpenAI(api_key=open_ai_api_key)
        if google_api_key is not None:
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel('gemini-pro')

    def chatgpt_completion(self, prompt_text):
        messages = [
            { "role": "user", "content": prompt_text },
        ]
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=1000,
            top_p=1,)
        response_text = response.choices[0].message.content
        return response_text
    
    def gemini_completion(self, prompt_text):
        response = self.model.generate_content(prompt_text)
        return response.text

class Utils:
    @staticmethod
    def evaluate_single(result, answer):
        try:
            _ = float(result)
            if float(result) == float(answer):
                return 1
            else:
                return 0
        except:
            return 0

    @staticmethod
    def parse_zeroshot_chatgpt_output(chatgpt_output):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
            
        if '"' in chatgpt_output:
            chatgpt_output = chatgpt_output.split('"')[1]

        if "=" in chatgpt_output:
            chatgpt_output = chatgpt_output.split("=")[1]

        words = chatgpt_output.split()
        for word in words:
            if is_number(word):
                return word
        return float("inf")
    
    @staticmethod
    def calculator(chatgpt_output):
        if '%' in chatgpt_output:
            chatgpt_output = chatgpt_output.replace('%', '/100')
        if '$' in chatgpt_output:
            chatgpt_output = chatgpt_output.replace('$', '')

        try:
            # Evaluate the arithmetic expression
            result = eval(chatgpt_output)
        except Exception as e:
            return f"Error evaluating expression: {e}"
        
        return result
    

class SpeedLimitTimer:
    def __init__(self, second_per_step=3.1):
        self.record_time = time.time()
        self.second_per_step = second_per_step

    def step(self):
        time_div = time.time() - self.record_time
        if time_div <= self.second_per_step:
            time.sleep(self.second_per_step - time_div)
        self.record_time = time.time()

    def sleep(self, s):
        time.sleep(s)

class ExperimentRunner:
    def __init__(self, llm):
        self.llm = llm
        self.timer = SpeedLimitTimer(second_per_step=3.1)

    def zero_shot_prompt_chatgpt(self, model, test_dataset, prompt):
        data_list = []
        print("model", model)
        for idx, data in tqdm(test_dataset.iterrows()): 
            question = data["Operation"] 
            answer = data["Result"]
            operator = data["Operator"] 
            num_Digits = data["Num_Digits"]

            
            example_prompt =  prompt + " " + question 
            if 'gpt' in model:
                response = self.llm.chatgpt_completion(example_prompt)
            elif 'gemini' in model:
                response = self.llm.gemini_completion(example_prompt)
            else:
                raise ValueError("Invalid model")
            result = Utils.parse_zeroshot_chatgpt_output(response)

            correct = Utils.evaluate_single(result, answer)

            data_list.append({
                'Question': question,
                'Answer': answer,
                'Operator': operator,
                'Num_Digits': num_Digits,
                'Predicted': result,
                'Correct': correct
            })
            self.timer.step()

        df = pd.DataFrame(data_list)
        return df

    def toolformer_prompt_chatgpt(self, model, test_dataset, tool_prompt, context):
        data_list = []
        for idx, data in tqdm(test_dataset.iterrows()):
            question = data["Operation"] 
            answer = data["Result"]
            operator = data["Operator"] 
            num_Digits = data["Num_Digits"]

            example_prompt =  tool_prompt + context + " " + question 
            try:
                if 'gpt' in model:
                    response = self.llm.chatgpt_completion(example_prompt)
                elif 'gemini' in model:
                    response = self.llm.gemini_completion(example_prompt)
                else:
                    raise ValueError("Invalid model")
                calculate_out = response.split("[Calculator(")[1].split(")")[0]
                result = Utils.calculator(calculate_out)
            except:
                result =  float("inf")

            correct = Utils.evaluate_single(result, answer)

            data_list.append({
                'Question': question,
                'Answer': answer,
                'Operator': operator,
                'Num_Digits': num_Digits,
                'Predicted': result,
                'Correct': correct
            })
            self.timer.step()
            df = pd.DataFrame(data_list)
        return df

    def run_experiment(self, model, dataset, prompt, method, use_toolformer):
        if method == 'one-shot':
            if use_toolformer:
                return self.toolformer_prompt_chatgpt(model, dataset, prompt)
            else:
                return self.zero_shot_prompt_chatgpt(model, dataset, prompt)
        elif method == 'four-shot':
            # Implement four-shot method here
            pass
        else:
            print("Invalid method")


if __name__ == "__main__":
    load_dotenv('API_KEYS.env')
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_pro_key = os.getenv("GEMINI_API_KEY")
    llm = LLM(google_api_key = gemini_pro_key)
    runner = ExperimentRunner(llm)
    dataset = pd.read_csv('math_operations.csv')
    prompt = "Give me the ONLY NUMERICAL answer in the LAST LINE IN DOUBLE QUOTES for the given problem. Think throught the problem step by step."
    method = 'one-shot'
    name = 'COT'
    model = 'gemini-pro'
    use_toolformer = False

    results = runner.run_experiment(model, dataset[0:500], prompt, method, use_toolformer)
    print(results.head())
    results.to_csv(f'{name}_{model}_{method}.csv', index=False)