#!/usr/bin/env python
# coding=utf-8
from openai import OpenAI
from termcolor import colored
import time
import traceback

from toolbench.utils import process_system_message
from toolbench.inference.utils import react_parser, react_deparser

def completion_request(key, base_url, prompt,
                        model="qwen2-sft-mix", process_id=0, **args):
    json_data = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 1024,
        "stop": "<|endoftext|>",
        "echo": False,
        "extra_body": {
            "truncate_prompt_tokens": 7100, # 7680
        },
        **args
    }
    try:
        client = OpenAI(base_url=base_url, api_key=key)
        vllm_response = client.completions.create(**json_data)
        json_data = vllm_response.dict()
        return json_data

    except Exception as e:
        print("Unable to generate ChatCompletion response")
        traceback.print_exc()
        import pdb;  pdb.set_trace()
        return {"error": str(e), "total_tokens": 0}

class Qwen2Model:
    def __init__(
            self, 
            model="qwen2-sft-mix",
            openai_key="", 
            base_url=None) -> None:
        super().__init__()
        self.model = model
        self.conversation_history = []
        self.openai_key = openai_key
        self.base_url = base_url
        self.time = time.time()
        self.TRY_TIME = 6
        
    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "tool_calls" in message.keys():
                print_obj = print_obj + f"tool_calls: {message['tool_calls']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def get_prompt(self, conversations, tools):
        start_header_id = "<|im_start|>"
        end_header_id = "<|im_end|>"
        nl_tokens = '\n'

        if tools != []:
            functions = [tool['function'] for tool in tools]

        prompt = ''
        for conv in conversations:
            role = conv['role']  # llama3  roles=("system", "user", "function", "assistant") 不需要转化
            value = conv['content']
            if role == "system" and tools != []:
                value = process_system_message(value, functions)
            if role == "tool":
                role = "function"
            if role == "assistant":
                value = react_deparser(conv['content'], conv['tool_calls'][0]['function']["name"], conv['tool_calls'][0]['function']['arguments'])
            prompt += start_header_id + role + nl_tokens + value + end_header_id + nl_tokens
        
        prompt += start_header_id + 'assistant' + nl_tokens
        return prompt

    def parse(self, tools, process_id, **args):
        self.time = time.time()
        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(15)
            prompt = self.get_prompt(self.conversation_history, tools)
            # breakpoint()
            # if self.conversation_history[-1]['role'] == "tool":
            #     obs = self.conversation_history[-1]['content']
            #     if "Function executing" in obs:
            #         breakpoint()
            #     elif "No such function name:" in obs:
                    # breakpoint()
            response = completion_request(self.openai_key, self.base_url, prompt,
                                          model=self.model, process_id=process_id, **args)
            # chat_response = chat_completion_request(self.openai_key, self.base_url, self.conversation_history, 
            #                                         tools=tools, process_id=process_id, model=self.model, **args)
            try:
                total_tokens = response['usage']['total_tokens']
                message = response["choices"][0]["text"]
                # chat_message = chat_response["choices"][0]["message"]['content']
                # if message != chat_message:
                #     breakpoint()
                if process_id == 0:
                    print(f"[process({process_id})]total tokens: {total_tokens}")

                thought, action, action_input = react_parser(message)
                message = {
                    "role": "assistant",
                    "content": thought,
                    "tool_calls": [
                        {
                            "id": 0,  # 可能在conver_to_answer_format处有bug
                            "function": {
                                "name": action,
                                "arguments": action_input 
                            },
                            "type": "function"
                        }
                    ]
                }
                return message, 0, total_tokens
            except BaseException as e:
                print(f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
                traceback.print_exc()
                if response is not None:
                    print(f"[process({process_id})]OpenAI return: {response}")
            
        return {"role": "assistant", "content": str(response)}, -1, 0


if __name__ == "__main__":
    # can accept all huggingface LlamaModel family
    llm = Qwen2Model(
        model="qwen2-sft-mix",
        openai_key="EMPTY", 
        base_url="http://127.0.0.1:8086/v1/"
    )
    messages = [
        {'role': 'system', 'content': 'You are AutoGPT, you can use many tools(functions) to do the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is irreversible, you can\'t go back to one of the former state, if you want to restart the task, say "I give up and restart".\n2.All the thought is short, at most in 5 sentence.\n3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.\nLet\'s Begin!\nTask description: You should use functions to help handle the real time user querys. Remember:\n1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user,If you can\'t handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.\n2.Do not use origin tool names, use only subfunctions\' names.\nYou have access of the following tools:\n1.weather\n'}, 
        {'role': 'user', 'content': "\nWhat's the weather like in San Francisco, Tokyo, and Paris?\nBegin!\n"},
        {'role': 'assisstant', 'content': "Thought: I need to call the \"get_current_weather\" function with the argument \"location\" set to \"San Francisco\" in order to retrieve the current weather in San Francisco. This will help me provide the user with the requested information about the weather in San Francisco, Tokyo, and Paris.\nAction: get_current_weather\nAction Input: {'location': 'San Francisco'}"},
        {'role': 'function', 'content': "{\"error\": \"\", \"response\": {\"location\": \"San Francisco\", \"temperature\": \"20°C\", \"weather\": \"Sunny\"}}"}
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    llm.change_messages(messages)
    output, error_code, token_usage = llm.parse(tools=tools, process_id=0)
    print(output)