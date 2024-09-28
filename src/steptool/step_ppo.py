# StepTool with PPO

import json
import time
from tqdm import tqdm
import os
import torch
from peft import LoraConfig

from argparse import ArgumentParser
from transformers import AutoTokenizer
from accelerate import Accelerator
from datasets import load_dataset

from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)
from step_ppotrainer import StepPPOTrainer
import wandb
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"]="false"

class StepToolPPOTrain():
    @staticmethod
    def parse_args():
        parser = ArgumentParser()
        parser.add_argument('--config_path', default="config/dpo-test.json", type=str, required=True, help='Path to the config file')
        parser.add_argument('--model_path', default="ToolBench/ToolLLaMA-2-7b-v2", type=str, help='Path to the model')
        parser.add_argument('--data_file', required=True, type=str, help='Path to the data file')
        parser.add_argument('--model_type', default="ToolLlama", type=str, help='Type of the model')
        parser.add_argument('--epochs', default=5, type=int, help='Number of epochs to train')
        parser.add_argument('--max_length', default=1024, type=int, help='Max length of the input')
        parser.add_argument('--max_context_len', default=4096, type=int, help='Max context length')
        parser.add_argument('--max_response_len', default=1200, type=int, help='Max response length')
        return parser.parse_args()

    def __init__(self, args):
        self.config_path = args.config_path
        self.model_path = args.model_path
        self.data_file = args.data_file
        self.max_length = args.max_length
        self.epochs = args.epochs
        self.max_length = args.max_length
        self.max_context_len = args.max_context_len
        self.max_response_len = args.max_response_len
        wandb_project = "StepTool_PPO"
        wandb_run_name = f"{args.model_type}"
        wandb.init(project=wandb_project, name=wandb_run_name)


    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
        )

    # Build step-grained input
    def formatting_func(self, examples):
        input_text = eval(examples["prompt"])
        response_text = eval(examples["response"])
        query_ids_list = []
        frag_mask_list = []
        for in_text, res_text in zip(input_text[:-1], response_text[:-1]):  # build the step-grained frag_mask
            in_text_ids = self.tokenizer.encode(in_text, return_tensors='pt').squeeze(0)
            res_text_ids = self.tokenizer.encode(res_text, return_tensors='pt').squeeze(0)
            frag_mask = torch.cat([torch.zeros_like(in_text_ids),torch.ones_like(res_text_ids)])
            query_ids_list.append(in_text_ids)
            query_ids_list.append(res_text_ids)
            frag_mask_list.append(frag_mask)
            
        in_text_ids = self.tokenizer.encode(input_text[-1], return_tensors='pt').squeeze(0)
        frag_mask = torch.zeros_like(in_text_ids)
        query_ids_list.append(in_text_ids)
        frag_mask_list.append(frag_mask)

        examples["query"] = torch.cat(query_ids_list)
        while len(examples["query"]) > self.max_context_len:
            examples["query"] = examples["query"][-self.max_context_len:]
        
        tmp_frag_mask = torch.cat(frag_mask_list)
        if len(tmp_frag_mask) > self.max_context_len:
            tmp_frag_mask = tmp_frag_mask[-self.max_context_len:]

        examples['response'] = self.tokenizer.encode(response_text[-1], return_tensors='pt').squeeze(0)
        if len(examples['response']) > self.max_response_len:
            examples['response'] = examples['response'][:self.max_response_len]
        
        examples['frag_mask'] = torch.cat([tmp_frag_mask, torch.ones_like(examples['response'])])
        examples["label"] = torch.tensor(eval(examples["reward"]), dtype=torch.float16)
        return examples
    
    def train(self, epochs: int = 1):
        base_dir = os.path.join('ckpts/', f'steptool_{args.model_type}'+str(int(time.time())))

        batch_steps = 0

        for epoch in range(epochs):
            print(f"==========================Epoch {epoch}==========================")
            for batch_id, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
                batch_steps += 1
                query_tensors_list, response_tensors_list = batch['query'], batch['response']
                frag_mask_list = batch['frag_mask']
                rewards_list = batch['label']
                stats = self.ppo_trainer.step(query_tensors_list, response_tensors_list, rewards_list, frag_mask_list)
                final_rewards_list = [rewards[-1] for rewards in rewards_list]
                
                self.ppo_trainer.log_stats(stats, batch, final_rewards_list, columns_to_log=[])

                if batch_steps % 100 == 0:
                    os.makedirs(base_dir, exist_ok=True)
                    self.ppo_trainer.save_pretrained(os.path.join(base_dir, f'batch-{batch_steps}'))
            os.makedirs(base_dir, exist_ok=True)
            self.ppo_trainer.save_pretrained(os.path.join(base_dir, f'epoch-{epoch}'))
                

    def run(self):
        set_seed(2024)
        
        with open(self.config_path, 'r') as config_f:
            config = json.load(config_f)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, 
                                                       device_map= {"": Accelerator().process_index})
        dataset = load_dataset('csv', data_files=self.data_file, delimiter='\t')

        peft_kwargs = config.get('peft_kwargs', {})
        peft_config = LoraConfig(**peft_kwargs)
        
        formatted_dataset = dataset.map(self.formatting_func, batched=False, load_from_cache_file=False)
        formatted_dataset.set_format(type="torch")
        train_dataset = formatted_dataset["train"]
        
        ppo_kwargs = config.get('ppo_kwargs', {})
        ppo_config = PPOConfig(**ppo_kwargs)

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            device_map="auto", 
            peft_config=peft_config, 
            torch_dtype=torch.bfloat16,
        )

        self.print_trainable_parameters(model)
        
        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            model.config.pad_token_id = model.config.eos_token_id
        
        self.ppo_trainer = StepPPOTrainer(
            config=ppo_config,
            model=model,
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator
        )

        self.train(epochs=args.epochs)


if __name__ == "__main__":
    args = StepToolPPOTrain.parse_args()
    StepToolPPOTrain = StepToolPPOTrain(args)
    StepToolPPOTrain.run()