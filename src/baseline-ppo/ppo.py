# PPO (Final Reward)

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
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)

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


class PPOTrain():
    @staticmethod
    def parse_args():
        parser = ArgumentParser()
        parser.add_argument('--config_path', default="config/dpo-test.json", type=str, required=True, help='Path to the config file')
        parser.add_argument('--model_path', default="ToolBench/ToolLLaMA-2-7b-v2", type=str, help='Path to the model')
        parser.add_argument('--data_file', required=True, type=str, help='Path to the data file')
        parser.add_argument('--model_type', default="ToolLlama", type=str, help='Type of the model')
        parser.add_argument('--epochs', default=3, type=int, help='Number of epochs to train')
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
        wandb_project = "baseline-PPO"
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

    def formatting_func(self, examples):
        input_text = examples["prompt"]
        examples["query"] = self.tokenizer.encode(input_text, return_tensors='pt').squeeze(0)

        max_context_len = 4096
        max_response_len = 1200
        while len(examples["query"]) > max_context_len:
            examples["query"] = examples["query"][-max_context_len:]
        

        examples['response'] = self.tokenizer.encode(examples["response"], return_tensors='pt').squeeze(0)
        if len(examples['response']) > max_response_len:
            examples['response'] = examples['response'][:self.max_response_len]
        examples["label"] = torch.tensor(eval(examples["reward"])[-1], dtype=torch.float16)
        return examples
    
    def train(self, epochs: int = 1):
        base_dir = os.path.join('ckpts/', f'baseline-ppo_'+str(int(time.time())))

        batch_steps = 0
        for epoch in range(epochs):
            print(f"==========================Epoch {epoch}==========================")
 
            for batch_id, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
                batch_steps += 1
                query_tensors, response_tensors = batch['query'], batch['response']
                rewards = batch['label']
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                self.ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=[])

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
            model.config.pad_token_id = model.config.eos_token_id
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator
        )

        self.train(epochs=args.epochs)


if __name__ == "__main__":
    args = PPOTrain.parse_args()
    PPOTrain = PPOTrain(args)
    PPOTrain.run()