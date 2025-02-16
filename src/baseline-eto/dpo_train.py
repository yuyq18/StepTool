import torch
import transformers
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from datasets import load_dataset
from transformers.integrations import deepspeed
from trl import (
    DPOTrainer,
    DPOConfig
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

@dataclass
class TrainingArguments(DPOConfig):
    beta: float = field(default=0.2, metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper."})
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Expanded maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    
@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"

class DPOTrain():

    def __init__(self):
        pass

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
    
    def run(self):
        global local_rank

        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
        )
        model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

        device_map = "auto"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

        # train_dataset = self.get_dpo_dataset(self.data_file)
        dataset = load_dataset('csv', data_files=data_args.data_path, delimiter='\t')
        print(dataset.keys())
        train_val = dataset["train"].train_test_split(
            test_size=0.02, shuffle=True, seed=2024
        )
        train_dataset = train_val["train"]
        val_dataset = train_val["test"]

        # Set RoPE scaling factor
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            rope_scaling = {
                "factor": 2.0,
                "type": "linear"
            },
            use_cache = False
        )
        model_load_kwargs = {
            'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config = model_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **model_load_kwargs
        )
        
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        self.print_trainable_parameters(model)
        
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        dpo_trainer.train()
        dpo_trainer.save_model()


if __name__ == "__main__":
    DPOTrain_ = DPOTrain()
    DPOTrain_.run()