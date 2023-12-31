import os
import sys
from typing import List
import jsonlines
import bitsandbytes as bnb

import numpy as np
import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainerCallback, get_cosine_schedule_with_warmup


# class LogCallback(TrainerCallback):
#     def init(self, state):
#         self.state = state
#     def on_step_end(self, args, state, control, logs = None, logging_steps=1, **kwargs):
#         if state.global_step % logging_steps == 0:
#             # 매 logging_steps 스텝마다 training loss 출력
#             print(f"Step {state.global_step}: Train loss {state.log_history[-1]}")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(
    # model/data params
    base_model: str = "microsoft/phi-1_5",  # the only required argument
    data_path: str = "./",
    output_dir: str = "/content/drive/MyDrive/out/mbpp_first/",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    val_set_size: int = 2000,
    # lora hyperparams
    # lora_r: int = 16,
    lora_r: int =32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ['Wqkv', 'out_proj'],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "dlproject",
    wandb_run_name: str = "first",
    wandb_watch: str = "true",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "nothing",  # The prompt template to use, will default to alpaca.
    seed: int = 42,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # seed 설정
    set_seed(seed)

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # test할 때는 wandb 사용하지 않음
    # use_wandb = None
    wandb.init(project='dlproject')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    global model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"":0},
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"  # Allow batched inference
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            input = data_point["input"],
            label = data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt, add_eos_token=add_eos_token)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                input = data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        # inputs = data_point["input"]
        # targets = data_point["output"]
        # model_inputs = tokenizer(inputs, max_length=cutoff_len, padding="max_length", truncation=True, return_tensors="pt")
        # labels = tokenizer(targets, max_length=cutoff_len, padding="max_length", truncation=True, return_tensors="pt")
        # labels = labels["input_ids"]
        # labels[labels == tokenizer.pad_token_id] = -100
        # model_inputs["labels"] = labels
        return tokenized_full_prompt

    def generate_and_tokenize_prompt_for_test(data_point):
        full_prompt = prompter.generate_prompt(
            input = data_point["input"],
        )
        tokenized_full_prompt = tokenize(full_prompt, add_eos_token=add_eos_token)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                input = data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    # torch.nn.utils.convert_parameters(model.parameters(), dtype=torch.float)
    # model = prepare_model_for_int8_training(model)
    model.resize_token_embeddings(len(tokenizer))
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, config)

    # train_data = CodeDataset(data_path+'apps_train.jsonl')
    # val_data = CodeDataset(data_path + 'apps_test.jsonl')
    train_data = CodeDataset(data_path+'mbpp_train.jsonl')
    val_data = CodeDataset(data_path + 'mbpp_valid.jsonl')

    train_data.map(generate_and_tokenize_prompt)
    val_data.map(generate_and_tokenize_prompt)
    print("data 완료")
    # test_data.map(generate_and_tokenize_prompt)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        # callbacks=[LogCallback],
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            weight_decay=0.01, ## 새롭게 설정
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            logging_dir='./logs',
            optim="adamw_torch",
            lr_scheduler_type = 'cosine',
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=3 if val_set_size > 0 else None,
            save_steps=3,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=1, return_tensors="pt", padding=True
        )
    )


    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    model = model.merge_and_unload()

    # temp=[]
    # predictions = trainer.predict(test_data, max_length=1024)
    # preds = np.argmax(predictions.predictions, axis=-1)
    # for row in preds:
    #     temp.append({"pred":tokenizer.decode(row, skip_special_tokens=True)})
    # with jsonlines.open('./predict_check.jsonl', mode='w') as writer:
    #     for prefix in tqdm(temp):
    #         writer.write(prefix)

    # model.generate()

fire.Fire(train)

