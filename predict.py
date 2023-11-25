import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import jsonlines
from tqdm import tqdm
from datasets import load_dataset
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "microsoft/phi-1_5",
    lora_weights: str = "./out/first/",
    prompt_template: str = "no_instruction",  # The prompt template to use, will default to alpaca.
):
    # base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    # 이 부분 신경써야함
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    # if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    # elif device == "mps":
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    # else:
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model, device_map={"": device}, low_cpu_mem_usage=True
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #     )
    model.resize_token_embeddings(len(tokenizer))
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id  # unk
    model.config.bos_token_id = tokenizer.bos_token_id  # <s>
    model.config.eos_token_id = tokenizer.eos_token_id  # </s>
    

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        input,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=1024,
        **kwargs,
    ):  
        # few-shot시 사용
        # prompt = "### Input:\n" + origin_data[1668]['description'] + "\n\n### Response:" + data[0]['label'] + "\n" + "### Input:\n" + origin_data[3381]['description'] + "\n\n### Response:" + data[1]['label'] + "\n" + input
        
        # SFT시 사용
        prompt = prompter.generate_prompt(input=input)
        inputs = tokenizer(prompt, 
                            padding="max_length",
                            truncation = True,
                            max_length=1024,
                            return_tensors="pt",
                            )
        
                            
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=generate_params['input_ids'],
                attention_mask=generate_params['attention_mask'],
                generation_config=generate_params['generation_config'],
                # do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                # max_length=generate_params['max_length'],
                max_new_tokens=generate_params['max_new_tokens'],
            )
        output = tokenizer.decode(generation_output['sequences'][0], skip_special_tokens=True)
        return output

    data = []
    origin_data = load_dataset('codeparrot/apps')['test']
    with jsonlines.open('/root/eunki/baseline/data/codecontest/ver_2/train_ver1_test.jsonl') as reader:
        for pred_item in tqdm(reader):
            data.append(pred_item)
    for row in tqdm(data):
        # prompt = row['prompt']
        task_id = int(row['task_id'])
        prompt = origin_data[task_id]['description']
        # instruction = prompt + row['completion']
        instruction = prompt
        result = evaluate(instruction)
        row['pred'] = result
    with jsonlines.open('./pred_1epochdata.jsonl', mode='w') as writer:
        for prefix in tqdm(data):
            writer.write(prefix)

if __name__ == "__main__":
    fire.Fire(main)