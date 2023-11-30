import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import jsonlines
from tqdm import tqdm
from datasets import load_dataset
# from utils.callbacks import Iteratorize, Stream
# from utils.prompter import Prompter

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
    lora_weights: str = "/content/drive/MyDrive/out/first/checkpoint-30/",
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

    # model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    # model = PeftModel.from_pretrained(
    #     model,
    #     lora_weights,
    #     torch_dtype=torch.float16,
    # )
    model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id  # unk
    model.config.bos_token_id = tokenizer.bos_token_id  # <s>
    model.config.eos_token_id = tokenizer.eos_token_id  # </s>

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    data = []
    with jsonlines.open('./mbpp_test.jsonl') as reader:
        for pred_item in tqdm(reader):
            data.append(pred_item)

    def evaluate(
        input,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=512,
        **kwargs,
    ):
        # few-shot시 사용
        prompt = input + "\nAnswer:"

        # SFT시 사용
        # prompt = prompter.generate_prompt(input=input)
        inputs = tokenizer(prompt,
                            padding=False,
                            # truncation = True,
                            # max_length=1024,
                            return_tensors="pt",
                            )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        device = model.device
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
            # "max_length" : max_new_tokens
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


    for row in tqdm(data[2:]):
        instruction = row['description']
        result = evaluate(instruction)
        row['pred'] = result
    with jsonlines.open('/content/drive/MyDrive/out/first/predict_check_zeroshot_512_mbpp_QAformat.jsonl', mode='w') as writer:
        for prefix in tqdm(data[2:]):
            writer.write(prefix)

if __name__ == "__main__":
    fire.Fire(main)