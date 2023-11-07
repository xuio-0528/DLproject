import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy
import pandas as pd
import torch
import transformers

import jsonlines
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

data = load_dataset("codeparrot/apps")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

print(data)

# torch.set_default_device("cuda")
# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
# inputs = tokenizer('''```python
#     def print_prime(n):
#     """
#     Print all primes between 1 and n
#     """''', return_tensors="pt", return_attention_mask=False)

# outputs = model.generate(**inputs, max_length=200)
# text = tokenizer.batch_decode(outputs)[0]
# print(text)