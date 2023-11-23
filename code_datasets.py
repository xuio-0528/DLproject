from datasets import load_dataset
import json 
from transformers import AutoTokenizer
from tqdm import tqdm

print('시작')
data = load_dataset('codeparrot/apps')
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

train = []
test = []

print('train 시작')

for data_point in tqdm(data['train']):
    try:
      question = data_point['question']
      if len(tokenizer(question).input_ids) > 512:
          continue
      inputs = json.loads(data_point['input_output'])['inputs']
      output = json.loads(data_point['input_output'])['outputs']
      cnt = 0
      for solution in data_point['solutions']:
          if cnt == 5:
              break
          if len(tokenizer(solution).input_ids) > 512:
              continue
          cnt +=1
          train.append({'description' : question, 'solution' : solution, 'input' : inputs, 'output' : output})
    except:
        continue

with open("apps_train.jsonl" , encoding= "utf-8",mode="w") as file:
  for line in train:
    file.write(json.dumps(line))
    file.write('\n')

print('test 시작')

for data_point in tqdm(data['test']):
    try:
      question = data_point['question']
      if len(tokenizer(question).input_ids) > 512:
          continue
      inputs = data_point['input_output']['inputs']
      output = data_point['input_output']['outpus']
      cnt = 0
      test.append({'description' : question, 'solution' : "", 'input' : inputs, 'output' : output})
    except:
      continue

with open("apps_test.jsonl" , encoding= "utf-8",mode="w") as file:
    for line in test:
        file.write(json.dumps(line))
        file.write('\n')
    
print('완료')