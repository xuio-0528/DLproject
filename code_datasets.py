import torch
from torch.utils.data import Dataset
import json
from accelerate import Accelerator
from datasets import load_dataset

class CodeDataset(Dataset):
    def __init__(self, file_path):
        self.examples = []
        self.data = []

        with open(file_path, "r") as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
        for example in self.examples:
            self.data.append(
                {
                    "input" : self.data['description'],
                    "output" : example["solution"],
                }
            )

    def map(self, func):
        self.data = [func(item) for item in self.data]
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        return self.data[idx]