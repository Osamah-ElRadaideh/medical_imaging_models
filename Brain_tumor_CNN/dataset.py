import json
from pathlib import Path

class BrainTumor():
    def __init__(self):
        pass
    
    def get_item(self,index):
        return self.dataset[index]
    def get_dataset(self,subset):
        self.dataset = load_json(f'D:\\Datasets\\brain-tumor\\{subset}.json')
        return self.dataset


def load_json(path):
    with open(path) as file:
        loaded = json.load(file)
    return loaded