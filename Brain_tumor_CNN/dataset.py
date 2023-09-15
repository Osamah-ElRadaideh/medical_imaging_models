import json
from pathlib import Path

class BrainTumor():
    def __init__(self):
        self.base_path = Path(__file__).parent
  
    
    def get_item(self,index):
        return self.dataset[index]
    def get_dataset(self,subset):
        self.dataset = load_json(f'{self.base_path}\\{subset}.json')
        return self.dataset


def load_json(path):
    with open(path) as file:
        loaded = json.load(file)
    return loaded