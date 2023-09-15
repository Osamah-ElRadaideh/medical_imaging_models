import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from dataset import BrainTumor
import cv2
import lazy_dataset
from utils import collate
from model import Network
from einops import einops
from tqdm import tqdm
from pathlib import Path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
db = BrainTumor()

t_ds = db.get_dataset('testing_set')
t_ds = lazy_dataset.new(t_ds)
def prepare_example(example):
    path = example['file_path']
    img = cv2.imread(path,0)
    img = cv2.resize(img,(256,256))
    example['image'] = img
    return example

def prepare_dataset(dataset,batch_size=16):
    dataset = dataset.map(prepare_example)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(collate)
    return dataset


model = Network().to(device=device)
PATH = directory = Path(__file__).parent
PATH = torch.load("D:\\trained_models\\image_classification\\model.pth")
model.load_state_dict(PATH)
model.eval()
with torch.no_grad():
    t_ds = prepare_dataset(t_ds[350:],batch_size=1)
    correct = 0
    for batch in tqdm(t_ds):
        images = batch['image']
        images = torch.Tensor(einops.rearrange(images,'b h w -> b 1 h w ')).to(device=device)
        targets = torch.Tensor(batch['target_label']).to(device=device)
        outputs = model(images)
        if torch.argmax(outputs).item() == targets.item():
                correct += 1
    print(f'testing done...')
    print(f'model accuracy on testing set: {correct * 100 / len(t_ds)}')

