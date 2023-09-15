import torch
import numpy as np
from dataset import BrainTumor
import cv2
import lazy_dataset
from utils import collate
from model import Network
from einops import einops
from tqdm import tqdm
import torch.nn as nn
import os 
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
db = BrainTumor()

t_ds = db.get_dataset('training_set')
t_ds = lazy_dataset.new(t_ds)
v_ds = db.get_dataset('testing_set')
v_ds = lazy_dataset.new(v_ds)
def prepare_example(example):
    path = example['file_path']
    img = cv2.imread(path,0)
    img = cv2.resize(img,(256,256))
    example['image'] = img
    return example

def prepare_dataset(dataset,batch_size=16):
    dataset = dataset.shuffle()
    dataset = dataset.map(prepare_example)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(collate)
    return dataset

def trainer_function(model,loss_function,optimizer,dataset):
    model.train()
    epoch_loss = 0
    for index,batch in enumerate(tqdm(dataset)):
        images = batch['image']
        images = torch.Tensor(einops.rearrange(images,'b h w -> b 1 h w ')).to(device=device)
        targets = torch.tensor(batch['target_label']).to(device=device)
        outputs = model(images)
        optimizer.zero_grad()
        loss = loss_function(outputs,targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # if (index + 1) % 20 ==0:
        #     print(f'loss after {index+1} steps: {epoch_loss / (index + 1)}')
    return epoch_loss / len(dataset)


def validate(model,loss_function,dataset):
    model.eval()
    epoch_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataset):
            images = batch['image']
            images = torch.Tensor(einops.rearrange(images,'b h w -> b 1 h w ')).to(device=device)
            targets = torch.tensor(batch['target_label']).to(device=device)
            outputs = model(images)
            loss = loss_function(outputs,targets)
            epoch_loss += loss.item()
            if torch.argmax(outputs).item() == targets.item():
                correct += 1
    print(f'validation_accuracy: {correct * 100 / len(dataset)} ')
    return epoch_loss / len(dataset)



model = Network().to(device=device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
directory = Path(__file__).parent
PATH = os.path.join(directory,f'model.pth')
training_loss = 0
valid_loss = 0
min_val = 10
for epoch in range(250):
    train_ds = prepare_dataset(t_ds)
    valid_ds = prepare_dataset(v_ds[:350],batch_size=1)
    train_loss = trainer_function(model,loss_function,optimizer,train_ds)
    validation_loss = validate(model,loss_function,valid_ds)
    print(f'training loss after {epoch + 1} epochs: {train_loss}')
    print(f'validation loss after {epoch + 1} epochs: {validation_loss}')
    if validation_loss < min_val:
        min_val = validation_loss
        torch.save(model.state_dict(), PATH)
    training_loss += train_loss
    valid_loss += validation_loss
print(f'training done ...')
print(f'training loss: {training_loss / 250} ')
print(f'validation loss : {valid_loss / 250}')



