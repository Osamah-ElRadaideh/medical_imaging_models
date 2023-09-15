import torch
import numpy as np
from dataset import Chase
import cv2
import lazy_dataset
from utils import collate
from model import Unet
from tqdm import tqdm
from sacred import Experiment
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from random import random
import torch.nn.functional as F

ex = Experiment('retinal vessel', save_git_info=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
db = Chase()
sw = SummaryWriter()
print(f'device set to {device}')

@ex.config
def defaults():
    lr = 1e-3
    batch_size=2
    steps_per_eval = 20
    num_epochs = 200
    use_fp16 = True


def prepare_example(example):
    img_path = example['image_path']
    if random() < 0.5:
        mask_path = example['1st_target']
    else:
        mask_path = example['2nd_target']
    img = cv2.imread(img_path,1).astype(np.float32) / 255.0
    mask = cv2.imread(mask_path,0).astype(np.float32) / 255.0
    example['image'] = img
    example['mask'] = mask
    return example

def prepare_dataset(dataset,batch_size=16):
    dataset = dataset.shuffle()
    dataset = dataset.map(prepare_example)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(collate)
    return dataset

    



@ex.automain
def main(batch_size,lr,steps_per_eval,num_epochs,use_fp16):
    steps = 0
    ds = db.get_dataset('training_set')
    t_ds = ds[0:18]
    v_ds = ds[18:]
    t_ds = lazy_dataset.new(t_ds)
    v_ds = lazy_dataset.new(v_ds)
    model = Unet().to(device=device)
    scaler = GradScaler()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    min_val = 10e7
    running_loss = 0 

    for epoch in range(num_epochs):
        train_ds = prepare_dataset(t_ds, batch_size)
        valid_ds = prepare_dataset(v_ds,batch_size=1)
        model.train()
        epoch_loss = 0
        for index,batch in enumerate(tqdm(train_ds)):
            optimizer.zero_grad()
            images = batch['image']
            images = torch.tensor(np.array(images)).to(device)
            targets = torch.tensor(np.array(batch['mask'])).to(device=device)
            targets.shape
       
            with autocast(enabled=use_fp16):
                outputs = model(images)
                loss = loss_function(outputs.squeeze(dim=1),targets[:,:,:outputs.shape[-1]])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            epoch_loss += loss.item()
            running_loss += loss.item()
            
            if steps % steps_per_eval == 0:
                sw.add_images("training/target_mask", targets[:,None,:,:].cpu(),global_step=steps)
                sw.add_images("training/estimated_mask", F.sigmoid(outputs).cpu().detach()
                              ,global_step=steps)
                model.eval()
                with torch.no_grad():
                    for batch in tqdm(valid_ds):
                        valid_images = batch['image']
                        valid_images = torch.tensor(np.array(valid_images)).to(device)
                        valid_targets = torch.tensor(np.array(batch['mask'])).to(device=device)
                        valid_outputs = model(valid_images)
                        v_loss = loss_function(valid_outputs.squeeze(dim=1),valid_targets[:,:,:outputs.shape[-1]])

                print(f'training loss after {steps} batches  {running_loss/ (steps + 1)}')
                sw.add_scalar("training/running_loss",running_loss/(steps + 1),steps)

                print(f'validation loss after {steps} batches: {v_loss.item()}')
                sw.add_scalar("validation/loss",v_loss.item(), steps)
                sw.add_image("validation/target_mask", valid_targets,dataformats='CHW',global_step=steps)
                sw.add_image("validation/estimated_mask", F.sigmoid(valid_outputs).squeeze(dim=1),dataformats='CHW',global_step=steps)                

                if v_loss.item() < min_val:
                    min_val = v_loss.item()
                    torch.save(model.state_dict(), 'ckpt_best_loss.pth')
                torch.save(model.state_dict(), 'ckpt_latest.pth')
            steps +=1
            model.train()
        print(f'epoch {epoch} loss: {epoch_loss / len(train_ds) :.2f}')
        sw.add_scalar('training/epoch_loss', epoch_loss, epoch)

if __name__== '__main__':
    main()


