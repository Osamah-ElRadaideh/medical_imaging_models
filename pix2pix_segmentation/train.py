import numpy as np
import torch
import torch.nn as nn
from models import Generator, Discriminator,gen_loss, disc_loss
from tqdm import tqdm
import cv2
from utils import collate
import lazy_dataset
from sacred import Experiment
from dataset import Kvasir
import copy
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
sw = SummaryWriter()
#load the files
ex = Experiment('pix2pix_segmentation',save_git_info=False)


@ex.config
def defaults():
    batch_size = 8
    d_lr = 0.00005
    g_lr = 0.0001
    steps_per_eval = 100 # after how many steps to evluate the model and add images to the tensorboard
    use_fp16 = True # True for half precision, useful for smaller GPUs
    load_ckpt=True

@ex.capture
def load_img(example):
    img = cv2.imread(example['image_path'])
    mask = cv2.imread(example['mask_path'], 0)
    mask = cv2.resize(mask,(512, 512))
    img = np.flip(img, axis=-1)
    img = (img / 255.0).astype(np.float32)
    img = cv2.resize(img, (512, 512))

    example['image'] = np.transpose(img,(-1, 0, 1))
    example['mask'] = np.round(mask / 255).astype(np.float32)
    return example



@ex.capture
def prepare_dataset(dataset,batch_size):
    if isinstance(dataset,list):
        dataset = lazy_dataset.new(dataset)
    dataset = dataset.map(load_img)
    dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size=batch_size, drop_last=True)
    dataset = dataset.map(collate)
    return dataset

@ex.automain
def main(batch_size,d_lr,g_lr, steps_per_eval,use_fp16,load_ckpt):
    #model hyperparamters
    #per the LSGAN paper, beta1 os set to 0.5
    scaler = torch.cuda.amp.GradScaler()

    db = Kvasir()

    t_ds = db.get_dataset('training_set')
    v_ds = db.get_dataset('validation_set')
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    steps = 0
    path = 'ckpt_latest.pth'
    gen.train()
    disc.train()
    g_optim = torch.optim.Adam(gen.parameters(),lr=g_lr)
    d_optim = torch.optim.Adam(disc.parameters(), lr=d_lr)

    if load_ckpt:
        states = torch.load('ckpt_latest.pth')
        gen.load_state_dict(states['generator'])
        disc.load_state_dict(states['discriminator'])
        g_optim.load_state_dict(states['generator_optimizer'])
        d_optim.load_state_dict(states['discriminator_optimizer'])
        steps = states['steps']
        print('checkpoint loading complete ... ')

    aux_criterion = nn.L1Loss()
    for epoch in range(10000):
        epoch_g_loss = 0
        epoch_d_loss = 0
        train_ds = prepare_dataset(t_ds, batch_size=batch_size)
        valid_ds = prepare_dataset(v_ds, batch_size=1)
        for index,batch in enumerate(tqdm(train_ds)):
            images = batch['image']
            mask = batch['mask']
            images = torch.tensor(np.array(images)).to(device)
            mask = torch.tensor(np.array(mask)).to(device).unsqueeze(dim=1)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                fakes = gen(images)


            #*********************  
            
            # discriminator step

            #*********************
            
            with torch.cuda.amp.autocast(enabled=use_fp16):
                    d_real,_ = disc(mask)
                    with torch.no_grad():
                        d_fake,_ = disc(fakes)
                    loss_d = disc_loss(d_real, d_fake) * 0.5

                    scaler.scale(loss_d).backward()
                    scaler.step(d_optim)
                    scaler.update()
              
                    epoch_d_loss += loss_d.item()
                    d_optim.zero_grad()


            #******************* 
            # generator step
        
            #*******************
            with torch.cuda.amp.autocast(enabled=use_fp16):
                d_real, fm_d = disc(mask)
                fakes = fakes.requires_grad_(True)
                d_fake, fm_g = disc(fakes)
                fm_loss = 0
                for g,d in zip(fm_g,fm_g):

                   fm_loss += torch.mean(torch.abs(g - d))
                loss_g = gen_loss(d_fake) + aux_criterion(fakes, mask) * 100 + fm_loss
                scaler.scale(loss_g).backward()
                scaler.step(g_optim)
                scaler.update()
                epoch_g_loss += loss_g.item()
                g_optim.zero_grad()
                
           
            if steps % steps_per_eval == 0:
                gen.eval()
                disc.eval()
                for batch in tqdm(valid_ds[0:2]):
                    images = batch['image']
                    mask = batch['mask']
                    images = torch.tensor(np.array(images)).to(device)
                    mask = torch.tensor(np.array(mask)).to(device).unsqueeze(dim=1)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=False):
                            generated = gen(images)
                            d_fake, _ = disc(generated)
                            d_real, _ = disc(mask)
                            g_loss = gen_loss(d_fake)
                            d_loss = disc_loss(d_real, d_fake)

            


                    sw.add_scalar("validation/fake_image_prediction",torch.mean(d_fake).item(),steps)
                    sw.add_scalar("validation/real_image_prediction",torch.mean(d_real).item(),steps)
                    sw.add_images("validation/mask_images", mask,steps)
                    sw.add_images("validation/real_images", images,steps)
                    sw.add_images("validation/generated_images", generated,steps)



                
                torch.save({
                    'steps': steps,
                    'generator': gen.state_dict(),
                    'generator_optimizer': g_optim.state_dict(),
                    'discriminator': disc.state_dict(),
                    'discriminator_optimizer': d_optim.state_dict(),
                    'generator_loss': g_loss,
                    'discriminator_loss': d_loss
                    }, path)
                
            steps +=1
            gen.train()
            disc.train()
        sw.add_scalar("training/generator_loss",epoch_g_loss/len(train_ds),epoch)
        sw.add_scalar("training/discriminator_loss",epoch_d_loss/len(train_ds),epoch)
        
  