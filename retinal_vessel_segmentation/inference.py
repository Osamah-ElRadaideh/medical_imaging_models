from model import Unet
import torch 
import numpy as np
import cv2
import torch.nn.functional as F
from sacred import Experiment

ex = Experiment('inference', save_git_info=False)

@ex.config
def defaults():
    ckpt_path = 'ckpt_best_loss.pth'
    output_path = 'output.png'
@ex.automain
def main(img_path,ckpt_path, output_path):
    model = Unet()
    states = torch.load(ckpt_path)
    model.load_state_dict(states)
    img = cv2.imread(img_path).astype(np.float32) /255.0
    tensored = torch.from_numpy(img)[None, :, : ,:]
    model.eval()
    with torch.inference_mode():
        segmented = F.sigmoid(model(tensored)).squeeze().numpy()
    cv2.imwrite(output_path, segmented * 255)
