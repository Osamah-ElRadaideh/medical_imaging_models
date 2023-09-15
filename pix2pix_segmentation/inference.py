from models import Generator
import torch 
import numpy as np
import cv2
from sacred import Experiment

ex = Experiment('inference',save_git_info=False)

@ex.config
def defaults():
    ckpt_name = 'ckpt_latest_mn.pth'
    output_name = 'output.png'


@ex.automain
def main(ckpt_name, img_path, output_name):
    model = Generator()
    states = torch.load(ckpt_name)   
    model.load_state_dict(states['generator'])

    img = cv2.imread(img_path)
    img = (img / 255.0).astype(np.float32)

    tensored = torch.from_numpy(img).permute(-1, 0, 1).unsqueeze(dim=0)
    model.eval()
    with torch.inference_mode():
        segmented = (model(tensored)).squeeze().numpy()

    cv2.imwrite(output_name,segmented * 255)

