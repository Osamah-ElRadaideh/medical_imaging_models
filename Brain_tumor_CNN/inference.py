import torch
import cv2
from model import Network
from einops import einops
import torch.nn.functional as F
from sacred import Experiment
from ResNet import ResNet

ex = Experiment('process_image',save_git_info=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@ex.config
def defaults():
      ckpt_name = 'ckpt_best_loss.pth'
      use_resnet=False


@ex.automain
def main(image_path,use_resnet,ckpt_name):
    if use_resnet:
        model = ResNet().to(device)
    else:
        model = Network().to(device=device)
    state = torch.load(ckpt_name)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
            image = cv2.imread(image_path,0)
            image = cv2.resize(image,(256,256))
            image = einops.rearrange(image,'b f -> 1 1 b f')
            image = torch.Tensor(image).to(device=device)
            output = model(image)
            probs = F.softmax(output,dim=1)[0]
            probs = torch.mul(probs,100)
            print(f'classification probabilities:')
            print(f'glioma: {probs[0]}',f'meningioma: {probs[1]}',f'no tumor: {probs[2]}',f'pituitary tumor: {probs[3]}',sep='\n')
