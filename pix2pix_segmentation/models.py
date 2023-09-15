import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def center_crop(tensor,reference_tensor):
    """
    center crops the encoded image to match the shape of the decoded one,
      needed for images not having shape of 2^n x 2^m 
    """
    height = reference_tensor.shape[2]
    width = reference_tensor.shape[3]
    heightStartIdx = ((tensor.shape[2] +1) - height) / 2
    widthStartIdx = ((tensor.shape[3] +1) - width) / 2
    return tensor[:,:,int(heightStartIdx):int(heightStartIdx+height), int(widthStartIdx):int(widthStartIdx+width)]


class down_block(nn.Module):
    #encoding path of the Unet
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(down_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class up_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,stride=2,padding=0):
        super(up_block,self).__init__()
        self.trans = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)
    def forward(self,x):
        x = self.trans(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x 


class conv_block(nn.Module):
    #discriminator's conv block
    def __init__(self, in_channels, out_channels,act='relu',padding = 1):
        super().__init__()
        assert act.lower() in ['relu', 'none'], f'expected activation to be either relu or none got {act} instead.'
        self.act = act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2,padding=padding)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.act =='relu':
            return self.relu(x)
        else:
            return x


class Generator(nn.Module):
    #the Unet model
    def __init__(self,channels=[3,32,64,128,256,512]):
        super(Generator,self).__init__()
        self.conv1 = down_block(channels[0], channels[1])

        self.maxpool= nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = down_block(channels[1], channels[2])

        self.conv3 = down_block(channels[2], channels[3])

        self.conv4 = down_block(channels[3], channels[4])

        self.conv5 = down_block(channels[4], channels[5])

        self.up1 = up_block(channels[5], channels[4])
        self.up2 = up_block(channels[4], channels[3])
        self.up3 = up_block(channels[3], channels[2])
        self.up4 = up_block(channels[2], channels[1])
        self.up5 = up_block(channels[1], channels[0])
        self.upconv1 = down_block(channels[5], channels[4])
        self.upconv2 = down_block(channels[4], channels[3])
        self.upconv3 = down_block(channels[3], channels[2])
        self.upconv4 = down_block(channels[2], channels[1])
        self.upconv5 = nn.Conv2d(channels[1], 1,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out1 = self.conv1(x)
        state1 = self.maxpool(out1)

        out2 = self.conv2(state1)
        state2 = self.maxpool(out2)

        out3 = self.conv3(state2)
        state3 = self.maxpool(out3)

        out4 = self.conv4(state3)
        state4 = self.maxpool(out4)
        out5 = self.conv5(state4)
        state5 = self.up1(out5)

        out6 = self.upconv1(torch.cat([state5, center_crop(out4, state5)], dim=1))
        state6 = self.up2(out6)
        out7 = self.upconv2(torch.cat([state6, center_crop(out3, state6)], dim=1))
        state7 = self.up3(out7)
        out8 = self.upconv3(torch.cat([state7, center_crop(out2, state7)], dim=1))
        state8 = self.up4(out8)
        out9 = self.upconv4(torch.cat([state8, center_crop(out1, state8)], dim=1))
        outputs = self.upconv5(out9)
      
        return self.sigmoid(outputs)
    


class Discriminator(nn.Module):
    #patch gan discriminator
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(1, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 64)
        self.conv4 = conv_block(64, 32)

        self.conv5 = conv_block(32,1, act='None',padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        features = []
        x = self.conv1(x)
        features.append(x)
        x = self.conv2(x)
        features.append(x)
        x = self.conv3(x)
        features.append(x)
        x = self.conv4(x)
        features.append(x)
        x = self.conv5(x)
        return x.squeeze(),features
            



def gen_loss(fake_outs):
    loss = 0.5 * torch.mean((fake_outs - 1) ** 2)

    return loss


def disc_loss(real_outs, fake_outs):
    d_loss = 0.5 * torch.mean((real_outs - 1)**2)
    g_loss = 0.5 * torch.mean(fake_outs ** 2)

    return d_loss + g_loss


def logistic_g_loss(d_generated_outputs: Tensor) -> Tensor:
    """
    Logistic generator loss.
    Assumes input is D(G(x)), or in our case, D(W(z)).
    `disc_outputs` of shape (bs,)
    """
    # d_generated_outputs = torch.sigmoid(d_generated_outputs)
    # loss = torch.log(1 - d_generated_outputs).mean()
    loss = F.softplus(-d_generated_outputs).mean()
    return loss

def logistic_d_loss(d_real_outputs, d_generated_outputs):
    """
    Logistic discriminator loss.
    `d_real_outputs` (bs,): D(x), or in our case D(c)
    `d_generated_outputs` (bs,): D(G(x)), or in our case D(W(z))
    D attempts to push real samples as big as possible (as close to 1.0 as possible), 
    and push fake ones to 0.0
    """
    # d_real_outputs = torch.sigmoid(d_real_outputs)
    # d_generated_outputs = torch.sigmoid(d_generated_outputs)
    # loss = -( torch.log(d_real_outputs) + torch.log(1-d_generated_outputs) )
    # loss = loss.mean()
    term1 = F.softplus(d_generated_outputs) 
    term2 = F.softplus(-d_real_outputs)
    return (term1 + term2).mean()
