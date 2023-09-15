import torch
import torch.nn as nn 
import mlflow
class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock,self).__init__()
        self.conv= nn.Conv2d(in_channels,out_channels,kernel_size=5,padding='same')
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.relu(x)
        return x 


class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = ConvBlock(1,16)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = ConvBlock(16,32)
        self.conv3 = ConvBlock(32,64)
        self.conv4 = ConvBlock(64,32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192,1024)
        self.fc2= nn.Linear(1024,4)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x 
