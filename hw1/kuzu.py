"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)

    def forward(self, input):
        output = self.fc1(input.view(-1, 28*28))
        return F.log_softmax(output, dim=1)
        

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.fc1 = nn.Linear(28*28, 250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, input):
        input = input.view(-1, 28*28) 
        output = torch.tanh(self.fc1(input))
        output = self.fc2(torch.tanh(output))
        return F.log_softmax(output, dim=1)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1=nn.Conv2d(in_channels = 1, out_channels = 64,kernel_size = 5, padding=2) 
        self.max_pool=nn.MaxPool2d(2,2) 
        self.conv2=nn.Conv2d(in_channels =64,out_channels = 24,kernel_size = 5, padding=2) 
        self.fc_layer_1=nn.Linear(1176,230)
        self.fc_layer_2=nn.Linear(230,10)


    def forward(self, input):
        input = F.relu(self.conv1(input))
        input = self.max_pool(input)
        input = F.relu(self.conv2(input))
        input = self.max_pool(input)
        input = input.view(input.size(0), -1)  #flattening the inputs. 
        input = F.relu(self.fc_layer_1(input))
        input = self.fc_layer_2(input)
        input = F.log_softmax(input, dim=1)  
        return input  
