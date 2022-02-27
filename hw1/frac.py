"""
   frac.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full2Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full2Net, self).__init__()
        self.lay1 = nn.Linear(2, hid)
        self.lay2 = nn.Linear(hid, hid)
        self.lay3 = nn.Linear(hid, 1)

    def forward(self, input):
        input = input.view(-1, 2)
        self.hid1 = torch.tanh(self.lay1(input))
        self.hid2 = torch.tanh(self.lay2(self.hid1))
        return torch.sigmoid(self.lay3(self.hid2))

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.lay1 = nn.Linear(2, hid)
        self.lay2 = nn.Linear(hid, hid)
        self.lay3 = nn.Linear(hid, hid)
        self.lay4 = nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.lay1(input))
        self.hid2 = torch.tanh(self.lay2(self.hid1))
        self.hid3 = torch.tanh(self.lay3(self.hid2))
        return torch.sigmoid(self.lay4(self.hid3))

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.lay1 = nn.Linear(2, num_hid)
        self.lay2 = nn.Linear(num_hid+2, num_hid)
        self.lay3 = nn.Linear(num_hid+num_hid+2, 1)        

    def forward(self, input):
        self.hid1 = torch.tanh(self.lay1(input))
        layer2Input = torch.cat([input, self.hid1], dim = 1)
        self.hid2 = torch.tanh(self.lay2(layer2Input))
        outInput = torch.cat([self.hid2, self.hid1, input], dim = 1)
        return torch.sigmoid(self.lay3(outInput))
