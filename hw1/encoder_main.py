"""
   encoder_main.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse

from encoder_model import EncModel, plot_hidden

from encoder import star16, ch34


# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target',type=str,default='input',help='input, star16 or ch34')
parser.add_argument('--dim',type=int,default=9,help='input dimension')
parser.add_argument('--plot',default=False,action='store_true',help='show intermediate plots')
parser.add_argument('--epochs',type=int, default=1000000, help='max epochs')
parser.add_argument('--stop',type=float, default=0.02, help='loss to stop at')
parser.add_argument('--lr', type=float, default=0.4, help='learning rate')
parser.add_argument('--mom',type=float, default=0.9, help='momentum')
parser.add_argument('--init',type=float, default=0.0001, help='initial weights')
parser.add_argument('--cuda',default=False,action='store_true',help='use cuda')
args = parser.parse_args()

# choose CPU or CUDA
if args.cuda:
    device = 'cuda'
else:
    device = 'cpu'

# load specified target values
if args.target == 'input':
    target = torch.eye(args.dim)
elif args.target == 'star16':
    target = star16
elif args.target == 'ch34':
    target = ch34
else:
    print('Unknown target: %s' %args.target )
    exit()
    
num_in  = target.size()[0]
num_out = target.size()[1]

print('rows:%d, cols:%d' %(num_in,num_out))

# input is one-hot with same number of rows as target
input = torch.eye(num_in)

print(input[0][0])
# print(input[0][1])
# print(input[0][2])

# print(input[1][0])
# print(input[1][1])
# print(input[1][2])

# print(input[2][0])
# print(input[2][1])
# print(input[2][2])

xor_dataset  = torch.utils.data.TensorDataset(input,target)
train_loader = torch.utils.data.DataLoader(xor_dataset,batch_size=num_in)

# create neural network according to model specification
net = EncModel(num_in,2,num_out).to(device) # CPU or GPU

# initialize weights, but set biases to zero
net.in_hid.weight.data.normal_(0,args.init)
net.hid_out.weight.data.normal_(0,args.init)
net.in_hid.bias.data.zero_()
net.hid_out.bias.data.zero_()

# SGD optimizer
optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=args.mom)

# plot only at selected epochs
def plot_epoch( epoch ):
    return epoch in [50,100,150,200,300,500,700,1000,
                     1500,2000,3000,5000,7000,10000,
                     15000,20000,30000,50000,70000,100000,
                     150000,200000,300000,500000,700000,1000000]

loss = 1.0
epoch = 0
while epoch < args.epochs and loss > args.stop:
    epoch = epoch+1
    for batch_id, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # zero the gradients
        output = net(data)    # apply network
        loss = F.binary_cross_entropy(output,target)
        loss.backward()       # compute gradients
        optimizer.step()      # update weights
        if epoch % 10 == 0:
            print('ep%3d: loss = %7.4f' % (epoch, loss.item()))
        if args.plot and plot_epoch( epoch ):
            plot_hidden(net)
            plt.show()

plot_hidden(net)
plt.show()
