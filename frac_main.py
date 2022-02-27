"""
   frac_main.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.utils.data
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from frac import Full2Net, Full3Net, DenseNet

def train(net, train_loader, optimizer):
    total=0
    correct=0
    for batch_id, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()    # zero the gradients
        output = net(data)       # apply network
        loss = F.binary_cross_entropy(output,target)
        loss.backward()          # compute gradients
        optimizer.step()         # update weights
        pred = (output >= 0.5).float()
        correct += (pred == target).float().sum()
        total += target.size()[0]
        accuracy = 100*correct/total

    if epoch % 100 == 0:
        print('ep:%5d loss: %6.4f acc: %5.2f' %
             (epoch,loss.item(),accuracy))

    return accuracy

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-16, end=16.1, step=0.02, dtype=torch.float32)
    yrange = torch.arange(start=-16, end=16.1, step=0.02, dtype=torch.float32)

    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batch norm, dropout back again

        if layer == 1:
            hid = net.hid1
        elif layer == 2:
            hid = net.hid2
        else:
            hid = net.hid3

        hid = (hid >= 0).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,(hid.cpu()[:,node]).view(yrange.size()[0],xrange.size()[0]), cmap='Wistia', shading='auto')


def graph_output(net):
    xrange = torch.arange(start=-16,end=16.01,step=0.02,dtype=torch.float32)
    yrange = torch.arange(start=-16,end=16.01,step=0.02,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batch norm, dropout back again

        pred = (output >= 0.5).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]),
                       cmap='Wistia', shading='auto')

# command-line arguments
parser = argparse.ArgumentParser()
#parser.add_argument('--layer', type=int, default=1, help='number of hidden layers')
parser.add_argument('--net',type=str,default='full2',help='full2, full3 or dense')
parser.add_argument('--init', type=float,default=0.10, help='initial weight size')
parser.add_argument('--hid', type=int,default='10', help='number of hidden units')
parser.add_argument('--lr', type=float,default=0.01, help='learning rate')
parser.add_argument('--epoch', type=int,default='200000', help='max training epochs')
args = parser.parse_args()

df = pd.read_csv('fractal.csv')

data = torch.tensor(df.values,dtype=torch.float32)

num_input = data.shape[1] - 1

full_input  = data[:,0:num_input]
full_target = data[:,num_input:num_input+1]

train_dataset = torch.utils.data.TensorDataset(full_input,full_target)
train_loader  = torch.utils.data.DataLoader(train_dataset,
                                 batch_size=train_dataset.__len__())

# choose network architecture
if args.net == 'full2':
    net = Full2Net(args.hid)
elif args.net == 'full3':
    net = Full3Net(args.hid)
else:
    net = DenseNet(args.hid)
    
if list(net.parameters()):
    # initialize weight values
    for m in list(net.parameters()):
        m.data.normal_(0,args.init)

    graph_output(net)
    plt.scatter(full_input[:,0],full_input[:,1],
                c=1-full_target[:,0],cmap='RdYlBu')
    plt.savefig('./plot/fractal.png')

    # use Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr,
                                 weight_decay=0.00001)

    # training loop
    epoch = 0
    count = 0
    while epoch < args.epoch and count < 2000:
        epoch = epoch+1
        accuracy = train(net, train_loader, optimizer)
        if accuracy == 100:
            count = count+1
        else:
            count = 0

    # graph hidden units
    if args.hid <= 30:
        for layer in [1,2,3]:
            if layer < 3 or args.net == 'full3':
                for node in range(args.hid):
                    graph_hidden(net, layer, node)
                    plt.scatter(full_input[:,0],full_input[:,1],
                                c=1-full_target[:,0],cmap='RdYlBu')
                    plt.savefig('./plot/hid_%s_%d_%d_%d.png' \
                                % (args.net, args.hid, layer, node))

    # graph output unit
    graph_output(net)
    plt.scatter(full_input[:,0],full_input[:,1],
                c=1-full_target[:,0],cmap='RdYlBu')
    plt.savefig('./plot/out_%s_%d.png' %(args.net, args.hid))
