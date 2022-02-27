"""
   anbn.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import random

class lang_anbn:
    def __init__(self,num_class=2,length=9):
        self.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.num_class = num_class
        self.max_length = length
        
    def get_one_example(self):
        seq = [0]
        prob = []
        state = []

        for n in range(5):
            length = np.random.randint(1,self.max_length+1)

            state.append(0)
            # string of (length-1) A's
            for j in range(1,length):
                # assume A, B occur with frequency (0.8,0.2)
                prob.append([0.8, 0.2]+([0]*(self.num_class-2)))
                seq.append(0) # A actually occurs
                state.append(j)
            # first B (either A or B could have occurred)
            prob.append([0.8, 0.2]+([0]*(self.num_class-2)))
            seq.append(1) # B actually occurs
            state.append(length)
            # string of (length-1) B's
            for j in range(1,length):
                prob.append([0,1]+([0]*(self.num_class-2)))
                seq.append(1)
                state.append(length-j)
            # string of C's, D's, etc.
            for k in range(2,self.num_class):
                for j in range(length):
                    prob.append(([0]*(k))+[1]+([0]*(self.num_class-k-1)))
                    seq.append(k)
                    state.append(length-j)

            prob.append([1]+([0]*(self.num_class-1)))
            seq.append(0)
        state.append(0)

        return seq, prob, state

    def get_sequence(self):
        seq_raw, prob, state = self.get_one_example()
        seq = torch.as_tensor(seq_raw)
        input = F.one_hot(seq[0:-1],num_classes=self.num_class).float()
        
        target = torch.from_numpy(np.asarray(prob)).float()
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)
        return input, seq, target, state

    def print_outputs(self,epoch,seq,state,hidden,target,output):
        hidden_np = hidden.squeeze().numpy()
        log_prob = F.log_softmax(output, dim=-1)
        prob_out = torch.exp(log_prob)
        prob_out_np = prob_out.squeeze().numpy()
        print('-----')
        print('color = ',*state,sep='')
        symbol = [self.chars[index] for index in seq.squeeze().tolist()]
        print('symbol= '+''.join(symbol))
        print('label = ',*(seq.squeeze().tolist()),sep='')
        print('hidden activations and output probabilities:')
        for k in range(len(state)-1):
            print(self.chars[seq[k+1]],hidden_np[k,:],prob_out_np[k,:])
        print('epoch: %d' %epoch)
        print('error: %1.4f' %torch.mean((prob_out - target)
                                        *(prob_out - target)))
