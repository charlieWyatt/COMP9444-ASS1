"""
   reber.py
   COMP9444, CSE, UNSW
   adapted from:
   http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/reberGrammar.php
"""

import torch
import torch.nn.functional as F
import numpy as np

class lang_reber:

    def __init__(self,embed=False,length=4):
        self.embed = embed
        self.min_length = length
        # assign a number to each transition symbol
        self.chars = 'BTSXPVE'
        # finite state machine for non-embedded Reber Grammar
        self.graph = [[(1,5),('T','P')], [(1,2),('S','X')], \
                      [(3,5),('S','X')], [(6,),('E')], \
                      [(3,2),('V','P')], [(4,5),('V','T')] ]
        
    def get_one_example(self,min_length=5):
        seq = [0]
        prob = []
        state = [0]
        node = 0
        while node != 6:
            state.append(node+1)
            this_prob = np.zeros(7)
            transitions = self.graph[node]
            if (len(seq) < self.min_length - 2) and (node == 2 or node == 4):
                # choose transition to force a longer sequence
                i = 1
                this_prob[self.chars.find(transitions[1][1])] = 1 
            else:
                # choose transition randomly
                i = np.random.randint(0, len(transitions[0]))
                for ch in transitions[1]:
                    this_prob[self.chars.find(ch)] = 1./len(transitions[1])

            prob.append(this_prob)
            seq.append(self.chars.find(transitions[1][i]))
            node = transitions[0][i]

        return seq, prob, state

    def get_one_embed_example(self,min_length=9):
        seq_mid, prob_mid, state_mid = self.get_one_example(min_length-4)
        i = np.random.randint(0,2)
        if i == 0:
            first = 1
            f1 = 1
            f4 = 0
            state_mid = [s+2 for s in state_mid]
            state = [0,1] + state_mid + [9,18]
        else:
            first = 4
            f1 = 0
            f4 = 1
            state_mid = [s+10 for s in state_mid]
            state = [0,1] + state_mid + [17,18]
        seq = [0,first] + seq_mid  + [first,6]
        prob = [(0,0.5,0,0,0.5,0,0),(1,0,0,0,0,0,0)] + \
                prob_mid + [(0,f1,0,0,f4,0,0),(0,0,0,0,0,0,1)]
        return seq, prob, state

    def get_sequence(self):
        if self.embed:
            seq_raw, prob, state = self.get_one_embed_example(self.min_length)
        else:
            seq_raw, prob, state = self.get_one_example(self.min_length)

        # convert numpy array to torch tensor
        seq = torch.as_tensor(seq_raw)
        input = F.one_hot(seq[0:-1],num_classes=7).float()
        target = torch.from_numpy(np.asarray(prob)).float()
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)
        return input, seq, target, state

    def print_outputs(self,epoch,seq,state,hidden,target,output):
        log_prob = F.log_softmax(output, dim=2)
        prob_out = torch.exp(log_prob)
        hidden_np = hidden.squeeze().numpy()
        target_np = target.squeeze().numpy()
        prob_out_np = prob_out.squeeze().numpy()
        print('-----')
        symbol = [self.chars[index] for index in seq.squeeze().tolist()]
        if self.embed:
            print('state = ',*state,sep=' ')
        else:
            print('state = ',*state,sep='')
        print('symbol= '+''.join(symbol))
        print('label = ',*(seq.squeeze().tolist()),sep='')
        print('true probabilities:')
        print('     B    T    S    X    P    V    E')
        for k in range(len(state)-1):
            print(state[k+1],target_np[k,:])
        print('hidden activations and output probabilities [BTSXPVE]:')
        for k in range(len(state)-1):
            print(state[k+1],hidden_np[k,:],prob_out_np[k,:])
        #print(prob_out.squeeze().numpy())
        print('epoch: %d' %epoch)
        if self.embed:
            prob_out_mid   = prob_out[:,2:-3,:]
            prob_out_final = prob_out[:,-2,:]
            target_mid   = target[:,2:-3,:]
            target_final = target[:,-2,:]
            print('error: %1.4f' %torch.mean((prob_out_mid - target_mid)
                                            *(prob_out_mid - target_mid)))
            print('final: %1.4f' %torch.mean((prob_out_final - target_final)
                                            *(prob_out_final - target_final)))
        else:
            print('error: %1.4f' %torch.mean((prob_out - target)
                                            *(prob_out - target)))
