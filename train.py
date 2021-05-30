import os
import re
import time
from numpy.matrixlib.defmatrix import matrix
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils.lib import get_detach_from
from utils.lib import list_hiddens2torch_tensor
from utils.lib import pretty_time

from net.model import Encoder_GRU

import utils.load_acc

max_length = 20

def load_dataset():

    earthquake = utils.load_acc.get_earthquake_data()[:]

    dataset = utils.load_acc.get_dataset() / 1000

    train_dataset, test_dataset = dataset[:21], dataset[21:]

    return train_dataset, test_dataset


def load_matrix():
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    import preprocess

    stiffness_o, mass_o = preprocess.task()

    ss_s, ss_m = MinMaxScaler(), MinMaxScaler()

    ss_s.fit(stiffness_o);ss_m.fit(mass_o)

    stiffness, mass = ss_s.transform(stiffness_o), ss_m.transform(mass_o)

    matrix = torch.tensor(np.append(stiffness, mass, axis=1))

    matrix = matrix.transpose(1,0)

    matrix = torch.unsqueeze(matrix, dim=0).to(torch.float32)

    return matrix

def train(net, train_dataset, matrix, epochs, lr):
    element_dim = train_dataset[0].shape[-1]

    criterion =  nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    start_time = time.time()

    loss_list = []

    length = len(train_dataset[0])

    start_time = time.time()

    print("Start Training!" ,time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    for epoch in range(epochs):
        wave_id = 0
        for data in train_dataset:
            i = 0
            current_loss = 0
            while i + max_length < len(data):
                
                inputs, targets = data[i:i+max_length], data[i+1:i+1+max_length]

                net.zero_grad()
                
                outputs = net(inputs, matrix) 

                loss = criterion(outputs, targets)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)

                optimizer.step()

                current_loss += loss.item()

                i += max_length

                loss_list.append(loss.item())

            scheduler.step()
            wave_id += 1
            print("epoch: ", epoch, 
            "time: ", pretty_time(time.time() - start_time), 
            "wave_id:", wave_id,
            "current_loss: ", current_loss, 
            )

    print("Training Finished !")

    plt.plot(loss_list)
    plt.savefig("Loss" + str(time.strftime("_%a_%b_%d_%H_%M_%S_%Y", time.localtime())) + ".png")

    return 


def test(net, test_dataset, matrix):

    criterion =  nn.MSELoss()

    total_loss = 0

    for data in test_dataset:
        pre = torch.tensor([])
        i = 0
        test_time_zeros = time.time()
        with torch.no_grad():
            while i + max_length < len(data):
                inputs, targets = data[i:i+max_length], data[i+1:i+1+max_length]

                outputs = net(inputs, matrix)

                loss = criterion(outputs, targets)

                total_loss += loss.item()

                pre = torch.cat((pre, outputs))

                i += max_length

        print("Time Cost: ", time.time() - test_time_zeros, 's')
    
        plt.plot(pre.squeeze(dim=1).detach().numpy())

        plt.show()

        plt.plot(data.squeeze(dim=1).detach().numpy())

        plt.show()

    print("Total Loss: ", total_loss)

    return 


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', default=20, type=int)
parser.add_argument('--lr', default=0.0005, type =float)
parser.add_argument('--hidden_size', default=300, type=int)
parser.add_argument('--logits_size', default=20, type=int)
parser.add_argument('--num_layers', default=2, type=int)

def run():

    args = parser.parse_args()

    net = Encoder_GRU(
        inputs_size=154, 
        matrix_dim=144, 
        hidden_size=args.hidden_size, 
        targets_size=154, 
        logits_size=args.logits_size, 
        num_layers=args.num_layers
        )
    
    print("PreProcess & Load Data ...")

    train_dataset, test_dataset = load_dataset()
    
    matrix = load_matrix()
    
    train(net.to(device), train_dataset.to(device), matrix, args.epoch, args.lr)
    
    #test()

    return 

if __name__ == "__main__":
    run()

