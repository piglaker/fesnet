import os
import re
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import pandas as pd
import matplotlib.pyplot as plt

class FesNet(nn.Module):
    """
    
    """
    def __init__(self, element_dim, step_size=100, hidden_dim=153, nhead=9, num_layers=1):
        super(FesNet, self).__init__()
        #self.embedding_dim = embedding_dim
        #self.hidden_dim = hidden_dim

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=element_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=element_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    
    def forward(self, structures, sentences):
        memory = self.transformer_encoder(structures)

        out = self.transformer_decoder(sentences, memory)

        return out


def pretty_time(time, degree=1):
    return str(int(time // 60)) + "m" + str(round(time % 60, degree)) if time > 60 else round(time, degree)


def test_train(net, src, tgt, epoch=50):
    
    criterion =  nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.005)

    start_time = time.time()

    loss_list = []

    print(src.shape, tgt.shape)

    for i in range(epoch):
        total_loss = 0.

        probs = net(src, tgt)

        net.zero_grad()
  
        loss = criterion(probs.reshape(-1), tgt.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if (i+1) % 5 == 0:
            time_past = time.time() - start_time
            cur_loss = total_loss
            loss_list.append(cur_loss)
            total_loss = 0
            print("epoch: ", i, " time: ", pretty_time(time_past), "loss: ", cur_loss)

    plt.plot(loss_list)

    plt.show()


def run():
    fesnet = FesNet(element_dim=153)

    structures = torch.rand(6, 1, 153)
    
    sentences = torch.rand(101, 1,153)

    out = fesnet(structures, sentences)

    print(out.shape)

    test_train(net=fesnet, src=structures, tgt=sentences)

    return


if __name__ == "__main__":
    run()
