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

class LSTM(nn.Module):
    """

    """
    def __init__(self, element_dim, hidden_dim, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        #self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers)

        self.rnn = nn.GRU(element_dim, hidden_dim, num_layers)

        self.fc = nn.Linear(element_dim, vocab_size)

    def forward(self, sentence):
        lstm_out, _ = self.rnn(sentence, 1, -1)
        out = F.log_softmax(self.fc(lstm_out.view(len(sentence), -1)), dim=1)
        return out
    
    def predict(self, start, max_length):
        x = self.word_embeddings(start).unsqueeze(0).view(1, 1, -1)

        #hidden = (torch.zeros(self.num_layers, 1, net.hidden_dim), torch.zeros(self.num_layers, 1, net.hidden_dim))
        hidden = torch.zeros(self.num_layers, 1, net.hidden_dim)
        from copy import copy
        lstm_out = copy(x)

        for i in range(max_length-1):
            x, hidden = self.rnn(x, hidden)
            lstm_out = torch.cat([lstm_out, x])

        out = F.log_softmax(self.fc(lstm_out.view(max_length, -1)), dim=1)

        out = torch.argmax(out,dim=1)

        return out.tolist()[:max_length-1]


class FesNet(nn.Module):
    """
    
    """
    def __init__(self, element_dim, nhead=17, num_layers=6):
        super(FesNet, self).__init__()
        #self.embedding_dim = embedding_dim
        #self.hidden_dim = hidden_dim

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=element_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=element_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    
    def forward(self, src, tgt):
        memory = self.transformer_encoder(src)

        out = self.transformer_decoder(tgt, memory)

        return out


def pretty_time(time, degree=1):
    return str(int(time // 60)) + "m" + str(round(time % 60, degree)) if time > 60 else round(time, degree)


def test_train(net, data, epoch=50):
    
    criterion =  nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.05)

    start_time = time.time()

    loss_list = []

    max_length = 100

    length = len(data)

    for epoch in range(50):
        i = 0
        current_loss = 0
        while i + max_length < len(data):
        
            inputs, targets = data[i:i+max_length], data[i+1:i+1+max_length]

            inputs, targets = torch.unsqueeze(inputs, dim=1), torch.unsqueeze(targets, dim=1)

            predicts = net(inputs, targets)

            net.zero_grad()
        
            loss = criterion(predicts, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)

            optimizer.step()

            current_loss += loss.item()

            i += max_length

            loss_list.append(loss.item())
    
            print("epoch: ", epoch, "current_loss: ", current_loss)

    plt.plot(loss_list)

    plt.show()


def run():
    fesnet = FesNet(element_dim=153)

    structures = torch.rand(101, 1, 153)
    
    sentences = torch.rand(101, 1,153)

    out = fesnet(structures, sentences)

    print(out.shape)

    data = torch.rand(200, 153)

    test_train(net=fesnet, data=data)

    return


if __name__ == "__main__":
    run()
