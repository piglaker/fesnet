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
    def __init__(self, element_dim, step_size, embedding_dim, hidden_dim):
        super(FesNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(element_dim, hidden_dim)

    
    def forward(self, sentences, length_list, targets):
        out = self.get_lstm_feature(sentences, length_list)

        return self.crf(torch.transpose(out, 0, 1), torch.transpose(targets, 0, 1), torch.transpose(self.get_mask(length_list), 0, 1))

    def predict(self, sentences, length_list):
        out = self.get_lstm_feature(sentences, length_list)

        return self.crf.decode(torch.transpose(out, 0, 1), torch.transpose(self.get_mask(length_list), 0, 1))




def run():
    return



if __name__ == "__main__":
    run()
