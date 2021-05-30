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

# utils
def get_detach_from(hiddens):
    return [hidden.clone().detach() for hidden in hiddens]

def list_hiddens2torch_tensor(hiddens):
    re = torch.tensor([])
    for hidden in hiddens:
        re = torch.cat([re, hidden], dim=-1)
    return re

class myGRUCell(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(myGRUCell, self).__init__()
        self.inputs_size = inputs_size
        self.hidden_size = hidden_size
        self.r_layer_i = nn.Linear(self.inputs_size, self.hidden_size)
        self.r_layer_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.z_layer_i = nn.Linear(self.inputs_size, self.hidden_size)
        self.z_layer_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.n_layer_i = nn.Linear(self.inputs_size, self.hidden_size)
        self.n_layer_h = nn.Linear(self.hidden_size, self.hidden_size)

        #self.fc = nn.Linear(hidden_size, target_size)

    def forward(self, input, hidden):
        #print(input.shape, hidden.shape)
        r = torch.sigmoid(self.r_layer_i(input) + self.r_layer_h(hidden))
        z = torch.sigmoid(self.z_layer_i(input) + self.r_layer_h(hidden))
        n = torch.tanh(self.n_layer_i(input) + torch.mul(r, self.n_layer_h(hidden)))
        next_hidden = torch.mul((1-z), n) + torch.mul(z, hidden)

        # extra full connect
        #y = self.fc(next_hidden)

        return next_hidden, next_hidden

#follow just for test model

class Encoder(nn.Module):
    def __init__(self, input_dim=153, logit_size=100, kernel_wins=[3, 4, 5]):
        super(Encoder, self).__init__()

        self.convs = nn.ModuleList([nn.Conv1d(input_dim, logit_size, size) for size in kernel_wins])

        self.dropout = nn.Dropout(0.6)

        self.fc = nn.Linear(len(kernel_wins)*logit_size, logit_size)

    def forward(self, x):
        con_x = [conv(x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        
        fc_x = torch.cat(pool_x, dim=1)

        fc_x = self.dropout(fc_x)

        fc_x = fc_x.squeeze(-1)

        logit = self.fc(fc_x)

        return logit.unsqueeze(0)

#for regress max_acc


class Decoder(nn.Module):
    def __init__(self, inputs_size=400, matrix_dim=144, hidden_size=100, targets_size=154, logits_size=20, num_layers=2):
        super(Decoder, self).__init__()
        self.inputs_size = inputs_size
        self.matrix_dim = matrix_dim
        self.hidden_size = hidden_size
        self.targets_size = targets_size
        self.logits_size = logits_size
        self.num_layers = num_layers

        self.encoder = Encoder(input_dim=self.matrix_dim,
                               logit_size=self.logits_size, kernel_wins=[3, 4, 5])

        self.myGRUCell = nn.ModuleList([myGRUCell(inputs_size=self.inputs_size, hidden_size=self.hidden_size)] +
                                       [myGRUCell(inputs_size=self.hidden_size, hidden_size=self.hidden_size) for i in range(num_layers - 1)])

    def init_hiddens(self):
        return [torch.zeros(1, 1, self.hidden_size) for i in range(self.num_layers)]

    def impl(self, y, hiddens):
        for i, grucell in enumerate(self.myGRUCell):
            y, hiddens[i] = grucell(y, hiddens[i])

        return y, hiddens

    def forward(self, y_, features, hiddens_down):

        logits = features

        inputs = torch.cat([y_, logits], dim=2)

        y_down, hiddens_down = self.impl(inputs, hiddens_down)

        y_down = y_down.squeeze(dim=2)

        return y_down, hiddens_down


class Encoder_GRU(nn.Module):
    """
    inputs : L_length, N_batch, H_elementdim
    """

    def __init__(self, inputs_size=154, matrix_dim=144, hidden_size=20, targets_size=154, logits_size=10, num_layers=2):
        super(Encoder_GRU, self).__init__()
        self.inputs_size = inputs_size + logits_size
        self.matrix_dim = matrix_dim
        self.hidden_size = hidden_size
        self.targets_size = targets_size
        self.logits_size = logits_size
        self.num_layers = num_layers

        self.encoder = Encoder(input_dim=self.matrix_dim,
                               logit_size=self.logits_size, kernel_wins=[3, 4, 5])

        self.decoder = Decoder(
            inputs_size=self.hidden_size + self.logits_size,
            matrix_dim=self.matrix_dim,
            hidden_size=self.hidden_size,
            targets_size=self.targets_size,
            logits_size=self.logits_size,
            num_layers=2
        )

        self.myGRUCell = nn.ModuleList([myGRUCell(inputs_size=self.inputs_size, hidden_size=self.hidden_size)] +
                                       [myGRUCell(inputs_size=self.hidden_size, hidden_size=self.hidden_size) for i in range(num_layers - 1)])

        self.fc = nn.Linear(self.hidden_size, self.hidden_size)

        self.mlp = nn.Linear(self.hidden_size, self.targets_size)

    def init_hiddens(self):
        return [torch.zeros(1, 1, self.hidden_size) for i in range(self.num_layers)]

    def impl(self, y, hiddens):
        for i, grucell in enumerate(self.myGRUCell):
            y, hiddens[i] = grucell(y, hiddens[i])
        return y, hiddens

    def forward(self, inputs, matrix):
        inputs = inputs.unsqueeze(dim=1)

        hiddens_up = self.init_hiddens()

        hiddens_down = self.decoder.init_hiddens()

        logits = self.encoder(matrix)

        y_up, hiddens_up = self.impl(
            torch.cat((inputs[0], logits), dim=2), hiddens_up)

        y_up = self.fc(y_up)

        features = self.decoder.encoder(matrix)

        y_down, hiddens_down = self.decoder(
            y_up,
            features,
            hiddens_down
        )

        y_down = self.mlp(y_down)

        for i in range(1, inputs.shape[0]):
            y_up_semi, hiddens_up = self.impl(
                torch.cat((inputs[i], logits), dim=2), hiddens_up)

            y_up_ = self.fc(y_up_semi)

            y_down_, hiddens_down = self.decoder(
                y_up_,
                features,
                hiddens_down
            )

            y_down = torch.cat([y_down, self.mlp(y_down_)], dim=0)

            #y_up = torch.cat([y_up, y_up_], dim=0)

        return y_down

class LSTM(nn.Module):
    """
    inputs : L_length, N_batch, H_elementdim
    """
    def __init__(self, element_dim, hidden_dim, output_size, num_layers=2):
        super().__init__()
        self.element_dim = element_dim

        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        #self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers)

        self.rnn = nn.GRU(element_dim, hidden_dim, num_layers)

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        lstm_out, _ = self.rnn(sentence.view(len(sentence), 1, -1))
        
        out = self.fc(lstm_out.view(len(sentence), -1))
        
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
