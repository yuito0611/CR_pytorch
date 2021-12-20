import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary 
import numpy as np
import matplotlib.pyplot as plt
import pprint

from CharToIndex import CharToIndex
from MyDatasets import BaseDataset as MyDataset
from MyCustomLayer import TenHotEncodeLayer 


import time
import math


#hot encode用
class Proofreader(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,n_layers):
        super(Proofreader, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = TenHotEncodeLayer(output_size)
        self.rnn = nn.RNN(input_size, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.to(self.device)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size).to(self.device)

        x = self.encoder(x)
        out, hidden = self.rnn(x, hidden)
        out = out[:,-1,:]
        out = self.fc(out)
        
        return out

#char2vec使用用
class Proofreader(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,n_layers):
        super(Proofreader, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = CharToVec()
        self.rnn = nn.RNN(5, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.to(device)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size).to(device)

        x = self.encoder(x)
        out, hidden = self.rnn(x, hidden)
        out = out[:,-1,:]
        out = self.fc(out)
        
        return out


#char2vec使用+候補文字群の圧縮にRNN使用
class Proofreader(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,n_layers):
        super(Proofreader, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = CharToVec()
        vector_size = 5
        sub_rnn_output = 128
        self.sub_rnn = nn.RNN(vector_size, self.hidden_dim, batch_first=True)
        self.rnn = nn.RNN(sub_rnn_output, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.to(device)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size).to(device)
        sub_hidden = self.init_hidden(batch_size).to(device)

        x = self.encoder(x) #(1,5,10,5)

        sub_out_list=[]
        for _x in x: #(5,10,5)
            _out,sub_hidden = self.sub_rnn(_x,sub_hidden)
            sub_out_list.append(_out)
        sub_out = torch.stack(sub_out_list,0)

        out, hidden = self.rnn(x, hidden)
        out = out[:,-1,:]
        out = self.fc(out)
        
        return out


#Word2Vec 加重平均
class Proofreader(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,n_layers):
        super(Proofreader, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = CharToVec()

        self.rnn = nn.RNN(5, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.to(device)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size).to(device)

        x = self.encoder(x) 

        out, hidden = self.rnn(x, hidden)
        out = out[:,-1,:]
        out = self.fc(out)
        
        return out









from MyCustomLayer import CharToVectorLayer4 as CharToVec


#char2vec使用+候補文字群の圧縮にRNN使用
class Proofreader(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,n_layers):
        super(Proofreader, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = CharToVec()
        vector_size = 5
        sub_rnn_output = 128
        self.sub_rnn = nn.RNN(vector_size, self.hidden_dim, batch_first=True)
        self.rnn = nn.RNN(sub_rnn_output, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.to(device)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size).to(device)
        sub_hidden = self.init_hidden(5).to(device)

        x = self.encoder(x) #(1,5,10,5)
        sub_out_list=[]
        for _x in x: #(5,10,5)
            _out,sub_hidden = self.sub_rnn(_x,sub_hidden)
            sub_out_list.append(_out[:,-1,:])
        sub_out = torch.stack(sub_out_list,0)

        out, hidden = self.rnn(sub_out, hidden)
        out = out[:,-1,:]
        out = self.fc(out)
        
        return out