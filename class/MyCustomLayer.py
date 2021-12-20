import torch
import torch.nn as nn
import numpy as np

class TenHotEncodeLayer(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self,x):
        hot_out = torch.zeros(x.size(0),x.size(1),self.num_tokens)
        for N in range(x.size(0)):
            for T in range(x.size(1)):
                for F in x[N,T]:
                    hot_out[N,T,F.long()]=1
        return hot_out.to(self.device)




#文字(インデックス化された状態)をベクトルに埋め込む (特徴量が5*10=50次元)
class CharToVectorLayer1(nn.Module):
    def __init__(self):
        super().__init__()
        self.vec_of_char = torch.load("/net/nfs2/export/home/ohno/CR_pytorch/data/main/char2vec")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        output = []
        for timestep_item in x:
            timesteps_list = []
            for feature_item in timestep_item:
                features_list = []
                for f in feature_item:
                    features_list.append(self.vec_of_char[f.item()])
                timesteps_list.append(torch.cat(features_list,0))
            output.append(torch.stack(timesteps_list,0))

        return torch.stack(output,0).to(self.device)


#文字(インデックス化された状態)をベクトルに埋め込む (特徴量を第一候補のみで5次元)
class CharToVectorLayer2(nn.Module):
    def __init__(self):
        super().__init__()
        self.vec_of_char = torch.load("/net/nfs2/export/home/ohno/CR_pytorch/data/main/char2vec")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        output = []
        for timestep_item in x:
            timesteps_list = []
            for feature_item in timestep_item:
                features_list = []
                for f in feature_item:
                    features_list.append(self.vec_of_char[f.item()])
                timesteps_list.append(torch.cat(features_list,0))
            output.append(torch.stack(timesteps_list,0))

        return torch.stack(output,0).to(self.device)



#文字(インデックス化された状態)をベクトルに埋め込む (10個のベクトルを足し合わせて平均をとる)
class CharToVectorLayer3(nn.Module):
    def __init__(self):
        super().__init__()
        self.vec_of_char = torch.load("/net/nfs2/export/home/ohno/CR_pytorch/data/main/char2vec")

        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        output = []
        for timestep_item in x:
            timesteps_list = []
            for feature_item in timestep_item:
                vec = torch.zeros((5))
                for f in feature_item:
                    vec.add_(self.vec_of_char[f.item()])
                vec.div_(10)
                timesteps_list.append(vec)
            output.append(torch.stack(timesteps_list,0))

        return torch.stack(output,0).to(self.device)



#文字(インデックス化された状態)をベクトルに埋め込む (特徴量が(10,5)の多次元)
class CharToVectorLayer4(nn.Module):
    def __init__(self):
        super().__init__()
        self.vec_of_char = torch.load("/net/nfs2/export/home/ohno/CR_pytorch/data/main/char2vec")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        #(1,5,10)
        output = []
        for timestep_item in x:
            timesteps_list = []
            for feature_item in timestep_item:
                features_list = []
                for f in feature_item:
                    features_list.append(self.vec_of_char[f.item()])
                timesteps_list.append(torch.stack(features_list,0))
            output.append(torch.stack(timesteps_list,0))

        return torch.stack(output,0).to(self.device)

#文字(インデックス化された状態)をベクトルに埋め込む (特徴量が(10,5)の多次元)
# ※RNNに第一候補を最後に読み込ませるため逆順
class CharToVectorLayer4_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.vec_of_char = torch.load("/net/nfs2/export/home/ohno/CR_pytorch/data/main/char2vec")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        #(1,5,10)
        output = []
        for timestep_item in x:
            timesteps_list = []
            for feature_item in timestep_item:
                features_list = []
                for f in reversed(feature_item):
                    features_list.append(self.vec_of_char[f.item()])
                timesteps_list.append(torch.stack(features_list,0))

            output.append(torch.stack(timesteps_list,0))

        return torch.stack(output,0).to(self.device)




#文字(インデックス化された状態)をベクトルに埋め込む (10個のベクトルの加重平均)
class CharToVectorLayer5(nn.Module):
    def __init__(self):
        super().__init__()
        self.vec_of_char = torch.load("/net/nfs2/export/home/ohno/CR_pytorch/data/main/char2vec")

        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.w = [10,9,8,7,6,5,4,3,2,1]

    def forward(self,x):
        output = []
        for timestep_item in x:
            timesteps_list = []
            for feature_item in timestep_item:
                vec_list = []
                for f in feature_item:
                    vec_list.append(self.vec_of_char[f.item()])
                vec = torch.stack(vec_list,0)

                average = torch.from_numpy(np.average(vec.detach().numpy(), axis=0, weights=self.w).astype(np.float32)).clone()
                timesteps_list.append(average)
            output.append(torch.stack(timesteps_list,0))

        return torch.stack(output,0).to(self.device)



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
            sub_out_list.append(_out[:,-1:])
        sub_out = torch.stack(sub_out_list,0)

        out, hidden = self.rnn(x, hidden)
        out = out[:,-1,:]
        out = self.fc(out)

        return out









#文字(インデックス化された状態)をベクトルに埋め込む (特徴量をDense層で圧縮)
class Proofreader(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,n_layers):
        super(Proofreader, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = CharToVec()

        self.dence = nn.Linear(5, 1)
        self.rnn = nn.RNN(10, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.to(device)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size).to(device)

        x = self.encoder(x)
        x = self.dence(x)   #(feature,vector_size)を(feature,1)へ
        x = torch.squeeze(x,dim=-1) #１の次元削除
        out, hidden = self.rnn(x, hidden)
        out = out[:,-1,:]
        out = self.fc(out)

        return out
