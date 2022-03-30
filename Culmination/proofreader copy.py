from sqlite3 import Time
import torch
import torch.nn as nn
import numpy as np
from CharToIndex import CharToIndex
from MyDatasets import Cross_Validation
from torch import optim
from torch.utils.data import DataLoader
from MyCustomLayer import WeightedTenHotEncodeLayer
import torch.nn.functional as F
from TimeChecker import TimeChecker

#5文字の中心を予測
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-4
        self.p_val_idx = torch.zeros((self.length+4,10),dtype=torch.long)
        self.p_ans_idx = torch.zeros(self.length+4,dtype=torch.long)
        self.d_ans     = torch.zeros(self.length+4,dtype=torch.long)
        self.device = device

        for i_r,chars in enumerate(data['value']):
            for i_c, idx in enumerate(map(self.char2index.get_index,chars)):
                self.p_val_idx[i_r][i_c] = idx

        for i,char in enumerate(data['answer']):
            self.p_ans_idx[i] = self.char2index.get_index(char)
            self.d_ans[i] = 1 if self.p_val_idx[i][0] == self.p_ans_idx[i] else 0 #検出器用、OCR第一出力と答えが等しければ１、異なれば０


        #距離値付きのten_hot_encodeにvalueを変換
        distances = np.nan_to_num(data['distance'])
        self.distanced_ten_hot_encoded_value = torch.full((self.length+6,len(self.char2index)),0,dtype=torch.float)
        for row,indicies in enumerate(self.p_val_idx):
            for id_distance,id_value in enumerate(indicies):
                self.distanced_ten_hot_encoded_value[row][id_value]=distances[row][id_distance]


    def __len__(self):
        return self.length


    def __getitem__(self,index):
        p_inp  = self.p_val_idx[index:index+5,0].to(device)
        p_tar = self.p_ans_idx[index+2].to(device)
        d_ans = self.d_ans[index+2].to(device)
        distance = self.distanced_ten_hot_encoded_value[index+2].to(device)
        return distance,d_ans,p_inp,p_tar


class Proofreader(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,n_layers):
        super(Proofreader, self).__init__()

        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(output_size,embedding_dim=256)
        self.rnn = nn.RNN(256, self.hidden_dim, batch_first=True,bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_dim*2, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.to(self.device)


    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim)
        return hidden


    def forward(self, x, distance):
        batch_size = x.size(0)
        x = self.embedding(x.long())
        hidden = self.init_hidden(batch_size).to(self.device)
        out, hidden = self.rnn(x, hidden)
        out = out[:,2,:]
        out = self.dropout(out)
        out = self.fc(out)

        out.mul_(distance)
        return out

chars_file_path = r"/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/all_chars_3812.npy"
datas_file_path = r"/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/katsuji_distance.npz"
tokens = CharToIndex(chars_file_path)
data = np.load(datas_file_path,allow_pickle=True)

EMBEDDING_DIM = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 64
VOCAB_SIZE = len(tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MyDataset(data,chars_file_path,device=device)



def train(proofreader,train_dataloader,learning_rate=0.001):
    p_criterion = nn.CrossEntropyLoss()
    p_optim = optim.Adam(proofreader.parameters(), lr=learning_rate)
    batch_size = next(iter(train_dataloader))[0].size(0)
    p_running_loss = 0
    p_runnning_accu = 0

    proofreader.train()
    for d_x,d_y,p_x,p_y in train_dataloader:
        #修正器の処理
        p_output = proofreader(p_x,d_x)
        p_tmp_loss = p_criterion(p_output, p_y) #損失計算
        p_prediction = p_output.data.max(1)[1] #予測結果
        p_runnning_accu += p_prediction.eq(p_y.data).sum().item()/batch_size
        p_optim.zero_grad() #勾配初期化
        p_tmp_loss.backward(retain_graph=True) #逆伝播
        p_optim.step()  #重み更新
        p_running_loss += p_tmp_loss.item()

    p_loss = p_running_loss/len(train_dataloader)
    p_accu = p_runnning_accu/len(train_dataloader)

    return p_loss,p_accu

def eval(proofreader,valid_dataloader):
    batch_size = next(iter(valid_dataloader))[0].size(0)
    p_runnning_accu = 0
    proofreader.eval()

    for d_x,d_y,p_x,p_y in valid_dataloader:
        #修正器の処理
        p_output = proofreader(p_x,d_x)
        p_prediction = p_output.data.max(1)[1] #予測結果
        p_runnning_accu += p_prediction.eq(p_y.data).sum().item()/batch_size

    p_accu = p_runnning_accu/len(valid_dataloader)

    return p_accu






p_val_accu_record = []

##学習


train_dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
proofreader = Proofreader(VOCAB_SIZE, hidden_dim=HIDDEN_SIZE, output_size=VOCAB_SIZE, n_layers=1)

epochs = 100
# epochs = 10

p_acc_record=[]
p_loss_record=[]
acc_record = []
stopwatch = TimeChecker()
stopwatch.start()

for epoch in range(1,epochs+1):
    #進捗表示
    print(f'\r{epoch}', end='')

    p_loss,p_accu = train(proofreader,train_dataloader,learning_rate=0.01)



    if epoch%10==0:
        print(f'\r epoch:[{epoch:3}/{epochs}]| {stopwatch.stop()}')
        print(f'  Proof   | loss:{p_loss:.5}, accu:{p_accu:.5}')
        stopwatch.start()

torch.save(proofreader.state_dict(), "/net/nfs2/export/home/ohno/CR_pytorch/Culmination/Learned_models/pre_trained_proof")




