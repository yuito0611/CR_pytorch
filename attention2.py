
#候補文字10個をDTHE->sigmoid->最終層で結合することでattention、OCR第一候補のみで予測、5文字の中から中央１文字予測、embedding、PreTrained
import torch
import torch.nn as nn
import numpy as np
from CharToIndex import CharToIndex
from MyDatasets import Cross_Validation
from torch import optim
from torch.utils.data import DataLoader


#5文字の中心を予測
class OCR1Dataset(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-4
        self.val_idx = torch.zeros((self.length+4,10),dtype=torch.long)
        self.ans_idx = torch.zeros(self.length+4,dtype=torch.long)
        self.device = device

        for i_r,chars in enumerate(data['value']):
            for i_c, idx in enumerate(map(self.char2index.get_index,chars)):
                self.val_idx[i_r][i_c] = idx

        for i,char in enumerate(data['answer']):
            self.ans_idx[i] = self.char2index.get_index(char)

        #距離値付きのten_hot_encodeにvalueを変換
        distances = np.nan_to_num(data['distance'])
        self.distanced_ten_hot_encoded_value = torch.full((self.length+4,len(self.char2index)),0,dtype=torch.float)
        for row,indicies in enumerate(self.val_idx):
            for id_distance,id_value in enumerate(indicies):
                self.distanced_ten_hot_encoded_value[row][id_value]=distances[row][id_distance]


    def __len__(self):
        return self.length


    def __getitem__(self,index):
        input  = self.val_idx[index:index+5,0].to(self.device)
        target = self.ans_idx[index+2].to(self.device)
        distance = self.distanced_ten_hot_encoded_value[index+2].to(self.device)
        return input,target,distance



chars_file_path = r"/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/all_chars_3812.npy"
datas_file_path = r"/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/tegaki_distance.npz"
katsuji_file_path = r"/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/katsuji_distance.npz"
tokens = CharToIndex(chars_file_path)

data = np.load(datas_file_path,allow_pickle=True)
katsuji_data = np.load(katsuji_file_path,allow_pickle=True)
EMBEDDING_DIM = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 64
VOCAB_SIZE = len(tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tegaki_dataset = OCR1Dataset(data,chars_file_path,device=device)
katsuji_dataset = OCR1Dataset(katsuji_data,chars_file_path,device=device)



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


    def forward(self, x,distance):
        batch_size = x.size(0)
        x = self.embedding(x.long())
        hidden = self.init_hidden(batch_size).to(self.device)
        out, hidden = self.rnn(x, hidden)
        out = out[:,2,:]
        out = self.dropout(out)
        out = self.fc(out)
        pred = out.mul(distance)
        return pred


import time
import math
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(proofreader,train_dataloader,learning_rate=0.001):
    p_criterion = nn.CrossEntropyLoss()
    p_optim = optim.Adam(proofreader.parameters(), lr=learning_rate)
    batch_size = next(iter(train_dataloader))[0].size(0)
    p_running_loss = 0
    p_runnning_accu = 0

    proofreader.train()
    for x,y,distance in train_dataloader:
        #修正器の処理
        p_output = proofreader(x,distance)
        p_tmp_loss = p_criterion(p_output, y) #損失計算
        p_prediction = p_output.data.max(1)[1] #予測結果
        p_runnning_accu += p_prediction.eq(y.data).sum().item()/batch_size
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

    for proof_x,proof_y,distance in valid_dataloader:
        #修正器の処理
        p_output = proofreader(proof_x,distance)
        p_prediction = p_output.data.max(1)[1] #予測結果
        p_runnning_accu += p_prediction.eq(proof_y.data).sum().item()/batch_size

    p_accu = p_runnning_accu/len(valid_dataloader)

    return p_accu




cross_validation = Cross_Validation(tegaki_dataset)
k_num = cross_validation.k_num #デフォルトは10
# k_num = 1
p_acc_record=[]
p_loss_record=[]

##学習
for i in range(k_num):
    train_dataset,valid_dataset = cross_validation.get_datasets(k_idx=i)

    print(f'Cross Validation: k=[{i+1}/{k_num}]')

    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True) #訓練データのみシャッフル
    valid_dataloader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
    pre_train_dataloader = DataLoader(katsuji_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    proofreader = Proofreader(VOCAB_SIZE, hidden_dim=HIDDEN_SIZE, output_size=VOCAB_SIZE, n_layers=1)
    epochs = 100
    # epochs = 1

    start = time.time() #開始時間の設定
    print('Starting PreTraining...')
    for epoch in range(1,epochs+1):
        print(f'\r{epoch}', end='')
        p_loss,p_accu = train(proofreader,pre_train_dataloader,learning_rate=0.01)

    print(f'pretrained acc: {p_accu}, loss: {p_loss}')
    print('End PreTraining!')
    print('Starting Main Training')
    for epoch in range(1,epochs+1-50):
        #進捗表示
        print(f'\r{epoch}', end='')

        p_loss,p_accu = train(proofreader,train_dataloader,learning_rate=0.01)
        p_val_accu = eval(proofreader,valid_dataloader)

        if epoch%10==0:
            print(f'\r epoch:[{epoch:3}/{epochs}]| {timeSince(start)}')
            print(f'  Proof   | loss:{p_loss:.5}, accu:{p_accu:.5}, val_accu:{p_val_accu:.5}')
            start = time.time() #開始時間の設定

    #学習結果の表示


    p_loss_record.append(p_loss)
    p_acc_record.append(p_val_accu)


print(f'=================================================')
print(f'Proof \nacc: {p_acc_record}')
print(f'acc average: {np.mean(p_acc_record)}')
print(f'loss: {p_loss_record}')
print(f'loss average: {np.mean(p_loss_record)}')






