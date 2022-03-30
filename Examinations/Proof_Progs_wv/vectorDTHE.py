#DistancedTenHotEncoding,出力はベクトル9文字の中心を予測

import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import gensim

#自作クラス
from CharToIndex import CharToIndex
from MyDatasets import Cross_Validation
from TimeChecker import TimeChecker


#9文字の中心を予測
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,c2v,device=torch.device('cpu')):
        self.data = data
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-8
        self.val_idx = torch.zeros((self.length+8,10),dtype=torch.long)
        self.ans_idx = torch.zeros(self.length+8,dtype=torch.long)
        self.c2v = c2v
        self.device = device


        for row,chars in enumerate(data['value']):
            for col,idx in enumerate(map(self.char2index.get_index,chars)):
                self.val_idx[row][col] = idx


        for i,char in enumerate(data['answer']):
            self.ans_idx[i] = self.char2index.get_index(char)

        #距離値付きのten_hot_encodeにvalueを変換
        distances = data['distance']
        self.distanced_ten_hot_encoded_value = torch.zeros((data['value'].shape[0],len(self.char2index)),dtype=torch.float)
        for row,indicies in enumerate(self.val_idx):
            for id_distance,id_value in enumerate(indicies):
                self.distanced_ten_hot_encoded_value[row][id_value]=distances[row][id_distance]

    def __len__(self):
        return self.length


    def __getitem__(self,index):
        assert index < self.length
        inp = self.distanced_ten_hot_encoded_value[index:index+9].to(device)
        tar = torch.from_numpy(self.c2v.wv[self.data['answer'][index+4]]).to(device)

        return inp,tar

class Proofreader(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,n_layers):
        super(Proofreader, self).__init__()

        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rnn = nn.RNN(input_size, self.hidden_dim, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim*2, output_size)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.to(self.device)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim)
        return hidden

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size).to(self.device)
        out, hidden = self.rnn(x.float(), hidden)
        out = out[:,4,:]
        out = self.dropout(out)
        out = self.fc(out)

        return out

class CosSimLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        losses = torch.cosine_similarity(outputs,targets,dim=1)
        losses = torch.sigmoid(-losses*5)
        loss = torch.mean(losses)
        return loss



def train(model,train_dataloader,learning_rate=0.001):
    criterion = CosSimLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = next(iter(train_dataloader))[0].size(0)
    running_loss = 0
    col_cnt = 0
    all_cnt = 0
    model.train()

    for i,(x,y) in enumerate(train_dataloader):
        output = model(x)
        loss = criterion(output, y) #損失計算
        optimizer.zero_grad() #勾配初期化
        loss.backward(retain_graph=True) #逆伝播
        optimizer.step()  #重み更新
        running_loss += loss.item()
        for pred,ans in zip(output,y):
            p = char2vector.wv.most_similar([pred.to('cpu').detach().numpy().copy()],[],1)[0][0]
            a = char2vector.wv.most_similar([ans.to('cpu').detach().numpy().copy()],[],1)[0][0]
            if p==a:
                col_cnt += 1
            all_cnt += 1

    loss_result = running_loss/len(train_dataloader)
    accuracy_result = col_cnt/all_cnt

    return loss_result,accuracy_result


def eval(model,valid_dataloader,char2vector):
    col_cnt = 0
    all_cnt = 0
    batch_size = next(iter(valid_dataloader))[0].size(0)
    model.eval()
    for x,y in valid_dataloader:
        output = model(x)
        for pred,ans in zip(output,y):
            p = char2vector.wv.most_similar([pred.to('cpu').detach().numpy().copy()],[],1)[0][0]
            a = char2vector.wv.most_similar([ans.to('cpu').detach().numpy().copy()],[],1)[0][0]
            if p==a:
                col_cnt += 1
            all_cnt += 1

    return col_cnt/all_cnt



chars_file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/all_chars_3815.npy"
datas_file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/tegaki_distance.npz"
c2v_file_path   = "/net/nfs2/export/home/ohno/CR_pytorch/Wrod2Vec/context_2/word2vec_1024.model"
tokens = CharToIndex(chars_file_path)
data = np.load(datas_file_path,allow_pickle=True)

EMBEDDING_DIM = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 64
VOCAB_SIZE = len(tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
char2vector = gensim.models.Word2Vec.load(c2v_file_path)
tegaki_dataset = MyDataset(data,chars_file_path,char2vector,device=device)


cross_validation = Cross_Validation(tegaki_dataset)
k_num = cross_validation.k_num #デフォルトは10
k_num = 1

p_acc_record=[]
p_loss_record=[]
clock = TimeChecker()#開始時間の設定
print('vector DETH epochs 1000,learning rate 0.01')
##学習
for i in range(k_num):
    train_dataset,valid_dataset = cross_validation.get_datasets(k_idx=i)

    print(f'Cross Validation: k=[{i+1}/{k_num}]')

    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True) #訓練データのみシャッフル
    valid_dataloader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
    proofreader = Proofreader(VOCAB_SIZE,HIDDEN_SIZE,1024,1)

    # epochs = 1
    epochs = 100

    clock.start()

    for epoch in range(1,epochs+1):
        #進捗表示
        print(f'\r{epoch}', end='')


        p_loss,p_accu = train(proofreader,train_dataloader,learning_rate=0.01)


        if epoch%10==0:
            p_val_accu = eval(proofreader,valid_dataloader,char2vector)
            print(f'\r epoch:[{epoch:3}/{epochs}]| {clock.stop()}')
            print(f'  Proof   | loss:{p_loss:.5}, accu:{p_accu:.5}, val_accu:{p_val_accu:.5}')
            clock.start() #開始時間の設定

    #学習結果の表示


    p_loss_record.append(p_loss)
    p_acc_record.append(p_val_accu)

print(f'=================================================')

print(f'Proof \nacc: {p_acc_record}')
print(f'loss: {p_loss_record}')

