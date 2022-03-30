
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from CharToIndex import CharToIndex
from DistancedDatasets import Distanced_TenHot_Dataset_set5 as MyDataset
from MyDatasets import Cross_Validation

import time
import math





chars_file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/all_chars_3812.npy"
datas_file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/katsuji_distance.npz"
# chars_file_path = r"data\tegaki_katsuji\all_chars_3812.npy"
# file_path = r"data\tegaki_katsuji\tegaki_distance.npz"

tokens = CharToIndex(chars_file_path)
data = np.load(datas_file_path,allow_pickle=True)

EMBEDDING_DIM = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 64
VOCAB_SIZE = len(tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
katsuji_dataset = MyDataset(data,chars_file_path,device=device)





def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def show_ans_pred(answers,predictions):
    for ans,pred in zip(answers,predictions):
        correct = '✓' if ans.item() == pred.item() else '✗'
        print(f'{tokens.get_decoded_char(ans.item())}{tokens.get_decoded_char(pred.item()):2} {correct}',end=' ')
    print()



def train(model,train_dataloader,learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = next(iter(train_dataloader))[0].size(0)
    running_loss = 0
    accuracy = 0

    model.train()
    for i,(x,y) in enumerate(train_dataloader):
        output = model(x)
        loss = criterion(output, y) #損失計算
        prediction = output.data.max(1)[1] #予測結果
        accuracy += prediction.eq(y.data).sum().item()/batch_size
        optimizer.zero_grad() #勾配初期化
        loss.backward(retain_graph=True) #逆伝播
        optimizer.step()  #重み更新
        running_loss += loss.item()

    loss_result = running_loss/len(train_dataloader)
    accuracy_result = accuracy/len(train_dataloader)

    return loss_result,accuracy_result


class Proofreader(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,n_layers):
        super(Proofreader, self).__init__()

        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rnn = nn.RNN(output_size, self.hidden_dim, batch_first=True,bidirectional=True)
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
        out = out[:,-1,:]
        out = self.dropout(out)
        out = self.fc(out)

        return out






##学習
train_dataloader=DataLoader(katsuji_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True) #訓練データのみシャッフル
model = Proofreader(VOCAB_SIZE, hidden_dim=HIDDEN_SIZE, output_size=VOCAB_SIZE, n_layers=1)

epochs = 100
# epochs = 10
acc_record=[]
loss_record=[]
start = time.time() #開始時間の設定

for epoch in range(1,epochs+1):
    #進捗表示
    i = (epoch-1)%10
    pro_bar = ('=' * i) + (' ' * (10 - i))
    print('\r[{0}] {1}%'.format(pro_bar, i / 10 * 100.), end='')


    loss,acc = train(model,train_dataloader,learning_rate=0.01)



    if epoch%10==0:
        print(f'\repoch:[{epoch:3}/{epochs}] | {timeSince(start)} - loss: {loss:.7},  accuracy: {acc:.7}')
        start = time.time() #開始時間の設定

model_save_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/pre_trained_proof_distance.pth"

torch.save(model.state_dict(), model_save_path)


