
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from CharToIndex import CharToIndex
from MyDatasets import Cross_Validation
# from MyCustomLayer import TenHotEncodeLayer
import torch.nn.functional as F


class BinaryClassDataset(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.len = len(data[0])
        self.device = device

        values = data[0]
        self.val_idx = []
        for chars in values:
            indexes = []
            for idx in map(self.char2index.get_index,chars):
                indexes.append(idx)
            self.val_idx.append(indexes)

        answers = data[1]
        for idx in map(self.char2index.get_index,answers):
            self.ans_idx.append(idx)


    def __len__(self):
        return self.len


    def __getitem__(self,idx):
        assert idx < self.len
        inp = self.val_idx[idx]
        #OCRの第一候補と答えが等しければ１、等しくなければ０
        if self.val_idx[idx][0] == self.ans_idx[idx]:
            tar = 1
        else:
            tar = 0
        return torch.tensor(inp).to(self.device),torch.tensor(tar).to(self.device)



# chars_file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/all_chars_3812.npy"
# file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/tegaki.npy"

chars_file_path = r"data\tegaki_katsuji\all_chars_3812.npy"
file_path = r"data\tegaki_katsuji\tegaki.npy"

tokens = CharToIndex(chars_file_path)
data = np.load(file_path,allow_pickle=True)

EMBEDDING_DIM = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 64
VOCAB_SIZE = len(tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tegaki_dataset = BinaryClassDataset(data,chars_file_path,device=device)


class TenHotEncodeLayer(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        hot_out = torch.zeros(x.size(0),self.num_tokens)

        for N in range(x.size(0)):
            for F in x[N]:
                hot_out[N,F.long()]=1
        return hot_out.to(self.device)


class Net(nn.Module):
  def __init__(self,encode_size):
    super(Net, self).__init__()
    self.encoder = TenHotEncodeLayer(encode_size)
    self.fc1 = nn.Linear(encode_size, 128)
    self.fc2 = nn.Linear(128, 2)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, x):
    x = self.encoder(x)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


def show_detail(x,y,p):
    for _x,_y,_p in zip(x,y,p): #バッチ
        #xの表示
        for __x in _x:
            print(tokens.get_decoded_char(int(__x.item())),end='')
        print('\t\t(予想、答え)=(',end='')
        if _p.item() == 1:
            print('〇,',end='')
        if _p.item() == 0:
            print('Ｘ,',end='')
        if _y.item() == 1:
            print('〇',end='')
        if _y.item() == 0:
            print('Ｘ',end='')

        if _p.item() == _y.item():
            print(')--> 正解')
        else:
            print(')--> 不正解')



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

def eval(model,valid_dataloader,is_show_detail=False):
    THRESHOLD = 0.1
    accuracy = 0
    batch_size = next(iter(valid_dataloader))[0].size(0)
    confusion_matrix = torch.zeros(2,2)
    threshold = torch.full((batch_size,2),THRESHOLD).to(device)
    model.eval()

    for x,y in valid_dataloader:
        output = model(x)
        output = F.softmax(output,dim=1)
        compare = (output.data > threshold).float()
        prediction = compare.data.max(1)[1] #予測結果

        accuracy += prediction.eq(y.data).sum().item()/batch_size

        if is_show_detail:
          show_detail(x,y,prediction)

        for y_true,y_pred in zip(y,prediction):
          confusion_matrix[y_true,y_pred]+=1
    return accuracy/len(valid_dataloader), confusion_matrix



import time
import math
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


final_accuracies = []
final_losses = []

cross_validation = Cross_Validation(tegaki_dataset)
k_num = cross_validation.k_num #デフォルトは10
# k_num=1
confusion_matrix = torch.zeros(2,2)

##学習
for i in range(k_num):
    train_dataset,valid_dataset = cross_validation.get_datasets(k_idx=i)

    print(f'Cross Validation: k=[{i+1}/{k_num}]')

    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True) #訓練データのみシャッフル
    valid_dataloader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
    model = Net(encode_size=len(tokens))

    epochs = 100
    # epochs = 1
    acc_record=[]
    loss_record=[]
    start = time.time() #開始時間の設定

    for epoch in range(1,epochs+1):
        #進捗表示
        i = (epoch-1)%10
        pro_bar = ('=' * i) + (' ' * (10 - i))
        print('\r[{0}] {1}%'.format(pro_bar, i / 10 * 100.), end='')


        loss,acc = train(model,train_dataloader,learning_rate=0.01)

        valid_acc,conf_mat = eval(model,valid_dataloader)
        loss_record.append(loss)
        acc_record.append(valid_acc)
        confusion_matrix += conf_mat


        if epoch%10==0:
            print(f'\repoch:[{epoch:3}/{epochs}] | {timeSince(start)} - loss: {loss:.7},  accuracy: {acc:.7},  valid_acc: {valid_acc:.7}',end=',  ')

            #recall,precision計算
            tp = conf_mat[1,1]
            tn = conf_mat[0,0]
            fp = conf_mat[0,1]
            fn = conf_mat[1,0]
            print(f'precision: {tn/(tn+fn):.7},    recall: {tn/(tn+fp):.7}')
            start = time.time() #開始時間の設定

    #学習結果の表示
    # eval(model,valid_dataloader,is_show_detail=True)

    print(f'final_loss: {loss_record[-1]:.7},   final_accuracy:{acc_record[-1]:.7}\n\n')

    final_accuracies.append(acc_record[-1])
    final_losses.append(loss_record[-1])
print('======================================================')
print('Detector THE Compare Threshold: 0.1')
print(f'accuracies: {final_accuracies}')
print(f'losses: {final_losses}')

print(f'accu average: {np.mean(final_accuracies)}')
print(f'loss average: {np.mean(final_losses)}')
print(f'confusion matrix: {confusion_matrix}')


tp = confusion_matrix[1][1]
tn = confusion_matrix[0][0]
fp = confusion_matrix[0][1]
fn = confusion_matrix[1][0]

recall_posi = tp/(tp+fn)
precision_posi = tp/(tp+fp)

recall_neg = tn/(tn+fp)
precision_neg = tn/(tn+fn)
print(f'recall_posi:{recall_posi}')
print(f'precision_posi:{precision_posi}')
print(f'recall_neg:{recall_neg}')
print(f'precision_neg:{precision_neg}')





