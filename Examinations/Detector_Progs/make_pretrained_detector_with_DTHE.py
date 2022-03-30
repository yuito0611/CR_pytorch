
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from CharToIndex import CharToIndex
from MyDatasets import Cross_Validation
import torch.nn.functional as F


class Distanced_TenHot_Dataset_set(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])
        self.device = device

        values = data['value']
        for chars in values:
            indexes = []
            for idx in map(self.char2index.get_index,chars):
                indexes.append(idx)
            self.val_idx.append(indexes)

        answers = data['answer']
        for idx in map(self.char2index.get_index,answers):
            self.ans_idx.append(idx)


        #距離値付きのten_hot_encodeにvalueを変換
        distances = data['distance']
        self.distanced_ten_hot_encoded_value = np.zeros(shape=(values.shape[0],len(self.char2index)),dtype=np.float32)
        for row,indicies in enumerate(self.val_idx):
            for id_distance,id_value in enumerate(indicies):
                self.distanced_ten_hot_encoded_value[row][id_value]=distances[row][id_distance]


    def __len__(self):
        return self.length


    def __getitem__(self,idx):
        assert idx < self.length
        out_val = self.distanced_ten_hot_encoded_value[idx]

        #OCRの第一候補と答えが等しければ１、等しくなければ０
        if self.val_idx[idx][0] == self.ans_idx[idx]:
            out_ans = 1
        else:
            out_ans = 0
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)


chars_file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/all_chars_3812.npy"
tokens = CharToIndex(chars_file_path)
data_file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/katsuji_distance.npz"
data = np.load(data_file_path,allow_pickle=True)

EMBEDDING_DIM = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 64
VOCAB_SIZE = len(tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Distanced_TenHot_Dataset_set(data,chars_file_path,device=device)


class Net(nn.Module):
  def __init__(self,encode_size):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(encode_size, 128)
    self.fc2 = nn.Linear(128, 2)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, x):
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



def eval(model,valid_dataloader,threshold_value,is_show_detail=False):
    accuracy = 0
    batch_size = next(iter(valid_dataloader))[0].size(0)
    confusion_matrix = torch.zeros(2,2)
    threshold = torch.full((batch_size,2),threshold_value).to(device)
    model.eval()

    for x,y in valid_dataloader:
        output = model(x)
        output = F.softmax(output,dim=1)
        compare = (output.data > threshold).long()
        prediction = compare[:,1] #予測結果
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




##学習

dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
model = Net(encode_size=len(tokens))

epochs = 100


for epoch in range(1,epochs+1):

    loss,acc = train(model,dataloader,learning_rate=0.01)

model_save_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/pre_trained_detector_distance.pth"

torch.save(model.state_dict(), model_save_path)







