
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from CharToIndex import CharToIndex
from DistancedDatasets import Distanced_TenHot_Dataset_sest7 as MyDataset
from MyDatasets import Cross_Validation
from MyCustomLayer import WeightedTenHotEncodeLayer

import time
import math


chars_file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/all_chars_3812.npy"
tokens = CharToIndex(chars_file_path)
file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/tegaki_distance.npz"
data = np.load(file_path,allow_pickle=True)
# chars_file_path = r"data\tegaki_katsuji\all_chars_3812.npy"
# tokens = CharToIndex(chars_file_path)
# file_path = r"data\tegaki_katsuji\tegaki_distance.npz"
# data = np.load(file_path,allow_pickle=True)

EMBEDDING_DIM = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 64
VOCAB_SIZE = len(tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tegaki_dataset = MyDataset(data,chars_file_path,device=device)


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


def eval(model,valid_dataloader,is_show_ans_pred=False):
    accuracy = 0
    batch_size = next(iter(valid_dataloader))[0].size(0)
    model.eval()
    for x,y in valid_dataloader:
        output = model(x)
        prediction = output.data.max(1)[1] #予測結果
        accuracy += prediction.eq(y.data).sum().item()/batch_size
        if is_show_ans_pred:
            ans_pred_list=show_ans_pred(y,prediction)
            print(ans_pred_list)

    return accuracy/len(valid_dataloader)


#hot encode用
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



def get_correct_char(model,valid_dataloader,correct_char):
    accuracy = 0
    batch_size = next(iter(valid_dataloader))[0].size(0)
    model.eval()
    for x,y in valid_dataloader:
        output = model(x)
        prediction = output.data.max(1)[1] #予測結果
        accuracy += prediction.eq(y.data).sum().item()/batch_size

        for correct,idx in zip(prediction.eq(y.data),y.data):
            if correct:
                correct_char[idx]+=1


    return accuracy/len(valid_dataloader),correct_char



final_accuracies = []
final_losses = []
correct_char=torch.zeros(len(tokens),dtype=torch.int)

cross_validation = Cross_Validation(tegaki_dataset)
k_num = cross_validation.k_num #デフォルトは10
# k_num = 1

text_file = open("output.txt","wt") #結果の保存


##学習
for i in range(k_num):
    train_dataset,valid_dataset = cross_validation.get_datasets(k_idx=i)

    print(f'Cross Validation: k=[{i+1}/{k_num}]')
    text_file.write(f'Cross Validation: k=[{i+1}/{k_num}]')

    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True) #訓練データのみシャッフル
    valid_dataloader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
    model = Proofreader(VOCAB_SIZE, hidden_dim=HIDDEN_SIZE, output_size=VOCAB_SIZE, n_layers=1)
    # model.load_state_dict(torch.load("data/tegaki_katsuji/pre_trained_model.pth"))

    epochs = 100
    acc_record=[]
    loss_record=[]
    start = time.time() #開始時間の設定

    for epoch in range(1,epochs+1):
        #進捗表示
        i = (epoch-1)%10
        pro_bar = ('=' * i) + (' ' * (10 - i))
        print('\r[{0}] {1}%'.format(pro_bar, i / 10 * 100.), end='')


        loss,acc = train(model,train_dataloader,learning_rate=0.01)

        valid_acc = eval(model,valid_dataloader)

        loss_record.append(loss)
        acc_record.append(valid_acc)

        if epoch%10==0:
            print(f'\repoch:[{epoch:3}/{epochs}] | {timeSince(start)} - loss: {loss:.7},  accuracy: {acc:.7},  valid_acc: {valid_acc:.7}')
            text_file.write(f'\repoch:[{epoch:3}/{epochs}] | {timeSince(start)} - loss: {loss:.7},  accuracy: {acc:.7},  valid_acc: {valid_acc:.7}')

            start = time.time() #開始時間の設定

    acc,correct_char=get_correct_char(model,valid_dataloader,correct_char)


    print(f'final_loss: {loss_record[-1]:.7},   final_accuracy:{acc_record[-1]:.7}\n\n')
    text_file.write(f'\nfinal_loss: {loss_record[-1]:.7},   final_accuracy:{acc_record[-1]:.7}\n\n')

    final_accuracies.append(acc_record[-1])
    final_losses.append(loss_record[-1])

print(f'=================================================')
print(f'accuracies: {final_accuracies}')
print(f'losses: {final_losses}')

print(f'accu average: {np.mean(final_accuracies)}')
print(f'loss average: {np.mean(final_losses)}')

text_file.write(f'=================================================')
text_file.write(f'\naccuracies: {final_accuracies}')
text_file.write(f'\nlosses: {final_losses}')
text_file.write(f'\naccu average: {np.mean(final_accuracies)}')
text_file.write(f'\nloss average: {np.mean(final_losses)}')

text_file.close()


