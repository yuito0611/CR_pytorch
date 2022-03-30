import torch
import torch.nn as nn
import numpy as np
from CharToIndex import CharToIndex
from MyDatasets import Cross_Validation
from torch import optim, softmax
from torch.utils.data import DataLoader
from MyCustomLayer import WeightedTenHotEncodeLayer
import torch.nn.functional as F
from TimeChecker import TimeChecker

# #5文字の中心を予測
# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self,data,chars_file_path,device=torch.device('cpu')):
#         self.data = data
#         self.char2index = CharToIndex(chars_file_path)
#         self.length = len(data['answer'])-4
#         self.p_val_idx = torch.zeros((self.length+4,10),dtype=torch.long)
#         self.p_ans_idx = torch.zeros(self.length+4,dtype=torch.long)
#         self.d_ans     = torch.zeros(self.length+4,dtype=torch.long)
#         self.device = device

#         for i_r,chars in enumerate(data['value']):
#             for i_c, idx in enumerate(map(self.char2index.get_index,chars)):
#                 self.p_val_idx[i_r][i_c] = idx

#         for i,char in enumerate(data['answer']):
#             self.p_ans_idx[i] = self.char2index.get_index(char)
#             self.d_ans[i] = 1 if self.p_val_idx[i][0] == self.p_ans_idx[i] else 0 #検出器用、OCR第一出力と答えが等しければ１、異なれば０


#         #距離値付きのten_hot_encodeにvalueを変換
#         distances = np.nan_to_num(data['distance'])
#         self.distanced_ten_hot_encoded_value = torch.full((self.length+6,len(self.char2index)),0,dtype=torch.float)
#         for row,indicies in enumerate(self.p_val_idx):
#             for id_distance,id_value in enumerate(indicies):
#                 self.distanced_ten_hot_encoded_value[row][id_value]=distances[row][id_distance]


#     def __len__(self):
#         return self.length


#     def __getitem__(self,index):
#         p_inp  = self.p_val_idx[index:index+5,0].to(device)
#         p_tar = self.p_ans_idx[index+2].to(device)
#         d_ans = self.d_ans[index+2].to(device)
#         distance = self.distanced_ten_hot_encoded_value[index+2].to(device)
#         return distance,d_ans,p_inp,p_tar


##BinaryCrossEntropy用
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-4
        self.p_val_idx = torch.zeros((self.length+4,10),dtype=torch.long)
        self.p_ans_idx = torch.zeros(self.length+4,dtype=torch.long)
        self.d_ans     = torch.zeros((self.length+4,2),dtype=torch.float)
        self.device = device

        for i_r,chars in enumerate(data['value']):
            for i_c, idx in enumerate(map(self.char2index.get_index,chars)):
                self.p_val_idx[i_r][i_c] = idx

        for i,char in enumerate(data['answer']):
            self.p_ans_idx[i] = self.char2index.get_index(char)
            idx = 1 if self.p_val_idx[i][0] == self.p_ans_idx[i] else 0 #検出器用、OCR第一出力と答えが等しければ１、異なれば０
            self.d_ans[i][idx] = 1

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



def train(detector,train_dataloader,learning_rate=0.001):
    # d_criterion = nn.CrossEntropyLoss()
    d_criterion = nn.BCELoss()
    softmax = nn.Softmax(dim=1)
    d_optim = optim.Adam(detector.parameters(), lr=learning_rate)
    batch_size = next(iter(train_dataloader))[0].size(0)
    d_running_loss = 0
    d_runnning_accu = 0

    detector.train()
    for d_x,d_y,p_x,p_y in train_dataloader:
        #検出器の処理
        d_output = detector(d_x)

        d_tmp_loss = d_criterion(softmax(d_output), d_y) #損失計算
        d_prediction = d_output.data.max(1)[1] #予測結果
        d_runnning_accu += d_prediction.eq(d_y.data.max(1)[1]).sum().item()/batch_size
        d_optim.zero_grad() #勾配初期化
        d_tmp_loss.backward(retain_graph=True) #逆伝播
        d_optim.step()  #重み更新
        d_running_loss += d_tmp_loss.item()

    d_loss = d_running_loss/len(train_dataloader)
    d_accu = d_runnning_accu/len(train_dataloader)

    return d_loss,d_accu



class Detector(nn.Module):
  def __init__(self,encode_size):
    super(Detector, self).__init__()
    self.fc1 = nn.Linear(encode_size, 128)
    self.fc2 = nn.Linear(128, 2)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


stopwatch = TimeChecker()


##学習

train_dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True) #訓練データのみシャッフル
detector = Detector(encode_size=len(tokens))

epochs = 100
d_acc_record=[]
d_loss_record=[]
recall_posis = []
recall_negs = []
precision_posis = []
precision_negs = []

stopwatch.start()

for epoch in range(1,epochs+1):
    #進捗表示
    print(f'\r{epoch}', end='')

    d_loss,d_accu = train(detector,train_dataloader,learning_rate=0.01)
    d_loss_record.append(d_loss)


    if epoch%10==0:
        print(f'\r epoch:[{epoch:3}/{epochs}]| {stopwatch.stop()}')
        print(f'  Detector| loss:{d_loss:.5}, accu:{d_accu:.5}')
        stopwatch.start()

torch.save(detector.state_dict(), "/net/nfs2/export/home/ohno/CR_pytorch/Culmination/Learned_models/pre_trained_detector_b")





