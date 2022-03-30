import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from CharToIndex import CharToIndex
from MyDatasets import Cross_Validation
import torch.nn.functional as F


class Distanced_TenHot_Dataset_For_Main_set5(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-4
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

        detec_val = self.distanced_ten_hot_encoded_value[idx+4]
        proof_val = self.distanced_ten_hot_encoded_value[idx:idx+5]
        proof_ans = self.ans_idx[idx+4]

        ocr = self.val_idx[idx+4][0] #OCR第一候補

        #OCRの第一候補と答えが等しければ１、等しくなければ０
        if ocr == self.ans_idx[idx+4]:
            detec_ans = 1
        else:
            detec_ans = 0

        detec_val = torch.tensor(detec_val).to(self.device)
        detec_ans = torch.tensor(detec_ans).to(self.device)
        proof_val = torch.tensor(proof_val).to(self.device)
        proof_ans = torch.tensor(proof_ans).to(self.device)
        ocr       = torch.tensor(ocr).to(self.device)

        return detec_val,detec_ans,proof_val,proof_ans,ocr



chars_file_path = r"/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/all_chars_3812.npy"
datas_file_path = r"/net/nfs2/export/home/ohno/CR_pytorch/data/tegaki_katsuji/tegaki_distance.npz"

tokens = CharToIndex(chars_file_path)

data = np.load(datas_file_path,allow_pickle=True)

EMBEDDING_DIM = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 64
VOCAB_SIZE = len(tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tegaki_dataset = Distanced_TenHot_Dataset_For_Main_set5(data,chars_file_path,device=device)


def train(detector,proofreader,train_dataloader,learning_rate=0.001):
    d_criterion = nn.CrossEntropyLoss()
    p_criterion = nn.CrossEntropyLoss()

    d_optim = optim.Adam(detector.parameters(), lr=learning_rate)
    p_optim = optim.Adam(proofreader.parameters(), lr=learning_rate)

    batch_size = next(iter(train_dataloader))[0].size(0)
    d_running_loss = 0
    p_running_loss = 0

    d_runnning_accu = 0
    p_runnning_accu = 0

    detector.train()
    proofreader.train()
    for i,(detec_x,detec_y,proof_x,proof_y,_) in enumerate(train_dataloader):
        #検出器の処理
        d_output = detector(detec_x)
        d_tmp_loss = d_criterion(d_output, detec_y) #損失計算
        d_prediction = d_output.data.max(1)[1] #予測結果
        d_runnning_accu += d_prediction.eq(detec_y).sum().item()/batch_size
        d_optim.zero_grad() #勾配初期化
        d_tmp_loss.backward(retain_graph=True) #逆伝播
        d_optim.step()  #重み更新
        d_running_loss += d_tmp_loss.item()


        #修正器の処理
        p_output = proofreader(proof_x)
        p_tmp_loss = p_criterion(p_output, proof_y) #損失計算
        p_prediction = p_output.data.max(1)[1] #予測結果
        p_runnning_accu += p_prediction.eq(proof_y.data).sum().item()/batch_size
        p_optim.zero_grad() #勾配初期化
        p_tmp_loss.backward(retain_graph=True) #逆伝播
        p_optim.step()  #重み更新
        p_running_loss += p_tmp_loss.item()

    p_loss = p_running_loss/len(train_dataloader)
    p_accu = p_runnning_accu/len(train_dataloader)
    d_loss = d_running_loss/len(train_dataloader)
    d_accu = d_runnning_accu/len(train_dataloader)

    return d_loss,d_accu,p_loss,p_accu


def eval(detector,proofreader,valid_dataloader):
    batch_size = next(iter(valid_dataloader))[0].size(0)

    d_runnning_accu = 0
    p_runnning_accu = 0

    detector.eval()
    proofreader.eval()

    for detec_x,detec_y,proof_x,proof_y,_ in valid_dataloader:
        #検出器の処理
        d_output = detector(detec_x)
        d_prediction = d_output.data.max(1)[1] #予測結果
        d_runnning_accu += d_prediction.eq(detec_y).sum().item()/batch_size


        #修正器の処理
        p_output = proofreader(proof_x)
        p_prediction = p_output.data.max(1)[1] #予測結果
        p_runnning_accu += p_prediction.eq(proof_y.data).sum().item()/batch_size

    p_accu = p_runnning_accu/len(valid_dataloader)
    d_accu = d_runnning_accu/len(valid_dataloader)

    return d_accu,p_accu



def examination(detector,proofreader,valid_dataloader,show_out=False):
    confusion_matrix = torch.zeros(2,2)
    batch_size = next(iter(valid_dataloader))[0].size(0)

    runnning_accu = 0
    threshold = torch.full((batch_size,2),0.5).to(device)

    detector.eval()
    proofreader.eval()


    for detec_x,_,proof_x,proof_y,ocr_pred in valid_dataloader:
        #検出器の処理
        d_output = detector(detec_x)
        d_output = F.softmax(d_output,dim=1)
        compare_threshold = (d_output > threshold).long()
        flg_ocr = compare_threshold[:,1] #ocrの出力を使用するか
        ocr_pred.mul_(flg_ocr)

        #修正器の処理
        p_output = proofreader(proof_x)
        rnn_pred = p_output.data.max(1)[1] #RNNの予測結果
        flg_rnn = torch.logical_not(flg_ocr,out=torch.empty(batch_size,dtype=torch.long).to(device))#rnnの出力を使用するか
        rnn_pred.mul_(flg_rnn)

        prediction = torch.add(ocr_pred,rnn_pred)
        runnning_accu += prediction.eq(proof_y.data).sum().item()/batch_size

        if show_out:
            for idx in proof_y.data:
                print(tokens.get_decoded_char(idx),end='')
            print()
            for idx in prediction:
                print(tokens.get_decoded_char(idx),end='')
            print()

    accuracy = runnning_accu/len(valid_dataloader)
    return accuracy




import time
import math
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


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


result_output_file_path = r"/net/nfs2/export/home/ohno/CR_pytorch/results/Main/main_DTHE_epoch500.txt"
text_file = open(result_output_file_path,"a") #結果の保存


cross_validation = Cross_Validation(tegaki_dataset)
k_num = cross_validation.k_num #デフォルトは10
# k_num=1

acc_record = []
d_acc_record=[]
d_loss_record=[]
p_acc_record=[]
p_loss_record=[]

##学習
for i in range(k_num):
    train_dataset,valid_dataset = cross_validation.get_datasets(k_idx=i)

    print(f'Cross Validation: k=[{i+1}/{k_num}]')
    text_file.write(f'Cross Validation: k=[{i+1}/{k_num}]\n')

    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True) #訓練データのみシャッフル
    valid_dataloader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
    proofreader = Proofreader(VOCAB_SIZE, hidden_dim=HIDDEN_SIZE, output_size=VOCAB_SIZE, n_layers=1)
    detector = Detector(encode_size=len(tokens))

    epochs = 500

    start = time.time() #開始時間の設定

    for epoch in range(1,epochs+1):
        #進捗表示
        print(f'\r{epoch}', end='')

        d_loss,d_accu,p_loss,p_accu = train(detector,proofreader,train_dataloader,learning_rate=0.01)
        d_val_accu,p_val_accu = eval(detector,proofreader,valid_dataloader)


        if epoch%10==0:
            print(f'\r epoch:[{epoch:3}/{epochs}]| {timeSince(start)}')
            print(f'  Detector| loss:{d_loss:.5}, accu:{d_accu:.5}, val_accu:{d_val_accu:.5}')
            print(f'  Proof   | loss:{p_loss:.5}, accu:{p_accu:.5}, val_accu:{p_val_accu:.5}')
            text_file.write(f'\r epoch:[{epoch:3}/{epochs}]\n')
            text_file.write(f'  Detector| loss:{d_loss:.5}, accu:{d_accu:.5}, val_accu:{d_val_accu:.5}\n')
            text_file.write(f'  Proof   | loss:{p_loss:.5}, accu:{p_accu:.5}, val_accu:{p_val_accu:.5}\n')
            start = time.time() #開始時間の設定

    #学習結果の表示
    accuracy = examination(detector,proofreader,valid_dataloader,show_out=False)

    d_loss_record.append(d_loss)
    d_acc_record.append(d_val_accu)
    p_loss_record.append(p_loss)
    p_acc_record.append(p_val_accu)
    acc_record.append(accuracy)
    print(f' examin accuracy:{acc_record[-1]:.7}\n\n')
    text_file.write(f' examin accuracy:{acc_record[-1]:.7}\n\n')


print(f'=================================================')
print(f'Detector \nacc: {d_acc_record}')
print(f'loss: {d_loss_record}')
print(f'acc average: {np.mean(d_acc_record)}')
print(f'Proof \nacc: {p_acc_record}')
print(f'loss: {p_loss_record}')
print(f'Examin \nacc: {acc_record}')
print(f'accu average: {np.mean(acc_record)}')

text_file.write(f'=================================================\n\n')
text_file.write(f'Detector \nacc: {d_acc_record}\n')
text_file.write(f'loss: {d_loss_record}\n')
text_file.write(f'acc average: {np.mean(d_acc_record)}\n')
text_file.write(f'Proof \nacc: {p_acc_record}\n')
text_file.write(f'loss: {p_loss_record}\n')
text_file.write(f'Examin \nacc: {acc_record}\n')
text_file.write(f'accu average: {np.mean(acc_record)}\n')

text_file.close()





