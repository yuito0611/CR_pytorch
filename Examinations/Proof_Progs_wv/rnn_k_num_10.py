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
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-8
        self.val_idx = torch.zeros(self.length+8,dtype=torch.long)
        self.ans_idx = torch.zeros(self.length+8,dtype=torch.long)
        self.device = device

        for i,char in enumerate(data['value']):
            self.val_idx[i] = self.char2index.get_index(char[0])

        for i,char in enumerate(data['answer']):
            self.ans_idx[i] = self.char2index.get_index(char)


    def __len__(self):
        return self.length


    def __getitem__(self,index):
        assert index < self.length
        return self.val_idx[index:index+9].to(device),self.ans_idx[index+4].to(device)


class Proofreader(nn.Module):
    def __init__(self, hidden_dim, class_num,n_layers,tokens,c2v):
        super(Proofreader, self).__init__()

        self.output_size = class_num
        self.hidden_dim = hidden_dim
        self.n_layers  = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean_vector = self.make_mean_vector(class_num,tokens,c2v)


        self.rnn = nn.RNN(1024, self.hidden_dim, batch_first=True,bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_dim*2, class_num)
        self.fc2 = nn.Linear(class_num, class_num)

        self.to(self.device)



    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim)
        return hidden


    #意味ベクトルの作成
    def make_mean_vector(self,class_num,tokens,c2v):
        vec_size = c2v.vector_size
        m_vec = torch.zeros((class_num,vec_size),dtype=float)
        for i,char in enumerate(tokens.table):
            try:
                m_vec_i = np.copy(c2v.wv[char.decode()])
                m_vec[i] = torch.from_numpy(m_vec_i)
            except KeyError:
                m_vec[i] = 0
        return m_vec


    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size).to(self.device)

        x_m_vec = self.mean_vector[x[:,4]] #5文字目の意味ベクトル
        x_m_vec = torch.squeeze(x_m_vec)
        attention_mat = torch.zeros(size=(batch_size,self.mean_vector.shape[0]))

        for batch,vec in enumerate(x_m_vec):
            x_m_mat = torch.mul(vec,torch.ones_like(self.mean_vector)) #行列計算のためx_m_vecを列方向に複製
            cos_sims_x = torch.nn.functional.cosine_similarity(self.mean_vector,x_m_mat,dim=1) #cos類似度計算
            attention_mat[batch] =  torch.sigmoid(cos_sims_x).detach().clone() #シグモイド

        embed_x = self.mean_vector[x].float().to(device)
        out, hidden = self.rnn(embed_x, hidden)
        out = out[:,4,:]
        out = self.dropout(out)
        out = self.fc(out)
        out = torch.mul(out,attention_mat.to(device))
        out = self.fc2(out)

        return out

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

    return accuracy/len(valid_dataloader)

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
tegaki_dataset = MyDataset(data,chars_file_path,device=device)
char2vector = gensim.models.Word2Vec.load(c2v_file_path)


cross_validation = Cross_Validation(tegaki_dataset)
k_num = cross_validation.k_num #デフォルトは10
#k_num = 1
p_acc_record=[]
p_loss_record=[]
clock = TimeChecker()#開始時間の設定

##学習
for i in range(k_num):
    train_dataset,valid_dataset = cross_validation.get_datasets(k_idx=i)

    print(f'Cross Validation: k=[{i+1}/{k_num}]')

    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True) #訓練データのみシャッフル
    valid_dataloader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
    proofreader = Proofreader(HIDDEN_SIZE,VOCAB_SIZE,1,tokens,char2vector)

    # epochs = 1
    epochs = 100

    clock.start()

    for epoch in range(1,epochs+1):
        #進捗表示
        print(f'\r{epoch}', end='')


        p_loss,p_accu = train(proofreader,train_dataloader,learning_rate=0.01)

        if epoch%10==0:
            p_val_accu = eval(proofreader,valid_dataloader)
            print(f'\r epoch:[{epoch:3}/{epochs}]| {clock.stop()}')
            print(f'  Proof   | loss:{p_loss:.5}, accu:{p_accu:.5}, val_accu:{p_val_accu:.5}')
            clock.start() #開始時間の設定

    #学習結果の表示


    p_loss_record.append(p_loss)
    p_acc_record.append(p_val_accu)
 

print(f'=================================================')

print(f'Proof \nacc: {p_acc_record}')
print(f'loss: {p_loss_record}')

