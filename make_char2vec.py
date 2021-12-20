
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math


import sys
sys.path.append("/net/nfs2/export/home/ohno/CR_pytorch/class")
import CharToIndex


#GPUチェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#学習用データ
origin_file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/main/origin.npy"
origin = np.load(origin_file_path,allow_pickle=True)


corpus = origin[0]

#すべての文字リスト
chars_file_path = "/net/nfs2/export/home/ohno/CR_pytorch/data/main/all_chars.npy"
tokens = CharToIndex.CharToIndex(chars_file_path)

print(len(tokens))

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


window_size = 2
idx_pairs = []
# for each sentence
for chars in corpus:
    indices = [tokens.get_index(char) for char in chars]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array



def get_input_layer(word_idx):
    x = torch.zeros(len(tokens)).float()
    x[word_idx] = 1.0
    return x


embedding_dims = 512
W1 = Variable(torch.randn(embedding_dims, len(tokens)).float(), requires_grad=True)
W2 = Variable(torch.randn(len(tokens), embedding_dims).float(), requires_grad=True)
num_epochs = 500
learning_rate = 0.001
loss_record = []

print('start learning ...')
start = time.time() #開始時間の設定

for epo in range(num_epochs+1):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())
        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)
        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()    


    if epo % 10 == 0:    
        print(f'Loss at epo {epo} -{timeSince(start)}: {loss_val/len(idx_pairs)}')
        loss_record.append(loss_val/len(idx_pairs))
        start = time.time() #開始時間の設定
        torch.save(W2,"/net/nfs2/export/home/ohno/CR_pytorch/data/main/char2vec_512")
        if loss_record[-1] < 0.1:
            print(f'learning is end at epoch:{epo}')
            break



torch.save(W2,"/net/nfs2/export/home/ohno/CR_pytorch/data/main/char2vec_512")


import matplotlib.pyplot as plt

fig = plt.figure()
fig1 = fig.add_subplot(1, 2, 1)
fig1.plot(loss_record,color="tomato")
fig1.set_title("loss")



plt.show()


plt.savefig("/net/nfs2/export/home/ohno/CR_pytorch/make_char2vec512.png")



