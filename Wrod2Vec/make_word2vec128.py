
from gensim.models import word2vec
import numpy as np
from TimeChecker import TimeChecker


EPOCHS = 1000
print('Start making Dataset...')


dataset=np.load("/net/nfs2/export/home/ohno/CR_pytorch/Wrod2Vec/dataset.npy",allow_pickle=True).tolist()
step=128

print('Finish making Dataset !')


running_time = TimeChecker()
running_time.start()

print('Start making Word2Vector...   step:',step)
step*=2

model = word2vec.Word2Vec(dataset,vector_size=128,min_count=1,window=2, epochs=EPOCHS,sg=1,compute_loss=True)

print('Finish making Word2Vector !')

print('実行時間： ',running_time.stop())

#モデルの保存
model.save("/net/nfs2/export/home/ohno/CR_pytorch/Wrod2Vec/word2vec_128.model")
print(model.get_latest_training_loss())

