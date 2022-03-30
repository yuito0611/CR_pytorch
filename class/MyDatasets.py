import sys
sys.path.append("/net/nfs2/export/home/ohno/CR_pytorch/class")
from CharToIndex import CharToIndex
import torch
import numpy as np

# Xが(10,)、Yが(1,)の文字からなるデータを半角統一してインデックス化して
# Xが(9,10)、Yが(1,)からなるデータセットにするクラス
class BaseDataset_set9(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.len = len(data[0])-8
        self.device = device

        values = data[0]
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
        out_val = self.val_idx[idx:idx+9]
        out_ans = self.ans_idx[idx+8]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)


# Xが(10,)、Yが(1,)の文字からなるデータを半角統一してインデックス化して
# Xが(7,10)、Yが(1,)からなるデータセットにするクラス
class BaseDataset_set7(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.len = len(data[0])-6
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
        out_val = self.val_idx[idx:idx+7]
        out_ans = self.ans_idx[idx+6]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)


# Xが(10,)、Yが(1,)の文字からなるデータを半角統一してインデックス化して
# Xが(5,10)、Yが(1,)からなるデータセットにするクラス
class BaseDataset_set5(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.len = len(data[0])-4
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
        out_val = self.val_idx[idx:idx+5]
        out_ans = self.ans_idx[idx+4]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)

# Xが(10,)、Yが(1,)の文字からなるデータを半角統一してインデックス化して
# Xが(3,10)、Yが(1,)からなるデータセットにするクラス
class BaseDataset_set3(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.len = len(data[0])-4
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
        out_val = self.val_idx[idx:idx+3]
        out_ans = self.ans_idx[idx+2]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)


class BaseDataset_set2(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.len = len(data[0])-2
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
        out_val = self.val_idx[idx:idx+2]
        out_ans = self.ans_idx[idx+1]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)




# Xが(10,)、Yが(1,)の文字からなるデータを半角統一してインデックス化して
# Xが(10,)、Yが(1,)からなるデータセットにするクラス
class BaseDataset_set1(torch.utils.data.Dataset):
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
        out_val = self.val_idx[idx]
        out_ans = self.ans_idx[idx]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)

# Xが(10,)、Yが(1,)の文字からなるデータを半角統一してインデックス化して
# Xが(5,10)、Yが(1,)からなるデータセットにするクラス
#ただし3文字をみて2文字目を予測する用
class CenterCharDataset_set3(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.len = len(data[0])-1
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
        out_val = self.val_idx[idx:idx+3]
        out_ans = self.ans_idx[idx+1]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)



# Xが(10,)、Yが(1,)の文字からなるデータを半角統一してインデックス化して
# Xが(5,10)、Yが(1,)からなるデータセットにするクラス
#ただし5文字をみて3文字目を予測する用
class CenterCharDataset_set5(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.len = len(data[0])-4
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
        out_val = self.val_idx[idx:idx+5]
        out_ans = self.ans_idx[idx+2]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)


# Xが(10,)、Yが(1,)の文字からなるデータを半角統一してインデックス化して
# Xが(5,10)、Yが(1,)からなるデータセットにするクラス
#ただし7文字をみて4文字目を予測する用
class CenterCharDataset_set7(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.len = len(data[0])-6
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
        out_val = self.val_idx[idx:idx+7]
        out_ans = self.ans_idx[idx+3]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)


# Xが(10,)、Yが(1,)の文字からなるデータを半角統一してインデックス化して
# Xが(5,10)、Yが(1,)からなるデータセットにするクラス
#ただし9文字をみて5文字目を予測する用
class CenterCharDataset_set9(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.len = len(data[0])-8
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
        out_val = self.val_idx[idx:idx+9]
        out_ans = self.ans_idx[idx+4]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)


#交差検証法　（データ全体を10個のまとまりに分割し、9個を訓練データ、1個を検証データに分けて返す）
class Cross_Validation():
    def __init__(self,dataset,k=10,shuffle=False):
        import random

        self.k_num=k
        self.dataset = dataset
        whole_length = len(self.dataset)
        i_sets = []
        pre_point = 0
        split_num = 10
        sets_length = int(whole_length/split_num)
        self.valid_ratio = 1 #訓練用：検証用＝９：１

        for i in range(split_num):
            i_sets.append(list(range(pre_point,pre_point+sets_length)))
            pre_point+=sets_length

        if shuffle == True:
            self.indices_sets = np.array(random.sample(i_sets,split_num))
        else :
            self.indices_sets = np.array(i_sets)



    def get_datasets(self,k_idx):
        import itertools

        ind = np.ones(10, dtype=bool)
        ind[k_idx] = False

        valid_indices = list(self.indices_sets[k_idx])
        train_indices = list(itertools.chain.from_iterable(self.indices_sets[ind.tolist()]))

        from torch.utils.data import Subset
        valid_dataset = Subset(self.dataset,valid_indices)
        train_dataset = Subset(self.dataset,train_indices)

        return train_dataset,valid_dataset
