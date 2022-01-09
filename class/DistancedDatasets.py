from CharToIndex import CharToIndex
import torch
import numpy as np

class Distanced_TenHot_Dataset_sest5(torch.utils.data.Dataset):
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
        out_val = self.distanced_ten_hot_encoded_value[idx:idx+5]
        out_ans = self.ans_idx[idx+4]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)