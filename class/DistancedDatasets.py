from CharToIndex import CharToIndex
import torch
import numpy as np


class Distanced_TenHot_Dataset_set1(torch.utils.data.Dataset):
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
        out_ans = self.ans_idx[idx]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)



class Distanced_TenHot_Dataset_set2(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-1
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
        out_val = self.distanced_ten_hot_encoded_value[idx:idx+2]
        out_ans = self.ans_idx[idx+1]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)



class Distanced_TenHot_Dataset_set3(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-2
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
        out_val = self.distanced_ten_hot_encoded_value[idx:idx+3]
        out_ans = self.ans_idx[idx+2]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)



class Distanced_TenHot_Dataset_set5(torch.utils.data.Dataset):
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



class Distanced_TenHot_Dataset_set7(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-6
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
        out_val = self.distanced_ten_hot_encoded_value[idx:idx+7]
        out_ans = self.ans_idx[idx+6]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)



class Distanced_TenHot_Dataset_set9(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.val_idx = []
        self.ans_idx = []
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-8
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
        out_val = self.distanced_ten_hot_encoded_value[idx:idx+9]
        out_ans = self.ans_idx[idx+8]
        return torch.tensor(out_val).to(self.device),torch.tensor(out_ans).to(self.device)

