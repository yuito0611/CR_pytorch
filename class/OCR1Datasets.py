#9文字の中心を予測
class OCR1Dataset9(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-8
        self.val_idx = torch.zeros((self.length+8,10),dtype=torch.long)
        self.ans_idx = torch.zeros(self.length+8,dtype=torch.long)
        self.device = device

        for i_r,chars in enumerate(data['value']):
            for i_c, idx in enumerate(map(self.char2index.get_index,chars)):
                self.val_idx[i_r][i_c] = idx

        for i,char in enumerate(data['answer']):
            self.ans_idx[i] = self.char2index.get_index(char)


        #距離値付きのten_hot_encodeにvalueを変換
        distances = np.nan_to_num(data['distance'])
        self.distanced_ten_hot_encoded_value = torch.full((self.length+8,len(self.char2index)),-0.5,dtype=torch.float)
        for row,indicies in enumerate(self.val_idx):
            for id_distance,id_value in enumerate(indicies):
                self.distanced_ten_hot_encoded_value[row][id_value]=distances[row][id_distance]



    def __len__(self):
        return self.length


    def __getitem__(self,index):
        input  = self.val_idx[index:index+9,0].to(self.device)
        target = self.ans_idx[index+4].to(self.device)
        distance = self.distanced_ten_hot_encoded_value[index+4].to(self.device)
        return input,target,distance



#7文字の中心を予測
class OCR1Dataset7(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-6
        self.val_idx = torch.zeros((self.length+6,10),dtype=torch.long)
        self.ans_idx = torch.zeros(self.length+6,dtype=torch.long)
        self.device = device

        for i_r,chars in enumerate(data['value']):
            for i_c, idx in enumerate(map(self.char2index.get_index,chars)):
                self.val_idx[i_r][i_c] = idx

        for i,char in enumerate(data['answer']):
            self.ans_idx[i] = self.char2index.get_index(char)


        #距離値付きのten_hot_encodeにvalueを変換
        distances = np.nan_to_num(data['distance'])
        self.distanced_ten_hot_encoded_value = torch.full((self.length+6,len(self.char2index)),0,dtype=torch.float)
        for row,indicies in enumerate(self.val_idx):
            for id_distance,id_value in enumerate(indicies):
                self.distanced_ten_hot_encoded_value[row][id_value]=distances[row][id_distance]



    def __len__(self):
        return self.length


    def __getitem__(self,index):
        input  = self.val_idx[index:index+7,0].to(self.device)
        target = self.ans_idx[index+3].to(self.device)
        distance = self.distanced_ten_hot_encoded_value[index+3].to(self.device)
        return input,target,distance

#5文字の中心を予測
class OCR1Dataset5(torch.utils.data.Dataset):
    def __init__(self,data,chars_file_path,device=torch.device('cpu')):
        self.data = data
        self.char2index = CharToIndex(chars_file_path)
        self.length = len(data['answer'])-4
        self.val_idx = torch.zeros((self.length+4,10),dtype=torch.long)
        self.ans_idx = torch.zeros(self.length+4,dtype=torch.long)
        self.device = device

        for i_r,chars in enumerate(data['value']):
            for i_c, idx in enumerate(map(self.char2index.get_index,chars)):
                self.val_idx[i_r][i_c] = idx

        for i,char in enumerate(data['answer']):
            self.ans_idx[i] = self.char2index.get_index(char)

        #距離値付きのten_hot_encodeにvalueを変換
        distances = np.nan_to_num(data['distance'])
        self.distanced_ten_hot_encoded_value = torch.full((self.length+4,len(self.char2index)),0,dtype=torch.float)
        for row,indicies in enumerate(self.val_idx):
            for id_distance,id_value in enumerate(indicies):
                self.distanced_ten_hot_encoded_value[row][id_value]=distances[row][id_distance]


    def __len__(self):
        return self.length


    def __getitem__(self,index):
        input  = self.val_idx[index:index+5,0].to(self.device)
        target = self.ans_idx[index+2].to(self.device)
        distance = self.distanced_ten_hot_encoded_value[index+2].to(self.device)
        return input,target,distance