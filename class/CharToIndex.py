import numpy as np
import mojimoji 

class CharToIndex():
    def __init__(self,chars_file_path):
        ori_chars = np.load(chars_file_path,allow_pickle=True) #バイト形式の文字（半角と全角）
        han_chars = []
        for char in ori_chars:
            if isinstance(char,str):
                han_chars.append(mojimoji.zen_to_han(char.tolist()).encode())
            if isinstance(char,bytes):
                han_chars.append(mojimoji.zen_to_han(char.decode()).encode())
        han_chars = np.unique(han_chars) #半角で統一
        self.table = {b'<UNK>':0}
        self.table.update({char:idx+1 for idx,char in enumerate(han_chars)}) #対応表
        self.char_num = len(self.table.items())

    def __str__(self):
        return str(self.table)

    def __len__(self):
        return len(self.table)



    #charを取得するがencodeされた形式 (例: b'\xe7\x9f\xa5'
    def get_char(self,index):
        if isinstance(index, int)==False:
            print("\033[31m"+"ERROR: index must be int"+" \033[0m")
            return b'<UNK>'

        try:
            return [k for k, v in self.table.items() if v == index][0]
        except IndexError:
            print("\033[31m"+"ERROR: No such index -->"+f" \033[0m{index}")
            return b'<UNK>'


    #日本語で取得 (例: b'\xe7\x9f\xa5'->'知'
    def get_decoded_char(self,index):
        if isinstance(index, int)==False:
            print("\033[31m"+"ERROR: index must be int"+" \033[0m")
            return None

        try:
            return [k for k, v in self.table.items() if v == index][0].decode()
        except IndexError:
            print("\033[31m"+"ERROR: No such index -->"+f" \033[0m{index}")
            return '<UNK>'


    #文字から対応するインデックスを取得
    def get_index(self,char):
        if isinstance(char,str):
            char = mojimoji.zen_to_han(char).encode()
        if isinstance(char,bytes):
            char = mojimoji.zen_to_han(char.decode()).encode()
        if isinstance(char,bytes)==False:
            print("\033[31m"+f"ERROR: arg must be <class 'str'> or <class 'bytes'>\nbut input arg is {char}"+" \033[0m")
            return None

        try:
            return self.table[char]
        except KeyError:
            print("\033[31m"+"ERROR: No such char -->"+f" \033[0m{char}")
            return 0

    def decoded_table(self):
        for i,item in enumerate(self.table.keys()):
            print(f'\'{item.decode()}\':{i}',end=', ')