import torch
import numpy as np

class measured_y_txt_dataset256_fast(torch.utils.data.Dataset):
    # 从txt文件中读矩阵
    def __init__(self,data,data_shape):

        self.data = data
        self.data_shape = data_shape
        
    def __getitem__(self,idx):

        matrix = self.data[0:self.data_shape[0],0:self.data_shape[1]]

        matrix = matrix.reshape(1,matrix.shape[0],matrix.shape[1])
        


        # return x,y
        return matrix,matrix #使数据类型和模型参数类型一样

    def __len__(self):

        return 1
class measured_y_txt_dataset(torch.utils.data.Dataset):
    # 从txt文件中读矩阵
    def __init__(self,txt_path):

        self.txt_path = txt_path
        print(f'building datasets {self.txt_path}')

    def __getitem__(self,idx):


        matrix = np.loadtxt(self.txt_path,dtype=np.float32,delimiter=",") # frame:文件

        matrix = matrix.reshape(1,matrix.shape[0],matrix.shape[1])
        
        print(f'max of measured_y:{matrix.max()}')


        x = torch.tensor(matrix)

        # return x,y
        return x,x #使数据类型和模型参数类型一样
        


    def __len__(self):

        return 1

class measured_y_txt_dataset256(torch.utils.data.Dataset):
    # 从txt文件中读矩阵
    def __init__(self,txt_path):

        self.txt_path = txt_path
        print(f'building datasets {self.txt_path}')

    def __getitem__(self,idx):
        sac = 1
        shape = [int(1356*sac),int(2040*sac)]

        a = np.loadtxt(self.txt_path,dtype=np.float32,delimiter=",") # frame:文件

        matrix = a[0:1024,0:1920]

        matrix = matrix.reshape(1,matrix.shape[0],matrix.shape[1])
        
        print(f'max of measured_y:{matrix.max()}')


        x = torch.tensor(matrix)

        # return x,y
        return x,x #使数据类型和模型参数类型一样
        


    def __len__(self):

        return 1
class measured_y_txt_dataset(torch.utils.data.Dataset):
    # 从txt文件中读矩阵
    def __init__(self,txt_path):

        self.txt_path = txt_path
        print(f'building datasets {self.txt_path}')

    def __getitem__(self,idx):


        matrix = np.loadtxt(self.txt_path,dtype=np.float32,delimiter=",") # frame:文件

        matrix = matrix.reshape(1,matrix.shape[0],matrix.shape[1])
        
        print(f'max of measured_y:{matrix.max()}')


        x = torch.tensor(matrix)

        # return x,y
        return x,x #使数据类型和模型参数类型一样
        


    def __len__(self):

        return 1