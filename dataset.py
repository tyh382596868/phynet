import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import copy
from source_target_transforms import *
class mydataset(torch.utils.data.Dataset):
    # 从txt文件中读矩阵
    def __init__(self,Amp,Pha,transform=None):

        self.Amp = (Amp)      
        self.Pha = (Pha)
        self.transform = transform

        
    def __getitem__(self,idx):

        self.Amp_copy = copy.deepcopy(self.Amp)
        self.Pha_copy = copy.deepcopy(self.Pha)
        print(self.Amp_copy.shape,self.Pha_copy.shape)
        
        if self.transform is not None:
            
            self.Amp_copy = Image.fromarray(self.Amp_copy)
            self.Pha_copy = Image.fromarray(self.Pha_copy)  
            # print(self.Amp_copy.shape,self.Pha_copy.shape)      
                
            self.Amp_copy, self.Pha_copy = self.transform((self.Amp_copy, self.Pha_copy))
            print(self.Amp_copy.shape,self.Pha_copy.shape)
        
        return self.Amp_copy,self.Pha_copy #使数据类型和模型参数类型一样

    def __len__(self):

        return 1 
    
import matplotlib.pyplot as plt

def my_save2image(matrix1, matrix2, image_path, cmap='viridis'):
    '''
    matrix1, matrix2: float32 [H,W] - 分别代表两个要显示的图像矩阵
    image_path: 保存图像的路径
    cmap: 颜色映射
    '''

    plt.clf()  # 清图。
    plt.cla()  # 清坐标轴
    plt.figure(figsize=(12, 6))  # 设定图像大小

    # 显示第一个图像
    plt.subplot(1, 2, 1)
    imgplot1 = plt.imshow(matrix1, cmap=cmap)
    plt.colorbar()  # 为第一个图像添加颜色条

    # 显示第二个图像
    plt.subplot(1, 2, 2)
    imgplot2 = plt.imshow(matrix2, cmap=cmap)
    plt.colorbar()  # 为第二个图像添加颜色条

    plt.savefig(image_path)  # 保存图像

if __name__ == '__main__':

    amp = np.loadtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/pi/amp/baseline/phase_diff_prop001.txt',dtype=np.float32,delimiter=',')
    pha = np.loadtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/pi/pha/baseline/phase_diff.txt',dtype=np.float32,delimiter=',')

    transform = transforms.Compose([
            RandomResizeFromSequence([[192,256],[192*4,256*4],[192*6,256*6],[192*8,256*8],[192*10,256*10],[192*5,256*5]]),
            RandomRotationFromSequence((360)),#[0, 90, 180, 270]
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor()])

    dataset = mydataset(amp,pha,transform=transform)

    # Amp,Pha = dataset[0]
    # my_save2image(Amp[0,:,:].numpy(),Pha[0,:,:].numpy(),'./combined_image.png', cmap='viridis')
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    
    epochs = 10
    for epoch in range(epochs):
        for i, (Amp,Pha) in enumerate(dataloader):
            print(Amp.shape,Pha.shape)

            my_save2image(Amp[0,0,:,:].numpy(),Pha[0,0,:,:].numpy(),f'./{epoch}_combined_image.png', cmap='viridis')
            
            
            
            


