import os
import math
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
from torchvision.io import read_image
import pandas
import cv2
from torchvision.transforms import ToTensor
import numpy as np
from torchvision import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.signal as signal


def my_readimage(image_path):
    '''
    读入图片:[0-255]array(256, 256, 3) ->[0,1]tensor torch.Size([1, 256, 256])
    '''
    imgcv = cv2.imread(image_path)
    # print(f'shape of image:{imgcv.shape}')
    # print(f'type of image:{imgcv.dtype}')
    # print(f'max of image:{imgcv.max()}')
    transform = transforms.Compose([
    transforms.ToTensor()
])
    
    imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY) #三通道转换为单通道
    # print(f'shape of image:{imgcv.shape}')
    # print(f'type of image:{imgcv.dtype}')
    # print(f'max of image:{imgcv.max()}')
    imgcvb = transform(imgcv) #将一个取值范围在[0,255]的numpy.ndarray图像转换为[0,1.0]的torch.FloadTensor图像，同时各维度的顺序也将自动调整。
    # print(f'shape of image:{imgcvb.shape}')
    # print(f'type of image:{imgcvb.dtype}')
    # print(f'max of image:{imgcvb.max()}')
    return imgcvb

def my_saveimage(matrix,image_path,cmap='viridis'):
          
    '''
    matrix:float32 [H,W]
    '''
    # plt.clf() # 清图。
    # plt.cla() # 清坐标轴

    # # print(f'shape of measured_y:{matrix.shape}')
    # # print(f'type of measured_y:{matrix.dtype}')
    # # print(f'max of measured_y:{matrix.max()}')
    # ax = plt.subplot()

    # im = ax.imshow(matrix,resample=True)

    # # create an Axes on the right side of ax. The width of cax will be 5%
    # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)

    # plt.colorbar(im, cax=cax)
    # plt.savefig(image_path)

    plt.clf() # 清图。
    plt.cla() # 清坐标轴

    # cmap1 = copy.copy(mpl.cm.viridis)
    # norm1 = mpl.colors.Normalize(vmin=0, vmax=100)
    # im1 = mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1)
    # ax = plt.subplot()
    imgplot = plt.imshow(matrix,cmap=cmap)
    plt.colorbar()

    plt.savefig(image_path)
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
def my_saveimage_plus(matrix,image_path):
          
    '''
    matrix:float32 [H,W]
    '''
    plt.figure(dpi=1000) # 清图。


    # print(f'shape of measured_y:{matrix.shape}')
    # print(f'type of measured_y:{matrix.dtype}')
    # print(f'max of measured_y:{matrix.max()}')
    ax = plt.subplot()

    im = ax.imshow(matrix,resample=True)

    # create an Axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.savefig(image_path)



      
def my_savetxt(matrix,txt_path):
    '''
    matrix:float32 [H,W]
    '''      

    np.savetxt(txt_path,matrix,fmt='%.10e',delimiter=",") #frame: 相位图 array:存入文件的数组

def my_readtxt(txt_path):
     matrix = np.loadtxt(txt_path,dtype=np.float32,delimiter=",") # frame:文件
     return matrix

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")

def visual_data(dataloader,root_path):

    for x,y in dataloader:
        print(f'shape of input [N,C,H,W]:{x.shape},{x.dtype} {x.max()}')
        print(f'shape of output:{y.shape},{x.dtype}')
        # print(f'all attribute of x:{dir(x)}')

        '''
        torchvision.utils.save_image(tensor, fp)
        # 参数
        # tensor(Tensor or list)：待保存的tensor数据（可以是上述处理好的grid）。如果给以一个四维的batch的tensor，将调用网格方法，然后再保存到本地。最后保存的图片是不带batch的。
        # fp：图片保存路径
        '''

        # torchvision.utils.save_image(x/x.max(),f'{root_path}/input.jpg')
        # torchvision.utils.save_image(y/y.max(),f'{root_path}/label.jpg')
        my_saveimage(x.reshape(x.shape[2],x.shape[3]),f'{root_path}/input.png')
        my_saveimage(y.reshape(x.shape[2],x.shape[3]),f'{root_path}/label.png')
        my_savetxt(x.reshape(x.shape[2],x.shape[3]),f'{root_path}/input.txt')
        my_savetxt(y.reshape(x.shape[2],x.shape[3]),f'{root_path}/label.txt')

        # print(f'label:{y}')

        break


# def fresnel_dfft(
#         inpt,  wavelength, nx, ny, xstart, ystart, xend, yend, distance
#         ):
    
#     '''
#         这段代码是用于计算光的菲涅耳衍射模式的。它使用了离散傅立叶变换（DFFT）来模拟光在一定距离后的强度分布。
#     '''

#     print('fresnel_dfft')


#     print(torch.max(inpt))

#     inpt = torch.exp(1j * inpt) #wrap到0-2pi

#     # wave number k
#     wave_num = 2*torch.pi / wavelength
    
#     # the axis in frequency space
#     qx = torch.linspace(0.25/xstart, 0.25/xend, nx) * nx
#     qy = torch.linspace(0.25/ystart, 0.25/yend, ny) * ny
    
#     mesh_qx, mesh_qy = torch.meshgrid(qx, qy)
    
#     # the propagation function
#     impulse_q = torch.exp(
#         (1j * wave_num * distance) * 
#         (1 - wavelength**2 * (mesh_qx**2 + mesh_qy**2))/2
#         )
    
#     inpt.to(device)
#     impulse_q.to(device)
    
#     part1 = torch.fft.fft2(inpt).to(device)
#     part2 = torch.fft.ifftshift(impulse_q).to(device)
    
#     diffraction = torch.fft.ifft2(part1 * part2)
#     intensity = torch.abs(diffraction) * torch.abs(diffraction)
    
#     return intensity

# def fresnel_dfft_guiyi(
#         inpt,  wavelength, nx, ny, xstart, ystart, xend, yend, distance
#         ):
    
#     '''
#         这段代码是用于计算光的菲涅耳衍射模式的。它使用了离散傅立叶变换（DFFT）来模拟光在一定距离后的强度分布。
#     '''

#     print('fresnel_dfft')


#     print(torch.max(inpt))

#     inpt = torch.exp(1j * inpt) #wrap到0-2pi

#     # wave number k
#     wave_num = 2*torch.pi / wavelength
    
#     # the axis in frequency space
#     qx = torch.linspace(0.25/xstart, 0.25/xend, nx) * nx
#     qy = torch.linspace(0.25/ystart, 0.25/yend, ny) * ny
    
#     mesh_qx, mesh_qy = torch.meshgrid(qx, qy)
    
#     # the propagation function
#     impulse_q = torch.exp(
#         (1j * wave_num * distance) * 
#         (1 - wavelength**2 * (mesh_qx**2 + mesh_qy**2))/2
#         )
    
#     inpt.to(device)
#     impulse_q.to(device)
    
#     part1 = torch.fft.fft2(inpt).to(device)
#     part2 = torch.fft.ifftshift(impulse_q).to(device)
    
#     diffraction = torch.fft.ifft2(part1 * part2)
#     intensity = torch.abs(diffraction) * torch.abs(diffraction)
    
#     return intensity / torch.max(intensity)

# def fresnel_dfft_wrap_2_2pi(
#         inpt,  wavelength, nx, ny, xstart, ystart, xend, yend, distance
#         ):
#     print('fresnel_dfft_wrap_2_2pi')
#     print(torch.max(inpt))

#     inpt = inpt%(2*torch.pi) #将输入的相位限制在0到2π之间。

#     print(torch.max(inpt))

#     inpt = torch.exp(1j * inpt) #将相位转换为复数形式，以便进行傅立叶变换。

#     # wave number k
#     wave_num = 2*torch.pi / wavelength #计算了光的波数，波数是波长的倒数。
    
#     # the axis in frequency space
#     qx = torch.linspace(0.25/xstart, 0.25/xend, nx) * nx
#     qy = torch.linspace(0.25/ystart, 0.25/yend, ny) * ny #qx 和 qy 是频率空间的轴，它们是通过对空间轴进行傅立叶变换得到的。
    
#     mesh_qx, mesh_qy = torch.meshgrid(qx, qy) #生成了一个二维网格，用于计算每个点的传播函数。
    
#     # the propagation function
#     impulse_q = torch.exp(
#         (1j * wave_num * distance) * 
#         (1 - wavelength**2 * (mesh_qx**2 + mesh_qy**2))/2
#         ) #计算了传播函数，它描述了光在传播过程中的相位变化。
    
#     inpt.to(device)
#     impulse_q.to(device)
    
#     part1 = torch.fft.fft2(inpt).to(device) #分别计算了输入光场和传播函数的傅立叶变换。
#     part2 = torch.fft.ifftshift(impulse_q).to(device) #分别计算了输入光场和传播函数的傅立叶变换。
    
#     diffraction = torch.fft.ifft2(part1 * part2) #计算了衍射模式，它是输入光场和传播函数的傅立叶变换的乘积的逆傅立叶变换。
#     intensity = torch.abs(diffraction) * torch.abs(diffraction) #算了衍射模式的强度，它是衍射模式的绝对值的平方。
    
#     return intensity




def result_visual(pred_ref_path,pred_sam_path,gt_ref_path,gt_sam_path,save_path,cmap='viridis'):
    pred_ref = my_readtxt(pred_ref_path)
    gt_ref = my_readtxt(gt_ref_path)

    pred_sam = my_readtxt(pred_sam_path) 
    gt_sam = my_readtxt(gt_sam_path) 

    pred_sam_pred_ref = sam_ref(pred_sam,pred_ref)
    pred_sam_gt_ref = sam_ref(pred_sam,gt_ref)
    gt_sam_pred_ref = sam_ref(gt_sam,pred_ref)
    gt_sam_gt_ref = sam_ref(gt_sam,gt_ref)
    


    my_saveimage(pred_sam_pred_ref,f'{save_path}/pred_sam_pred_ref.png',cmap)
    my_saveimage(pred_sam_gt_ref,f'{save_path}/pred_sam_gt_ref.png',cmap)
    my_saveimage(gt_sam_pred_ref,f'{save_path}/gt_sam_pred_ref.png',cmap)
    my_saveimage(gt_sam_gt_ref,f'{save_path}/gt_sam_gt_ref.png',cmap)

    # my_saveimage(gt_sam_gt_ref,f'{save_path}/{cmap}_gt_sam_gt_ref.png',cmap)


def sam_ref(sam,ref):
    diff = (sam - ref + 2.3)%(np.pi)
    diff = signal.medfilt(diff,(7,7)) #二维中值滤波    
    return diff

def sam_ref_2pi(sam,ref):
    diff = (sam - ref + 4.1)%(np.pi*2)
    # diff = (sam - ref)%(np.pi*2)
    diff = signal.medfilt(diff,(11,11)) #二维中值滤波    
    return diff

       



if __name__=='__main__':

    # cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
    #         'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    #         'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper','PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    #                   'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic','twilight', 'twilight_shifted', 'hsv','flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    #                   'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    #                   'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
    #                   'turbo', 'nipy_spectral', 'gist_ncar']

    ref_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/1536_1536_ref_pi_01_prop_pi/2024-02-01-18-15/img_txt_folder/9000_pred.txt'
    sam_path=  '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/1536_1536_sam_pi_01_prop_pi/2024-02-01-18-53/img_txt_folder/4500_pred.txt'
    gt_ref_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_ref_pi_01_prop_pi.txt'
    gt_sam_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_sam_pi_01_prop_pi.txt'
    
    # ref = my_readtxt(sam_path)
    # sam = my_readtxt(gt_sam_path)   
    # diff = (sam - ref)
    save_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result_visual/baseline'
    # my_saveimage(diff,save_path)  

    # for cmap in tqdm(cmaps):
    #     result_visual(ref_path,sam_path,gt_ref_path,gt_sam_path,save_path,cmap)
    result_visual(ref_path,sam_path,gt_ref_path,gt_sam_path,save_path)
    # ref = my_readtxt(ref_path)
    # sam = my_readtxt(gt_sam_path)
    # diff = (sam - ref + 2.3)%(np.pi)

    # import scipy.signal as signal
    # diff = signal.medfilt(diff,(7,7)) #二维中值滤波
    # save_path = './ref_ref.png'
    # my_saveimage(diff,save_path)
    # dif = my_readtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/512_512_imagenet_prop_pi.txt')
    # dif640 = np.pad(dif,(64,64),'constant')

    # noise = (np.random.random((512,512))+np.random.random((512,512))+np.random.random((512,512)))/3
    # ref = noise*np.pi
    # ref640 = np.pad(ref,(64,64),'constant')

    # noise = my_readtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_phase_ref_prop_pi.txt')
    # ref = noise[768-256:768+256,768-256:768+256]
    # ref640 = np.pad(ref,(64,64),'constant')

    # sam640 = (dif640+ref640)/np.pi
    # # # print(ref.shape())
    # # # print(ref640.shape())
    # # my_saveimage(ref640,'./temp.png')
    # my_saveimage(dif640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/dif640.png')
    # my_savetxt(dif640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/dif640.txt')

    # my_saveimage(ref640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/ref640.png')
    # my_savetxt(ref640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/ref640.txt')
    
    # my_saveimage(sam640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/sam640.png')
    # my_savetxt(sam640,'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/sam640.txt')

