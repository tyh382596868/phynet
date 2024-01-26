import os
import math

import numpy as np
import matplotlib.pyplot as plt

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

def my_saveimage(matrix,image_path):
          
    '''
    matrix:float32 [H,W]
    '''
    plt.clf() # 清图。
    plt.cla() # 清坐标轴

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


def my_saveimage_pro(matrix,image_path):
          
    '''
    matrix:float32 [H,W]
    '''
    plt.figure(figsize=(10,10),dpi=1000)
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


def prop(img,  dx=2.2e-6, dy=2.2e-6, lam=532e-9, dist=0.0788):
    '''
    1.注意传播距离，确保生成图像的传播距离与训练的传播距离一致，一般默认一致
    2.输入数据的维度应该为二维，如果是三维，就算值一样傅里叶变换后也不一样
    '''

    # print(torch.max(img))

    img_phase = img #*torch.pi

    # print(torch.max(img_phase))
    
    H = torch.exp(1j * img_phase) 
    fft_H = torch.fft.ifftshift(torch.fft.fft2(H))
    # H = torch.exp(1j * img) 
    # (Ny,Nx) = H.size()
    # fft_H = torch.fft.ifftshift(torch.fft.fft2(H)).to(device)
    # the axis in frequency space
    # qx = torch.linspace(0.25/xstart, 0.25/xend, nx) * nx
    # qy = torch.linspace(0.25/ystart, 0.25/yend, ny) * ny
    
    # qx = torch.range(1-Nx/2, Nx/2, 1).to(device)
    # qy = torch.range(1-Ny/2, Ny/2, 1).to(device)
    # # print(qx)
    # y, x = torch.meshgrid(qx, qy)
    # # print(f'mesh_qx{mesh_qx}')
    # # print(f'mesh_qy{mesh_qy}')
    (Ny,Nx) = H.shape[0],H.shape[1]
    
    qx = torch.range(1-Nx/2, Nx/2, 1).cuda()
    qy = torch.range(1-Ny/2, Ny/2, 1).cuda()
    y, x = torch.meshgrid(qy, qx)
    r=(2*torch.pi*x/(dx*Nx))**2+(2*torch.pi*y/(dy*Ny))**2

    k=2*torch.pi/lam

    kernel=torch.exp(-1*1j*torch.sqrt(k**2-r)*dist)

    fft_HH=fft_H[:,:]*kernel
    fft_HH=torch.fft.fftshift(fft_HH)

    Ud=torch.fft.ifft2(fft_HH)

    Id=Ud
    Id1=torch.angle(Ud)
    intensity = torch.abs(Id) * torch.abs(Id)
 
    # print(f'torch.max(intensity){torch.max(intensity)}')
    # print(f'intensity{intensity}')

    return intensity,Id1

def back_prop(img,int , dx=2.2e-6, dy=2.2e-6, lam=532e-9, dist=-1*0.0788):
    '''
    1.注意传播距离，确保生成图像的传播距离与训练的传播距离一致，一般默认一致
    2.输入数据的维度应该为二维，如果是三维，就算值一样傅里叶变换后也不一样
    '''

    # print(torch.max(img))

    img_phase = img #*torch.pi

    # print(torch.max(img_phase))
    
    H = torch.exp(1j * img_phase) * int
    fft_H = torch.fft.ifftshift(torch.fft.fft2(H))
    # H = torch.exp(1j * img) 
    # (Ny,Nx) = H.size()
    # fft_H = torch.fft.ifftshift(torch.fft.fft2(H)).to(device)
    # the axis in frequency space
    # qx = torch.linspace(0.25/xstart, 0.25/xend, nx) * nx
    # qy = torch.linspace(0.25/ystart, 0.25/yend, ny) * ny
    
    # qx = torch.range(1-Nx/2, Nx/2, 1).to(device)
    # qy = torch.range(1-Ny/2, Ny/2, 1).to(device)
    # # print(qx)
    # y, x = torch.meshgrid(qx, qy)
    # # print(f'mesh_qx{mesh_qx}')
    # # print(f'mesh_qy{mesh_qy}')
    (Ny,Nx) = H.shape[0],H.shape[1]
    
    qx = torch.range(1-Nx/2, Nx/2, 1).cuda()
    qy = torch.range(1-Ny/2, Ny/2, 1).cuda()
    y, x = torch.meshgrid(qy, qx)
    r=(2*torch.pi*x/(dx*Nx))**2+(2*torch.pi*y/(dy*Ny))**2

    k=2*torch.pi/lam

    kernel=torch.exp(-1*1j*torch.sqrt(k**2-r)*dist)

    fft_HH=fft_H[:,:]*kernel
    fft_HH=torch.fft.fftshift(fft_HH)

    Ud=torch.fft.ifft2(fft_HH)

    Id=Ud
    Id1=torch.angle(Ud)
    intensity = torch.abs(Id) * torch.abs(Id)
 
    # print(f'torch.max(intensity){torch.max(intensity)}')
    # print(f'intensity{intensity}')

    return intensity,Id1


   


if __name__=='__main__':
    
    # 将fast算法和net_model_v1得到的样品相位相减得到比较
    
    fast = 'D:\\tyh\\PhysenNet_git\\rawsam-ref.txt'
    net = 'D:\\tyh\\PhysenNet_git\\sam-ref.txt'

    int1 = my_readtxt(fast)
    int2 = my_readtxt(fast)
    
    dif = (int1-int2)
    my_saveimage_pro(dif,'D:\\tyh\\PhysenNet_git\\fast-net.png')
    my_savetxt(dif,'D:\\tyh\\PhysenNet_git\\fast-net.txt')
    

    
    
    '''
    # 将fast算法和net_model_v1得到的样品相位相减得到比较
    
    fast = 'D:\\tyh\\PhysenNet_git\\rawsam-ref_pi_4.1_fil.txt'
    net = 'D:\\tyh\\PhysenNet_git\\sam-ref_pi_4.1_fil.txt'

    int1 = my_readtxt(fast)
    int2 = my_readtxt(fast)
    
    dif = (int1-int2)
    my_saveimage_pro(dif,'D:\\tyh\\PhysenNet_git\\fast-net_pi_4.1_fil.png')
    my_savetxt(dif,'D:\\tyh\\PhysenNet_git\\fast-net_pi_4.1_fil.txt')    
    
    # 将FAST预测的样品畸变相位与光纤畸变相位相减得到样品相位
    
    img1 = 'D:\\tyh\\PhysenNet_git\\traindata\\gt\\1536_1536_phase_sam_prop_pi.txt'
    img2 = 'D:\\tyh\\PhysenNet_git\\traindata\\gt\\1536_1536_phase_ref_prop_pi.txt'
    maskpath = 'D:\\tyh\\FAST-main\\FAST-main\\mask.txt'
    int1 = my_readtxt(img1)
    int2 = my_readtxt(img2)

    rawmask = my_readtxt(maskpath)
    a = 128*12
    c = 128*12
    shape = [a,c]
    b = 470
    d = 150
    mask = rawmask[0+d :shape[0]+d, 0+b:shape[1]+b]
    
    dif = (int1-int2)
    my_saveimage_pro(dif,'D:\\tyh\\PhysenNet_git\\rawsam-ref.png')
    my_savetxt(dif,'D:\\tyh\\PhysenNet_git\\rawsam-ref.txt')
    
    dif = (int1-int2)/np.pi
    my_saveimage_pro(dif,'D:\\tyh\\PhysenNet_git\\rawsam-ref_pi.png')
    my_savetxt(dif,'D:\\tyh\\PhysenNet_git\\rawsam-ref_pi.txt')
    
    dif = ((int1-int2+4.1)/np.pi)*mask
    my_saveimage_pro(dif,'D:\\tyh\\PhysenNet_git\\rawsam-ref_pi_4.1.png')
    my_savetxt(dif,'D:\\tyh\\PhysenNet_git\\rawsam-ref_pi_4.1.txt')
    
    import scipy.signal as signal
    dif = signal.medfilt(dif,(11,11)) #二维中值滤波
    my_saveimage_pro(dif,'D:\\tyh\\PhysenNet_git\\rawsam-ref_pi_4.1_fil.png')
    my_savetxt(dif,'D:\\tyh\\PhysenNet_git\\rawsam-ref_pi_4.1_fil.txt')
    
    
    # 将net_model_v1预测的样品畸变相位与光纤畸变相位相减得到样品相位
    
    img1 = 'D:\\tyh\PhysenNet_git\\result\\1536_1536_phase_sam_prop_pi\\2024-01-26-13-56\\img_txt_folder\\3500_pred.txt'
    img2 = 'D:\\tyh\PhysenNet_git\\result\\1536_1536_phase_ref_prop_pi\\2024-01-25-15-43\\img_txt_folder\\2000_pred.txt'
    maskpath = 'D:\\tyh\\FAST-main\\FAST-main\\mask.txt'
    int1 = my_readtxt(img1)
    int2 = my_readtxt(img2)
    
    rawmask = my_readtxt(maskpath)
    a = 128*12
    c = 128*12
    shape = [a,c]
    b = 470
    d = 150
    mask = rawmask[0+d :shape[0]+d, 0+b:shape[1]+b]
    
    dif = (int1-int2)
    my_saveimage_pro(dif,'D:\\tyh\\PhysenNet_git\\sam-ref.png')
    my_savetxt(dif,'D:\\tyh\\PhysenNet_git\\sam-ref.txt')
    
    dif = (int1-int2)/np.pi
    my_saveimage_pro(dif,'D:\\tyh\\PhysenNet_git\\sam-ref_pi.png')
    my_savetxt(dif,'D:\\tyh\\PhysenNet_git\\sam-ref_pi.txt')
    
    dif = ((int1-int2+4.1)/np.pi)*mask
    my_saveimage_pro(dif,'D:\\tyh\\PhysenNet_git\\sam-ref_pi_4.1.png')
    my_savetxt(dif,'D:\\tyh\\PhysenNet_git\\sam-ref_pi_4.1.txt')
    
    import scipy.signal as signal
    dif = signal.medfilt(dif,(11,11)) #二维中值滤波
    my_saveimage_pro(dif,'D:\\tyh\\PhysenNet_git\\sam-ref_pi_4.1_fil.png')
    my_savetxt(dif,'D:\\tyh\\PhysenNet_git\\sam-ref_pi_4.1_fil.txt')
    '''

    