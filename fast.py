import sys 
sys.path.append("/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet") 
from library import mkdir,my_readtxt,my_saveimage,my_savetxt,sam_ref_2pi
import torch
from tqdm import tqdm
import scipy.io as scio 
import cv2

import numpy as np
## Reconstruct the phase of the sample

def prop(H, dx=2.2e-6, dy=2.2e-6, lam=532e-9, dist=0.0788,device='cuda:0'):
    """Angular Spectrum Method for Light Field Propagation Function

    Args:
        H (Complex Light Field): Input field (for example, a simple plane wave)
        dx (_type_, optional): Spatial sampling interval (meters). Defaults to 2.2e-6.
        dy (_type_, optional): Spatial sampling interval (meters). Defaults to 2.2e-6.
        lam (_type_, optional): Wavelength (meters). Defaults to 532e-9.
        dist (float, optional): Propagation distance (meters). Defaults to 0.0788.
        device (str, optional): GPU or CPU device. Defaults to 'cuda:0'.

    Returns:
        Ud: Complex Light Field

    1.注意传播距离，确保生成图像的传播距离与训练的传播距离一致，一般默认一致
    2.输入数据的维度应该为二维，如果是三维，就算值一样傅里叶变换后也不一样
    """    

    fft_H = torch.fft.fftshift(torch.fft.fft2(H))
    (Ny,Nx) = H.shape[0],H.shape[1]
    
    qx = torch.range(1-Nx/2, Nx/2, 1).to(device)
    qy = torch.range(1-Ny/2, Ny/2, 1).to(device)
    y, x = torch.meshgrid(qy, qx)
    r=(2*torch.pi*x/(dx*Nx))**2+(2*torch.pi*y/(dy*Ny))**2


    k=2*torch.pi/lam

    kernel=torch.exp(-1*1j*torch.sqrt(k**2-r)*dist)

    fft_HH=fft_H[:,:]*kernel
    fft_HH=torch.fft.fftshift(fft_HH)

    Ud=torch.fft.ifft2(fft_HH)



    return Ud

def PaddingImage(img,original_width,original_height,target_width, target_height):
    """Pad the image with zeros to expand to a fixed size.

    Args:
        img (_type_): input little image
        original_width (_type_): width of original image
        original_height (_type_): height of original image
        target_width (_type_): width of target image
        target_height (_type_): height of target image

    Returns:
        extended_image: Image after pixel padding
    """    

    x_padding = target_width - original_width
    y_padding = target_height - original_height
    # 使用copyMakeBorder函数在原始图片的周围添加像素
    # blk_constant参数指定添加的像素颜色，这里是0（黑色）
    extended_image = cv2.copyMakeBorder(img, y_padding//2, y_padding//2, x_padding//2, x_padding//2, cv2.BORDER_CONSTANT, value=0)
    return extended_image   


ispadding = 'False' # 图片是否扩充
save_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/fast/result/baseline'
mkdir(save_path)

epoch1ss = [200]#恢复光纤自身畸变相位迭代次数
b = 0.2

for epoch1s in epoch1ss:

    device = 'cuda'
    data = scio.loadmat('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/data.mat')
    # print('data:\n',data)     		    #大致看一下data的结构
    # print('datatype:\n',type(data)) 	#看一下data的类型
    # print('keys:\n',data.keys)  		#查看data的键，这里验证一下是否需要加括号
    # print('keys:\n',data.keys())		#当然也可以用data.values查看值
    # print(data['amp_facet_ref'])      		    #查看数据集
    # print('target shape\n',data['amp_facet_ref'].shape)
    # print('target type\n',type(data['amp_facet_ref']))

# 1.光纤自身畸变
    amp_far_ref_a = data['amp_far_ref_a']
    amp_far_ref_b = data['amp_far_ref_b']
    my_saveimage(amp_far_ref_a, f'{save_path}/amp_far_ref_a.png')
    my_saveimage(amp_far_ref_b, f'{save_path}/amp_far_ref_b.png')

# 2.光纤掩膜
    mask = data['amp_facet_ref'] # 光纤端面的掩膜
    mask[mask < 0.02] = 0
    mask[mask > 0.0001] = 1
    kernel = np.ones((11,11),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    my_saveimage(mask, f'{save_path}/mask.png')

# 3.样品畸变
    A_sam = (data['amp_far_sam']) #距离1样品畸变的相位
    my_saveimage(A_sam, f'{save_path}/amp_far_sam.png')

    if ispadding=='True':
        A_sam = PaddingImage(A_sam,512,512,2560,1920)
    print(A_sam.shape)    

# 4.形成散斑的传播距离
    zs = data['zs'] #光纤自身畸变的两个距离

    z = zs[:,0] #样品畸变的距离
    print(f'zs:{zs}')
    print(f'z:{z}')


    A = torch.empty(amp_far_ref_a.shape[0],amp_far_ref_a.shape[1],2).to(device)
    A[:,:,0] = torch.tensor(amp_far_ref_a).to(device) #距离1光纤自身畸变的散斑
    A[:,:,1] = torch.tensor(amp_far_ref_b).to(device) #距离2光纤自身畸变的散斑
    A_sam = torch.tensor(A_sam).to(device)
    mask = torch.tensor(mask).to(device)
    zs = torch.tensor(zs).to(device)
    z = torch.tensor(z).to(device)



    init_phase = torch.zeros_like(mask).to(device) #光纤端面初始相位
    Uo = mask*torch.exp(1j*init_phase).to(device) #光纤端面初始复光场

    Un = torch.zeros(mask.shape[0],mask.shape[1],2,dtype=torch.cdouble).to(device) #光纤端面初始复光场
    
    print('光纤畸变相位恢复开始！！！')
    for epoch1 in tqdm(range(epoch1s)):

        for i in range(2):
            Ui = prop(Uo,dist = zs[:,i])
            Ua = A[:,:,i]*Ui/torch.abs(Ui)
            Um = prop(Ua,dist = -zs[:,i])
            Uo = ((1+b)*Um - Uo)*mask + Uo - b*Um
            Un[:,:,i] = Uo

        Uo = torch.mean(Un, 2)
    print('光纤畸变相位恢复结束！！！')

    phase_ref = torch.angle(Uo) #恢复的光纤端面的相位作为恢复样品畸变相位的初始相位
    epoch2s = 40
    uo = mask * torch.exp(1j * phase_ref) #样品畸变初始复光场

    print('样品畸变相位恢复开始！！！')
    for epoch2 in tqdm(range(epoch2s)):

        ui = prop(uo,dist=z)
        ua = A_sam*ui/torch.abs(ui)
        um = prop(ua,dist=-z)
        uo = ((1+0.2)*um - uo)*mask + uo - 0.2*um
    phase_sam = torch.angle(uo)
    print('样品畸变相位恢复结束！！！')

    my_saveimage(phase_ref.cpu().detach().numpy(),f'{save_path}/{epoch1s}_phase_ref.png')
    my_savetxt(phase_ref.cpu().detach().numpy(),f'{save_path}/{epoch1s}_phase_ref.txt')

    my_saveimage(phase_sam.cpu().detach().numpy(),f'{save_path}/{epoch1s}_phase_sam.png')
    my_savetxt(phase_sam.cpu().detach().numpy(),f'{save_path}/{epoch1s}_phase_sam.txt')

    my_saveimage(sam_ref_2pi(phase_sam.cpu().detach().numpy(),phase_ref.cpu().detach().numpy()),f'{save_path}/{epoch1s}_sam_ref_2pi.png')
    my_savetxt(sam_ref_2pi(phase_sam.cpu().detach().numpy(),phase_ref.cpu().detach().numpy()),f'{save_path}/{epoch1s}_sam_ref_2pi.txt')