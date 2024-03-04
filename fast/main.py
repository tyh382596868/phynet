import sys 
sys.path.append("/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet") 
from library import mkdir,my_readtxt,my_saveimage,my_savetxt
import torch
from tqdm import tqdm
import scipy.io as scio 
## Reconstruct the phase of the sample

def prop(H,  dx=2.2e-6, dy=2.2e-6, lam=532e-9, dist=0.0788,device='cuda:0'):
    '''
    1.注意传播距离，确保生成图像的传播距离与训练的传播距离一致，一般默认一致
    2.输入数据的维度应该为二维，如果是三维，就算值一样傅里叶变换后也不一样
    '''

    fft_H = torch.fft.ifftshift(torch.fft.fft2(H))
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

    # Id=Ud
    # Id1=torch.angle(Ud)
    # intensity = torch.abs(Id) * torch.abs(Id)

    return Ud



# device = 'cuda'
# data = scio.loadmat('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet/fast/data.mat')
# # print('data:\n',data)     		    #大致看一下data的结构
# # print('datatype:\n',type(data)) 	#看一下data的类型
# # print('keys:\n',data.keys)  		#查看data的键，这里验证一下是否需要加括号
# # print('keys:\n',data.keys())		#当然也可以用data.values查看值
# # print(data['amp_facet_ref'])      		    #查看数据集
# # print('target shape\n',data['amp_facet_ref'].shape)
# # print('target type\n',type(data['amp_facet_ref']))
# A = torch.empty(data['amp_facet_ref'].shape[0],data['amp_facet_ref'].shape[1],2)
# # print(type(A),A.shape)
# A[:,:,0] = torch.tensor(data['amp_far_ref_a'])
# A[:,:,1] = torch.tensor(data['amp_far_ref_b'])
# print(type(A),A.shape)
# mask = data['amp_facet_ref']

# mask[mask < 0.02] = 0
# mask[mask > 0.0001] = 1
# zs = torch.tensor(data['zs'])
# print(zs)
# import cv2

# import numpy as np


# kernel = np.ones((11,11),np.uint8)

# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# my_saveimage(A[:,:,0], './amp_far_ref_a.png')
# my_saveimage(A[:,:,1], './amp_far_ref_b.png')
# my_saveimage(mask, './mask.png')

# mask = torch.tensor(mask).to(device)
# init_phase = torch.zeros_like(mask).to(device)
# A = A.to(device)
# zs = zs.to(device)

# Uo = mask*torch.exp(1j*init_phase)
# Un = torch.zeros(data['amp_facet_ref'].shape[0],data['amp_facet_ref'].shape[1],2).to(device)

# epoch1s = 2500
# b = 0.2
# for epoch1 in tqdm(range(epoch1s)):

#     for i in range(2):
#         # print((zs[:,i]))
#         # print((zs[i]))
#         Ui = prop(Uo,dist = zs[:,i])
#         Ua = A[:,:,i]*Ui/torch.abs(Ui)
#         Um = prop(Ua,dist = -zs[:,i])
#         Uo = ((1+b)*Um - Uo)*mask + Uo - b*Um
#         Un[:,:,i] = Uo

#         # break

#     Uo = torch.mean(Un, 2)

#     # break

    

# phase_ref = torch.angle(Uo)


# gt_sam = my_readtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/sam_pi.txt')
# my_savetxt(gt_sam*2-phase_ref.cpu().detach().numpy(),  './gtsam-ref.txt')
# my_saveimage(gt_sam*2-phase_ref.cpu().detach().numpy(), './gtsam-ref.png')


# # epoch2s = 40
# # z = zs[:,0]

# # # device = 'cuda'
# # # A_sam = my_readtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/exp/1536_1536_sam_pi_01_prop_pi.txt')
# # # A_sam = torch.tensor(A_sam).to(device)
# # # # phase_ref = my_readtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_ref_pi_01_prop_pi.txt')
# # # # phase_ref = torch.tensor(phase_ref).to(device)
# # # phase_ref = torch.zeros_like(A_sam)
# # # mask = my_readtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_mask.txt')
# # # mask = torch.tensor(mask).to(device)

# # A_sam = torch.tensor(data['amp_far_sam']).to(device)

# # uo = mask * torch.exp(1j * phase_ref)
# # # print(torch.sum(uo))
# # for epoch2 in tqdm(range(epoch2s)):

# #     ui = prop(uo,dist=z)
# #     ua = A_sam*ui/torch.abs(ui)
# #     um = prop(ua,dist=-z)
# #     uo = ((1+0.2)*um - uo)*mask + uo - 0.2*um

# # phase_sam = torch.angle(uo)
# # my_saveimage(phase_sam.cpu().detach().numpy(),'./phase_sam.png')
# # my_saveimage(phase_ref.cpu().detach().numpy(),'./phase_ref.png')
# # my_saveimage(phase_sam.cpu().detach().numpy()-phase_ref.cpu().detach().numpy(),'./sam_ref.png')