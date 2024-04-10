import numpy as np
import matplotlib.pyplot as plt
import random
import os

import sys 
sys.path.append("D:\\tyh\phynet")


from os.path import join, getsize

import numpy as np
from library import my_saveimage,mkdir,my_savetxt
from prop import propcomplex
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import cv2
from config.parameter import Parameter,import_class
from trainer.trainer import train_epoch_complex
import argparse
from copy import deepcopy
import time
from library import (my_readtxt,mkdir,visual_data,my_saveimage,my_savetxt,my_save2image)
from dataset import mydataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from source_target_transforms import *
def create_circles_in_rectangle_within_circle(height, width, a, b, number_of_cores, core_radius, fiber_radius,ispha='pha',scale = 'pi'):
    # 创建一个空白图像
    
    image = np.zeros((height, width))
    
    # 大圆的圆心和大圆内部像素值的设置
    fiber_center = (height // 2, width // 2)
    yy, xx = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((xx - fiber_center[1])**2 + (yy - fiber_center[0])**2)
    is_inside_fiber = distance_from_center <= fiber_radius
    
    # 大圆内的像素赋予0到π的随机值
    if ispha=='amp':
        image[is_inside_fiber] = 0
    elif ispha=='pha':
        if scale == 'pi':
            image[is_inside_fiber] = np.random.uniform(0, np.pi, is_inside_fiber.sum())
        elif scale == '2pi':
            image[is_inside_fiber] = np.random.uniform(0, 2*np.pi, is_inside_fiber.sum())
    
    
    # 计算长方形内小圆的均匀分布
    rectangle_center = (height // 2, width // 2)
    rectangle_top_left = (rectangle_center[0] - a // 2, rectangle_center[1] - b // 2)
    
    # 长方形内小圆的间距
    cores_per_row = int(np.sqrt(number_of_cores * b / a))
    cores_per_col = number_of_cores // cores_per_row
    spacing_x = b // (cores_per_row + 1)
    spacing_y = a // (cores_per_col + 1)

    # 生成小圆，并赋予每个小圆内的像素相同的随机值
    for i in range(1, cores_per_col + 1):
        for j in range(1, cores_per_row + 1):
            center_x = rectangle_top_left[1] + j * spacing_x
            center_y = rectangle_top_left[0] + i * spacing_y
            # 判断小圆整体是否在大圆内
            if (np.abs(center_x - (fiber_center[1]))+9)**2 + (np.abs(center_y - (fiber_center[0]))+9)**2 <= fiber_radius**2:
                if ispha=='amp':
                    core_phase_value = 1
                elif ispha=='pha':
                    if scale == 'pi':
                        core_phase_value = np.random.uniform(0, np.pi)
                    elif scale == '2pi':
                        core_phase_value = np.random.uniform(0, 2*np.pi)
                for y in range(-core_radius, core_radius + 1):
                    for x in range(-core_radius, core_radius + 1):
                        if x**2 + y**2 <= core_radius**2:
                            image[center_y + y, center_x + x] = core_phase_value
    
    return image

def mkdir(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        print("make dirs")

    else:
        print("dirs exists")

def getData(para):
    
    # 新参数

        

        
    num = para.num
    print(f'num of core:{num}')
    scale = para.scale
    print(f'scale of angle:{scale}')
    
    
    data = {
        1600: (1600, 800), #纤芯数量与光纤束的直径
        3000: (3000, 1096),
        6000: (6000, 1550),
        10000: (10000, 2000),
        15000: (15000, 2550),
        10:(10,100),
        100:(100,200),
        200:(200,282),
        500:(500,448),
        1000:(1000,632)
    }
    
    
    # 使用新参数
    rootpath = f'../simulateData/simulate_data/hio/{para.constraint}/{para.scale}/{para.num}/{para.dist}/{para.fi}'
    
    # 相机的像素
    height = int(data[num][1]*para.fi)#para.image_height
    width = int(data[num][1]*para.fi)#para.image_width

    # 光纤的圆心和半径，决定了光纤间的间隙
    fiber_center = (width/2, height/2)
    fiber_radius = data[num][1]/2
    
    if data[num][0] < 100:
        
        fang = fiber_radius
        
    elif data[num][0] < 3000:
        fang = fiber_radius+10
        
    else:
        fang = fiber_radius+30

    a, b = int(fang*2), int(fang*2)  # 长方形的尺寸
    core_radius = 9
    number_of_cores = num
    
    mkdir(f"{rootpath}")
    
    pha = create_circles_in_rectangle_within_circle(height, width, a, b, number_of_cores, core_radius, fiber_radius,ispha='pha',scale=scale)
    if para.resize['flag'] == True:
        pha = cv2.resize(pha, (para.resize['size'],para.resize['size']), interpolation=cv2.INTER_CUBIC)
    np.savetxt(f'{rootpath}/{number_of_cores}_pha_simulate.txt',pha,fmt='%.10e',delimiter=',')
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(pha, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
    plt.colorbar()
    plt.savefig(f'{rootpath}/{number_of_cores}_pha_simulate.png',dpi=800)#
    print('pha Done!!')

    amp = create_circles_in_rectangle_within_circle(height, width, a, b, number_of_cores, core_radius, fiber_radius,ispha='amp')
    if para.resize['flag'] == True:
        amp = cv2.resize(amp, (para.resize['size'],para.resize['size']), interpolation=cv2.INTER_CUBIC)
    np.savetxt(f'{rootpath}/{number_of_cores}_amp_simulate.txt',amp,fmt='%.10e',delimiter=',')
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(amp, cmap='viridis')  # 使用HSV色彩映射以更好地显示相位
    plt.colorbar()
    plt.savefig(f'{rootpath}/{number_of_cores}_amp_simulate.png',dpi=800)#
    print('amp Done!!')

    # 2.光纤掩膜
    mask = deepcopy(pha)
    mask[mask < 0.02] = 0
    mask[mask > 0.0001] = 1
    kernel = np.ones((11,11),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    my_saveimage(mask, f'{rootpath}/{number_of_cores}_mask_simulate.png')

    
    # generate speckle
    dist =  para.dist
    print(f'dist of prop:{dist}')
    pha = torch.tensor(pha)
    amp = torch.tensor(amp)
    
    Uo = amp*torch.exp(1j*pha) #光纤端面初始复光场
    Ui = propcomplex(Uo,dist=dist,device='cpu')

    speckle = torch.abs(Ui)

    dist_prop = str(dist).replace('.','')
    
    my_saveimage(speckle,f'{rootpath}/{number_of_cores}_speckle_prop{dist_prop}_simulate.png',dpi=800)
    my_savetxt(speckle,f'{rootpath}/{number_of_cores}_speckle_prop{dist_prop}_simulate.txt')
    print('speckle Done!!')  
    
    return pha.numpy(),amp.numpy(),mask,speckle.numpy() 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--opt', type=str, default='D:\\tyh\phynet\option\simulate_complex.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    para = Parameter(args.opt)    
    
    pha,amp_gt,mask_gt,speckle  = getData(para)
    save_path = f'./ResultIterativePhaseRetrieval/simulate/MCF/hio/{para.constraint}/{para.scale}/{para.num}/{para.dist}/{para.fi}'
    mkdir(save_path)

    epoch1s = 2000#恢复光纤自身畸变相位迭代次数



    # 1.光纤自身畸变

    amp_far_ref_a = speckle
    my_saveimage(amp_far_ref_a, f'{save_path}/amp_far_ref_a.png')


    # 2.光纤掩膜
    mask = pha
    mask[mask < 0.02] = 0
    mask[mask > 0.0001] = 1
    kernel = np.ones((11,11),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    my_saveimage(mask, f'{save_path}/mask.png')

    # 4.形成散斑的传播距离
    zs = para.dist

    device = 'cuda:0'
    A = torch.empty(amp_far_ref_a.shape[0],amp_far_ref_a.shape[1],1).to(device)
    A[:,:,0] = torch.tensor(amp_far_ref_a).to(device) #距离1光纤自身畸变的散斑

    mask = torch.tensor(mask).to(device)
    amp_gt = torch.tensor(amp_gt).to(device)
    zs = torch.tensor(zs).to(device)

    init_phase = torch.zeros_like(mask).to(device) #远场初始相位
    Uo = A[:,:,0]*torch.exp(1j*init_phase).to(device) #远场初始复光场
    # Uo = mask*torch.exp(1j*init_phase).to(device) #光纤端面初始复光场

    Un = torch.zeros(mask.shape[0],mask.shape[1],1,dtype=torch.cdouble).to(device) #光纤端面初始复光场

    print('光纤畸变相位恢复开始！！！')
    loss = []
    for epoch1 in tqdm(range(epoch1s)):

        for i in range(1):
            Ui = propcomplex(Uo,dist = -1*zs,device=device)
            Ua = amp_gt*Ui/torch.abs(Ui)
            Um = propcomplex(Ua,dist = zs,device=device)
            Uo = A[:,:,0]*Um/torch.abs(Um)
            # Uo = mask*Um/torch.abs(Um)
            # ((1+b)*Um - Uo)*mask + Uo - b*Um 
            Un[:,:,i] = Ua

        Uo = torch.mean(Un[:,:,0:1], 2) #输入的是一张图像的话，相当于直接把Un[:,:,0]赋给了Uo；输入的是一张图像的话，相当于直接把Un[:,:,0]和Un[:,:,1]求平均赋给了Uo
        phase_ref = torch.angle(Uo) #恢复的光纤端面的相位作为恢复样品畸变相位的初始相位
        loss.append(np.mean(np.abs(pha-phase_ref.cpu().detach().numpy())))
    print('光纤畸变相位恢复结束！！！')

    phase_ref = torch.angle(Uo) #恢复的光纤端面的相位作为恢复样品畸变相位的初始相位


    my_saveimage(phase_ref.cpu().detach().numpy(),f'{save_path}/{epoch1s}_phase_ref.png')
    my_savetxt(phase_ref.cpu().detach().numpy(),f'{save_path}/{epoch1s}_phase_ref.txt')

    my_saveimage(np.mod((pha-phase_ref.cpu().detach().numpy()),2*np.pi),f'{save_path}/{epoch1s}_refloss_pi.png')
    my_savetxt(np.mod((pha-phase_ref.cpu().detach().numpy()),2*np.pi),f'{save_path}/{epoch1s}_refloss_pi.txt')

    # 绘制loss曲线图
    plt.figure(figsize=(10, 6))

    # 绘制平均loss
    plt.plot(range(1, epoch1s + 1), loss, label='Average Loss', linestyle='--')

    # 设置图表标题和标签
    plt.title('Loss Over迭代ations for Multiple Experiments')
    plt.xlabel('迭代ation')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # 保存图表为PNG图片
    plt.savefig(f'{save_path}_loss_over_iterations.png')  