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
from trainer.trainer import train_epoch_complex,train_epoch_complex_gs
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
    rootpath = f'../simulateData/simulate_data/{para.constraint}/{para.scale}/{para.num}/{para.dist}/{para.fi}'
    
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
def result_record(current_epoch,writer,loss_value,best_loss,net,flattened_pred_pha,pred_Speckle,Speckle,pha_gt,img_txt_folder,weight_folder):
    '''
    This function is used to record the results of the training process.


    Parameters:
    None


    Return:
    None

    '''
    step = current_epoch 

    # 记录loss
    if step % 50 == 0:
        # tb记录loss
        writer.add_scalar('training loss',
                        loss_value.item(),
                        step)
        
        phase_diff = np.abs(pha_gt-((flattened_pred_pha).cpu().detach().numpy()))

        writer.add_scalar('相位差',
                        np.mean(phase_diff),
                        step)
        
        
        # 记录最好的模型权重
        # 保存loss值最小的网络参数
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(net.state_dict(), f"{weight_folder}/best_model.pth")

    # 记录中间结果图片
    if step % 3000 == 0:
                dpi = 800
                my_saveimage(flattened_pred_pha.cpu().detach().numpy(),f'{img_txt_folder}/{step}_PredPha.png',dpi=dpi)
                my_savetxt(flattened_pred_pha.cpu().detach().numpy(),f'{img_txt_folder}/{step}_PredPha.txt')

                my_saveimage(pred_Speckle.cpu().detach().numpy(),f'{img_txt_folder}/{step}_PredAmp.png',dpi=dpi)
                my_savetxt(pred_Speckle.cpu().detach().numpy(),f'{img_txt_folder}/{step}_PredAmp.txt')

                my_saveimage((Speckle[0, 0, :, :]-pred_Speckle).cpu().detach().numpy(),f'{img_txt_folder}/{step}_AmpLoss.png',dpi=dpi)
                my_savetxt((Speckle[0, 0, :, :]-pred_Speckle).cpu().detach().numpy(),f'{img_txt_folder}/{step}_AmpLoss.txt')
                
                my_saveimage(pha_gt-((flattened_pred_pha).cpu().detach().numpy()),f'{img_txt_folder}/{step}_PhaLoss.png',dpi=dpi)
                my_savetxt(pha_gt-((flattened_pred_pha).cpu().detach().numpy()),f'{img_txt_folder}/{step}_PhaLoss.txt')

    if step % 40 == 0:
        # 80的时候显存差不多满了
        torch.cuda.empty_cache()
        
    if step % 9000 == 0:
        plt.clf()  # 清图。
        plt.cla()  # 清坐标轴
        plt.figure(figsize=(12, 6))  # 设定图像大小

        # 显示第一个图像
        plt.subplot(2, 2, 1)
        imgplot1 = plt.imshow(flattened_pred_pha.cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()  # 为第一个图像添加颜色条

        # 显示第二个图像
        plt.subplot(2, 2, 2)
        imgplot2 = plt.imshow(pred_Speckle.cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()  # 为第二个图像添加颜色条

        # 显示第一个图像
        plt.subplot(2, 2, 3)
        imgplot1 = plt.imshow(pha_gt-((flattened_pred_pha).cpu().detach().numpy()), cmap='viridis')
        plt.colorbar()  # 为第一个图像添加颜色条

        # 显示第二个图像
        plt.subplot(2, 2, 4)
        imgplot2 = plt.imshow((Speckle[0, 0, :, :]-pred_Speckle).cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()  # 为第二个图像添加颜色条

        plt.savefig(f'{img_txt_folder}/{step}_result.png',dpi=800)  # 保存图像                
    # my_save2image(Speckle[0,0,:,:].numpy(),Pha[0,0,:,:].numpy(),f'./{epoch}_combined_image.png', cmap='viridis') 
    
    


            




def train_simulate(para,loop=None):
    
    if loop is not None:
        para.num = loop
        print(f'loop of core:{para.num}')
    
    pha_gt,amp_gt,mask_gt,speckle_gt  = getData(para)
    
    print(type(pha_gt))
    print(type(amp_gt))
    print(type(mask_gt))
    print(type(speckle_gt))

    save_path = f'.'
    
    localtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())    
 
    result_folder = f'../Resultimulate/{para.train_epoch}/{para.constraint}/{para.scale}/{para.num}/{para.dist}/{para.fi}/{localtime}'

    tb_folder = f'{result_folder}/tb_folder'
    weight_folder = f"{result_folder}/weight_folder"
    img_txt_folder = f'{result_folder}/img_txt_folder'    

    # 最好的loss初始化为无穷大
    best_loss = float('inf')

    # 随机种子和实验设备
    torch.manual_seed(para.seed)
    device = torch.device(para.device) #分布式训练时用默认
    
    # hypara
    batch_size = para.batch_size
    lr = para.lr
    epochs = para.epochs
    print(f'batch_size,lr,epochs:{batch_size},{lr},{epochs}')   
    
    # 1.dataset
    transform = transforms.Compose([
            # RandomResizeFromSequence([[192,256],[192*4,256*4],[192*6,256*6],[192*8,256*8],[192*10,256*10],[192*5,256*5]]),
            # RandomRotationFromSequence((360)),#[0, 90, 180, 270]
            # RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            ToTensor()])    
    train_data = mydataset(speckle_gt,pha_gt,transform)

    # 2.dataloader  
    dataloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    print('loading data') 

    # 3.model
    print(para.model['name'])
    model_name = para.model['name']
    
    modelnet = import_class('arch.'+model_name,model_name) 
    
    if  para.model['name'] == 'net_model_Dropout_full' or para.model['name'] == 'net_model_Dropout_full':
        print(para.model['name'])
        net  =  modelnet(drop=para.model['p'],scale=para.scale).to(device)                        
    else:
        net  =  modelnet().to(device) 
    print('creating model')

    # 4.loss and optimization
    if para.loss['name'] ==  'MSELoss':
        loss_mse = torch.nn.MSELoss()

    elif para.loss['name'] ==  'L1Loss':
        loss_mse = torch.nn.L1Loss()

    else:
        assert False, "未支持的损失函数类型。只支持 'MSELoss' 和 'L1Loss'。"


    optimizer = torch.optim.Adam(net.parameters(), lr = lr)

    print('creating loss and optimization')   
                     
    # 创建文件夹
    mkdir(tb_folder)
    mkdir(weight_folder)
    mkdir(img_txt_folder)
    
    hypar_file = open(f"{result_folder}/hypar.txt","w")
    # 记录训练开始前的超参数，网络结构，输入强度图，gt图像
    hypar_file.write(f'target： {para.target}\n')
    hypar_file.write(f'para.fi:{para.fi}\n')
    hypar_file.write(f'num of core {para.num}\n')
    hypar_file.write(f'scale of angle {para.scale}\n')
    hypar_file.write(f'dist of prop {para.dist}\n')
    hypar_file.write(f'batch_size {batch_size}\n')
    hypar_file.write(f'lr {lr}\n')
    hypar_file.write(f'epochs {epochs}\n')
    hypar_file.write(f'network\n{net}\n')
    
    hypar_file.close() 
    
    # tensorboard
    writer = SummaryWriter(tb_folder)
    amp_gt = torch.tensor(amp_gt).to(device)
    mask_gt = torch.tensor(mask_gt).to(device)

    print('starting loop')
    for current_epoch in tqdm(range(epochs)):
        if para.train_epoch == 'train_epoch_complex_gs':
            train_epoch_complex_gs(dataloader,net,loss_mse,optimizer,para,pha_gt,amp_gt,mask_gt,current_epoch,writer,weight_folder,img_txt_folder,best_loss)
        elif para.train_epoch == 'train_epoch_complex':
            train_epoch_complex(dataloader,net,loss_mse,optimizer,para,pha_gt,amp_gt,mask_gt,current_epoch,writer,weight_folder,img_txt_folder,best_loss)
            
        else:
            assert False, "未支持的训练方式。只支持 'train_epoch_complex_gs' 和 'train_epoch_complex'。"

        
        
        
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--opt', type=str, default='D:\\tyh\phynet\option\simulate_complex.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    para = Parameter(args.opt)
    
    train_simulate(para)

