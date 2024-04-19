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
import argparse
from copy import deepcopy
import time
from library import (my_readtxt,mkdir,visual_data,my_saveimage,my_savetxt,my_save2image)
from dataset import mydataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from source_target_transforms import *
from utils.compute_metric import compute_core_std_plot
from torch.cuda.amp import autocast as autocast
from utils.generate_mcf_simulate import mcf_simulate_plot

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--opt', type=str, default='D:\\tyh\phynet\option\simulate.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    para = Parameter(args.opt)

    pha_gt,amp_gt,mask_gt,speckle_gt  = mcf_simulate_plot(para)
    
    if para.noise['flag'] == True:
        noise = np.random.uniform(para.noise['mean'],para.noise['std'],size=speckle_gt.shape)
        speckle_gt = speckle_gt + noise
        print('add noise111111111111111111111111111111111111111111')

    # if para.after_resize['flag'] == True:
    #     print(type(pha_gt))
    #     print(type(amp_gt))
    #     print(type(mask_gt))
    #     print(type(speckle_gt))
    #     pha_gt = cv2.resize(pha_gt, (para.after_resize['size'],para.after_resize['size']), interpolation=cv2.INTER_CUBIC)
    #     amp_gt = cv2.resize(amp_gt, (para.after_resize['size'],para.after_resize['size']), interpolation=cv2.INTER_CUBIC)
    #     mask_gt = cv2.resize(mask_gt, (para.after_resize['size'],para.after_resize['size']), interpolation=cv2.INTER_CUBIC)
    #     speckle_gt = cv2.resize(speckle_gt, (para.after_resize['size'],para.after_resize['size']), interpolation=cv2.INTER_CUBIC)
    #     print('resize Done!!')
    


    save_path = f'.'
    
    localtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())    
 
    result_folder = f'../Resultimulate/{para.constraint}/{para.scale}/{para.num}/{para.dist}/{para.fi}/{localtime}'

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
    dataloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size)
    print('loading data') 

    # 3.model
    print(para.model['name'])
    model_name = para.model['name']
    modelnet = import_class('arch.'+para.model['filename'],para.model['classname']) 
    # modelnet = import_class('arch.'+model_name,model_name) 
    if  para.model['name'] == 'net_model_Dropout_full':
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
    hypar_file.write(f'epochs {para.noise["flag"]},mean {para.noise["mean"]},std {para.noise["std"]}\n')
    hypar_file.write(f'network\n{net}\n')
    
    hypar_file.close() 
    
    # tensorboard
    writer = SummaryWriter(tb_folder)
    amp_gt = torch.tensor(amp_gt).to(device)
    mask_gt = torch.tensor(mask_gt).to(device)

    print('starting loop')
    scaler = torch.cuda.amp.GradScaler()
    for current_epoch in tqdm(range(epochs)):
        for i, (Speckle,Pha) in enumerate(dataloader):
            Speckle = Speckle.to(device)
            Pha = Pha.to(device)
            
            optimizer.zero_grad()
            # forward proapation
            with torch.cuda.amp.autocast():
                
                pred_pha = net(Speckle) 
                
                flattened_pred_pha = pred_pha[0, 0, :, :] 
                if para.constraint == 'strong':
                    Uo = amp_gt*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场
                elif para.constraint == 'weak':
                    Uo = mask_gt*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场
                    
                Ui = propcomplex(Uo,dist=para.dist,device=device)            
                    
                pred_Speckle = torch.abs(Ui)  

                loss_mse_value = loss_mse(Speckle[0, 0, :, :].float(),pred_Speckle.float()) 
                
                # zero_matrix = torch.zeros_like(amp_gt).to(para.device)
                # one_matrix = torch.ones_like(amp_gt).to(para.device)                
                # loss_1 = loss_mse((flattened_pred_pha*(one_matrix-mask_gt)).float(),zero_matrix.float())
                
                loss_value =  loss_mse_value#0.02*loss_1+0.8*loss_mse_value

            # backward proapation 
            # loss_value.backward() 
            scaler.scale(loss_value).backward() 
            # optimizer.step()  
            scaler.step(optimizer) 
            scaler.update() 
                    
            # 实验记录

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
            if step % 300 == 0 or step+1 == epochs:
                        dpi = 800
                        my_saveimage(np.mod(flattened_pred_pha.cpu().detach().numpy(),2*np.pi),f'{img_txt_folder}/{step}_PredPhawrap2pi.png',dpi=dpi)
                        my_savetxt(np.mod(flattened_pred_pha.cpu().detach().numpy(),2*np.pi),f'{img_txt_folder}/{step}_PredPhawrap2pi.txt')

                        my_saveimage((flattened_pred_pha.cpu().detach().numpy()),f'{img_txt_folder}/{step}_PredPha.png',dpi=dpi)
                        my_savetxt((flattened_pred_pha.cpu().detach().numpy()),f'{img_txt_folder}/{step}_PredPha.txt')

                        my_saveimage(pred_Speckle.cpu().detach().numpy(),f'{img_txt_folder}/{step}_PredAmp.png',dpi=dpi)
                        my_savetxt(pred_Speckle.cpu().detach().numpy(),f'{img_txt_folder}/{step}_PredAmp.txt')

                        my_saveimage((((Speckle[0, 0, :, :]-pred_Speckle)).cpu().detach().numpy()),f'{img_txt_folder}/{step}_AmpLoss.png',dpi=dpi)
                        my_savetxt(((Speckle[0, 0, :, :]-pred_Speckle)).cpu().detach().numpy(),f'{img_txt_folder}/{step}_AmpLoss.txt')
                        
                        my_saveimage(np.mod(pha_gt-((flattened_pred_pha).cpu().detach().numpy()),2*np.pi),f'{img_txt_folder}/{step}_PhaLoss.png',dpi=dpi)
                        my_savetxt(np.mod(pha_gt-((flattened_pred_pha).cpu().detach().numpy()),2*np.pi),f'{img_txt_folder}/{step}_PhaLoss.txt')
                        
                        compute_core_std_plot(amp_gt.cpu().detach().numpy(),np.mod(pha_gt-((flattened_pred_pha).cpu().detach().numpy()),2*np.pi),f'{img_txt_folder}/{step}core_std.png',meanflag=True,labeledflag=True)
                        # compute_core_std_plot(amp_gt.cpu().detach().numpy(),np.mod(flattened_pred_pha.cpu().detach().numpy(),2*np.pi),f'{img_txt_folder}/{step}core_std.png',outputflag=True)
                        my_saveimage(np.mod((amp_gt*flattened_pred_pha).cpu().detach().numpy(),2*np.pi),f'{img_txt_folder}/{step}_Phamulmask.png',dpi=dpi)
                        plt.close()
                        
            # if step == 1000:
            #     compute_core_std_plot(amp_gt.cpu().numpy().detach(),flattened_pred_pha.cpu().detach().numpy(),f'{img_txt_folder}/core_std.png')
            if step % 40 == 0:
                # 80的时候显存差不多满了
                torch.cuda.empty_cache()
            # 记录中间结果图片

              
            if step+1 == epochs:
                plt.clf()  # 清图。
                plt.cla()  # 清坐标轴
                plt.figure(figsize=(12, 6))  # 设定图像大小

                # 显示第一个图像
                plt.subplot(2, 2, 1)
                imgplot1 = plt.imshow(np.mod(flattened_pred_pha.cpu().detach().numpy(),2*np.pi), cmap='viridis')
                plt.colorbar()  # 为第一个图像添加颜色条

                # 显示第二个图像
                plt.subplot(2, 2, 2)
                imgplot2 = plt.imshow(pred_Speckle.cpu().detach().numpy(), cmap='viridis')
                plt.colorbar()  # 为第二个图像添加颜色条

                # 显示第一个图像
                plt.subplot(2, 2, 3)
                imgplot1 = plt.imshow(np.mod(pha_gt-((flattened_pred_pha).cpu().detach().numpy()),2*np.pi), cmap='viridis')
                plt.colorbar()  # 为第一个图像添加颜色条

                # 显示第二个图像
                plt.subplot(2, 2, 4)
                imgplot2 = plt.imshow((Speckle[0, 0, :, :]-pred_Speckle).cpu().detach().numpy(), cmap='viridis')
                plt.colorbar()  # 为第二个图像添加颜色条

                plt.savefig(f'{img_txt_folder}/{step}_result.png',dpi=800)  # 保存图像                
            # my_save2image(Speckle[0,0,:,:].numpy(),Pha[0,0,:,:].numpy(),f'./{epoch}_combined_image.png', cmap='viridis')
                

  










