'''
训练过程速度提高

'''

import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from library import (my_readtxt,mkdir,visual_data,prop,my_saveimage,my_savetxt)
from unet import net_model_v1
from loss import TVLoss
from dataset import measured_y_txt_dataset256,measured_y_txt_dataset256_fast

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


    


#------------------------------------------------------------------------------
# main

if __name__ == "__main__":

    a = 128*12
    c = 128*12
    shape = [a,c]

    # 1.实验名称
    
    image_name = 'phase_diff_prop_pi'#phase_sam_prop_pi
    name = f'{shape[0]}_{shape[1]}_{image_name}' #读取0-pi归一化强度图txt文件名，
    # 2.用于相减参考的相位txt文件路径


    # 生成实验结果、tensorboard日志文件、权重文件保存的文件夹
    localtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    current_dir = os.getcwd()


    root_path = 'D:\\tyh\PhysenNet_git'
    measured_y_path_txt = f'D:\\tyh\\PhysenNet_git\\traindata\\exp\{name}.txt'
    gt_txt = f'D:\\tyh\\PhysenNet_git\\traindata\\gt\\{name}.txt' #读取的原相位图txt文件名，现为face，0-pi
    gt_matrix = my_readtxt(gt_txt)
    print(gt_matrix.dtype)

    result_folder = f'D:\\tyh\PhysenNet_git\\result\\{name}\\{localtime}'
    tb_folder = f'{result_folder}\\tb_folder'

    weight_folder = f"{result_folder}\\weight_folder"
    img_txt_folder = f'{result_folder}\\img_txt_folder'

    mkdir(tb_folder)
    mkdir(weight_folder)
    mkdir(img_txt_folder)

    # 最好的loss初始化为无穷大
    best_loss = float('inf')

    # 随机种子和实验设备
    torch.manual_seed(1)
    
    # tensorboard
    writer = SummaryWriter(tb_folder)

    # # prop的   默认  参数
    # dx = 2.2e-6
    # dy = dx
    # lam = 532e-9
    # d = 0.0788
    
    # hypar
    batch_size = 1
    lr = 0.1
    epochs = 4000

    hypar_file = open(f"{result_folder}/hypar.txt","w")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #分布式训练时用默认
    print(f"PyTorch is running on GPU: {torch.cuda.current_device()}")

    # 1.dataset

    data = np.loadtxt(measured_y_path_txt,dtype=np.float32,delimiter=",") # frame:文件
    data = torch.tensor(data).to(device)
    train_data = measured_y_txt_dataset256_fast(data,shape)

    # 2.dataloader
    
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size = batch_size, shuffle = True
        )
    print('loading data')

    # 3.model
    net = net_model_v1().to(device)

    print('creating model')
    
    
    # 4.loss and optimization
    loss_mse = torch.nn.MSELoss()
    loss_tv = TVLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.995)
    print('creating loss and optimization')

    # 记录训练开始前的超参数，网络结构，输入强度图，gt图像
    hypar_file.write(f'batch_size {batch_size}\n')
    hypar_file.write(f'lr {lr}\n')
    hypar_file.write(f'epochs {epochs}\n')
    hypar_file.write(f'network\n{net}\n')
    hypar_file.close()

# '''
    
    # 5.training
    print('starting loop')
 
    for current_epoch in tqdm(range(epochs)):
        # print(f"Epoch {current_epoch}\n-------------------------------")

        for batch,(x,y) in (enumerate(train_dataloader)):
            
            optimizer.zero_grad()
            pred_y = net(x)
            


            measured_y = prop(pred_y[0, 0, :, :])
            loss_mse_value = loss_mse(y.float(),measured_y.float())
            loss_value =  loss_mse_value

            # backward proapation

            loss_value.backward()
            optimizer.step()
            
            scheduler.step()

            

            # 实验记录

            step = current_epoch * len(train_dataloader) + batch

            # 记录loss
            if step % 50 == 0:
                # tb记录loss
                writer.add_scalar('training loss',
                                loss_value.item(),
                                step)
                
                phase_diff = np.abs(gt_matrix-((pred_y).cpu().detach().numpy().reshape(shape[0],shape[1])))

                writer.add_scalar('相位差',
                                np.mean(phase_diff),
                                step)
                last_lr = scheduler.get_last_lr()[-1]
                writer.add_scalar('lr',
                                last_lr,
                                step)
                
                
                # 记录最好的模型权重
                # 保存loss值最小的网络参数
                if loss_value < best_loss:
                    best_loss = loss_value
                    torch.save(net.state_dict(), f"{weight_folder}/best_model.pth")

            # 记录中间结果图片
            if step % 500 == 0:
                        my_saveimage(pred_y.cpu().detach().numpy().reshape(shape[0],shape[1]),f'{img_txt_folder}/{step}_pred.png')
                        my_savetxt(pred_y.cpu().detach().numpy().reshape(shape[0],shape[1]),f'{img_txt_folder}/{step}_pred.txt')

                        my_saveimage(measured_y.cpu().detach().numpy().reshape(shape[0],shape[1]),f'{img_txt_folder}/{step}_measured_y.png')
                        my_savetxt(measured_y.cpu().detach().numpy().reshape(shape[0],shape[1]),f'{img_txt_folder}/{step}_measured_y.txt')

                        my_saveimage((x-measured_y).cpu().detach().numpy().reshape(shape[0],shape[1]),f'{img_txt_folder}/{step}_{name}-measured_y.png')
                        my_savetxt((x-measured_y).cpu().detach().numpy().reshape(shape[0],shape[1]),f'{img_txt_folder}/{step}_{name}-measured_y.txt')
                        
                        my_saveimage(gt_matrix-((pred_y).cpu().detach().numpy().reshape(shape[0],shape[1])),f'{img_txt_folder}/{step}_gt-_pred.png')
                        my_savetxt(gt_matrix-((pred_y).cpu().detach().numpy().reshape(shape[0],shape[1])),f'{img_txt_folder}/{step}_gt-_pred.txt')

            if step % 40 == 0:
                # 80的时候显存差不多满了
                torch.cuda.empty_cache()
            
    print("Done!")






        




