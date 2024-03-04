'''
探索不同学习率的影响

'''

import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from library import (my_readtxt,mkdir,visual_data,prop,my_saveimage,my_savetxt)

from dataset import measured_y_txt_dataset256,measured_y_txt_dataset256_fast

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import argparse
from config.parameter import Parameter,import_class

    


#------------------------------------------------------------------------------
# main

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--opt', type=str, default='./config.yml', help='Path to the configuration file')
    args = parser.parse_args()
    para = Parameter(args.opt) 

    shape = [para.image_width,para.image_height]
    # print(shape)

    # 1.实验名称
    
    image_name = para.image_name#phase_diff_prop_pi,imagenet_prop_pi,0801_prop_pi,imagenet_prop_pi,imagenet256_prop_pi
    name = f'{shape[0]}_{shape[1]}_{image_name}' #读取0-pi归一化强度图txt文件名，
    print(name)


    # 生成实验结果、tensorboard日志文件、权重文件保存的文件夹
    localtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    
    measured_y_path_txt = f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/exp/{name}.txt'

    gt_txt = f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/{name}.txt' #读取的原相位图txt文件名，现为face，0-pi
    gt_matrix = my_readtxt(gt_txt)
    print(gt_matrix.dtype)

    result_folder = f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/{para.exp_name}/{name}/{localtime}'

    # 最好的loss初始化为无穷大
    best_loss = float('inf')

    # 随机种子和实验设备
    torch.manual_seed(para.seed)

    lrs = [0.1,0.01,  0.001,  0.0001,  0.00001,  0.000001, 0.0000001,0.00000001]
# [0.1,0.05,0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005,0.0000001,0.00000005,0.00000001]
    # for i in range(7):
    #     lrs.append(float('%.7f'% (0.1*(0.1**i))))
    #     lrs.append(float('%.7f'% (0.05*(0.1**i))))
        



    
    print(lrs)
    for lr in tqdm(lrs):
        print(type(lr))
        print(f'sam_{lr}')

        
        tb_folder = f'{result_folder}/tb_folder/{lr}'

        weight_folder = f"{result_folder}/weight_folder/{lr}"
        img_txt_folder = f'{result_folder}/img_txt_folder/{lr}'
    
        # tensorboard
        writer = SummaryWriter(tb_folder)
        
        # hypara
        batch_size = para.batch_size
        
        epochs = para.epochs
        print(batch_size,lr,epochs)
        hypar_file = open(f"{result_folder}/hypar.txt","w")
        device = torch.device(para.device) #分布式训练时用默认


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
        # print(para.model['name'])
        model_name = para.model['name']
        modelnet = import_class('arch.'+model_name,model_name)  
        net  =  modelnet().to(device)
        print('creating model')
        
        
        # 4.loss and optimization
        loss_mse = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.999)
        print('creating loss and optimization')

        # 记录训练开始前的超参数，网络结构，输入强度图，gt图像
        hypar_file.write(f'target： {para.target}\n')
        hypar_file.write(f'batch_size {batch_size}\n')
        hypar_file.write(f'lr {lr}\n')
        hypar_file.write(f'epochs {epochs}\n')
        hypar_file.write(f'network\n{net}\n')
        hypar_file.close()

        # 创建文件夹
        mkdir(tb_folder)
        mkdir(weight_folder)
        mkdir(img_txt_folder)
        
        # 5.training
        print('starting loop')
    
        for current_epoch in tqdm(range(epochs)):
            # print(f"Epoch {current_epoch}\n-------------------------------")

            for batch,(x,y) in (enumerate(train_dataloader)):
                
                optimizer.zero_grad()
                pred_y = net(x)
                


                measured_y = prop(pred_y[0, 0, :, :],dist=para.dist)
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
                if step % 3000 == 0:
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






        




