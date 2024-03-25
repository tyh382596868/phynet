'''
训练过程速度提高

'''

import os
import sys 
sys.path.append("/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet") 
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=128'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from library import (my_readtxt,mkdir,visual_data,prop,my_saveimage,my_savetxt)
from dataset import mydataset
from config.parameter import Parameter,import_class
from trainer.trainer import train_epoch

import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import argparse

def cropImage(img,image_width = 1920,image_height = 2560,crop_width = 1920,crop_height = 2048):
    
    # 图像尺寸 image_width = 1920,image_height = 2560
    
    
    # 截取矩形的尺寸 crop_width = 1920,crop_height = 2048
    
    
    # 计算截取矩形的左上角坐标
    x_coordinate = (image_width - crop_width) // 2
    y_coordinate = (image_height - crop_height) // 2
    # 截取图像
    cropped_img = img[x_coordinate:x_coordinate + crop_width,y_coordinate:y_coordinate + crop_height]
    
    return cropped_img

def getCropImage(file_path,crop_width = 1920,crop_height = 2048):
    # 读取并打印文件内容
    amp = np.loadtxt(file_path,dtype=np.float32,delimiter=',')
    print(f"内容来自文件 {filename}:")
    print(f'shape of data:{amp.shape}')
    print("---------------------")
    
    # 获取对应的相位路径
    pha_path = file_path.replace('amp', 'pha').replace('_prop001', '')
    # 打印对应的相位路径
    
    pha = np.loadtxt(pha_path,dtype=np.float32,delimiter=',')
    print(f"内容来自文件 {pha_path}:")
    print(f'shape of data:{pha.shape}')
    print("---------------------")

    # 截取图像
    cropped_amp = cropImage(amp,crop_width = crop_width,crop_height = crop_height)
    
    # 截取图像
    cropped_pha = cropImage(pha,crop_width = crop_width,crop_height = crop_height)
    

    
    return cropped_amp,cropped_pha    

   

# def train_epoch(train_dataloader,net,loss_mse,optimizer,para,current_epoch,cropped_pha,best_loss):
    
#     for batch,(x,y) in (enumerate(train_dataloader)):
        
#         optimizer.zero_grad()
#         # forward proapation
#         pred_y = net(x) 
        
#         flattened_pred_y = pred_y[0, 0, :, :]      
#         measured_y = prop(flattened_pred_y,dist=para.dist)
#         loss_mse_value = loss_mse(y.float(),measured_y.float())
#         loss_value =  loss_mse_value

#         # backward proapation
#         loss_value.backward()
        
#         optimizer.step()
        
#         return loss_value,flattened_pred_y
        
def train_epoch(train_dataloader,net,loss_mse,optimizer):
    for (x,y) in (train_dataloader):
        x = x.to(device)
        flattened_x = x[0, 0, :, :] 
        optimizer.zero_grad()
        # forward proapation
        pred_y = net(x) 
        
        flattened_pred_y = pred_y[0, 0, :, :]      
        measured_y = prop(flattened_pred_y,dist=para.dist)
        loss_mse_value = loss_mse(flattened_x.float(),measured_y.float())
        loss_value =  loss_mse_value

        # backward proapation
        loss_value.backward()
        
        optimizer.step()   

        return flattened_pred_y,measured_y,loss_value
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--opt', type=str, default='/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet/option/baseline2.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    para = Parameter(args.opt)
    
    

    localtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())    
 
    result_folder = f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/{para.exp_name}/{localtime}'

    tb_folder = f'{result_folder}/tb_folder'
    weight_folder = f"{result_folder}/weight_folder"
    img_txt_folder = f'{result_folder}/img_txt_folder'
    
    # 最好的loss初始化为无穷大
    best_loss = float('inf')

    # 随机种子和实验设备
    torch.manual_seed(para.seed)
    
    # hypara
    batch_size = para.batch_size
    lr = para.lr
    epochs = para.epochs
    print(batch_size,lr,epochs)
    # 初始化存储所有图片loss列表的列表
    all_experiment_losses = []
    
    device = torch.device(para.device) #分布式训练时用默认    
    
    # 定义文件夹路径
    folder_paths = para.folder_paths
    # 遍历每个文件夹
    for folder in folder_paths:
        # 检查文件夹是否存在
        if os.path.exists(folder):
            # 遍历文件夹内的所有文件
            for filename in os.listdir(folder):
                # 检查文件是否为.txt文件
                if filename.endswith('.txt'):
                    # 拼接完整文件路径
                    file_path = os.path.join(folder, filename)
                    cropped_amp,cropped_pha = getCropImage(file_path,crop_width = para.image_height,crop_height = para.image_width)
                    
                    # # 保存图像
                    # mkdir(folder.replace('amp','amp_cropped'))
                    # cropped_amp_path = file_path.replace('amp','amp_cropped').replace('txt','png')
                    # my_saveimage(cropped_amp,cropped_amp_path)
                    
                    # # 保存图像
                    # mkdir(folder.replace('amp','pha_cropped'))
                    # cropped_pha_path = file_path.replace('amp','pha_cropped').replace('txt','png')
                    # my_saveimage(cropped_pha,cropped_pha_path)

                    # 1.dataset

                    transform = transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(degrees=(0, 360)),
                    transforms.ToTensor(),
                    # transforms.Resize((720,720))
                    ])

                    train_data = mydataset(cropped_amp,cropped_pha,transform=transform)


                    # 2.dataloader
                    
                    train_dataloader = torch.utils.data.DataLoader(
                        train_data, batch_size = batch_size, shuffle = True
                        )
                    print('loading data')                    

                    # 3.model
                    print(para.model['name'])
                    model_name = para.model['name']
                    modelnet = import_class('arch.'+model_name,model_name) 
                    if  para.model['name'] == 'net_model_Dropout_full':
                        net  =  modelnet(drop=para.model['p']).to(device)                        
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
                    
                    # # 创建文件夹
                    data_name = folder.split('/amp/')[1] + f'_{filename[:-4]}'
                    

                    tb_folderimg = (f'{tb_folder}/{data_name}')
                    weight_folderimg = (f'{weight_folder}/{data_name}')
                    img_txt_folderimg = (f'{img_txt_folder}/{data_name}')                    
                    # 创建文件夹
                    mkdir(tb_folderimg)
                    mkdir(weight_folderimg)
                    mkdir(img_txt_folderimg)
                    hypar_file = open(f"{result_folder}/hypar.txt","w")
                    # 记录训练开始前的超参数，网络结构，输入强度图，gt图像
                    hypar_file.write(f'target： {para.target}\n')
                    hypar_file.write(f'batch_size {batch_size}\n')
                    hypar_file.write(f'lr {lr}\n')
                    hypar_file.write(f'epochs {epochs}\n')
                    hypar_file.write(f'network\n{loss_mse}\n')
                    hypar_file.write(f'network\n{net}\n')
                    
                    hypar_file.close()                    

                    # tensorboard
                    writer = SummaryWriter(tb_folderimg)
                    # 初始化一张图片的loss列表
                    experiment_losses = []    
                    
                    # 5.training
                    print('starting loop')


                    for current_epoch in tqdm(range(epochs)):
                        # print(f"Epoch {current_epoch}\n-------------------------------")

                        # for batch,(x,y) in (enumerate(train_dataloader)):
                            
                        #     optimizer.zero_grad()
                        #     # forward proapation
                        #     pred_y = net(x) 
                            
                        #     flattened_pred_y = pred_y[0, 0, :, :]      
                        #     measured_y = prop(flattened_pred_y,dist=para.dist)
                        #     loss_mse_value = loss_mse(y.float(),measured_y.float())
                        #     loss_value =  loss_mse_value

                        #     # backward proapation
                        #     loss_value.backward()
                            
                        #     optimizer.step()   
                                                                                 
                        # 实验记录
                        flattened_pred_y,measured_y,loss_value = train_epoch(train_dataloader,net,loss_mse,optimizer)
                        step = current_epoch 
                        experiment_losses.append(loss_value.item())
                        # 记录loss
                        if step % 50 == 0:
                            # tb记录loss
                            writer.add_scalar('training loss',
                                            loss_value.item(),
                                            step)
                            
                            phase_diff = np.abs(cropped_pha-((flattened_pred_y).cpu().detach().numpy()))

                            writer.add_scalar('相位差',
                                            np.mean(phase_diff),
                                            step)
                            
                            
                            # 记录最好的模型权重
                            # 保存loss值最小的网络参数
                            if loss_value < best_loss:
                                best_loss = loss_value
                                torch.save(net.state_dict(), f"{weight_folderimg}/best_model.pth")

                        # 记录中间结果图片
                        if step % 3000 == 0:
                                    my_saveimage(flattened_pred_y.cpu().detach().numpy(),f'{img_txt_folderimg}/{step}_PredPha.png')
                                    my_savetxt(flattened_pred_y.cpu().detach().numpy(),f'{img_txt_folderimg}/{step}_PredPha.txt')

                                    my_saveimage(measured_y.cpu().detach().numpy(),f'{img_txt_folderimg}/{step}_PredAmp.png')
                                    my_savetxt(measured_y.cpu().detach().numpy(),f'{img_txt_folderimg}/{step}_PredAmp.txt')

                                    my_saveimage(cropped_amp[:,:]-(measured_y).cpu().detach().numpy(),f'{img_txt_folderimg}/{step}_AmpLoss.png')
                                    my_savetxt(cropped_amp[:,:]-(measured_y).cpu().detach().numpy(),f'{img_txt_folderimg}/{step}_AmpLoss.txt')
                                    
                                    my_saveimage(cropped_pha-((flattened_pred_y).cpu().detach().numpy()),f'{img_txt_folderimg}/{step}_PhaLoss.png')
                                    my_savetxt(cropped_pha-((flattened_pred_y).cpu().detach().numpy()),f'{img_txt_folderimg}/{step}_PhaLoss.txt')

                        if step % 40 == 0:
                            # 80的时候显存差不多满了
                            torch.cuda.empty_cache()
                        
            # 将当前实验的loss列表添加到总列表中
            all_experiment_losses.append(experiment_losses)


                    
        else:
            print(f"文件夹 {folder} 不存在。")



    # 计算每个实验的平均loss
    average_losses = [np.mean(all_experiment_losses, axis=0)]

    # 打印每个实验的平均loss
    print(average_losses)
 
    # 绘制loss曲线图
    plt.figure(figsize=(10, 6))

    # # 绘制每个实验的loss列表
    # for i, losses in enumerate(all_experiment_losses):
    #     plt.plot(range(1, num_iterations + 1), losses, label=f'Experiment {i+1} Loss')

    # 绘制平均loss
    plt.plot(range(1, epochs + 1), average_losses[0], label='Average Loss', linestyle='--')

    # 设置图表标题和标签
    plt.title('Loss Over迭代ations for Multiple Experiments')
    plt.xlabel('迭代ation')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # 保存图表为PNG图片
    plt.savefig(f'{result_folder}\loss_over_iterations.png')    




        




