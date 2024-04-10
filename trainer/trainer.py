import torch
import sys
sys.path.append("D:\\tyh\phynet")
from prop import propcomplex
import numpy as np
import matplotlib.pyplot as plt
from library import (my_readtxt,mkdir,visual_data,my_saveimage,my_savetxt,my_save2image)
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
    
    

def train_epoch_baseline(dataloader,net,loss_mse,optimizer,para,pha_gt,amp_gt,mask_gt,current_epoch,writer,weight_folder,img_txt_folder,best_loss):
    '''
    This function is used to train the network for one epoch.


    Parameters:
    dataloader: dataloader of the dataset
    net: the network to be trained
    loss_mse: the loss function for the network
    optimizer: the optimizer for the network
    para: the parameters of the config file
    pha_gt: the ground truth of the phase
    amp_gt: the ground truth of the amplitude
    mask_gt: the ground truth of the mask
    current_epoch: the current epoch number
    writer: tensorboard writer
    weight_folder: the folder to save the weights
    img_txt_folder: the folder to save the images and txt files
    best_loss: the best loss value of the network


    Return:
    None

    '''
            
    for i, (Speckle,Pha) in enumerate(dataloader):
        Speckle = Speckle.to(para.device)
        Pha = Pha.to(para.device)
        
        optimizer.zero_grad()
        # forward proapation
        pred_pha = net(Speckle) 
        
        flattened_pred_pha = pred_pha[0, 0, :, :] 
        if para.constraint == 'strong':
            Uo = amp_gt*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场
        elif para.constraint == 'weak':
            Uo = mask_gt*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场
            
        Ui = propcomplex(Uo,dist=para.dist,device=para.device)            
                
        pred_Speckle = torch.abs(Ui)

        loss_mse_value = loss_mse(Speckle[0, 0, :, :].float(),pred_Speckle.float())
        loss_value =  loss_mse_value

        # backward proapation
        loss_value.backward()
        
        optimizer.step() 
            
            # 实验记录

        step = current_epoch 

        result_record(current_epoch,writer,loss_value,best_loss,net,flattened_pred_pha,pred_Speckle,Speckle,pha_gt,img_txt_folder,weight_folder)
            
def train_epoch_complex(dataloader,net,loss_mse,optimizer,para,pha_gt,amp_gt,mask_gt,current_epoch,writer,weight_folder,img_txt_folder,best_loss):
    '''
    This function is used to train the network for one epoch.


    Parameters:
    dataloader: dataloader of the dataset
    net: the network to be trained
    loss_mse: the loss function for the network
    optimizer: the optimizer for the network
    para: the parameters of the config file
    pha_gt: the ground truth of the phase
    amp_gt: the ground truth of the amplitude
    mask_gt: the ground truth of the mask
    current_epoch: the current epoch number
    writer: tensorboard writer
    weight_folder: the folder to save the weights
    img_txt_folder: the folder to save the images and txt files
    best_loss: the best loss value of the network


    Return:
    None

    '''
            
    for i, (Speckle,Pha) in enumerate(dataloader):
        Speckle = Speckle.to(para.device)
        Pha = Pha.to(para.device)
        
        optimizer.zero_grad()
        # forward proapation
        pred_complex = net(Speckle) 
        
        flattened_pred_mask = pred_complex[0, 0, :, :] 
        
        flattened_pred_pha = pred_complex[0, 1, :, :]

        # Uo = flattened_pred_mask*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场
        Uo = flattened_pred_mask*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场

            
        Ui = propcomplex(Uo,dist=para.dist,device=para.device)            
                
        pred_Speckle = torch.abs(Ui)

        zero_matrix = torch.zeros_like(flattened_pred_mask).to(para.device)
        one_matrix = torch.ones_like(flattened_pred_mask).to(para.device)
        loss_1 = loss_mse((flattened_pred_mask*(one_matrix-mask_gt)).float(),zero_matrix.float())
        # loss_3 = loss_mse((flattened_pred_pha*(one_matrix-mask_gt)).float(),zero_matrix.float())
        loss_2 = loss_mse(Speckle[0, 0, :, :].float(),pred_Speckle.float())
        loss_value =  loss_2+loss_1#+loss_3

        # backward proapation
        loss_value.backward()
        
        optimizer.step() 
            
            # 实验记录

        step = current_epoch 

        result_record(current_epoch,writer,loss_value,best_loss,net,flattened_pred_pha,pred_Speckle,Speckle,pha_gt,img_txt_folder,weight_folder)
        if step % 3000 == 0:
            my_saveimage(flattened_pred_mask.cpu().detach().numpy(),f'{img_txt_folder}/{step}_perdmask.png',dpi=800) 
                   
            my_saveimage((flattened_pred_mask*(one_matrix-mask_gt)).cpu().detach().numpy(),f'{img_txt_folder}/{step}_perdmask2.png',dpi=800)
            my_saveimage((flattened_pred_mask-mask_gt).cpu().detach().numpy(),f'{img_txt_folder}/{step}_perdmaskdiff.png',dpi=800)
            # my_saveimage((flattened_pred_mask*(one_matrix-mask_gt)).cpu().detach().numpy(),f'{img_txt_folder}/{step}_lingshi.png',dpi=800)
            
def train_epoch_complex_gs(dataloader,net,loss_mse,optimizer,para,pha_gt,amp_gt,mask_gt,current_epoch,writer,weight_folder,img_txt_folder,best_loss):
    '''
    This function is used to train the network for one epoch.


    Parameters:
    dataloader: dataloader of the dataset
    net: the network to be trained
    loss_mse: the loss function for the network
    optimizer: the optimizer for the network
    para: the parameters of the config file
    pha_gt: the ground truth of the phase
    amp_gt: the ground truth of the amplitude
    mask_gt: the ground truth of the mask
    current_epoch: the current epoch number
    writer: tensorboard writer
    weight_folder: the folder to save the weights
    img_txt_folder: the folder to save the images and txt files
    best_loss: the best loss value of the network


    Return:
    None

    '''
            
    for i, (Speckle,Pha) in enumerate(dataloader):
        Speckle = Speckle.to(para.device)
        Pha = Pha.to(para.device)
        
        optimizer.zero_grad()
        # forward proapation
        pred_complex = net(Speckle) 
        
        flattened_pred_mask = pred_complex[0, 0, :, :] 
        
        flattened_pred_pha = pred_complex[0, 1, :, :]

        # Uo = flattened_pred_mask*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场
        Uo = mask_gt*torch.exp(1j*flattened_pred_pha) #光纤端面初始复光场

            
        Ui = propcomplex(Uo,dist=para.dist,device=para.device)            
                
        pred_Speckle = torch.abs(Ui)
        
        Ua = Speckle[0, 0, :, :]*Ui/torch.abs(Ui)
        Um = propcomplex(Ua,dist = -1*para.dist,device=para.device)
                

        zero_matrix = torch.zeros_like(flattened_pred_mask).to(para.device)
        one_matrix = torch.ones_like(flattened_pred_mask).to(para.device)
        # loss_1 = loss_mse((flattened_pred_mask*(one_matrix-mask_gt)).float(),zero_matrix.float())
        loss_3 = loss_mse((flattened_pred_pha).float(),torch.angle(Um).float())
        loss_2 = loss_mse(Speckle[0, 0, :, :].float(),pred_Speckle.float())
        loss_value =  loss_2+loss_3

        # backward proapation
        loss_value.backward()
        
        optimizer.step() 
            
            # 实验记录

        step = current_epoch 

        result_record(current_epoch,writer,loss_value,best_loss,net,flattened_pred_pha,pred_Speckle,Speckle,pha_gt,img_txt_folder,weight_folder)
        if step % 3000 == 0:
            my_saveimage(flattened_pred_mask.cpu().detach().numpy(),f'{img_txt_folder}/{step}_perdmask.png',dpi=800) 
                   
            my_saveimage((flattened_pred_mask*(one_matrix-mask_gt)).cpu().detach().numpy(),f'{img_txt_folder}/{step}_perdmask2.png',dpi=800)
            my_saveimage((flattened_pred_mask-mask_gt).cpu().detach().numpy(),f'{img_txt_folder}/{step}_perdmaskdiff.png',dpi=800)
            # my_saveimage((flattened_pred_mask*(one_matrix-mask_gt)).cpu().detach().numpy(),f'{img_txt_folder}/{step}_lingshi.png',dpi=800)