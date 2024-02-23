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
# from unet import net_model_v1
from unetgood import UnetGenerator
from baseunet import UnetGeneratorDouble,UNet

from loss import TVLoss
from dataset import measured_y_txt_dataset256,measured_y_txt_dataset256_fast

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import argparse
from config.parameter import Parameter,import_class


def train_loop(para,train_dataloader,net,loss_mse,optimizer):

    for batch,(x,y) in (enumerate(train_dataloader)):
        
        optimizer.zero_grad()

        pred_y = net(x)        
        measured_y = prop(pred_y[0, 0, :, :],dist=para.dist)

        loss_mse_value = loss_mse(y.float(),measured_y.float())
        loss_value =  loss_mse_value

        # backward proapation

        loss_value.backward()
        optimizer.step()

