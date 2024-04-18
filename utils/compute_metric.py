import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

from scipy import ndimage
def compute_std(mask,pha_diff):
    """
    Compute the standard deviation of the phase difference between the masked region and the background.
    """
    pha = mask * pha_diff
    pha = np.mod(pha,2*np.pi)
    plt.figure(figsize=(6, 6))
    plt.imshow(pha,cmap='viridis')
    plt.colorbar()
    plt.show()
    
    non_zero_elements = pha[pha != 0]
    mean = np.mean(non_zero_elements)
    std = np.std(non_zero_elements)
    data_mode = mode(non_zero_elements)
    
    plt.figure(figsize=(8,6))
    plt.scatter(range(len(non_zero_elements)),non_zero_elements,marker='o',s=10,c='r')
    plt.title('Phase difference distribution')
    plt.xlabel('Index')
    plt.ylabel('Phase difference')
    plt.grid(True)
    
    # plt.axhline(y=mean,color='g',linestyle='-',label=f'Mean:{mean:.4f}')
    # plt.axhline(y=mean+std,color='b',linestyle='--',label=f'Mean+std:{mean+std:.4f}')
    # plt.axhline(y=mean-std,color='b',linestyle='--',label=f'Mean-std:{mean-std:.4f}')
    
    # plt.axhline(y=float(data_mode[0]),color='g',linestyle='-',label=f'Mode:{data_mode[0]:.4f}')
    
    
    plt.legend(loc='lower left')
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.hist(non_zero_elements,bins='auto',color='r',label='Histogram')
    
    ax = plt.gca()
    ax.set_xticks(np.arange(0,2*np.pi,0.4))
    
    plt.title('Phase difference distribution')
    plt.xlabel('Phase difference')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    return std
def compute_core_std(mask,pha_diff):
    '''
    Compute the standard deviation of the phase difference between the masked region and the background.
    
    '''
    pha = mask * pha_diff
    pha = np.mod(pha,2*np.pi)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(pha,cmap='viridis')
    # plt.colorbar()
    # plt.show()  
    
    scale = (pha > 0.01) & (pha < (2*np.pi-0.01))
    
    labeled_array, num_features = ndimage.label(scale)
    
    # plt.figure(figsize=(6, 6))
    # plt.imshow(labeled_array,cmap='viridis')
    # plt.colorbar()
    # plt.show()
    
    # print(num_features)
    
    output = np.zeros_like(pha)
    
    core_mean_values = []
    
    for i in range(1,num_features+1):
        mask_core = (labeled_array == i)
        core_mean_value = pha[mask_core].mean()
        core_mean_values.append(core_mean_value)
        output[mask_core] = core_mean_value
        
    mean = np.mean(core_mean_values)
    std = np.std(core_mean_values)    
    # print(mean_values)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(output,cmap='viridis')
    # plt.colorbar()
    # plt.show()
    
    # plt.figure(figsize=(8,6))
    # plt.scatter(range(len(core_mean_values)),core_mean_values,marker='o',s=10,c='r')
    
    
    # ax = plt.gca()
    # ax.set_yticks(np.arange(0,2*np.pi,0.4))
    # plt.title('Phase difference distribution')
    # plt.xlabel('Index')
    # plt.ylabel('Phase difference')
    # plt.grid(True)
    
    # plt.axhline(y=mean,color='g',linestyle='-',label=f'Mean:{mean:.4f}')
    # plt.axhline(y=mean+std,color='b',linestyle='--',label=f'Mean+std:{mean+std:.4f}')
    # plt.axhline(y=mean-std,color='b',linestyle='--',label=f'Mean-std:{mean-std:.4f}')
    # plt.axhline(y=std,color='black',linestyle='--',label=f'std:{std:.4f}')
    # plt.legend(loc='lower left')
    # plt.show()    
    return core_mean_values,mean,std,output,num_features,labeled_array
def compute_core_std_plot(mask,pha_diff,save_path,meanflag=None,outputflag=None,labeledflag=None):
    '''
    Compute the standard deviation of the phase difference between the masked region and the background.
    
    '''
    core_mean_values,mean,std,output,num_features,labeled_array = compute_core_std(mask,pha_diff)
    if meanflag is not None:
        plt.figure(figsize=(8,6))
        plt.scatter(range(len(core_mean_values)),core_mean_values,marker='o',s=10,c='r')
        
        ax = plt.gca()
        ax.set_yticks(np.arange(0,2*np.pi,0.4))
        plt.title('Phase difference distribution')
        plt.xlabel('Index')
        plt.ylabel('Phase difference')
        plt.grid(True)
        
        plt.axhline(y=mean,color='g',linestyle='-',label=f'Mean:{mean:.4f}')
        plt.axhline(y=mean+std,color='b',linestyle='--',label=f'Mean+std:{mean+std:.4f}')
        plt.axhline(y=mean-std,color='b',linestyle='--',label=f'Mean-std:{mean-std:.4f}')
        plt.axhline(y=std,color='black',linestyle='--',label=f'std:{std:.4f}')
        
        plt.legend(loc='lower left')
        plt.savefig(save_path) 
        
     
    if outputflag is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(output,cmap='viridis')
        plt.colorbar() 
        # plt.title(f'num_features:{num_features}')
        plt.savefig(save_path.replace('core_std','_PredPhaclean'))   
        
    if labeledflag is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(labeled_array,cmap='viridis')
        plt.colorbar()
        plt.title(f'num_features:{num_features}')
        plt.savefig(save_path.replace('core_std','labeled'))   
    
 
    

if __name__ == '__main__':
    # mask = np.loadtxt('D:\\tyh\simulateData\simulate_data\strong\\2pi\\10\\0.6\\4\\10_amp_simulate.txt',dtype=np.float32,delimiter=',')
    # pha_diff = np.loadtxt('D:\\tyh\Resultimulate\strong\\2pi\\10\\0.6\\4\\2024-04-10-16-50\img_txt_folder\\9000_PhaLoss.txt',dtype=np.float32,delimiter=',')   
    # save_path = 'D:\\tyh\\phynet\\utils\\std\\10.png'
    
    
    # mask = np.loadtxt('D:\\tyh\simulateData\simulate_data\strong\\2pi\\100\\0.6\\4\\100_amp_simulate.txt',dtype=np.float32,delimiter=',')
    # pha_diff = np.loadtxt('D:\\tyh\Resultimulate\strong\\2pi\\100\\0.6\\4\\2024-04-10-16-33\img_txt_folder\\3000_PhaLoss.txt',dtype=np.float32,delimiter=',')   
    # save_path = 'D:\\tyh\\phynet\\utils\\std\\100.png'    
    
    # mask = np.loadtxt('D:\\tyh\simulateData\simulate_data\strong\\2pi\\100\\0.6\\4\\100_amp_simulate.txt',dtype=np.float32,delimiter=',')
    # pha_diff = np.loadtxt('D:\\tyh\Resultimulate\strong\\2pi\\100\\0.6\\4\\2024-04-10-16-33\img_txt_folder\\3000_PhaLoss.txt',dtype=np.float32,delimiter=',')   
    # save_path = 'D:\\tyh\\phynet\\utils\\std\\100.png'  
    
    # mask = np.loadtxt('D:\\tyh\\simulateData\\simulate_data\\strong\\2pi\\100\\0.6\\2\\100_amp_simulate.txt',dtype=np.float32,delimiter=',')
    # pha_pred = np.loadtxt('D:\\tyh\GS\\100_predpha.txt',dtype=np.float32,delimiter=',')   
    # pha_gt = np.loadtxt('D:\\tyh\simulateData\simulate_data\strong\\2pi\\100\\0.6\\2\\100_pha_simulate.txt',dtype=np.float32,delimiter=',')   
    # pha_diff = np.mod(pha_pred-pha_gt,2*np.pi)   
    # save_path = 'D:\\tyh\\phynet\\utils\\std\\100.png'    
    # compute_core_std_plot(mask,pha_diff,save_path,meanflag=True)

    mask = np.loadtxt('D:\\tyh\\simulateData\\simulate_data\\strong\\2pi\\1000\\0.6\\2\\1000_amp_simulate.txt',dtype=np.float32,delimiter=',')
    pha_pred = np.loadtxt('D:\\tyh\Resultimulate\strong\\2pi\\1000\\0.6\\2\\2024-04-17-11-21\img_txt_folder\\1099_PredPha.txt',dtype=np.float32,delimiter=',')   
    pha_pred = pha_pred*mask
    
   
    pha_diff = np.mod(pha_pred,2*np.pi)   
    
    plt.figure(figsize=(6, 6))
    plt.imshow(pha_diff,cmap='viridis')
    plt.colorbar()
    plt.show()
   
