import sys 
sys.path.append("/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet") 
from library import result_visual,mkdir
if __name__=='__main__':
    num=155500
    name = f'sam_ref_dropout_full'#一、生成的文件存放路径，最好与二里一致1536_1536_sam_pi_01_prop_pi/2024-02-21-21-53
    
    # 二、读取的ref或Sam路径
    ref_path = f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/net_model_Dropout_full/1536_1536_ref_pi_01_prop_pi/2024-02-29-19-10/img_txt_folder/155500_pred.txt'
    sam_path=  f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/net_model_res/1536_1536_sam_pi_01_prop_pi/2024-02-23-21-00/img_txt_folder/79500_pred.txt'
    gt_ref_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_ref_pi_01_prop_pi.txt'
    gt_sam_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_sam_pi_01_prop_pi.txt'

    # ref = my_readtxt(sam_path)
    # sam = my_readtxt(gt_sam_path)   
    # diff = (sam - ref)
    save_path = f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result_visual/{name}/{num}'
    # my_saveimage(diff,save_path)  

    # for cmap in tqdm(cmaps):
    #     result_visual(ref_path,sam_path,gt_ref_path,gt_sam_path,save_path,cmap)
    mkdir(save_path)
    result_visual(ref_path,sam_path,gt_ref_path,gt_sam_path,save_path)
