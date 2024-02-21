import sys 
sys.path.append("/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet") 
from library import result_visual
if __name__=='__main__':

    name = 'baseline'
    ref_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/1536_1536_ref_pi_01_prop_pi/2024-02-01-18-15/img_txt_folder/9000_pred.txt'
    sam_path=  '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/1536_1536_sam_pi_01_prop_pi/2024-02-01-18-53/img_txt_folder/4500_pred.txt'
    gt_ref_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_ref_pi_01_prop_pi.txt'
    gt_sam_path = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/gt/1536_1536_sam_pi_01_prop_pi.txt'

    # ref = my_readtxt(sam_path)
    # sam = my_readtxt(gt_sam_path)   
    # diff = (sam - ref)
    save_path = f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result_visual/{name}'
    # my_saveimage(diff,save_path)  

    # for cmap in tqdm(cmaps):
    #     result_visual(ref_path,sam_path,gt_ref_path,gt_sam_path,save_path,cmap)
    result_visual(ref_path,sam_path,gt_ref_path,gt_sam_path,save_path)
