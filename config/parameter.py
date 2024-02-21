import argparse
import yaml

if __name__=='__main__':
    print('argparse')
    parse = argparse.ArgumentParser(description='training script')
    # print(parse)

    parse.add_argument('--file',type=str,default='/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet/config/config.yaml')

    opt = parse.parse_args()
    print(opt.file)

    with open(opt.file, 'r') as file:
        config = yaml.safe_load(file) 

    print(config.get('device')) 