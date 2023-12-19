import argparse, os, torch, inspect
from finetune_batch import Finetune
from utils import str2bool, save_result
import pandas as pd


"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--mode', type = str, default='finetune')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--subject', type=int, default=1)

    parser.add_argument('--data_set', type=str, default='SleepEDF', help='data_set')
    parser.add_argument('--root_dir', type=str, default='/home/dilab/data')

    parser.add_argument('--model_name', type=str, default='SA-TSC')
    parser.add_argument('--exp_name', type=str, default='aaai') 
    parser.add_argument('--file_name', type=str, default = '1288') #1283 for HAR 
    parser.add_argument('--file_epoch', type=int, default=100)
    parser.add_argument('--spt_lr', type=float, default=0.01)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--csv_save', type=str2bool, default=True)

    return check_args(parser.parse_args())

def check_args(args):
    try:
        assert args.mode == 'finetune' or args.mode == 'linear_eval'
    except:
        print('select mode (finetune or linear_eval)')

    return args


def main():
    args = parse_args()
    print(args)
    output = Finetune(args)
    
    output.train()
    acc, f1, class_f1, f_names = output.test()

    if args.csv_save == True:
        save_result('result_'+args.data_set+'.csv', args.data_set, f_names, acc, f1, class_f1, args.random_seed, args.epoch, args.spt_lr, os.path.join(args.model_name, args.data_set, args.exp_name, args.file_name, 'spatio_' + str(args.file_epoch) +'epoch.pth'))

if __name__ == '__main__':
    main()