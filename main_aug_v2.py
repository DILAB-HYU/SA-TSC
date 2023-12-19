import argparse, os, torch, inspect
from pretrain_aug_v2 import Graph_sleep
from utils import str2bool

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
   
    # training argument
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='SA-TSC')
    parser.add_argument('--exp_name', type=str, default='aaai')
    
    ## directory argument 
    parser.add_argument('--data_set', type=str, default='ISRUC', help='data_set')
    parser.add_argument('--root_dir', type=str, default='/home/dilab/data',help='Data directory ')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory name to save the model')

    parser.add_argument('--print_interval', type=int, default=32, help='Interval between print loss')

    parser.add_argument('--output_dim', type=int, default=128)
    parser.add_argument('--kernel_size', type=int, default=8) ## 3 for HAR 

    # hyperparameter argument 
    parser.add_argument('--lr', type=float, default=0.0001) 

    parser.add_argument('--ssl_distance_adj', type=str, default='gaussian_kernel')

    parser.add_argument('--gumbel', type=str2bool, default=True)
    parser.add_argument('--gumbel_only', type=str2bool, default=False)
    parser.add_argument('--gumbel_tmp', type=float, default=1.0)

    parser.add_argument('--adj_dist', type=str, default='gaussian_kernel')
    parser.add_argument('--adj_norm', type=str2bool, default=False)
    
    parser.add_argument('--alpha', type=float, default=1.0, help='hyperparamter for spectral loss')
    parser.add_argument('--beta', type=float, default=1.0, help='hyperparamter for agreement loss')
    parser.add_argument('--sg', type=str2bool, default=True, help='stop gradient operator')  
    parser.add_argument('--mu', type=float, default=100.0, help='gaussian kernel length scale parameter')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--aug_mode', type=str,  default='cross_domain', help='graph augmentation for self supervised learning') 
    parser.add_argument('--randomness', type=str2bool, default=False)




    # mode argument 
    parser.add_argument('--benchmark_mode', type=str2bool, default=True, help='auto tuner of pytorch') 
    parser.add_argument('--save', type=str2bool, default=True, help='whether to save or not') 
    
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    if args.save: 
        save_path = os.path.join(args.save_dir, args.model_name)
        if not os.path.exists(os.path.join(save_path, args.data_set)):

            os.makedirs(os.path.join(save_path, args.data_set))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')


    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    print(args)

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True
    
    print("model save at ", inspect.getfile(inspect.currentframe())[:-3])
    model = Graph_sleep(args)

    # launch the graph in a session
    model.train()
    print(" [*] Training finished!")

    


if __name__ == '__main__':
    main()
