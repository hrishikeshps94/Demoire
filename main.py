import argparse
import os
from train import Train
import torch
import time



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device used to train the model is {device}')
def main(arg_list):
    if arg_list.mode == 'train':
        train_pipe = Train(arg_list)
        train_pipe.run()
    elif arg_list.mode == 'test':
        pass
    else:
        raise ValueError('Use a valid mode')




if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Get command-line arguments.")
    main_path = os.path.dirname(os.path.abspath(__file__))
    arg_parser.add_argument('--mode', type=str, choices=['test', 'train'], default='train')
    arg_parser.add_argument('--checkpoint_folder', type=str, default='checkpoint')
    arg_parser.add_argument('--model_type', type=str, default='KTDN',help='Curretly availabel models are PDCRN,KTDN')
    arg_parser.add_argument('--train_path', type=str, default='/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/cvip/ds/train')
    arg_parser.add_argument('--test_path', type=str, default='/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/cvip/ds/val')
    arg_parser.add_argument('--im_shape', type=tuple, default=(64,64))
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=1000)
    arg_parser.add_argument('--LR', type=int, default=1e-3)
    arg_parser.add_argument('--num_filters', type=int, default=64)
    arg_parser.add_argument('--dilation_rates', type=tuple, default=(3, 2, 1, 1, 1, 1))
    arg_parser.add_argument('--nPyramidFilters', type=int, default=64)
    arg_parser.add_argument('--log_name', type=str, default='logger')
    arg_parser.add_argument('--in_ch', type=int, default=3)
    arg_parser.add_argument('--device', type=int, default=device)
    arg_list = arg_parser.parse_args()
    main(arg_list)