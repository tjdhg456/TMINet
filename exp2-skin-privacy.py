import subprocess
import numpy as np
from multiprocessing import Process
import os
import torch
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Basic Python Environment
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp_num', type=int, default=-1)
parser.add_argument('--gpu', type=str, default='0,1,2')
args = parser.parse_args()

exp_name = 'exp2-skin-privacy'
exp_num = args.exp_num

resume = False
lambda_g = 1.
G_name = 'resnet_9blocks'

## Evaluating (original model) -- 2x
if exp_num == 1:
    # RUNNING
    server = 'nipa'
    network_type = 'SD198'
    gpu = '0,1'
    batch_size = 24
    epoch_num = 200
    model_type = 'tf_efficientnet_b0'
    lr_g = 2e-4
    lr_d = 1e-4
    lr_c = 0.01
    lambda_i = 1.
    lambda_c = 1.
    lambda_g = 1.
    G_name = 'resnet_9blocks'
    privacy = True
    pretrained = True


script = '%s main_cls.py --exp_name %s --exp_num %d --server %s --network_type %s --privacy %s \
                         --gpu %s --batch_size %d --epoch_num %d --model_type %s --lr_g %f --lr_d %f --lr_c %f \
                         --lambda_c %f --lambda_g %f --resume %s \
                         --pretrained %s --G_name %s' \
                         %('python', exp_name, exp_num, server, network_type, privacy, \
                           gpu, batch_size, epoch_num, model_type, lr_g, lr_d, lr_c, \
                           lambda_c, lambda_g, resume, \
                           pretrained, G_name)

os.system(script)