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
args = parser.parse_args()

exp_name = 'exp1-obj-privacy'
exp_num = args.exp_num
resume = False
mask_epoch = 10

if exp_num == 1:
    # RUNNING
    server = 'gyuri'
    network_type = 'Food101'
    gpu = '0,1,2,3,4'
    batch_size = 6
    epoch_num = 200
    model_type = 'tf_efficientnet_b7'
    lr_g = 0.0001
    lr_d = 0.0001
    lr_c = 0.01
    lambda_i = 1.
    lambda_c = 1.
    split = True
    privacy = True

if exp_num == 2:
    # RUNNING
    server = 'giai'
    network_type = 'STL10'
    gpu = '0,1'
    batch_size = 36
    epoch_num = 100
    model_type = 'tf_efficientnet_b0'
    lr_g = 0.0003
    lr_d = 0.0001
    lr_c = 0.01

    lambda_i = 1.0
    lambda_c = 1.0
    split = True
    privacy = True

if exp_num == 3:
    server = 'gyuri'
    network_type = 'Food101'
    gpu = '0,1,2,3,4'
    batch_size = 14
    epoch_num = 200
    model_type = 'resnet50'
    lr_g = 0.0003
    lr_d = 0.0001
    lr_c = 0.01

    lambda_i = 1.
    lambda_c = 1.
    privacy = True

if exp_num == 4:
    server = 'giai'
    network_type = 'STL10'
    gpu = '0,1'
    batch_size = 36
    epoch_num = 200
    model_type = 'resnet50'
    lr_g = 0.0003
    lr_d = 0.0001
    lr_c = 0.01

    lambda_i = 1.0
    lambda_c = 1.0
    privacy = True

if exp_num == 5:
    # RUNNING
    server = 'nipa'
    network_type = 'Food101'
    gpu = '0,1'
    batch_size = 32
    epoch_num = 200
    model_type = 'tf_efficientnet_b7'
    lr_g = 0.0003
    lr_d = 0.0001
    lr_c = 0.01
    lambda_i = 1.
    lambda_c = 1.
    privacy = False

if exp_num == 6:
    # RUNNING
    server = 'nipa'
    network_type = 'STL10'
    gpu = '0,1'
    batch_size = 64
    epoch_num = 200
    model_type = 'efficientnet'
    lr_g = 0.0003
    lr_d = 0.0001
    lr_c = 0.01
    lambda_i = 1.
    lambda_c = 1.
    privacy = False

if exp_num == 7:
    server = 'nipa'
    network_type = 'Food101'
    gpu = '0,1'
    batch_size = 64
    epoch_num = 200
    model_type = 'resnet50'
    lr_g = 0.0003
    lr_d = 0.0001
    lr_c = 0.01
    lambda_i = 1.
    lambda_c = 1.
    privacy = False

if exp_num == 8:
    server = 'nipa'
    network_type = 'STL10'
    gpu = '0,1'
    batch_size = 64
    epoch_num = 200
    model_type = 'resnet50'
    lr_g = 0.0003
    lr_d = 0.0001
    lr_c = 0.01
    lambda_i = 1.
    lambda_c = 1.
    privacy = False


if exp_num == 11:
    # RUNNING
    server = 'nipa'
    network_type = 'Food101'
    gpu = '0,1'
    batch_size = 28
    epoch_num = 200
    model_type = 'tf_efficientnet_b7'
    lr_g = 0.0003
    lr_d = 0.0001
    lr_c = 0.01
    lambda_i = 1.
    lambda_c = 2.
    split = False
    privacy = True

if exp_num == 12:
    # RUNNING
    server = 'gyuri'
    network_type = 'Food101'
    gpu = '0,1,2,3,4'
    batch_size = 6
    epoch_num = 200
    model_type = 'tf_efficientnet_b7'
    lr_g = 0.0001
    lr_d = 0.0001
    lr_c = 0.01
    lambda_i = 1.
    lambda_c = 1.
    split = False
    privacy = True


script = '%s main_cls.py --exp_name %s --exp_num %d --server %s --network_type %s --privacy %s --split %s --mask_epoch %d \
                         --gpu %s --batch_size %d --epoch_num %d --model_type %s --lr_g %f --lr_d %f --lr_c %f \
                         --lambda_i %f --lambda_c %f --resume %s' \
                         %('python', exp_name, exp_num, server, network_type, privacy, split, mask_epoch, \
                           gpu, batch_size, epoch_num, model_type, lr_g, lr_d, lr_c, \
                           lambda_i, lambda_c, resume)

os.system(script)