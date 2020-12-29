import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import torch
import os
from torch.utils.data.sampler import BatchSampler
from collections import defaultdict
from tqdm import tqdm
import pickle
import os
from glob import glob
import argparse
from loader import dataset, intra_dataset

## Argparse
# Save and Resume Options
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--server', type=str, default='nipa')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

# Data path
if args.server == 'nipa':
    args.data_base = '/home/sung/dataset'
elif args.server == 'gyuri':
    args.data_base = '/data_2/sung/dataset'
elif args.server == 'lilka':
    args.data_base = '/SSD1/sung/dataset'
elif args.server == 'giai':
    args.data_base = '/mnt/giai4/sung/dataset'
else:
    raise('Select proper server')

for args.network_type in ['CIFAR10', 'CIFAR100', 'ImageNet', 'STL10']:
    # num_class
    if args.network_type == 'CelebA':
        args.num_class = 2
    else:
        raise ('Select proper network_type==Dataset type')

    args.data_root = os.path.join(args.data_base, args.network_type.lower())

    ## save
    tr_dataset = dataset(args, mode='train')
    intra_dataset(tr_dataset, args, mode='train')

    val_dataset = dataset(args, mode='val')
    intra_dataset(val_dataset, args, mode='val')

    test_dataset = dataset(args, mode='test')
    intra_dataset(test_dataset, args, mode='test')