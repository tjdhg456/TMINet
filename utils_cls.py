import argparse
import os
from glob import glob

def init_args_cls():
    ## Argparse
    # Save and Resume Options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resume_path', type=str, default='./result_high/CIFAR100_CBAM_34/kkk')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--exp_name', type=str, default='imp')
    parser.add_argument('--exp_num', type=int, default=-1)
    parser.add_argument('--server', type=str, default='nipa')
    parser.add_argument('--privacy', type=lambda x: (x.lower() == 'true'), default=True)
    parser.add_argument('--resume', type=lambda x:(x.lower() == 'true'), default=False)

    # Label Option
    parser.add_argument('--network_type', type=str, default='STL10')

    # Learning
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--validation', type=int, default=3)

    # Model
    # Generator
    parser.add_argument('--G_name', type=str, default='resnet_9blocks', choices=['resnet_9blocks','resnet_6blocks'])
    parser.add_argument('--G_input_nc', type=int, default=3)
    parser.add_argument('--G_output_nc', type=int, default=3)

    # Discriminator
    parser.add_argument('--temperature', type=float, default=0.5)

    # Classifier
    parser.add_argument('--model_type', type=str, default='tf_efficientnet_b0')
    parser.add_argument('--pretrained', type=lambda x: x.lower() == 'true', default=True)

    # Param
    parser.add_argument('--lambda_c', type=float, default=1.)
    parser.add_argument('--lambda_g', type=float, default=1.)
    parser.add_argument('--lambda_d', type=float, default=1.)

    parser.add_argument('--lr_g', type=float, default=0.0003)
    parser.add_argument('--lr_d', type=float, default=0.0001)
    parser.add_argument('--lr_c', type=float, default=0.01)

    args = parser.parse_args()

    # Embedding dimension
    args.D_out_emb = 512

    # num_class
    if args.network_type == 'ImageNet':
        args.input_size = 224
        args.num_class = 100
        args.avg_pool = True
    elif args.network_type == 'STL10':
        args.input_size = 224
        args.num_class = 10
        args.avg_pool = True
    elif args.network_type == 'Caltech101':
        args.input_size = 224
        args.num_class = 101
        args.avg_pool = True
    elif args.network_type == 'Food101':
        args.input_size = 224
        args.num_class = 101
        args.avg_pool = True
    elif args.network_type == 'OOD_Food':
        args.input_size = 224
        args.num_class = 10
        args.avg_pool = True
    elif args.network_type == 'OOD_ImageNet':
        args.input_size = 224
        args.num_class = 10
        args.avg_pool = True
    elif args.network_type == 'SD198':
        args.input_size = 224
        args.num_class = 198
        args.avg_pool = True
    else:
        raise('Select proper network_type==Dataset type')


    # Server
    if args.server == 'lilka':
        args.data_root = '/SSD1/sung/dataset'
    elif args.server == 'gyuri':
        args.data_root = '/data_2/sung/dataset'
    elif args.server == 'nipa':
        args.data_root = '/home/sung/dataset'
    elif args.server == 'giai':
        args.data_root = '/mnt/giai4/sung/dataset'

    # Data path
    args.data_root = os.path.join(args.data_root, args.network_type.lower())

    # Save folder
    args.save_folder = './result/%s/%s/%s' %(args.network_type.lower(), args.exp_name, str(args.exp_num))
    os.makedirs(args.save_folder, exist_ok=True)

    # resume
    if args.resume:
        args.path = os.path.join(args.save_folder, 'last.ckpt')

    return args


