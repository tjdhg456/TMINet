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
from copy import deepcopy
from data.augmentation import strong_augment
from PIL import Image
import albumentations
from albumentations import pytorch

def un_normalization(args):
    if args.network_type == 'CIFAR10':
        unnormalize = transforms.Compose([transforms.Normalize((0., 0., 0.), (1/0.2023, 1/0.1994, 1/0.2010)),
                                          transforms.Normalize((-0.4914, -0.4822, -0.4465), (1.,1.,1.))])

    elif args.network_type == 'CIFAR100':
        unnormalize = transforms.Compose([transforms.Normalize((0., 0., 0.), (1/0.2023, 1/0.1994, 1/0.2010)),
                                          transforms.Normalize((-0.4914, -0.4822, -0.4465), (1.,1.,1.))])

    elif args.network_type in ['ImageNet', 'STL10', 'Food101', 'SD198']:
        unnormalize = transforms.Compose([transforms.Normalize((0., 0., 0.), (1/0.229, 1/0.224, 1/0.225)),
                                          transforms.Normalize((-0.485, -0.456, -0.406), (1.,1.,1.))])

    elif (args.network_type == 'OOD_Food') or (args.network_type == 'OOD_ImageNet'):
        unnormalize = transforms.Compose([transforms.Normalize((0., 0., 0.), (1/0.5, 1/0.5, 1/0.5)),
                                          transforms.Normalize((-0.5, -0.5, -0.5), (1.,1.,1.))])
    else:
        raise('Select proper network_type')

    return unnormalize

def normalization(args):
    if args.network_type == 'CIFAR10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    elif args.network_type == 'CIFAR100':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    elif args.network_type in ['ImageNet', 'STL10', 'Food101', 'SD198']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    elif (args.network_type == 'OOD_Food') or (args.network_type == 'OOD_ImageNet'):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        raise('Select proper network_type')

    return normalize


def dataset(args, mode='train'):
    mode = mode.lower()
    normalize = normalization(args)
    if args.network_type == 'CIFAR10':
        if mode == 'train':
            transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
            ])

        else:
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
            ])

        dataset = torchvision.datasets.CIFAR10(args.data_root, train=(mode == 'train'), transform=transform)

    elif args.network_type == 'CIFAR100':
        if mode == 'train':
            transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
            ])

        else:
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
            ])

        dataset = torchvision.datasets.CIFAR100(args.data_root, train=(mode == 'train'), transform=transform)

    elif args.network_type == 'ImageNet':
        if mode == 'train':
            transform = transforms.Compose([
                        transforms.RandomSizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
            ])

        else:
            transform = transforms.Compose([
                        transforms.Scale(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
            ])

        dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_root,mode), transform=transform)

    elif args.network_type == 'Caltech101':
        if mode == 'train':
            transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        else:
            transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                normalize,
            ])

        dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_root, mode), transform=transform)

    elif args.network_type == 'STL10':
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

            mode = 'train'

        else:
            transform = transforms.Compose([
                        transforms.Scale(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize])

            mode = 'test'

        dataset = torchvision.datasets.STL10(args.data_root, split=mode, transform=transform)

    elif args.network_type == 'OOD_ImageNet':
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        else:
            transform = transforms.Compose([
                        transforms.Scale(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize])

        dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'ID', mode), transform=transform)

    elif args.network_type == 'OOD_Food':
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        else:
            transform = transforms.Compose([
                        transforms.Scale(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize])

        dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'ID', mode), transform=transform)

    elif args.network_type == 'Food101':
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

            mode = 'train'

        else:
            transform = transforms.Compose([
                        transforms.Scale(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize])

        dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_root, mode), transform=transform)


    elif args.network_type == 'SD198':

        class SD198_Dataset(Dataset):
            def __init__(self, args, transform=None, mode=None, fold=0):
                self.transform = transform
                self.mode = mode
                self.args = args

                self.dataset_root = args.data_root
                self.image_paths = []
                self.gts = []
                if self.mode == 'train':
                    f = open(os.path.join(self.dataset_root, '8_2_split', 'train_{}.txt'.format(fold)))
                elif self.mode == 'val':
                    f = open(os.path.join(self.dataset_root, '8_2_split', 'val_{}.txt'.format(fold)))

                lines = f.readlines()
                for line in lines:
                    image_path, gt = line.split(' ')
                    self.image_paths.append(image_path)
                    self.gts.append(int(gt))
                f.close()

            def __getitem__(self, index):
                img_path = self.dataset_root = os.path.join(self.args.data_root, 'images',
                                                            self.image_paths[index])
                input_img = Image.open(img_path).convert('RGB')
                if self.mode == 'train':
                    input_img = self.transform(input_img)
                elif self.mode == 'val' or self.mode == 'test':
                    input_img = self.transform(input_img)
                else:
                    raise ValueError

                target = torch.Tensor([self.gts[index]]).int().item()
                return input_img, target

            def __len__(self):
                return len(self.image_paths)

        def train_transforms():
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            return transform

        def val_transforms():
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])

        if mode == 'train':
            transform = train_transforms()
        elif mode == 'val':
            transform = val_transforms()
        else:
            raise('Select proper mode')

        return SD198_Dataset(args, transform=transform, mode=mode, fold=0)

    else:
        raise('Select proper network_type')

    return dataset

def save_label_dict(dataset, args, mode='train'):
    label_to_dict = defaultdict(list)

    print('Save %s - %s dataset' %(args.network_type, mode))
    for ix in tqdm(range(len(dataset))):
        _, c = dataset.__getitem__(ix)
        label_to_dict[str(c)] += [ix]

    with open(os.path.join(args.data_root, '%s_label.pkl' %mode), 'wb') as f:
        pickle.dump(label_to_dict, f)

    return label_to_dict


class intra_dataset(Dataset):
    def __init__(self, dataset, args, mode='train'):
        super(intra_dataset, self).__init__()
        self.dataset = dataset
        self.un_normalization = un_normalization(args)
        self.normalization = normalization(args)
        self.strong_augment = strong_augment(args)

        label_file = os.path.join(args.data_root, '%s_label.pkl' %mode)
        if os.path.isfile(label_file):
            print('Loading %s label dictionary' %mode)
            with open(label_file, 'rb') as f:
                self.label_dict = pickle.load(f)
        else:
            self.label_dict = save_label_dict(dataset, args, mode)

    def __getitem__(self, index):
        img, label = self.dataset.__getitem__(index)

        # For augmentation
        aug_img = deepcopy(img)
        chance = np.random.choice([0,1], 1, replace=False)

        if chance == 0:
            aug_img = self.un_normalization(aug_img)
            aug_img = torchvision.transforms.ToPILImage()(aug_img)
            aug_img = self.strong_augment(aug_img)
            aug_img = self.normalization(aug_img)
        else:
            different_id = np.random.choice(list(set(self.label_dict[str(label)]) - set([index])), 1, replace=False)
            aug_img, _ = self.dataset.__getitem__(int(different_id))

        valid = torch.Tensor(chance).float()
        valid = valid.item()
        return img, label, aug_img, valid

    def __len__(self):
        return len(self.dataset)


def data_loader(args, dataset, mode='train'):
    if (args.network_type == 'OOD_Food') or (args.network_type == 'Food101'):
        drop_last = True
    else:
        drop_last = False
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(mode == 'train'), num_workers=args.num_workers, drop_last=drop_last)
    return dataloader




