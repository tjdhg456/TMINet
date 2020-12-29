import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from module.network import *
from data.loader import un_normalization, normalization, dataset, intra_dataset, data_loader
from data.augmentation import strong_augment
import matplotlib.pyplot as plt
from torchvision.models import resnet34, resnet50
import warnings
import timm

warnings.filterwarnings(action='ignore')
class disguise_GAN(LightningModule):
    def __init__(self, args):
            super(disguise_GAN, self).__init__()

            if type(args) == dict:
                args = args['kwargs']['args']

            self.args = args

            # Classifier
            if self.args.model_type == 'resnet34':
                if args.pretrained:
                    self.classifier = resnet34(pretrained=True)
                    self.classifier.fc = nn.Linear(2048, self.args.num_class)
                else:
                    args.depth = 34
                    self.classifier = ResidualNet(args)

            elif self.args.model_type == 'resnet50':
                if args.pretrained:
                    self.classifier = resnet50(pretrained=True)
                    self.classifier.fc = nn.Linear(2048, self.args.num_class)
                else:
                    args.depth = 34
                    self.classifier = ResidualNet(args)

            elif 'efficientnet' in self.args.model_type:
                if args.pretrained:
                    self.classifier = timm.create_model(self.args.model_type, pretrained=True, num_classes=self.args.num_class)
                else:
                    self.classifier = timm.create_model(self.args.model_type, pretrained=False, num_classes=self.args.num_class)


            # Generator
            self.generator = define_G(args)

            # Discriminator
            self.discriminator = define_D(args)
            self.similar = nn.BCEWithLogitsLoss()

            # Util Functions
            self.un_normalization = un_normalization(args)
            self.normalization = normalization(args)
            self.num_class = args.num_class

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_prob, y):
        return nn.CrossEntropyLoss()(y_prob, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Main Torch function
        img, label, aug_img, valid = batch

        valid_G = torch.zeros_like(valid)

        generated_img = self(img)
        generated_aug = self(aug_img)

        if optimizer_idx == 0:
            g_loss = self.similar(self.discriminator(generated_img, generated_aug), valid_G)

            pred_label = self.classifier(generated_img)
            c_loss = self.adversarial_loss(pred_label, label)

            generator_loss = c_loss * self.args.lambda_c + g_loss * self.args.lambda_g
            tqdm_dict = {'g_loss': g_loss, 'c_loss': c_loss}
            log_dict = {'generator_loss': generator_loss, 'g_loss': g_loss, 'c_loss': c_loss}

            output = OrderedDict({
                'loss': generator_loss,
                'progress_bar': tqdm_dict,
                'log': log_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            d_loss = self.similar(self.discriminator(generated_img, generated_aug), valid)

            tqdm_dict = {'d_loss': d_loss}

            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_idx):
        img, label, _, _ = batch

        generated_img = self(img)

        generated_label = self.classifier(generated_img)

        val_loss = self.adversarial_loss(generated_label, label)
        val_acc_1 = self.accuracy(generated_label, label, topk=(1,))[0].cpu().detach().numpy().astype('float')

        if batch_idx == 0:
            try:
                i0 = np.transpose(self.un_normalization(img[0]).cpu().detach().numpy(), [1, 2, 0]).astype('float')
                i1 = np.transpose(self.un_normalization(img[1]).cpu().detach().numpy(), [1, 2, 0]).astype('float')
                i2 = np.transpose(self.un_normalization(img[2]).cpu().detach().numpy(), [1, 2, 0]).astype('float')

                g0 = np.array((np.transpose(generated_img[0].cpu().detach().numpy(), [1, 2, 0]) + 1) / 2).astype('float')
                g1 = np.array((np.transpose(generated_img[1].cpu().detach().numpy(), [1, 2, 0]) + 1) / 2).astype('float')
                g2 = np.array((np.transpose(generated_img[2].cpu().detach().numpy(), [1, 2, 0]) + 1) / 2).astype('float')

                plt.figure()
                fig, axs = plt.subplots(3, 2)

                axs[0, 0].imshow(i0, vmin=0, vmax=1.)
                axs[0, 0].set_title('Original Image')
                axs[0, 1].imshow(g0, vmin=0, vmax=1.)
                axs[0, 1].set_title('Generated Image')

                axs[1, 0].imshow(i1, vmin=0, vmax=1.)
                axs[1, 1].imshow(g1, vmin=0, vmax=1.)

                axs[2, 0].imshow(i2, vmin=0, vmax=1.)
                axs[2, 1].imshow(g2, vmin=0, vmax=1.)

                self.logger.experiment.log_image('val_generated_image', fig)

            except:
                print('skip!')

        return {'loss':val_loss, 'val_acc_1':val_acc_1}


    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_acc_1 = np.array([x['val_acc_1'] for x in outputs]).mean()

        log = {'val_loss' : val_loss, 'val_top1' : val_acc_1}

        return {'val_loss': val_loss, 'val_top1':val_acc_1, 'log': log}

    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.99
        opt_g = torch.optim.Adam(*[list(self.generator.parameters()) + list(self.classifier.parameters())], \
                                 lr=self.args.lr_g, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr_d, betas=(b1, b2))
        return [opt_g, opt_d], []


    @pl.data_loader
    def train_dataloader(self):
        tr_dataset = dataset(self.args, mode='train')
        tr_dataset = intra_dataset(tr_dataset, self.args, mode='train')
        tr_loader = data_loader(self.args, tr_dataset, mode='train')
        return tr_loader

    @pl.data_loader
    def val_dataloader(self):
        val_dataset = dataset(self.args, mode='val')
        val_dataset = intra_dataset(val_dataset, self.args, mode='val')
        val_loader = data_loader(self.args, val_dataset, mode='val')
        return val_loader
