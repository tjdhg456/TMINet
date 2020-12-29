import torch
import torch.nn as nn
# from torch.nn import init
import functools
# from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import timm
from torchvision.models import resnet34

###############################################################################
# MODEL SPLIT
###############################################################################
def split_model(args, model):

    if 'efficientnet' in args.model_type:
        module_list = []
        for name, param in model.named_children():
            if name == 'classifier':
                break
            else:
                module_list.append(param)

        base = nn.Sequential(*module_list)

    elif 'resnet' in args.model_type:
        module_list = []
        for name, param in model.named_children():
            if name == 'fc':
                break
            else:
                module_list.append(param)

        base = nn.Sequential(*module_list)

    return base


###############################################################################
# GENERATOR
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(args, ngf=64, norm='batch', use_dropout=False):

    norm_layer = get_norm_layer(norm_type=norm)

    if args.G_name == 'resnet_9blocks':
        netG = ResnetGenerator(args, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif args.G_name == 'resnet_6blocks':
        netG = ResnetGenerator(args, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.G_name)

    netG.apply(weights_init)
    return netG

##############################################################################
# Classes
##############################################################################
# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(
            self, args, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = args.G_input_nc
        self.output_nc = args.G_output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]

        for i in range(n_blocks):
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]

        head = [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, self.output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.base = nn.Sequential(*model)
        self.head = nn.Sequential(*head)


    def forward(self, input):
        base = self.base(input)
        output = self.head(base)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()

        padAndConv = {
            'reflect': [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'replicate': [
                nn.ReplicationPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'zero': [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
        }

        try:
            blocks = padAndConv[padding_type] + [
                norm_layer(dim),
                nn.ReLU(True)
            ] + [
                nn.Dropout(0.5)
            ] if use_dropout else [] + padAndConv[padding_type] + [
                norm_layer(dim)
            ]
        except:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


###############################################################################
# DISCRIMINATOR
###############################################################################
def define_D(args, ngf=64, norm='batch', use_dropout=False):
    norm_layer = get_norm_layer(norm_type=norm)

    netD = SiameseNet(args)
    netD.apply(weights_init)

    return netD

import torchvision
class SiameseNet(nn.Module):
    def __init__(self, args):
        super(SiameseNet, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=False)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.args = args

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, args.D_out_emb)

        self.cosine = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward_one(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)

        out1, out2 = F.normalize(out1, dim=1), F.normalize(out2, dim=1)

        out = self.cosine(out1, out2) / self.args.temperature
        return out


###############################################################################
# ResNet
###############################################################################
import torch
import torch.nn as nn
from torch.nn import init
from .cbam import *
from .bam import *
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        if len(x) == 2:
            x = x[0]

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        if len(x) == 2:
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        self.att_type = att_type

        # different model config between ImageNet and CIFAR
        if network_type in ["ImageNet", "STL10", "Caltech101", "OOD_Food", "OOD_ImageNet", "Food101"]:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(64*block.expansion)
            self.bam2 = BAM(128*block.expansion)
            self.bam3 = BAM(256*block.expansion)
            self.bam4 = BAM(512*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3, self.bam4 = None, None, None, None

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type in ["ImageNet", "STL10", "Caltech101", "Food101", "OOD_Food", "OOD_ImageNet"]:
            x = self.maxpool(x)

        # Layer 1
        if self.att_type == 'CBAM':
            x1, attn1 = self.layer1(x)
        elif self.att_type == 'BAM':
            x1 = self.layer1(x)
            x1 = self.bam1(x1)
        elif 'basic' in self.att_type:
            x1 = self.layer1(x)
            attn1 = None
        else:
            raise('Select proper attention type')

        # Layer2
        if self.att_type == 'CBAM':
            x2 = self.layer2(x1)
        elif self.att_type == 'BAM':
            x2 = self.layer2(x1)
            x2 = self.bam2(x2)
        elif self.att_type == 'basic':
            x2 = self.layer2(x1)
            attn2 = None
        else:
            raise('Select proper attention type')

        # Layer3
        if self.att_type == 'CBAM':
            x3 = self.layer3(x2)
        elif self.att_type == 'BAM':
            x3 = self.layer3(x2)
            x3 = self.bam3(x3)
        elif self.att_type == 'basic':
            x3 = self.layer3(x2)
            attn3 = None
        else:
            raise('Select proper attention type')

        # Layer 4
        if self.att_type == 'CBAM':
            x4 = self.layer4(x3)
        elif self.att_type == 'BAM':
            x4 = self.layer4(x3)
            x4 = self.bam4(x4)
        elif self.att_type == 'basic':
            x4 = self.layer4(x3)
            attn4 = None
        else:
            raise('Select proper attention type')

        # Pooling
        x_ = self.avgpool(x4)
        x_ = x_.view(x_.size(0), -1)
        x_out = self.fc(x_)

        return x_out


def ResidualNet(args):
    assert args.network_type in ["ImageNet", "CIFAR10", "CIFAR100", "STL10", "Caltech101", "OOD_Food", "OOD_ImageNet", "Food101"], \
        "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert args.depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if args.depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], args.network_type, args.num_class, args.att_type)

    elif args.depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], args.network_type, args.num_class, args.att_type)

    elif args.depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], args.network_type, args.num_class, args.att_type)

    elif args.depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], args.network_type, args.num_class, args.att_type)

    return model