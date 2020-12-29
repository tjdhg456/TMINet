from torchvision.transforms import transforms
import cv2
import numpy as np

np.random.seed(0)

def strong_augment(args):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=args.input_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.ToTensor()])
    return data_transforms
