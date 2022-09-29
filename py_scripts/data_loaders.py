
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torch 
import pandas as pd

'''from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter'''
from typing import List

#from torchvision.io import read_image

class Torch_Dataset(Dataset):
    def __init__(self, dataset_path, transform=None, target_transform=None, train=True, download=None, data_size=None, split_ind=None ):
        # download is just to be compatible with other functions. 
        # stored as a tuple of data and labels. Already processed as a torch tensor. 
        
        
        train_or_test = 'train' if train else "test"
        if split_ind is not None: 
            self.dataset_path = dataset_path+f"split_{train_or_test}_{split_ind}.pt"
        else: 
            self.dataset_path = dataset_path+f"all_data_{train_or_test}.pt"

        # loading all the images here as a large torch tensor rather than doing it one by one. 
        self.images, self.img_labels = torch.load(self.dataset_path)

        if data_size:
            print("USING A MUCH SMALLER DATASET SIZE!")
            self.images = self.images[:data_size]
            self.img_labels = self.img_labels[:data_size]
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #print(idx, self.images.shape, self.images[idx])
        
        image = self.images[idx] #[read_image(img_path)
        label = self.img_labels[idx]
        # TODO: store as int8 and then divide by 256 here. 
        
        if "MNIST" in self.dataset_path: # could do this on back end but would take much more memory
            image = image.type(torch.float)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        if image.dtype is torch.uint8:#"/ImageNet32/" in self.dataset_path or "/CIFAR10/" in self.dataset_path:
            image = image.type(torch.float)/255

        return image, label

###########

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, params, data_path="data/" ):
        super().__init__()
        self.batch_size = params.batch_size
        self.num_workers = params.num_workers
        self.data_path = data_path
        self.dataset_name = params.dataset_name
        self.train_dataset_size = params.dataset_size

        self.continual_learning = params.continual_learning
        self.curr_index = -1 # used in continual learning and will increment immediately
        if params.continual_learning: 
            self.num_data_splits=params.num_data_splits
            self.epochs_per_cl_task = params.epochs_per_cl_task

        # note these are for images in the range 0-1 not 0-255
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
            
        transforms_list = []
        # small amount of special processing here. 
        if "MNIST" in self.dataset_name: 
            # everything is already being saved as a Torch tensor as it is a custom dataset.    
            self.data_function = MNIST
            transforms_list.append(transforms.ToTensor())
            if params.normalize_n_transform_inputs:
                transforms_list.append( transforms.Normalize((0.5, ), (0.5,)) )
        elif params.dataset_name == "ImageNetFull":
            # custom processing here. 
            self.data_function = torchvision.datasets.ImageNet("../data/ImageNetFull/val/")
        else:
            self.data_function = Torch_Dataset

        if "CIFAR10" in params.dataset_name and params.normalize_n_transform_inputs:
                transforms_list+= [
                    transforms.Lambda(lambda x: x.type(torch.float)/255 ), 
                    #transforms.Normalize(cifar10_mean, cifar10_std) 
                    ]

        if params.use_convmixer_transforms and not params.normalize_n_transform_inputs:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(params.scale, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(num_ops=params.ra_n, magnitude=params.ra_m),
                transforms.ColorJitter(params.jitter, params.jitter, params.jitter),
                #transforms.ToTensor(),
                # the input is unint8. Then but it is already a tensor not a PIL so need to convert to float and decimal points
                transforms.Lambda(lambda x: x.type(torch.float)/255 ),
                #transforms.Normalize(cifar10_mean, cifar10_std),
                transforms.RandomErasing(p=params.reprob)
            ])

            self.test_transform = transforms.Compose([
                #transforms.ToTensor(),
                transforms.Lambda(lambda x: x.type(torch.float)/255 ),
                #transforms.Normalize(cifar10_mean, cifar10_std)
            ])

        else: 
            self.train_transform = transforms.Compose(transforms_list)
            self.test_transform = transforms.Compose(transforms_list)

    def setup(self, stage, train_shuffle=True, test_shuffle=True):
        
        self.train_shuffle = train_shuffle
        self.test_shuffle = test_shuffle
        
        if self.continual_learning: 
            # handling each of the different training splits. 
            self.train_datasets = [ self.data_function(
                self.data_path, train=True, transform=self.train_transform, split_ind = split_ind
            ) for split_ind in range(self.num_data_splits) ]
            self.test_datasets = [ self.data_function(
                self.data_path, train=False, transform=self.train_transform, split_ind = split_ind
            ) for split_ind in range(self.num_data_splits) ]

        elif "MNIST" in self.dataset_name:
            # todo: homogenize dataset size passing in
            self.train_data = self.data_function(
                self.data_path, train=True, download=True, transform=self.train_transform
                )
            self.test_data = self.data_function(
                self.data_path, train=False,download=True, transform=self.test_transform,
                )

        
        else: 
            self.train_data = self.data_function(
                self.data_path, train=True, download=True, transform=self.train_transform, data_size=self.train_dataset_size,
                )
            self.test_data = self.data_function(
                self.data_path, train=False,download=True, transform=self.test_transform,data_size=self.train_dataset_size,
                )

    def train_dataloader(self):
        # updating here, called before validation dataloader: 
        if self.continual_learning and self.trainer.current_epoch % self.epochs_per_cl_task == 0:

            if self.curr_index<(self.num_data_splits-1): # 5 datasets for now.
                # TODO: make this more general. 
                self.curr_index += 1
                print("SWITCHING DATASET BY INCREMENTING INDEX TO:", self.curr_index, "EPOCH IS:", self.trainer.current_epoch)

        if self.continual_learning: 
            self.train_data = self.train_datasets[self.curr_index]

        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers
        )

    def val_dataloader(self):
        if self.continual_learning:
            return [DataLoader(ds, batch_size=self.batch_size, shuffle=True , num_workers=self.num_workers) for ds in self.test_datasets[:self.curr_index+1]]
        else: 
            return DataLoader(
                self.test_data, batch_size=self.batch_size, shuffle=self.test_shuffle , num_workers=self.num_workers
            )