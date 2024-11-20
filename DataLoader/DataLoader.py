## Importing Libraries
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_valid_loader(data_dir, batch_size, augment
                           , random_seed, valid_size = 0.1, shuffle = True):
    '''
    This the utility funtion for loading the training and validation data
    ###***********************************************************************###
    data_dir = path to the data directory
    batch_size = number of samples per batch
    augment = whether to apply data augmentation or not
    random_seed = fix seed for reproducibility
    valid_size = percentage split of the training set used for validation
    shuffle = whether to shuffle the dataset
    ###***********************************************************************###
    '''
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize
        ])

    if augment:
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        train_transform = transforms.Compose([
                transforms.Resize((227,227)),
                transforms.ToTensor(),
                normalize,
            ])
    
    # load the data
    train_dataset = datasets.CIFAR10(root = data_dir, train = True,download=True, transform = train_transform)
    valid_dataset = datasets.CIFAR10(root = data_dir, train = True,download=True, transform = valid_transform)
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)

def get_test_loader(data_dir, batch_size, shuffle = True):
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    # define transform
    transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])

    dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

    return data_loader