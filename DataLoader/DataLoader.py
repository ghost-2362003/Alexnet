## Importing Libraries
import numpy
import pandas
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#setup the available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_train_valid_loader(data_dir, batch_size, augnment
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

