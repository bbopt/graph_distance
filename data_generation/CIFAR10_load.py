import torch
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from constants import DATASET_DIR, VALID_RATIO, NUM_WORKERS


def load_cifar10(batch_size):
    """
    Load the CIFAR-10 dataset and return a training set and a test set

    :param (int) batch_size : the batch size value

    :return: (DataLoader, DataLoader) the training and the test set
    """


    # Set seed for reproducible results
    random.seed(1)
    torch.manual_seed(1)

    # CIFAR10 training and validation dataset
    train_valid_data = torchvision.datasets.CIFAR10(root=DATASET_DIR, train=True, download=True,
                                                         transform=transforms.ToTensor())

    # Randomly draw 15k data, where 12k for training and 3k for validation
    #train_valid_data, discarded_data = torch.utils.data.random_split(train_valid_data, [15000, 35000])
    #train_data, valid_data = torch.utils.data.random_split(train_valid_data, [12000, 3000])
    #train_valid_data, discarded_data = torch.utils.data.random_split(train_valid_data, [5000, 45000])
    #train_data, valid_data = torch.utils.data.random_split(train_valid_data, [3000, 2000])
    train_valid_data, discarded_data = torch.utils.data.random_split(train_valid_data, [10000, 40000])
    train_data, valid_data = torch.utils.data.random_split(train_valid_data, [8000, 2000])

    # Randomly draw 2.5k for the test dataset
    # test_data = torchvision.datasets.FashionMNIST(root=DATASET_DIR,
    #                                  train=False,
    #                                  download=False,
    #                                  transform=transforms.Compose([
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize((0.5,), (0.5,))]))
    test_data = torchvision.datasets.CIFAR10(root=DATASET_DIR, train=False, download=False,
                                                  transform=transforms.ToTensor())
    #test_data, discarded_data = torch.utils.data.random_split(test_data, [2500, 7500])
    #test_data, discarded_data = torch.utils.data.random_split(test_data, [500, 9500])
    test_data, discarded_data = torch.utils.data.random_split(test_data, [2000, 8000])
    del discarded_data

    # Prepare data loaders for training, validation and testing datasets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, validation_loader, test_loader


def displayExamples(trainLoader):
    """
    Display some images with their label from the trainLoader

    :param (DataLoader) train_loader: training set

    :return: None
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # get some random training images
    dataiter = iter(trainLoader)
    images, labels = next(dataiter)
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(32)))
    # show images
    imshow(torchvision.utils.make_grid(images))
    

def imshow(img):
    """
    Plot the given image
    """
    img = img / 2 + 0.5 # unnormalize
    print(img.type())
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


### MAIN ###
if __name__ == '__main__':
    trainLoader, validLoader, testLoader = load_cifar10(batch_size=128)
    print("The train set contains {} images, in {} batches".format(len(trainLoader.dataset), len(trainLoader)))
    print("The test set contains {} images, in {} batches".format(len(testLoader.dataset), len(testLoader)))
    displayExamples(trainLoader)