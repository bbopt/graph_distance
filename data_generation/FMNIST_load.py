import torch
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from constants import DATASET_DIR, VALID_RATIO, NUM_WORKERS


class DatasetTransformer(torch.utils.data.Dataset):
    """
    Transform PIL Images into pytorch tensors
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


def load_fmnist(batch_size):
    """
    Load the Fashion MNIST dataset and return a training set and a test set

    :param (int) batch_size: the size of the batch

    :return: (DataLoader, DataLoader, DataLoader) the training, validation and test set
    """

    #nb_train = int((1 - VALID_RATIO) * len(train_valid_data))
    #nb_valid = int(VALID_RATIO * len(train_valid_data))
    #train_data, valid_data = random_split(train_valid_data, [nb_train, nb_valid])

    # Set seed for reproducible results
    random.seed(1)
    torch.manual_seed(1)

    # FashionMNIST training and validation dataset
    #train_valid_data = torchvision.datasets.FashionMNIST(root=DATASET_DIR,
    #                                         train=True,
    #                                         download=True,
    #                                         transform=transforms.Compose([
    #                                             transforms.ToTensor(),
    #                                             transforms.Normalize((0.5,), (0.5,))]))
    train_valid_data = torchvision.datasets.FashionMNIST(root=DATASET_DIR, train=True, download=True,
                                                         transform=transforms.ToTensor())

    # Randomly draw 15k data, where 12k for training and 3k for validation
    train_valid_data, discarded_data = torch.utils.data.random_split(train_valid_data, [15000, 45000])
    train_data, valid_data = torch.utils.data.random_split(train_valid_data, [12000, 3000])

    # Randomly draw 2.5k for the test dataset
    #test_data = torchvision.datasets.FashionMNIST(root=DATASET_DIR,
    #                                  train=False,
    #                                  download=False,
    #                                  transform=transforms.Compose([
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize((0.5,), (0.5,))]))
    test_data = torchvision.datasets.FashionMNIST(root=DATASET_DIR, train=False, download=False,
                                                  transform=transforms.ToTensor())
    test_data, discarded_data = torch.utils.data.random_split(test_data, [2500, 7500])
    del discarded_data

    # Prepare data loaders for training, validation and testing datasets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, validation_loader, test_loader


def displayExamples(trainLoader):
    """
    Display 10 images with their label from the trainLoader

    :param (DataLoader) train_loader: training set

    :return: None
    """
    nsamples=10
    classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    imgs, labels = next(iter(trainLoader))
    fig=plt.figure(figsize=(20,5),facecolor='w')
    for i in range(nsamples):
        ax = plt.subplot(1,nsamples, i+1)
        plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
        ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


### MAIN ###
if __name__ == '__main__':

    trainLoader, validLoader, testLoader = load_fmnist(batchSize=128)
    print("The train set contains {} images, in {} batches".format(len(trainLoader.dataset), len(trainLoader)))
    print("The validation set contains {} images, in {} batches".format(len(validLoader.dataset), len(validLoader)))
    print("The test set contains {} images, in {} batches".format(len(testLoader.dataset), len(testLoader)))
    displayExamples(trainLoader)
