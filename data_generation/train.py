import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import time
from constants import PRINT, EARLY_STOP

# For testing
from constants import *
from CIFAR10_load import *
from CNN_class import CNN

def trainOneEpoch(model, trainLoader, optimizer, device):
    """
    Train the model on the training set for a single epoch

    :param (nn.Sequential) model: a nn network
    :param (DataLoader) trainLoader: training set
    :param (torch.optim) optimizer: the optimizer to use
    :param (torch.device) device: cuda or cpu

    :return: (nn.Sequential) the trained model
    """
    model.train()
    fLoss = torch.nn.CrossEntropyLoss()
    trainingLoss = 0
    for batch, (X, y) in enumerate(trainLoader):
        X, y = X.to(device), y.to(device)
        # Forward pass
        pred = model(X)
        loss = fLoss(pred, y)
        trainingLoss += loss.item()
        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def train(model, trainLoader, validLoader, device, nbEpochs, opt, lr_exp, optimParams=None):
    """
    Train on all epochs the model on the training set and select the best trained model
    based on accuracy on the validation set

    :param (nn.Sequential) model: a nn network
    :param (DataLoader) trainLoader: training set
    :param (DataLoader) validLoader: validation set
    :param (torch.device) device: cuda or cpu
    :param (int) nbEpochs: number of epochs
    :param (str) opt: the name of the optimizer to use
    :param (int) lr_exp: the value of the learning rate exponent (learning rate = 10 ** lr_exp)
    :param (list) optimParams: list of param for the optimizer

    :return: (nn.Sequential) the trained model
    """

    # Learning rate
    learningRate = 10**lr_exp

    if optimParams == None:
        optimizer = getattr(optim, opt)(model.parameters(), lr=learningRate)
    else:
        if opt == "ADAM":
            weight_decay_hp = 10 ** optimParams[2]
            optimizer = optim.Adam(model.parameters(), lr=learningRate, betas=(optimParams[0], optimParams[1]),
                                   weight_decay=weight_decay_hp)
        elif opt == "ASGD":
            ASGD_lambd = 10 ** optimParams[0]   # Decay term (lambd)
            ASGD_t0 = 10 ** optimParams[2]  # Starting average (t0)
            optimizer = optim.ASGD(model.parameters(), lr=learningRate, lambd=ASGD_lambd, alpha=optimParams[1],
                                   t0=ASGD_t0)
        elif opt == "Adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=learningRate, lr_decay=optimParams[0],
                                      initial_accumulator_value=optimParams[1], eps=optimParams[2])
        elif opt == "RMSprop":
            weight_decay_hp = 10 ** optimParams[2]
            optimizer = optim.RMSprop(model.parameters(), lr=learningRate, momentum=optimParams[0],
                                      alpha=optimParams[1], weight_decay=weight_decay_hp)

        elif opt == "SGD":
            weight_decay_hp = 10 ** optimParams[2]
            optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=optimParams[0],
                                      dampening=optimParams[1], weight_decay=weight_decay_hp)

    sched = scheduler.ReduceLROnPlateau(optimizer, 'min')

    bestAcc = 0

    listTrainAcc = []
    listTrainLoss = []
    listValLoss = []
    listValAcc = []

    stop = False  # For early stopping
    epoch = 1

    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model)

    t0 = time.time()
    executionTime = 0

    while (not stop) and (epoch <= nbEpochs):  # loop over the dataset multiple times
        if PRINT:
            print("> Epoch {}".format(epoch))

            # Train
        model = trainOneEpoch(model, trainLoader, optimizer, device)

        # Accuracy on training set
        trainAcc, trainLoss = accuracy(model, trainLoader, device)
        listTrainLoss.append(trainLoss)
        listTrainAcc.append(trainAcc)

        # Accuracy on validation set
        valAcc, valLoss = accuracy(model, validLoader, device)
        listValLoss.append(valLoss)
        listValAcc.append(valAcc)

        executionTime = time.time() - t0

        if PRINT:
            print("\tExecution time: {:.2f}s, Train accuracy: {:.2f}%, Val accuracy: {:.2f}%".format(executionTime,
                                                                                                     trainAcc, valAcc))

        if valAcc > bestAcc:
            bestAcc = valAcc
            bestModel = model

        # Early stopping
        if EARLY_STOP:
            if (epoch >= nbEpochs / 5) and (bestAcc < 20):
                stop = True
                if PRINT:
                    print("\tEarly stopped")

        # Scheduler
        sched.step(valLoss)

        epoch += 1

    return bestModel


def accuracy(model, loader, device):
    """
    Return the accuracy of the model on training, validation or test set

    :param (nn.Sequential) model: a nn network
    :param (DataLoader) loader: the given dataset
    :param (torch.device) device: cuda or cpu

    :return: (float, float) the accuracy of the model on the set in % and the mean loss
    """
    # Disable graph and gradient computations for speed and memory efficiency
    with torch.no_grad():
        model.eval()
        loss, nbCorrect = 0.0, 0
        size = len(loader.dataset)
        fLoss = torch.nn.CrossEntropyLoss()
        for (X, y) in loader:
            X, y = X.to(device), y.to(device)
            # Forward pass
            pred = model(X)
            # the class with the highest value is what we choose as prediction
            _, predicted = torch.max(pred.data, 1)
            # Losses and nb of correct predictions
            loss += fLoss(pred, y).item()
            nbCorrect += (predicted == y).sum().item()
    return float(nbCorrect / size) * 100, loss / size  # accuracy and mean loss



if __name__ == "__main__":

    # Small script for testing if the training and test properly works
    trainLoader, validLoader, testLoader = load_cifar10(batch_size=128)
    inputSize, inputChannel, numClasses = INPUT_SIZE_CIFAR, INPUT_CHANNELS_CIFAR, NUM_CLASSES_CIFAR

    a = [
        (64, 3, 1, 1, True),  # Conv1: 32 filters, 3x3 kernel, stride 1, padding 1, MaxPool 2x2
        (128, 3, 1, 1, True),  # Conv2: 64 filters, 3x3 kernel, stride 1, padding 1, MaxPool 2x2
        (256, 3, 1, 1, True)  # Conv3: 128 filters, 3x3 kernel, stride 1, padding 1, MaxPool 2x2
    ]
    b = [256]
    model = CNN(3, 1, a, b, 0.1, "ReLU", inputSize, numClasses, inputChannel)

    # Decide whether CPU or GPU is used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cast model to proper hardware (CPU or GPU)
    model = model.to(device)

    model = train(model, trainLoader, validLoader, device, 25, "ASGD", -2, [-4, 0.75, 6])

    print(accuracy(model, testLoader, device)[0])