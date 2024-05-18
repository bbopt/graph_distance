import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCC(nn.Module):
    def __init__(self, input_size, output_size, activation, nb_layers, list_units_per_layer, dropout):
        """
        Initialize a FCC.

        :param (int) input_size: size of the input
        :param (int) output_size: size of the outpout, number of classes
        :param (str) activation: the name of the activation function to use
        :param (int) nb_layers: number of hidden layers
        :param (List) list_units_per_layer: list of units number for each layers
        :param (float) dropout: dropout rate
        """
        super(FCC, self).__init__()

        # activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Tanh()

        self.input_size = input_size
        self.output_size = output_size
        self.nb_layers = nb_layers
        self.list_units_per_layer = list_units_per_layer
        self.dropout = dropout

        self.classifier = self.construct_network()

    def construct_network(self):
        """
        Construct a FCC
        """
        layers = []
        in_features = self.input_size

        for i in range(self.nb_layers):
            layers += [nn.Linear(in_features, self.list_units_per_layer[i])]
            layers += [self.activation]
            layers += [nn.Dropout(self.dropout)]
            in_features = self.list_units_per_layer[i]
        layers += [nn.Linear(in_features, self.output_size)]
        layers += [nn.LogSoftmax(dim=1)]

        self.classifier = nn.Sequential(*layers)
        return self.classifier

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x