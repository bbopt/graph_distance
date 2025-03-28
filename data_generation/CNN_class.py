import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    def __init__(self, num_conv_layers, num_full_layers, list_param_conv_layers, list_param_full_layers, dropout_rate,
                 activation, initial_image_size, total_classes, number_input_channels):
        """
            Initialize a CNN.
            We suppose that the initial image size is 32.
        """

        super(CNN, self).__init__()

        # activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Tanh()

        self.dropout = dropout_rate

        self.init_im_size = initial_image_size
        self.total_classes = total_classes
        self.in_size_first_full_layer = -1
        self.num_conv_layers = num_conv_layers
        self.num_full_layers = num_full_layers
        self.param_conv = list_param_conv_layers
        self.param_full = list_param_full_layers
        self.number_input_channels = number_input_channels

        assert num_conv_layers == len(list_param_conv_layers), 'len(list_param_conv_layers) != num_conv_layers'
        for i in range(num_conv_layers):
            assert len(list_param_conv_layers[i]) == 5, 'Problem with number of parameters of the convolutional layer ' \
                                                        'num %r' % i
            assert num_full_layers == len(list_param_full_layers), 'num_full_layers != len(list_param_full_layers)'

        self.features, self.classifier = self.construct_network()

    def construct_network(self):
        """
        Construct a CNN.

        list_param_conv_layers = [(n_out_channel, kernel_size, stride, padding, do_pool)]
        list_param_full_layers = [n_output_layer1,...]
        """
        layers = []
        n_in_channel = self.number_input_channels
        # construct the convolutional layers
        for i in range(self.num_conv_layers):
            params_i = self.param_conv[i]
            n_out_channel = params_i[0]
            kernel_size = params_i[1]
            stride = params_i[2]
            padding = params_i[3]

            layer = nn.Conv2d(n_in_channel, n_out_channel, kernel_size, stride=stride,
                              padding=padding)
            # nn.init.xavier_uniform_(layer.weight)
            layers += [layer]
            layers += [self.activation]
            layers += [nn.Dropout2d(self.dropout)]
            layers += [nn.BatchNorm2d(n_out_channel)]

            pooling = params_i[-1]
            if pooling:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            n_in_channel = n_out_channel

        if self.num_conv_layers > 0:
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        self.features = nn.Sequential(*layers)
        layers = []
        # print('Done with conv layer')

        # construct the full layers
        if not self.param_conv:
            size_input = self.number_input_channels * (self.init_im_size ** 2)
        else:
            size_input = (self.get_input_size_first_lin_layer() ** 2) * self.param_conv[-1][0]

        self.in_size_first_full_layer = size_input

        for i in range(self.num_full_layers):
            layer = nn.Linear(size_input, self.param_full[i])
            nn.init.xavier_uniform_(layer.weight)
            layers += [layer]
            size_input = self.param_full[i]
            layers += [self.activation]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.BatchNorm1d(self.param_full[i])]

        layer = nn.Linear(size_input, self.total_classes)
        # nn.init.xavier_uniform_(layer.weight)
        layers += [layer]
        layers += [nn.BatchNorm1d(self.total_classes)]
        # layers.append(nn.LogSoftmax(dim=1))
        self.classifier = nn.Sequential(*layers)
        return self.features, self.classifier

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(-1, self.in_size_first_full_layer)
        x = self.classifier(x)
        return x

    def get_input_size_first_lin_layer(self):
        current_size = self.init_im_size
        for i in range(self.num_conv_layers):
            temp = (current_size - self.param_conv[i][1] + 2 * self.param_conv[i][3]) / self.param_conv[i][2] + 1
            current_size = np.floor(temp)
            # Pooling
            pooling = self.param_conv[i][-1]
            if pooling:  # If poolling is True
                if current_size > 1:
                    current_size = np.floor(current_size / 2)
            current_size = int(current_size)

        return current_size