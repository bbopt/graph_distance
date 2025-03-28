import sys

import torch
import datetime
import random
import numpy as np
import pandas as pd
from CNN_class import CNN
from CIFAR10_load import load_cifar10
from train import train, accuracy
from constants import *
import openpyxl


def eval_perf(opt, nb_layers, units_layers, lr_exp, opt_hps, dropout, act="ReLU", batch_size=128, nb_epoch=25):
    """
    Function that constructs and train a FCC model w.r.t. to its HPs
    :param opt: string
    :param nb_layers: int
    :param units_layers: list of ints
    :param lr_exp: real (exponent) [-5, -1]
    :param opt_hps: list of real nbs
    :param dropout: [0,1]
    :param act: string
    :param batch_size: int (optional)
    :param nb_epoch: int (optional)
    :return:
    """

    # Load data
    train_loader, valid_loader, test_loader = load_cifar10(batch_size=batch_size)

    # Number of fully connected layers is fixed to 1 with 256 units
    conv_params = [(units_layers[i], 3, 1, 1, True) for i in range(nb_layers)]
    model = CNN(nb_layers, 1, conv_params, [256], dropout, act, INPUT_SIZE_CIFAR, NUM_CLASSES_CIFAR, INPUT_CHANNELS_CIFAR)

    # Decide whether CPU or GPU is used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cast model to proper hardware (CPU or GPU)
    model = model.to(device)

    # Training : HPs are 1) optimizer, 2) learning rate (lr), 3) HPs related to the optimizer
    model = train(model, train_loader, valid_loader, device, nb_epoch, opt, lr_exp, opt_hps)

    # Return performance
    return accuracy(model, test_loader, device)[0]


def random_dropout(nb_layers, units, pt_id, max_nb_total_units):
    nb_total_units = 0
    for i in range(nb_layers):
        nb_total_units += units[0]

    random.seed(pt_id + nb_total_units)
    # Decreed domain
    return random.uniform(0, nb_total_units/(2*max_nb_total_units))


if __name__ == '__main__':

    #variants = ["variant1", "variant2", "variant3", "variant4", "variant5"]  # variants simultaneously
    #variants = ["variant5"]  # for testing generation of excel files
    variants = ["variant3"]

    # size are generated 1-by-1 with a nb amongst 1 (very small),2 (small),3 (medium) and 4 (large)
    # size=1 <-> 0.25, size=2 <-> 0.5, size=3 <-> 1, size=4 <-> 1.25
    size = 4
    size_scaler = 0.25*size + 0.25
    size_scaler = size_scaler/4
    nb_epoch_setup = 25

    for variant in variants:
        data_list = []
        optimizers = []
        nb_max_layers_per_opt = {}

        if variant == "variant1" or variant == "variant2":
            optimizers = ["ASGD"]
            nb_max_layers_per_opt = {"ASGD": 3}

        elif variant == "variant3":
            optimizers = ["ADAM", "ASGD"]
            nb_max_layers_per_opt = {"ASGD": 2, "ADAM": 2}

        elif variant == "variant4":
            optimizers = ["ADAM", "ASGD"]
            nb_max_layers_per_opt = {"ASGD": 2, "ADAM": 3}

        elif variant == "variant5":
            optimizers = ["ADAM", "ASGD"]
            nb_max_layers_per_opt = {"ASGD": 2, "ADAM": 3}

        for optimizer in optimizers:

            # Loop on the subproblems, assigned by hidden layers
            for i in range(nb_max_layers_per_opt[optimizer]):

                nb_layers = i + 1
                random.seed(i + int(variant[-1]) + size)  # random seed based on nb of layers and variant number
                if variant == "variant1":
                    nb_pts_per_sub = int((20 + nb_layers * 20)*size_scaler)

                elif variant == "variant2" or variant == "variant3" or variant == "variant4":
                    nb_pts_per_sub = int((80 + nb_layers * 20)*size_scaler)

                elif variant == "variant5":
                    nb_pts_per_sub = int((100 + nb_layers * 20)*size_scaler)

                # For testing the generation of excel files
                #nb_pts_per_sub = 2

                # Random learning rates exp values
                lr_exp_values = [random.uniform(-5, -1) for p in range(nb_pts_per_sub)]

                # Random numbers of units
                units_values = np.zeros(shape=(nb_layers, nb_pts_per_sub))
                for k in range(nb_layers):
                    # Old for FCC
                    #units_values[k] = [random.randint(5, 25) for l in range(nb_pts_per_sub)]
                    if k == 0:
                        units_values[k] = [random.randint(32, 64) for l in
                                           range(nb_pts_per_sub)]  # First layer: 32-64 filters
                    elif k == 1:
                        units_values[k] = [random.randint(64, 128) for l in
                                           range(nb_pts_per_sub)]  # Second layer: 64-128 filters
                    elif k == 2:
                        units_values[k] = [random.randint(128, 256) for l in
                                           range(nb_pts_per_sub)]  # Third layer: 128-256 filters

                if optimizer == "ASGD":

                    # ASGD HPs are fixed
                    if variant == "variant1":
                        # Random weight decay term (lambd) exp values
                        lambd_exp_values = [-4]*nb_pts_per_sub
                        # Random power update (alpha)
                        alpha_values = [0.75]*nb_pts_per_sub
                        # Random starting avg (t0) exp values
                        t0_exp_values = [6]*nb_pts_per_sub

                    # ASGD HPs are randomized for variant 2,3,4,5
                    else:
                        # Random weight decay term (lambd) exp values
                        lambd_exp_values = [random.uniform(-5, 0) for o in range(nb_pts_per_sub)]
                        # Random power update (alpha)
                        alpha_values = [random.uniform(0, 1) for o in range(nb_pts_per_sub)]
                        # Random starting avg (t0) exp values
                        t0_exp_values = [random.uniform(3, 9) for o in range(nb_pts_per_sub)]

                    # Evaluate random points : constructs units
                    for j in range(nb_pts_per_sub):
                        # Units for constructing and evaluation performance with eval_perf()
                        if nb_layers == 1:
                            units = [int(units_values[0][j])]

                            if variant == "variant5":
                                dropout = random_dropout(nb_layers, units, j, max_nb_total_units=448)
                            else:
                                dropout = 0

                            performance = eval_perf(optimizer, nb_layers, units, lr_exp_values[j],
                                                    [lambd_exp_values[j], alpha_values[j], t0_exp_values[j]], dropout,
                                                    act="ReLU", batch_size=128, nb_epoch=nb_epoch_setup)
                            data_list.append([optimizer, nb_layers, units[0], "EXC", "EXC", lr_exp_values[j],
                                              lambd_exp_values[j], alpha_values[j], t0_exp_values[j],
                                              "EXC", "EXC", "EXC", dropout,
                                              performance])

                        elif nb_layers == 2:
                            units = [int(units_values[0][j]), int(units_values[1][j])]

                            if variant == "variant5":
                                dropout = random_dropout(nb_layers, units, j, max_nb_total_units=448)
                            else:
                                dropout = 0

                            performance = eval_perf(optimizer, nb_layers, units, lr_exp_values[j],
                                                    [lambd_exp_values[j], alpha_values[j], t0_exp_values[j]], dropout,
                                                    act="ReLU", batch_size=128, nb_epoch=nb_epoch_setup)
                            data_list.append([optimizer, nb_layers, units[0], units[1], "EXC", lr_exp_values[j],
                                              lambd_exp_values[j], alpha_values[j], t0_exp_values[j],
                                              "EXC", "EXC", "EXC", dropout,
                                              performance])

                        elif nb_layers == 3:
                            units = [int(units_values[0][j]), int(units_values[1][j]), int(units_values[2][j])]

                            if variant == "variant5":
                                dropout = random_dropout(nb_layers, units, j, max_nb_total_units=448)
                            else:
                                dropout = 0

                            performance = eval_perf(optimizer, nb_layers, units, lr_exp_values[j],
                                                    [lambd_exp_values[j], alpha_values[j], t0_exp_values[j]], dropout,
                                                    act="ReLU", batch_size=128, nb_epoch=nb_epoch_setup)
                            data_list.append([optimizer, nb_layers, units[0], units[1], units[2], lr_exp_values[j],
                                              lambd_exp_values[j], alpha_values[j], t0_exp_values[j],
                                              "EXC", "EXC", "EXC", dropout,
                                              performance])

                # ADAM is used for variant 3,4,5, and its HPs are always randomized
                elif optimizer == "ADAM":

                    # Random beta1_value
                    beta1_values = [random.uniform(0.5, 0.9) for o in range(nb_pts_per_sub)]
                    # Random beta2_value
                    beta2_values = [random.uniform(0.5, 0.9999) for o in range(nb_pts_per_sub)]
                    # Random epsilon
                    eps_exp_values = [random.uniform(-10, -7) for o in range(nb_pts_per_sub)]

                    # Evaluate random points : constructs units
                    for j in range(nb_pts_per_sub):
                        # Units for constructing and evaluation performance with eval_perf()
                        if nb_layers == 1:
                            units = [int(units_values[0][j])]

                            if variant == "variant5":
                                dropout = random_dropout(nb_layers, units, j, max_nb_total_units=448)
                            else:
                                dropout = 0

                            performance = eval_perf(optimizer, nb_layers, units, lr_exp_values[j],
                                                    [beta1_values[j], beta2_values[j], eps_exp_values[j]], dropout,
                                                    act="ReLU", batch_size=128, nb_epoch=nb_epoch_setup)
                            data_list.append([optimizer, nb_layers, units[0], "EXC", "EXC", lr_exp_values[j],
                                              "EXC", "EXC", "EXC",
                                              beta1_values[j], beta2_values[j], eps_exp_values[j], dropout,
                                              performance])

                        elif nb_layers == 2:
                            units = [int(units_values[0][j]), int(units_values[1][j])]

                            if variant == "variant5":
                                dropout = random_dropout(nb_layers, units, j, max_nb_total_units=448)
                            else:
                                dropout = 0

                            performance = eval_perf(optimizer, nb_layers, units, lr_exp_values[j],
                                                    [beta1_values[j], beta2_values[j], eps_exp_values[j]], dropout,
                                                    act="ReLU", batch_size=128, nb_epoch=nb_epoch_setup)
                            data_list.append([optimizer, nb_layers, units[0], units[1], "EXC", lr_exp_values[j],
                                              "EXC", "EXC", "EXC",
                                              beta1_values[j], beta2_values[j], eps_exp_values[j], dropout,
                                              performance])

                        elif nb_layers == 3:
                            units = [int(units_values[0][j]), int(units_values[1][j]), int(units_values[2][j])]

                            if variant == "variant5":
                                dropout = random_dropout(nb_layers, units, j, max_nb_total_units=448)
                            else:
                                dropout = 0

                            performance = eval_perf(optimizer, nb_layers, units, lr_exp_values[j],
                                                    [beta1_values[j], beta2_values[j], eps_exp_values[j]], dropout,
                                                    act="ReLU", batch_size=128, nb_epoch=nb_epoch_setup)
                            data_list.append([optimizer, nb_layers, units[0], units[1], units[2], lr_exp_values[j],
                                              "EXC", "EXC", "EXC",
                                              beta1_values[j], beta2_values[j], eps_exp_values[j], dropout,
                                              performance])

        # variant 1 : ['l', 'u1', 'u2', 'u3', 'lr', 'accuracy']
        if variant == "variant1":
            idx = [1, 2, 3, 4, 5, 13]
            data_list_int1 = [list(np.array(row,  dtype=object)[idx]) for row in data_list]

            # Cast list into a dataframe
            df = pd.DataFrame(data_list_int1, columns=['l', 'u1', 'u2', 'u3', 'lr', 'accuracy'])
            # Separate df into subproblems and log to Excel
            df_sub1 = df[df['l'] == 1]
            df_sub2 = df[df['l'] == 2]
            df_sub3 = df[df['l'] == 3]

            with pd.ExcelWriter("data_variant1_size" + str(size) + ".xlsx") as writer:
                df_sub1.to_excel(writer, sheet_name='sub1', index=False)
                df_sub2.to_excel(writer, sheet_name='sub2', index=False)
                df_sub3.to_excel(writer, sheet_name='sub3', index=False)

        # variant 2 : ['l', 'u1', 'u2', 'u3', 'lr', 'lambda', 'alpha', 't0', 'accuracy']
        elif variant == "variant2":
            idx = [1, 2, 3, 4, 5, 6, 7, 8, 13]
            data_list_int2 = [list(np.array(row, dtype=object)[idx]) for row in data_list]

            # Cast list into a dataframe
            df = pd.DataFrame(data_list_int2, columns=['l', 'u1', 'u2', 'u3', 'lr', 'lambda', 'alpha', 't0', 'accuracy'])
            # Separate df into subproblems and log to Excel
            df_sub1 = df[df['l'] == 1]
            df_sub2 = df[df['l'] == 2]
            df_sub3 = df[df['l'] == 3]

            with pd.ExcelWriter("data_variant2_size" + str(size) + ".xlsx") as writer:
                df_sub1.to_excel(writer, sheet_name='sub1', index=False)
                df_sub2.to_excel(writer, sheet_name='sub2', index=False)
                df_sub3.to_excel(writer, sheet_name='sub3', index=False)

        # variant 3 : ['o', 'l', 'u1', 'u2', 'lr', 'lambda', 'alpha', 't0', 'beta1', 'beta2', 'eps', 'accuracy']
        elif variant == "variant3":
            idx = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13]
            data_list_int3 = [list(np.array(row, dtype=object)[idx]) for row in data_list]

            # Cast list into a dataframe
            df = pd.DataFrame(data_list_int3, columns=['o', 'l', 'u1', 'u2', 'lr', 'lambda', 'alpha', 't0', 'beta1', 'beta2', 'eps', 'accuracy'])
            # Separate df into subproblems and log to Excel
            df_ASGD_l1 = df[(df['o'] == "ASGD") & (df['l'] == 1)]
            df_ASGD_l2 = df[(df['o'] == "ASGD") & (df['l'] == 2)]
            df_ADAM_l1 = df[(df['o'] == "ADAM") & (df['l'] == 1)]
            df_ADAM_l2 = df[(df['o'] == "ADAM") & (df['l'] == 2)]

            with pd.ExcelWriter("data_variant3_size" + str(size) + ".xlsx") as writer:
                df_ASGD_l1.to_excel(writer, sheet_name='sub1_ASGD', index=False)
                df_ASGD_l2.to_excel(writer, sheet_name='sub2_ASGD', index=False)
                df_ADAM_l1.to_excel(writer, sheet_name='sub1_ADAM', index=False)
                df_ADAM_l2.to_excel(writer, sheet_name='sub2_ADAM', index=False)

        # variant 4 : ['o', 'l', 'u1', 'u2', 'u3', 'lr', 'lambda', 'alpha', 't0', 'beta1', 'beta2', 'eps', 'accuracy']
        elif variant == "variant4":
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
            data_list_int4 = [list(np.array(row, dtype=object)[idx]) for row in data_list]

            # Cast list into a dataframe
            df = pd.DataFrame(data_list_int4, columns=['o', 'l', 'u1', 'u2', 'u3', 'lr', 'lambda', 'alpha', 't0', 'beta1', 'beta2', 'eps', 'accuracy'])
            # Separate df into subproblems and log to Excel
            df_ASGD_l1 = df[(df['o'] == "ASGD") & (df['l'] == 1)]
            df_ASGD_l2 = df[(df['o'] == "ASGD") & (df['l'] == 2)]
            df_ADAM_l1 = df[(df['o'] == "ADAM") & (df['l'] == 1)]
            df_ADAM_l2 = df[(df['o'] == "ADAM") & (df['l'] == 2)]
            df_ADAM_l3 = df[(df['o'] == "ADAM") & (df['l'] == 3)]

            with pd.ExcelWriter("data_variant4_size" + str(size) + ".xlsx") as writer:
                df_ASGD_l1.to_excel(writer, sheet_name='sub1_ASGD', index=False)
                df_ASGD_l2.to_excel(writer, sheet_name='sub2_ASGD', index=False)
                df_ADAM_l1.to_excel(writer, sheet_name='sub1_ADAM', index=False)
                df_ADAM_l2.to_excel(writer, sheet_name='sub2_ADAM', index=False)
                df_ADAM_l3.to_excel(writer, sheet_name='sub3_ADAM', index=False)

        # variant 5 : ['o', 'l', 'u1', 'u2', 'u3', 'lr', 'lambda', 'alpha', 't0', 'beta1', 'beta2', 'eps', 'p', 'accuracy']
        elif variant == "variant5":

            # Cast list
            df = pd.DataFrame(data_list, columns=['o', 'l', 'u1', 'u2', 'u3', 'lr', 'lambda', 'alpha', 't0', 'beta1', 'beta2', 'eps', 'p', 'accuracy'])
            # Separate df into subproblems and log to Excel
            df_ASGD_l1 = df[(df['o'] == "ASGD") & (df['l'] == 1)]
            df_ASGD_l2 = df[(df['o'] == "ASGD") & (df['l'] == 2)]
            df_ADAM_l1 = df[(df['o'] == "ADAM") & (df['l'] == 1)]
            df_ADAM_l2 = df[(df['o'] == "ADAM") & (df['l'] == 2)]
            df_ADAM_l3 = df[(df['o'] == "ADAM") & (df['l'] == 3)]

            with pd.ExcelWriter("data_variant5_size" + str(size) + ".xlsx") as writer:
            #with pd.ExcelWriter("variant5_data_test.xlsx") as writer:
                df_ASGD_l1.to_excel(writer, sheet_name='sub1_ASGD', index=False)
                df_ASGD_l2.to_excel(writer, sheet_name='sub2_ASGD', index=False)
                df_ADAM_l1.to_excel(writer, sheet_name='sub1_ADAM', index=False)
                df_ADAM_l2.to_excel(writer, sheet_name='sub2_ADAM', index=False)
                df_ADAM_l3.to_excel(writer, sheet_name='sub3_ADAM', index=False)

