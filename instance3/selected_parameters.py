import sys
import datetime
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.import_data import load_data_optimizers, prep_naive_data
from utils.draw import draw_subpbs_candidates_instance3
from problems_instance3 import instance3_bb_wrapper

if __name__ == '__main__':

    load_RMSE_dataset = True  # True if RMSEs have already been computed: useful for formatting final figure
    nb_rand_trials = 30

    # Model: choose between "IDW" and "KNN"
    model = "IDW"

    # Setup
    optimizers = ["ASGD", "ADAM"]
    nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 2}
    nb_pts = [100, 120, 140]  # nb_pts for subproblems {1,2,3}, not matter the optimizer : no need to specify the opt
    nb_test_pts = [25, 30, 35]
    nb_valid_pts = [25, 30, 35]
    nb_max_var = 11  # 12 - 1, with u3 removed
    var_inc_sub = {("ASGD", 1): [2, 4, 5, 6, 7], ("ASGD", 2): [2, 3, 4, 5, 6, 7],
                   ("ADAM", 1): [2, 4, 8, 9, 10], ("ADAM", 2): [2, 3, 4, 8, 9, 10]}
    nb_var_sub = {("ASGD", 1): 5, ("ASGD", 2): 6, ("ADAM", 1): 5, ("ADAM", 2): 6}

    # Best parameters
    x_best_naive_IDW = [2.61066e-05, 2.71553, 0.000484097, 0.000229857, 9.24545e-08, 0, 0, 2.65616, 1.24037e-07, 3.06573e-07, 0, 92.154, 1.39001, 1.17084e-07, 5.74629e-05, 5.95153e-08, 0, 1.38136e-07, 0.47074, 9.74467e-07, 1.26138e-06, 7.09046e-08]
    x_best_graph_IDW = [30.1889, 2.50003, 999.999, 242.704, 998.835, 1000, 999.999, 884.139, 0, 18.3991, 0, 0, 999.998, 1.33758e-07, 0.202883, 0.00711024, 0.000318189, 23.8456, 7.33904e-06]
    x_best_naive_KNN = [0.00217033, 400.252, 1.68182e-06, 3.90824e-05, 9.21491e-06, 7.7606e-06, 7.01465e-07, 0.624878, 0.000917052, 4.38702e-05, 8.43823e-05, 903.918, 10.3381, 103.512, 93.5844, 196.627, 0.000197105, 0.000615527, 0.400975, 0.00732583, 0.00401179, 7.70802e-06, 8, 1, 7, 6]
    x_best_graph_KNN = [900.009, 202.222, 79.6939, 43.1, 0.74478, 0.30695, 570.879, 1.5827, 999.98, 690.908, 2.25, 0.4295, 922.115, 907.74, 897.772, 5.383, 0.0082, 400.771, 24.01, 2]

    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        load_data_optimizers('data_instance3.xlsx', optimizers, nb_sub_per_optimizer, nb_max_var, nb_pts, nb_test_pts,
                             nb_valid_pts)

    # Prepare data for naive approach
    X_train_naive, y_train_naive, X_valid_naive, y_valid_naive, X_test_naive, y_test_naive = \
        prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, 'instance3')

    if load_RMSE_dataset:
        # Load test
        if model == "IDW":
            (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std) = \
                np.loadtxt("log_IDW_both_selected_parameters.txt")
        elif model == "KNN":
            (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std) = \
                np.loadtxt("log_KNN_both_selected_parameters.txt")
    else:
        RMSE_naive = np.zeros((220-5+1, nb_rand_trials))
        RMSE_graph = np.zeros((220-5+1, nb_rand_trials))
        for j in range(nb_rand_trials):
            random.seed(j)

            # Training data for naive : in the loop to allows reset for each seed-trial (we use pop on these lists)

            idx_train_ASGD_l1 = (np.where((X_train[:, 0] == "ASGD") & (X_train[:, 1] == 1))[0]).tolist()
            idx_train_ASGD_l2 = (np.where((X_train[:, 0] == "ASGD") & (X_train[:, 1] == 2))[0]).tolist()
            idx_train_ADAM_l1 = (np.where((X_train[:, 0] == "ADAM") & (X_train[:, 1] == 1))[0]).tolist()
            idx_train_ADAM_l2 = (np.where((X_train[:, 0] == "ADAM") & (X_train[:, 1] == 2))[0]).tolist()
            idx_train_subs = {("ASGD", 1): idx_train_ASGD_l1, ("ASGD", 2): idx_train_ASGD_l2,
                              ("ADAM", 1): idx_train_ADAM_l1, ("ADAM", 2): idx_train_ADAM_l2}

            # Add at least one point of each sub prob in the initial partial dataset : where()[0] returns the list
            idx_train_subs_partial = {("ASGD", 1): [idx_train_ASGD_l1.pop(random.randrange(len(idx_train_ASGD_l1)))],
                                      ("ASGD", 2): [idx_train_ASGD_l2.pop(random.randrange(len(idx_train_ASGD_l2)))],
                                      ("ADAM", 1): [idx_train_ADAM_l1.pop(random.randrange(len(idx_train_ADAM_l1)))],
                                      ("ADAM", 2): [idx_train_ADAM_l2.pop(random.randrange(len(idx_train_ADAM_l2)))]}

            draw = [1, 2, 3, 4]
            i = 0
            while len(draw) != 0:

                # Create partial data set for naive approach
                X_train_naive_partial = {("ASGD", 1): X_train[np.ix_(idx_train_subs_partial["ASGD", 1], var_inc_sub["ASGD", 1])],
                                         ("ASGD", 2): X_train[np.ix_(idx_train_subs_partial["ASGD", 2], var_inc_sub["ASGD", 2])],
                                         ("ADAM", 1): X_train[np.ix_(idx_train_subs_partial["ADAM", 1], var_inc_sub["ADAM", 1])],
                                         ("ADAM", 2): X_train[np.ix_(idx_train_subs_partial["ADAM", 2], var_inc_sub["ADAM", 2])]}
                y_train_naive_partial = {("ASGD", 1): y_train[idx_train_subs_partial["ASGD", 1]],
                                         ("ASGD", 2): y_train[idx_train_subs_partial["ASGD", 2]],
                                         ("ADAM", 1): y_train[idx_train_subs_partial["ADAM", 1]],
                                         ("ADAM", 2): y_train[idx_train_subs_partial["ADAM", 2]]}

                if model == "IDW":
                    # Create parametrized bb for naive approach
                    IDW_naive_bb = instance3_bb_wrapper(X_train_naive_partial, y_train_naive_partial,
                                                        X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "IDW")
                    RMSE_naive[i, j] = IDW_naive_bb(*x_best_naive_IDW)

                    # Create parametrized bb for graph approach
                    idx_train_partial_graph = idx_train_subs_partial["ASGD", 1] + idx_train_subs_partial["ASGD", 2] +\
                                              idx_train_subs_partial["ADAM", 1] + idx_train_subs_partial["ADAM", 2]
                    X_train_graph_partial = X_train[idx_train_partial_graph, :]
                    y_train_graph_partial = y_train[idx_train_partial_graph]
                    IDW_graph_bb = instance3_bb_wrapper(X_train_graph_partial, y_train_graph_partial,
                                                        X_test, y_test, nb_max_var, nb_var_sub, "graph", "IDW")
                    RMSE_graph[i, j] = IDW_graph_bb(*x_best_graph_IDW)

                if model == "KNN":
                    # Create parametrized bb for naive approach
                    KNN_naive_bb = instance3_bb_wrapper(X_train_naive_partial, y_train_naive_partial,
                                                        X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "KNN")
                    RMSE_naive[i, j] = KNN_naive_bb(*x_best_naive_KNN)

                    # Create parametrized bb for graph approach
                    idx_train_partial_graph = idx_train_subs_partial["ASGD", 1] + idx_train_subs_partial["ASGD", 2] + \
                                            idx_train_subs_partial["ADAM", 1] + idx_train_subs_partial["ADAM", 2]
                    X_train_graph_partial = X_train[idx_train_partial_graph, :]
                    y_train_graph_partial = y_train[idx_train_partial_graph]
                    KNN_graph_bb = instance3_bb_wrapper(X_train_graph_partial, y_train_graph_partial,
                                                        X_test, y_test, nb_max_var, nb_var_sub, "graph", "KNN")
                    RMSE_graph[i, j] = KNN_graph_bb(*x_best_graph_KNN)

                # Determine next draw is amongst which subpbs and check if while-loop is done
                draw = draw_subpbs_candidates_instance3(idx_train_subs["ASGD", 1], idx_train_subs["ASGD", 2],
                                                        idx_train_subs["ADAM", 1], idx_train_subs["ADAM", 2])

                if len(draw) != 0:
                    # 0=sub1, 1=sub1, 2=sub3
                    subpb_draw = random.choice(draw)
                    # Update
                    i = i + 1

                    if subpb_draw == 1:
                        next_index = idx_train_subs["ASGD", 1].pop()
                        idx_train_subs_partial["ASGD", 1].append(next_index)
                    elif subpb_draw == 2:
                        next_index = idx_train_subs["ASGD", 2].pop()
                        idx_train_subs_partial["ASGD", 2].append(next_index)
                    elif subpb_draw == 3:
                        next_index = idx_train_subs["ADAM", 1].pop()
                        idx_train_subs_partial["ADAM", 1].append(next_index)
                    elif subpb_draw == 4:
                        next_index = idx_train_subs["ADAM", 2].pop()
                        idx_train_subs_partial["ADAM", 2].append(next_index)

        # Output results in log file
        RMSE_graph_mean = RMSE_graph.mean(axis=1)
        RMSE_graph_std = RMSE_graph.std(axis=1)
        RMSE_naive_mean = RMSE_naive.mean(axis=1)
        RMSE_naive_std = RMSE_naive.std(axis=1)

        # Save test in case figures have to be modified
        if model == "IDW":
            np.savetxt("log_IDW_both_selected_parameters.txt",
                       (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std))

        elif model == "KNN":
            np.savetxt("log_KNN_both_selected_parameters.txt",
                       (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std))

    # Setup
    fig = plt.figure(dpi=100, tight_layout=True)
    plt.rc('axes', labelsize=28)
    plt.rc('legend', fontsize=28)
    plt.rc('xtick', labelsize=26)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=26)  # fontsize of the tick label
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    nb_plot = 220-5
    plt.plot(range(nb_plot), RMSE_naive_mean[0:nb_plot], linestyle='-', marker='s', markersize=2, label='Sub')
    plt.fill_between(range(nb_plot), RMSE_naive_mean[0:nb_plot] - RMSE_naive_std[0:nb_plot],
                     RMSE_naive_mean[0:nb_plot] + RMSE_naive_std[0:nb_plot], color="tab:blue", alpha=0.30)
    plt.plot(range(nb_plot), RMSE_graph_mean[0:nb_plot], linestyle='-', marker='^', markersize=2, label='Graph')
    plt.fill_between(range(nb_plot), RMSE_graph_mean[0:nb_plot] - RMSE_graph_std[0:nb_plot],
                     RMSE_graph_mean[0:nb_plot] + RMSE_graph_std[0:nb_plot], color="orange", alpha=0.30)
    plt.xlabel('Training dataset size')
    plt.ylabel('RMSE test')
    plt.ylim((7.5, 50))
    plt.legend(loc="upper right")
    #plt.savefig("instance3_IDW.pdf", bbox_inches='tight', format='pdf')
    plt.show()
