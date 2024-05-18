import sys
import datetime
import random
import numpy as np
from matplotlib import pyplot as plt

from utils.import_data import load_data_optimizers, prep_naive_data
from utils.draw import draw_subpbs_candidates_instance4_5
from problems_instance4 import instance4_bb_wrapper

if __name__ == '__main__':

    load_RMSE_dataset = True  # True if RMSEs have already been computed: useful for formatting final figure
    nb_rand_trials = 30

    # Model: choose between "IDW" and "KNN"
    model = "IDW"

    # Setup
    optimizers = ["ASGD", "ADAM"]
    nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 3}
    nb_pts = [100, 120, 140]
    nb_test_pts = [25, 30, 35]
    nb_valid_pts = [25, 30, 35]
    nb_max_var = 12
    var_inc_sub = {("ASGD", 1): [2, 5, 6, 7, 8], ("ASGD", 2): [2, 3, 5, 6, 7, 8],
                   ("ADAM", 1): [2, 5, 9, 10, 11], ("ADAM", 2): [2, 3, 5, 9, 10, 11],
                   ("ADAM", 3): [2, 3, 4, 5, 9, 10, 11]}
    nb_var_sub = {("ASGD", 1): 5, ("ASGD", 2): 6, ("ADAM", 1): 5, ("ADAM", 2): 6, ("ADAM", 3): 7}

    # Best parameters
    x_best_naive_IDW = [6.24743e-06, 1.81619, 0.000340702, 5.776e-06, 0.000121916, 0, 7.12439e-09, 0.844217, 1.00119e-07, 1.15241e-05, 3.06912e-08, 97.0525, 1.37657, 8.13872e-07, 4.28585e-06, 3.73813e-07, 0, 2.76438e-07, 1.32248, 1.09992e-07, 2.7641e-07, 9.1327e-08, 6.33001e-06, 1.5348e-07, 0, 1.15008, 4.35709e-08, 1.21737e-06, 0]
    x_best_graph_IDW = [960.544, 10, 916.86, 1000, 3.06424, 999.888, 1000, 1.5, 1.25636e-05, 721.373, 0.775113, 0.000210357, 1.27347e-06, 2.04072e-06, 999.999, 3.0895e-07, 0.360407, 0.00299164, 0.00569467, 37.9814, 1.7398e-05]
    x_best_naive_KNN = [0.00330341, 1.32845, 0.00439633, 0.0011544, 0, 0.000429643, 3.15247e-05, 1.08098, 3.59739, 0.00313888, 0.000294585, 9.17391, 1.28799, 0.000114471, 3.24964e-05, 0.00460077, 0.000331102, 0.000300558, 0.956119, 0.00150471, 1.05646e-06, 0.00269876, 0.00191142, 4.72403e-05, 2.73454e-06, 0.73586, 0.0513265, 0.014077, 0.000160786, 7, 1, 25, 5, 7]
    x_best_graph_KNN = [29.9627, 10.2, 10.9385, 18.9958, 3.26902, 6.72139, 9.16873, 1.50077, 306.161, 9.04198, 12.765, 0.186892, 4.837e-05, 0.0193345, 95.0474, 7.70068, 0.979647, 0.0536753, 102.423, 281.719, 2.52636, 5]

    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        load_data_optimizers('data_instance4.xlsx', optimizers, nb_sub_per_optimizer, nb_max_var, nb_pts, nb_test_pts,
                             nb_valid_pts)

    # Prepare data for naive approach
    X_train_naive, y_train_naive, X_valid_naive, y_valid_naive, X_test_naive, y_test_naive = \
        prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, 'instance4')

    if load_RMSE_dataset:
        # Load test
        if model == "IDW":
            (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std) = \
                np.loadtxt("log_IDW_both_selected_parameters.txt")
        elif model == "KNN":
            (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std) = \
                np.loadtxt("log_KNN_both_selected_parameters.txt")

    else:
        RMSE_naive = np.zeros((290-6+2, nb_rand_trials))
        RMSE_graph = np.zeros((290-6+2, nb_rand_trials))
        for j in range(nb_rand_trials):
            random.seed(j)

            # Training data for naive : in the loop to allows reset for each seed-trial (we use pop on these lists)
            idx_train_ASGD_l1 = (np.where((X_train[:, 0] == "ASGD") & (X_train[:, 1] == 1))[0]).tolist()
            idx_train_ASGD_l2 = (np.where((X_train[:, 0] == "ASGD") & (X_train[:, 1] == 2))[0]).tolist()
            idx_train_ADAM_l1 = (np.where((X_train[:, 0] == "ADAM") & (X_train[:, 1] == 1))[0]).tolist()
            idx_train_ADAM_l2 = (np.where((X_train[:, 0] == "ADAM") & (X_train[:, 1] == 2))[0]).tolist()
            idx_train_ADAM_l3 = (np.where((X_train[:, 0] == "ADAM") & (X_train[:, 1] == 3))[0]).tolist()
            idx_train_subs = {("ASGD", 1): idx_train_ASGD_l1, ("ASGD", 2): idx_train_ASGD_l2,
                              ("ADAM", 1): idx_train_ADAM_l1, ("ADAM", 2): idx_train_ADAM_l2,
                              ("ADAM", 3): idx_train_ADAM_l3}

            # Add at least one point of each sub prob in the initial partial dataset : where()[0] returns the list
            idx_train_subs_partial = {("ASGD", 1): [idx_train_ASGD_l1.pop(random.randrange(len(idx_train_ASGD_l1)))],
                                      ("ASGD", 2): [idx_train_ASGD_l2.pop(random.randrange(len(idx_train_ASGD_l2)))],
                                      ("ADAM", 1): [idx_train_ADAM_l1.pop(random.randrange(len(idx_train_ADAM_l1)))],
                                      ("ADAM", 2): [idx_train_ADAM_l2.pop(random.randrange(len(idx_train_ADAM_l2)))],
                                      ("ADAM", 3): [idx_train_ADAM_l3.pop(random.randrange(len(idx_train_ADAM_l3)))]}

            draw = [1, 2, 3, 4, 5]
            i = 0
            while len(draw) != 0:

                # Create partial data set for naive approach
                X_train_naive_partial = {("ASGD", 1): X_train[np.ix_(idx_train_subs_partial["ASGD", 1], var_inc_sub["ASGD", 1])],
                                         ("ASGD", 2): X_train[np.ix_(idx_train_subs_partial["ASGD", 2], var_inc_sub["ASGD", 2])],
                                         ("ADAM", 1): X_train[np.ix_(idx_train_subs_partial["ADAM", 1], var_inc_sub["ADAM", 1])],
                                         ("ADAM", 2): X_train[np.ix_(idx_train_subs_partial["ADAM", 2], var_inc_sub["ADAM", 2])],
                                         ("ADAM", 3): X_train[np.ix_(idx_train_subs_partial["ADAM", 3], var_inc_sub["ADAM", 3])]}

                y_train_naive_partial = {("ASGD", 1): y_train[idx_train_subs_partial["ASGD", 1]],
                                         ("ASGD", 2): y_train[idx_train_subs_partial["ASGD", 2]],
                                         ("ADAM", 1): y_train[idx_train_subs_partial["ADAM", 1]],
                                         ("ADAM", 2): y_train[idx_train_subs_partial["ADAM", 2]],
                                         ("ADAM", 3): y_train[idx_train_subs_partial["ADAM", 3]]}

                if model == "IDW":
                    # Create parametrized bb for naive approach
                    IDW_naive_bb = instance4_bb_wrapper(X_train_naive_partial, y_train_naive_partial,
                                                        X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "IDW")
                    RMSE_naive[i, j] = IDW_naive_bb(*x_best_naive_IDW)

                    # Create parametrized bb for graph approach
                    idx_train_partial_graph = idx_train_subs_partial["ASGD", 1] + idx_train_subs_partial["ASGD", 2] +\
                                              idx_train_subs_partial["ADAM", 1] + idx_train_subs_partial["ADAM", 2] +\
                                              idx_train_subs_partial["ADAM", 3]
                    X_train_graph_partial = X_train[idx_train_partial_graph, :]
                    y_train_graph_partial = y_train[idx_train_partial_graph]
                    IDW_graph_bb = instance4_bb_wrapper(X_train_graph_partial, y_train_graph_partial,
                                                        X_test, y_test, nb_max_var, nb_var_sub, "graph", "IDW")
                    RMSE_graph[i, j] = IDW_graph_bb(*x_best_graph_IDW)

                elif model == "KNN":
                    # Create parametrized bb for naive approach
                    KNN_naive_bb = instance4_bb_wrapper(X_train_naive_partial, y_train_naive_partial,
                                                        X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "KNN")
                    RMSE_naive[i, j] = KNN_naive_bb(*x_best_naive_KNN)

                    # Create parametrized bb for graph approach
                    idx_train_partial_graph = idx_train_subs_partial["ASGD", 1] + idx_train_subs_partial["ASGD", 2] + \
                                              idx_train_subs_partial["ADAM", 1] + idx_train_subs_partial["ADAM", 2] +\
                                                idx_train_subs_partial["ADAM", 3]
                    X_train_graph_partial = X_train[idx_train_partial_graph, :]
                    y_train_graph_partial = y_train[idx_train_partial_graph]
                    KNN_graph_bb = instance4_bb_wrapper(X_train_graph_partial, y_train_graph_partial,
                                                        X_test, y_test, nb_max_var, nb_var_sub, "graph", "KNN")
                    RMSE_graph[i, j] = KNN_graph_bb(*x_best_graph_KNN)

                # Determine next draw is amongst which subpbs and check if while-loop is done
                draw = draw_subpbs_candidates_instance4_5(idx_train_subs["ASGD", 1], idx_train_subs["ASGD", 2],
                                                          idx_train_subs["ADAM", 1], idx_train_subs["ADAM", 2],
                                                          idx_train_subs["ADAM", 3])

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
                    elif subpb_draw == 5:
                        next_index = idx_train_subs["ADAM", 3].pop()
                        idx_train_subs_partial["ADAM", 3].append(next_index)

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

    nb_plot = 290-6
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
    #plt.savefig("instance4_IDW.pdf", bbox_inches='tight', format='pdf')
    plt.show()
