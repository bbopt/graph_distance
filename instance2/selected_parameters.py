import sys
import datetime
import random
import numpy as np
from matplotlib import pyplot as plt

from utils.import_data import load_data_fix_optimizer, prep_naive_data
from utils.draw import draw_subpbs_candidates_instance1_2
from problems_instance2 import instance2_bb_wrapper

if __name__ == '__main__':

    load_RMSE_dataset = True  # True if RMSEs have already been computed: useful for formatting final figure
    nb_rand_trials = 30

    # Model: choose between "IDW" and "KNN"
    model = "KNN"

    nb_pts = [100, 120, 140]
    nb_test_pts = [25, 30, 35]
    nb_valid_pts = [25, 30, 35]
    nb_sub = 3
    nb_max_var = 8
    nb_var_sub = [5, 6, 7]
    var_inc_sub = [[1, 4, 5, 6, 7], [1, 2, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]  # included variables in each subproblem

    # Best parameters
    x_best_naive_IDW = [0.000118041, 11.0095, 0.00203427, 0.000454869, 2.7655e-06, 0, 0, 0.989989, 0, 2.74047e-05, 0, 0, 0, 3.498e-09, 1.02804, 0, 8.15094e-06, 1.28858e-08]
    x_best_graph_IDW = [10.0001, 10.0007, 999.489, 3.49128e-09, 0, 0, 0.00328981, 2.95877e-07, 7.87017e-07, 0]
    x_best_naive_KNN = [0.00084898, 1.05695, 0.0978699, 0.122922, 0.0094295, 0, 0.00983246, 489.794, 0.051403, 102.097, 0.006341, 1.0012, 0.080818, 0.017435, 206.008, 50.996, 0.526982, 0.060274, 7, 1, 24]
    x_best_graph_KNN = [11.0898, 12.058, 7.9969, 0, 0.00771, 6.19327, 22.2049, 10.607, 0.38868, 0.00408, 3]

    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_fix_optimizer('data_instance2.xlsx', nb_sub,
                                                                                 nb_max_var, nb_pts, nb_test_pts,
                                                                                 nb_valid_pts)

    # Prepare data for naive approach
    X_train_naive, y_train_naive, X_valid_naive, y_valid_naive, X_test_naive, y_test_naive = \
        prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, 'instance2')

    if load_RMSE_dataset:
        # Load test
        if model == "IDW":
            (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std) = \
                np.loadtxt("log_IDW_both_selected_parameters.txt")
        elif model == "KNN":
            (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std) = \
                np.loadtxt("log_KNN_both_selected_parameters.txt")
    else:
        RMSE_naive = np.zeros((180-3+1, nb_rand_trials))
        RMSE_graph = np.zeros((180-3+1, nb_rand_trials))
        for j in range(nb_rand_trials):
            random.seed(j)

            # Training data for naive : in the loop to allows reset for each seed-trial (we use pop on these lists)
            idx_train_subs1 = (np.where(X_train[:, 2] == "EXC")[0]).tolist()
            idx_train_subs2 = (np.where((X_train[:, 2] != "EXC") & (X_train[:, 3] == "EXC"))[0]).tolist()
            idx_train_subs3 = (np.where((X_train[:, 2] != "EXC") & (X_train[:, 3] != "EXC"))[0]).tolist()
            idx_train_subs = {"sub1": idx_train_subs1, "sub2": idx_train_subs2, "sub3": idx_train_subs3}

            # Add at least one point of each sub prob in the initial partial dataset : where()[0] returns the list
            idx_train_subs_partial = {"sub1": [idx_train_subs1.pop(random.randrange(len(idx_train_subs1)))],
                                      "sub2": [idx_train_subs2.pop(random.randrange(len(idx_train_subs2)))],
                                      "sub3": [idx_train_subs3.pop(random.randrange(len(idx_train_subs3)))]}

            draw = [1, 2, 3]
            i = 0
            while len(draw) != 0:

                # Create partial data set for naive approach
                X_train_naive_partial = {"sub1": X_train[np.ix_(idx_train_subs_partial["sub1"], var_inc_sub[0])],
                                         "sub2": X_train[np.ix_(idx_train_subs_partial["sub2"], var_inc_sub[1])],
                                         "sub3": X_train[np.ix_(idx_train_subs_partial["sub3"], var_inc_sub[2])]}
                y_train_naive_partial = {"sub1": y_train[idx_train_subs_partial["sub1"]],
                                         "sub2": y_train[idx_train_subs_partial["sub2"]],
                                         "sub3": y_train[idx_train_subs_partial["sub3"]]}

                if model == "IDW":
                    # Create parametrized bb for naive approach
                    IDW_naive_bb = instance2_bb_wrapper(X_train_naive_partial, y_train_naive_partial,
                                                             X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "IDW")
                    RMSE_naive[i, j] = IDW_naive_bb(*x_best_naive_IDW)

                    # Create parametrized bb for graph approach
                    idx_train_partial_graph = idx_train_subs_partial["sub1"] + idx_train_subs_partial["sub2"] + idx_train_subs_partial["sub3"]
                    X_train_graph_partial = X_train[idx_train_partial_graph, :]
                    y_train_graph_partial = y_train[idx_train_partial_graph]
                    IDW_graph_bb = instance2_bb_wrapper(X_train_graph_partial, y_train_graph_partial,
                                                             X_test, y_test, nb_max_var, nb_var_sub, "graph", "IDW")
                    RMSE_graph[i, j] = IDW_graph_bb(*x_best_graph_IDW)

                elif model == "KNN":
                    # Create parametrized bb for naive approach
                    KNN_naive_bb = instance2_bb_wrapper(X_train_naive_partial, y_train_naive_partial,
                                                        X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "KNN")
                    RMSE_naive[i, j] = KNN_naive_bb(*x_best_naive_KNN)

                    # Create parametrized bb for graph approach
                    idx_train_partial_graph = idx_train_subs_partial["sub1"] + idx_train_subs_partial["sub2"] + \
                                              idx_train_subs_partial["sub3"]
                    X_train_graph_partial = X_train[idx_train_partial_graph, :]
                    y_train_graph_partial = y_train[idx_train_partial_graph]
                    KNN_graph_bb = instance2_bb_wrapper(X_train_graph_partial, y_train_graph_partial,
                                                        X_test, y_test, nb_max_var, nb_var_sub, "graph", "KNN")
                    RMSE_graph[i, j] = KNN_graph_bb(*x_best_graph_KNN)

                # Determine next draw is amongst which subpbs and check if while-loop is done
                draw = draw_subpbs_candidates_instance1_2(idx_train_subs["sub1"], idx_train_subs["sub2"],
                                                          idx_train_subs["sub3"])

                if len(draw) != 0:
                    # 0=sub1, 1=sub1, 2=sub3
                    subpb_draw = random.choice(draw)
                    # Update
                    i = i + 1

                    if subpb_draw == 1:
                        next_index = idx_train_subs["sub1"].pop()
                        idx_train_subs_partial["sub1"].append(next_index)
                    elif subpb_draw == 2:
                        next_index = idx_train_subs["sub2"].pop()
                        idx_train_subs_partial["sub2"].append(next_index)
                    elif subpb_draw == 3:
                        next_index = idx_train_subs["sub3"].pop()
                        idx_train_subs_partial["sub3"].append(next_index)

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

    # Figure setup
    fig = plt.figure(dpi=100, tight_layout=True)
    plt.rc('axes', labelsize=28)
    plt.rc('legend', fontsize=28)
    plt.rc('xtick', labelsize=26)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=26)  # fontsize of the tick label
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    nb_plot = 180-3
    plt.plot(range(nb_plot), RMSE_naive_mean[0:nb_plot], linestyle='-', marker='s', markersize=2, label='Sub')
    plt.fill_between(range(nb_plot), RMSE_naive_mean[0:nb_plot] - RMSE_naive_std[0:nb_plot],
                     RMSE_naive_mean[0:nb_plot] + RMSE_naive_std[0:nb_plot], color="tab:blue", alpha=0.30)
    plt.plot(range(nb_plot), RMSE_graph_mean[0:nb_plot], linestyle='-', marker='^', markersize=2,
             label='Graph')
    plt.fill_between(range(nb_plot), RMSE_graph_mean[0:nb_plot] - RMSE_graph_std[0:nb_plot],
                     RMSE_graph_mean[0:nb_plot] + RMSE_graph_std[0:nb_plot], color="orange", alpha=0.30)
    plt.xlabel('Training dataset size')
    plt.ylabel('RMSE test')
    plt.ylim((7.5, 50))
    plt.legend(loc="upper right")
    #plt.savefig("instance2_IDW.pdf", bbox_inches='tight', format='pdf')
    plt.show()