import sys
import datetime
import random
import numpy as np
from matplotlib import pyplot as plt


from utils.import_data import load_data_optimizers, prep_naive_data
from utils.draw import draw_subpbs_candidates_instance4_5
from problems_instance5 import instance5_bb_wrapper


if __name__ == '__main__':

    load_RMSE_dataset = True  # True if RMSEs have already been computed: useful for formatting final figure
    nb_rand_trials = 30

    # Model: choose between "IDW" and "KNN"
    model = "KNN"

    # Setup
    optimizers = ["ASGD", "ADAM"]
    nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 3}
    nb_pts = [120, 140, 160]
    nb_test_pts = [30, 35, 40]
    nb_valid_pts = [30, 35, 40]
    nb_max_var = 13  # add the dropout
    var_inc_sub = {("ASGD", 1): [2, 5, 6, 7, 8, 12], ("ASGD", 2): [2, 3, 5, 6, 7, 8, 12],
                   ("ADAM", 1): [2, 5, 9, 10, 11, 12], ("ADAM", 2): [2, 3, 5, 9, 10, 11, 12],
                   ("ADAM", 3): [2, 3, 4, 5, 9, 10, 11, 12]}
    nb_var_sub = {("ASGD", 1): 6, ("ASGD", 2): 7, ("ADAM", 1): 6, ("ADAM", 2): 7, ("ADAM", 3): 8}

    # Best parameters
    x_best_naive_IDW = [9.88644e-06, 115.112, 1.15111e-05, 57.2049, 0, 2.14856e-07, 0, 1.51894e-08, 6.30149, 2.11e-12, 6.89e-12, 2.57e-12, 1.89485e-09, 3.72e-09, 9.96627, 2.79252e-09, 1.5375e-10, 0, 1.13584e-09, -9e-14, 4.09518e-06, 1.1049, 1.835e-11, 1.363e-11, 3.3132e-10, 0, 3e-14, 1.8228e-10, 1.44395e-05, 6.7535, 2.75411e-08, 6.78755e-09, 0, 0]
    x_best_graph_IDW = [13.9619, 27.134, 999.999, 599.987, 11.3778, 1000, 660.003, 992.007, 0, 958.484, 1000, 0, 0, 0, 74.0753, 0.00589456, 0, 0, 0.161019, 0, 0, 0.238256]
    x_best_naive_KNN = [1.60063e-05, 1.03939, 0.109924, 300.198, 0.00134277, 99.9831, 0.00707337, 0.000104207, 1.002, 0.0188928, 0.000494609, 0.0502207, 0.00254688, 4.79437e-05, 1.04246, 0.0370575, 0.00815143, 0.0026641, 0.000391668, 0.000371317, 0.00115088, 10.8504, 0.00984824, 0.0931623, 11.9758, 99.9888, 0.00104944, 1.64384e-05, 0.00117252, 0.987888, 0.00508727, 0.032226, 0.0174028, 0.00196256, 4, 5, 3, 4, 5]
    x_best_graph_KNN = [10, 980, 12.5, 10.5, 283, 0.2, 180, 551.5, 601, 999, 905, 1, 0, 8, 1000, 485, 1000, 98, 500, 904, 399, 822, 6]

    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        load_data_optimizers('data_instance5.xlsx', optimizers, nb_sub_per_optimizer, nb_max_var, nb_pts, nb_test_pts,
                             nb_valid_pts)

    # Prepare data for naive approach
    X_train_naive, y_train_naive, X_valid_naive, y_valid_naive, X_test_naive, y_test_naive = \
        prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, 'instance5')

    if load_RMSE_dataset:
        # Load test
        if model == "IDW":
            (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std) = \
                np.loadtxt("log_IDW_both_selected_parameters.txt")
        elif model == "KNN":
            (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std) = \
                np.loadtxt("log_KNN_both_selected_parameters.txt")
    else:
        RMSE_naive = np.zeros((340-6+2, nb_rand_trials))
        RMSE_graph = np.zeros((340-6+2, nb_rand_trials))
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
                    IDW_naive_bb = instance5_bb_wrapper(X_train_naive_partial, y_train_naive_partial,
                                                        X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "IDW")
                    RMSE_naive[i, j] = IDW_naive_bb(*x_best_naive_IDW)

                    # Create parametrized bb for graph approach
                    idx_train_partial_graph = idx_train_subs_partial["ASGD", 1] + idx_train_subs_partial["ASGD", 2] +\
                                              idx_train_subs_partial["ADAM", 1] + idx_train_subs_partial["ADAM", 2] +\
                                              idx_train_subs_partial["ADAM", 3]
                    X_train_graph_partial = X_train[idx_train_partial_graph, :]
                    y_train_graph_partial = y_train[idx_train_partial_graph]
                    IDW_graph_bb = instance5_bb_wrapper(X_train_graph_partial, y_train_graph_partial,
                                                        X_test, y_test, nb_max_var, nb_var_sub, "graph", "IDW")
                    RMSE_graph[i, j] = IDW_graph_bb(*x_best_graph_IDW)

                elif model == "KNN":
                    # Create parametrized bb for naive approach
                    KNN_naive_bb = instance5_bb_wrapper(X_train_naive_partial, y_train_naive_partial,
                                                        X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "KNN")
                    RMSE_naive[i, j] = KNN_naive_bb(*x_best_naive_KNN)

                    # Create parametrized bb for graph approach
                    idx_train_partial_graph = idx_train_subs_partial["ASGD", 1] + idx_train_subs_partial["ASGD", 2] + \
                                              idx_train_subs_partial["ADAM", 1] + idx_train_subs_partial["ADAM", 2] + \
                                              idx_train_subs_partial["ADAM", 3]
                    X_train_graph_partial = X_train[idx_train_partial_graph, :]
                    y_train_graph_partial = y_train[idx_train_partial_graph]
                    KNN_graph_bb = instance5_bb_wrapper(X_train_graph_partial, y_train_graph_partial,
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

    nb_plot = 340-6
    plt.plot(range(nb_plot), RMSE_naive_mean[0:nb_plot], linestyle='-', marker='s', markersize=2, label='Sub')
    plt.fill_between(range(nb_plot), RMSE_naive_mean[0:nb_plot] - RMSE_naive_std[0:nb_plot],
                     RMSE_naive_mean[0:nb_plot] + RMSE_naive_std[0:nb_plot], color="tab:blue", alpha=0.30)
    plt.plot(range(nb_plot), RMSE_graph_mean[0:nb_plot], linestyle='-', marker='^', markersize=2, label='Graph')
    plt.fill_between(range(nb_plot), RMSE_graph_mean[0:nb_plot] - RMSE_graph_std[0:nb_plot],
                     RMSE_graph_mean[0:nb_plot] + RMSE_graph_std[0:nb_plot], color="orange", alpha=0.30)
    plt.xlabel('Training dataset size')
    plt.ylabel('RMSE test')
    #plt.ylim((12.5, 42.5))
    plt.ylim((7.5, 50))
    plt.legend(loc="upper right")
    #plt.savefig("instance5_IDW.pdf", bbox_inches='tight', format='pdf')
    plt.show()