import sys

import random
import numpy as np
import PyNomad
from matplotlib import pyplot as plt

from models.IDW_class import IDW
from utils.distances import general_distance, graph_structured_distance
from utils.draw import draw_subpbs_candidates_instance1_2
from utils.import_data import load_data_fix_optimizer, prep_naive_data
from problems_instance1 import instance1_bb_wrapper


if __name__ == '__main__':

    load_RMSE_dataset = True  # True if RMSEs have already been computed: useful for formatting final figure
    nb_train_pts = 30  # 30
    nb_rand_trials = 10  # 10
    nb_evaluations_NOMAD = 250 # 250

    # Model: choose between "IDW" and "KNN"
    model = "IDW"

    # Setup
    nb_pts = [40, 60, 80]
    nb_test_pts = [10, 15, 20]
    nb_valid_pts = [10, 15, 20]
    nb_sub = 3
    nb_max_var = 5
    nb_var_sub = [2, 3, 4]
    var_inc_sub = [[1, 4], [1, 2, 4], [1, 2, 3, 4]]  # included variables in each subproblem

    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_fix_optimizer('data_instance1.xlsx', nb_sub,
                                                                                 nb_max_var, nb_pts, nb_test_pts,
                                                                                 nb_valid_pts)

    # Prepare data for naive approach
    X_train_naive, y_train_naive, X_valid_naive, y_valid_naive, X_test_naive, y_test_naive = \
        prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, "instance1")

    if load_RMSE_dataset:
        # Load test
        if model == "IDW":
            (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std) = \
                np.loadtxt("log_IDW_both_optimized_parameters.txt")
        elif model == "KNN":
            (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std) = \
                np.loadtxt("log_KNN_both_optimized_parameters.txt")

    else:
        RMSE_naive_test = np.zeros((90-3+1, nb_rand_trials))
        RMSE_graph_test = np.zeros((90-3+1, nb_rand_trials))
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
            while len(draw) != 0 and i <= nb_train_pts :

                # -------------------------------------------------#
                # Create partial data set for naive approach
                X_train_naive_partial = {"sub1": X_train[np.ix_(idx_train_subs_partial["sub1"], var_inc_sub[0])],
                                         "sub2": X_train[np.ix_(idx_train_subs_partial["sub2"], var_inc_sub[1])],
                                         "sub3": X_train[np.ix_(idx_train_subs_partial["sub3"], var_inc_sub[2])]}
                y_train_naive_partial = {"sub1": y_train[idx_train_subs_partial["sub1"]],
                                         "sub2": y_train[idx_train_subs_partial["sub2"]],
                                         "sub3": y_train[idx_train_subs_partial["sub3"]]}

                if model == "IDW":
                    # Create parametrized bb for adjusting parameters
                    IDW_naive_bb = instance1_bb_wrapper(X_train_naive_partial, y_train_naive_partial, X_valid_naive, y_valid_naive, nb_max_var, nb_var_sub, "naive", "IDW")
                    def IDW_naive_bb_nomad(x):
                        try:
                            # (w11, w12, w21, w22, w23, w31, w32, w33, w34)
                            # (u1, lr; u1, u2, lr; u1, u2, u3, lr)
                            f = IDW_naive_bb(x.get_coord(0), x.get_coord(1),
                                             x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                             x.get_coord(5), x.get_coord(6), x.get_coord(7), x.get_coord(8))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # (w11, w12; w21, w22, w23; w31, w32, w33, w34) = (u1, lr; u1, u2, lr; u1, u2, u3, lr)
                    x0 = [0, 1, 0, 0, 1, 0, 0, 0, 1]
                    lb = [0, 1e-6, 0, 0, 1e-6, 0, 0, 0, 1e-6]
                    ub = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

                    params = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL " + str(nb_evaluations_NOMAD), "LH_SEARCH 25 3",
                              "DISPLAY_DEGREE 0", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ"]
                    result = PyNomad.optimize(IDW_naive_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    #w11, w12, w21, w22, w23, w31, w32, w33, w34 = x_best[0], x_best[1], x_best[2], x_best[3], x_best[4], x_best[5], x_best[6], x_best[7], x_best[8]

                    # Naive test
                    IDW_naive_bb_test = instance1_bb_wrapper(X_train_naive_partial, y_train_naive_partial, X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "IDW")
                    RMSE_naive_test[i, j] = IDW_naive_bb_test(*x_best)
                    # -------------------------------------------------#

                    # -------------------------------------------------#
                    # Create parametrized bb for graph approach
                    idx_train_partial_graph = idx_train_subs_partial["sub1"] + idx_train_subs_partial["sub2"] + idx_train_subs_partial["sub3"]
                    X_train_graph_partial = X_train[idx_train_partial_graph, :]
                    y_train_graph_partial = y_train[idx_train_partial_graph]

                    IDW_graph_bb = instance1_bb_wrapper(X_train_graph_partial, y_train_graph_partial, X_valid, y_valid, nb_max_var, nb_var_sub, "graph", "IDW")
                    def IDW_graph_bb_nomad(x):
                        try:
                            # BB : (theta_u2, theta_u3, w1, w2, w3, w4, w5)
                            f = IDW_graph_bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                             x.get_coord(5), x.get_coord(6))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # (theta_u2, theta_u3, w1=l, w2=u1, w3=u2, w4=u3, w5=lr)
                    x0 = [10, 10, 1, 0, 0, 0, 1]
                    lb = [10, 10, 0, 0, 0, 0, 1e-6]
                    ub = [1000, 1000, 1000, 1000, 1000, 1000, 1000]

                    params = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL " + str(nb_evaluations_NOMAD), "LH_SEARCH 25 3",
                              "DISPLAY_DEGREE 0", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ"]
                    result = PyNomad.optimize(IDW_graph_bb_nomad, x0, lb, ub, params)
                    x_best_graph = result['x_best']
                    #a1, a2, a3, a4, a5, a6, a7 = x_best_graph[0], x_best_graph[1], x_best_graph[2], x_best_graph[3], x_best_graph[4], x_best_graph[5], x_best_graph[6]

                    # Graph test
                    IDW_graph_bb_test = instance1_bb_wrapper(X_train_graph_partial, y_train_graph_partial, X_test, y_test, nb_max_var, nb_var_sub, "graph", "IDW")
                    RMSE_graph_test[i, j] = IDW_graph_bb_test(*x_best_graph)
                    # -------------------------------------------------#

                elif model == "KNN":
                    # Construct parametrized bb w.r.t to dataset
                    KNN_naive_bb = instance1_bb_wrapper(X_train_naive_partial, y_train_naive_partial, X_valid_naive,
                                                        y_valid_naive, nb_max_var, nb_var_sub, "naive", "KNN")

                    # Construct parametrized bb for nomad
                    def KNN_naive_bb_nomad(x):
                        try:
                            # (w11, w12, w21, w22, w23, w31, w32, w33, w34, K1, K2, K3)
                            # (u1, lr, u1, u2, lr, u1, u2, u3, lr, K1, K2, K3)
                            f = KNN_naive_bb(x.get_coord(0), x.get_coord(1),
                                             x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                             x.get_coord(5), x.get_coord(6), x.get_coord(7), x.get_coord(8),
                                             int(x.get_coord(9)), int(x.get_coord(10)), int(x.get_coord(11)))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation


                    # Setup for nomad
                    # (w11, w12, w21, w22, w23, w31, w32, w33, w34, K1, K2, K3) = (u1, lr, u1, u2, lr, u1, u2, u3, lr, K1, K2, K3)
                    x0 = [0, 1, 0, 0, 1, 0, 0, 0, 1, 5, 5, 5]
                    lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
                    ub = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 20, 30, 40]
                    input_type = "BB_INPUT_TYPE (R R R R R R R R R I I I)"
                    params = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL " + str(nb_evaluations_NOMAD), "LH_SEARCH 25 3",
                              "DISPLAY_DEGREE 0", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ"]
                    result = PyNomad.optimize(KNN_naive_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    for k in range(3):
                        x_best[-(k + 1)] = int(x_best[-(k + 1)])

                    # Naive test
                    KNN_naive_bb_test = instance1_bb_wrapper(X_train_naive_partial, y_train_naive_partial, X_test_naive,
                                                             y_test_naive, nb_max_var, nb_var_sub, "naive", "KNN")
                    RMSE_naive_test[i, j] = KNN_naive_bb_test(*x_best)
                    # -------------------------------------------------#

                    # -------------------------------------------------#
                    # Create parametrized bb for graph approach
                    idx_train_partial_graph = idx_train_subs_partial["sub1"] + idx_train_subs_partial["sub2"] + \
                                              idx_train_subs_partial["sub3"]
                    X_train_graph_partial = X_train[idx_train_partial_graph, :]
                    y_train_graph_partial = y_train[idx_train_partial_graph]

                    KNN_graph_bb = instance1_bb_wrapper(X_train_graph_partial, y_train_graph_partial, X_valid, y_valid,
                                                        nb_max_var, nb_var_sub, "graph", "KNN")


                    # Create blackbox for Pynomad
                    def KNN_graph_bb_nomad(x):
                        try:
                            # BB : (theta_u2, theta_u3, w1, w2, w3, w4, w5, K)
                            f = KNN_graph_bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                             x.get_coord(5), x.get_coord(6), int(x.get_coord(7)))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation


                    # Setup for nomad
                    # (theta_u2, theta_u3, w1=l, w2=u1, w3=u2, w4=u3, w5=lr, K)
                    x0 = [10, 10, 1, 0, 0, 0, 1, 5]
                    lb = [10, 10, 0, 0, 0, 0, 0, 1]
                    ub = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 50]
                    input_type = "BB_INPUT_TYPE (R R R R R R R I)"

                    params = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL " + str(nb_evaluations_NOMAD), "LH_SEARCH 25 3",
                              "DISPLAY_DEGREE 0", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ"]
                    result = PyNomad.optimize(KNN_graph_bb_nomad, x0, lb, ub, params)
                    x_best_graph = result['x_best']
                    x_best_graph[-1] = int(x_best_graph[-1])

                    # Graph test
                    KNN_graph_bb_test = instance1_bb_wrapper(X_train_graph_partial, y_train_graph_partial, X_test, y_test,
                                                             nb_max_var, nb_var_sub, "graph", "KNN")
                    RMSE_graph_test[i, j] = KNN_graph_bb_test(*x_best_graph)
                    # -------------------------------------------------#

                # Determine next draw is amongst which subpbs and check if while-loop is done
                draw = draw_subpbs_candidates_instance1_2(idx_train_subs["sub1"], idx_train_subs["sub2"], idx_train_subs["sub3"])

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

        RMSE_graph_mean = RMSE_graph_test.mean(axis=1)
        RMSE_graph_std = RMSE_graph_test.std(axis=1)
        RMSE_naive_mean = RMSE_naive_test.mean(axis=1)
        RMSE_naive_std = RMSE_naive_test.std(axis=1)

        # Save test in case figures have to be modified
        if model == "IDW":
            #np.savetxt("log_IDW_both_optimized_parameters.txt",
            #           (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std))
            a = 1

        elif model == "KNN":
            #np.savetxt("log_KNN_both_optimized_parameters.txt",
            #           (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std))
            a = 2



    # Setup
    fig = plt.figure(dpi=100, tight_layout=True)
    plt.rc('axes', labelsize=24)
    plt.rc('legend', fontsize=24)
    plt.rc('xtick', labelsize=20)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the tick label
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    nb_plot = nb_train_pts
    plt.plot(range(nb_plot), RMSE_naive_mean[0:nb_plot], linestyle='-', marker='s', markersize=2, label='Sub')
    plt.fill_between(range(nb_plot), RMSE_naive_mean[0:nb_plot] - RMSE_naive_std[0:nb_plot],
                     RMSE_naive_mean[0:nb_plot] + RMSE_naive_std[0:nb_plot], color="tab:blue", alpha=0.30)
    plt.plot(range(nb_plot), RMSE_graph_mean[0:nb_plot], linestyle='-', marker='^', markersize=2, label='Graph')
    plt.fill_between(range(nb_plot), RMSE_graph_mean[0:nb_plot] - RMSE_graph_std[0:nb_plot],
                     RMSE_graph_mean[0:nb_plot] + RMSE_graph_std[0:nb_plot], color="orange", alpha=0.30)
    plt.xlabel('Training dataset size')
    plt.ylabel('RMSE on the test dataset')
    # plt.ylim((12.5, 42.5))
    plt.ylim((7.5, 52.5))
    plt.legend(loc="upper right")
    #plt.savefig("instance1_IDW_optimized.pdf", bbox_inches='tight', format='pdf')
    plt.show()

