import sys
import datetime
import random
import numpy as np
import math
from matplotlib import pyplot as plt


from utils.import_data import load_data_optimizers, prep_naive_data, hybrid_data_optimizers
from utils.draw import draw_subpbs_candidates_variant4_5
from utils.problems_instances_setup import scalar_div_list, variant_size_setup
from problems_variant5 import variant5_bb_wrapper


def read_last_row_discard_first_two(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if not lines:
            return []  # Return empty if file is empty
        last_line = lines[-1].strip().split()
        return [float(x) for x in last_line[2:]]  # Discard first two and convert to float


if __name__ == '__main__':

    load_RMSE_dataset = True  # True if RMSEs have already been computed: useful for formatting final figure
    nb_rand_trials = 20
    model = "IDW"
    architecture = "MLP"
    variant = "variant5"
    seed = 0
    size = 3

    # Setup
    (nb_pts, nb_test_pts, nb_valid_pts) = variant_size_setup(variant, size)
    nb_pts_train = 2*(nb_test_pts[0]+nb_valid_pts[0] + nb_test_pts[1]+nb_valid_pts[1]) + (nb_test_pts[2] + nb_valid_pts[2]) - 5

    optimizers = ["ASGD", "ADAM"]
    nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 3}
    nb_max_var = 13  # add the dropout
    var_inc_sub = {("ASGD", 1): [2, 5, 6, 7, 8, 12], ("ASGD", 2): [2, 3, 5, 6, 7, 8, 12],
                   ("ADAM", 1): [2, 5, 9, 10, 11, 12], ("ADAM", 2): [2, 3, 5, 9, 10, 11, 12],
                   ("ADAM", 3): [2, 3, 4, 5, 9, 10, 11, 12]}
    nb_var_sub = {("ASGD", 1): 6, ("ASGD", 2): 7, ("ADAM", 1): 6, ("ADAM", 2): 7, ("ADAM", 3): 8}
    var_sub_hybrid = {"ASGD": [1, 2, 3, 5, 6, 7, 8, 12],
                      "ADAM": [1, 2, 3, 4, 5, 9, 10, 11, 12]}

    # Build filenames
    data_file_naive = f"log_{variant}_size{size}_naive_{model}_{architecture}.{seed}.txt"
    data_file_graph = f"log_{variant}_size{size}_graph_{model}_{architecture}.{seed}.txt"
    data_file_hybrid = f"log_{variant}_size{size}_hybrid_{model}_{architecture}.{seed}.txt"

    # Read and extract data
    x_best_naive_IDW = read_last_row_discard_first_two(data_file_naive)
    x_best_graph_IDW = read_last_row_discard_first_two(data_file_graph)
    x_best_hybrid_IDW = read_last_row_discard_first_two(data_file_hybrid)

    # Load data
    data_file = "data_" + variant + "_size" + str(size) + "_" + architecture + ".xlsx"
    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        load_data_optimizers(data_file, optimizers, nb_sub_per_optimizer, nb_max_var, nb_pts, nb_test_pts,
                             nb_valid_pts)

    # Prepare data for naive approach
    X_train_naive, y_train_naive, X_valid_naive, y_valid_naive, X_test_naive, y_test_naive = \
        prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, variant)

    # Prepare data for hybrid approach
    X_train_hybrid, y_train_hybrid, X_valid_hybrid, y_valid_hybrid, X_test_hybrid, y_test_hybrid = \
        hybrid_data_optimizers(X_train, y_train, X_valid, y_valid, X_test, y_test, var_sub_hybrid)

    if load_RMSE_dataset:
        (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std, RMSE_hybrid_mean, RMSE_hybrid_std) = \
            np.loadtxt("selected_parameters_" + variant + "_size" + str(size) + "_" + model + "_" + architecture + ".txt")

    else:
        RMSE_naive = np.zeros((nb_pts_train, nb_rand_trials))
        RMSE_graph = np.zeros((nb_pts_train, nb_rand_trials))
        RMSE_hybrid = np.zeros((nb_pts_train, nb_rand_trials))

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

                # Create partial data set for graph approach
                idx_train_partial_graph = idx_train_subs_partial["ASGD", 1] + idx_train_subs_partial["ASGD", 2] + \
                                          idx_train_subs_partial["ADAM", 1] + idx_train_subs_partial["ADAM", 2] + \
                                          idx_train_subs_partial["ADAM", 3]
                X_train_graph_partial = X_train[idx_train_partial_graph, :]
                y_train_graph_partial = y_train[idx_train_partial_graph]

                # Create partial data set for hybrid approach
                idx_train_partial_hybrid_ASGD = idx_train_subs_partial["ASGD", 1] + idx_train_subs_partial["ASGD", 2]
                idx_train_partial_hybrid_ADAM = idx_train_subs_partial["ADAM", 1] + idx_train_subs_partial["ADAM", 2] + idx_train_subs_partial["ADAM", 3]
                X_train_hybrid_partial = {"ASGD": X_train[np.ix_(idx_train_partial_hybrid_ASGD, var_sub_hybrid["ASGD"])],
                                          "ADAM": X_train[np.ix_(idx_train_partial_hybrid_ADAM, var_sub_hybrid["ADAM"])]}
                y_train_hybrid_partial = {"ASGD": y_train[idx_train_partial_hybrid_ASGD],
                                          "ADAM": y_train[idx_train_partial_hybrid_ADAM]}

                if model == "IDW":
                    # Create parametrized bb for naive approach
                    IDW_naive_bb = variant5_bb_wrapper(X_train_naive_partial, y_train_naive_partial,
                                                        X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "IDW")
                    RMSE_naive[i, j] = IDW_naive_bb(*x_best_naive_IDW)

                    # Create parametrized bb for graph approach
                    IDW_graph_bb = variant5_bb_wrapper(X_train_graph_partial, y_train_graph_partial,
                                                        X_test, y_test, nb_max_var, nb_var_sub, "graph", "IDW")
                    RMSE_graph[i, j] = IDW_graph_bb(*x_best_graph_IDW)

                    # Create parametrized bb for hybrid approach
                    IDW_hybrid_bb_test = variant5_bb_wrapper(X_train_hybrid_partial, y_train_hybrid_partial,
                                                             X_test_hybrid, y_test_hybrid, nb_max_var, nb_var_sub,
                                                             "hybrid", "IDW")
                    RMSE_hybrid[i, j] = IDW_hybrid_bb_test(*x_best_hybrid_IDW)

                elif model == "KNN":
                    continue

                # Determine next draw is amongst which subpbs and check if while-loop is done
                draw = draw_subpbs_candidates_variant4_5(idx_train_subs["ASGD", 1], idx_train_subs["ASGD", 2],
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
        RMSE_hybrid_mean = RMSE_hybrid.mean(axis=1)
        RMSE_hybrid_std = RMSE_hybrid.std(axis=1)

        # Save test in case figures have to be modified
        np.savetxt("selected_parameters_" + variant + "_size" + str(size) + "_" + model + "_" + architecture + ".txt",
                   (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std, RMSE_hybrid_mean, RMSE_hybrid_std))




    # Setup
    fig = plt.figure(dpi=100, tight_layout=True)
    plt.rc('axes', labelsize=28)
    plt.rc('legend', fontsize=28)
    plt.rc('xtick', labelsize=26)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=26)  # fontsize of the tick label
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    nb_plot = nb_pts_train-7
    # Sub
    plt.plot(range(nb_plot), RMSE_naive_mean[0:nb_plot], linestyle='-', marker='s', markersize=2, label='Sub')
    plt.fill_between(range(nb_plot), RMSE_naive_mean[0:nb_plot] - np.sqrt(RMSE_naive_std[0:nb_plot]),
                     RMSE_naive_mean[0:nb_plot] + RMSE_naive_std[0:nb_plot], color="tab:blue", alpha=0.20)
    # Graph
    plt.plot(range(nb_plot), RMSE_graph_mean[0:nb_plot], linestyle='-', marker='^', markersize=2, label='Meta')
    plt.fill_between(range(nb_plot), RMSE_graph_mean[0:nb_plot] - RMSE_graph_std[0:nb_plot],
                     RMSE_graph_mean[0:nb_plot] + RMSE_graph_std[0:nb_plot], color="orange", alpha=0.20)
    # Hybrid
    plt.plot(range(nb_plot), RMSE_hybrid_mean[0:nb_plot], linestyle='-', marker='^', markersize=2, label='Hybrid')
    plt.fill_between(range(nb_plot), RMSE_hybrid_mean[0:nb_plot] - RMSE_hybrid_std[0:nb_plot],
                     RMSE_hybrid_mean[0:nb_plot] + RMSE_hybrid_std[0:nb_plot], color="green", alpha=0.20)
    #
    plt.xlabel('Training dataset size')
    plt.ylabel('RMSE test')
    #plt.yscale("log")
    #plt.ylim((12.5, 42.5))
    #plt.ylim((7.5, 50))
    plt.legend(loc="upper right")
    #plt.savefig("instance5_IDW.pdf", bbox_inches='tight', format='pdf')
    plt.show()