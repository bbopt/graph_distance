import sys
import datetime
import random
import numpy as np
import math

from matplotlib import pyplot as plt

from utils.import_data import load_data_fix_optimizer, prep_naive_data
from utils.draw import draw_subpbs_candidates_variant1_2
from utils.problems_instances_setup import scalar_div_list, variant_size_setup
from problems_variant2 import variant2_bb_wrapper


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
    architecture = "CNN"
    variant = "variant2"
    seed = 0
    size = 3

    # Setup
    (nb_pts, nb_test_pts, nb_valid_pts) = variant_size_setup(variant, size)
    nb_pts_train = sum(nb_test_pts + nb_valid_pts) - 3

    nb_sub = 3
    nb_max_var = 8
    nb_var_sub = [5, 6, 7]
    var_inc_sub = [[1, 4, 5, 6, 7], [1, 2, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]  # included variables in each subproblem

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
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_fix_optimizer(data_file, nb_sub,
                                                                                 nb_max_var, nb_pts, nb_test_pts,
                                                                                 nb_valid_pts)

    # Prepare data for naive approach
    X_train_naive, y_train_naive, X_valid_naive, y_valid_naive, X_test_naive, y_test_naive = \
        prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, variant)

    if load_RMSE_dataset:
        (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std, RMSE_hybrid_mean, RMSE_hybrid_std) = \
            np.loadtxt(
                "selected_parameters_" + variant + "_size" + str(size) + "_" + model + "_" + architecture + ".txt")

    else:
        RMSE_naive = np.zeros((nb_pts_train, nb_rand_trials))
        RMSE_graph = np.zeros((nb_pts_train, nb_rand_trials))
        RMSE_hybrid = np.zeros((nb_pts_train, nb_rand_trials))

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

                # Create partial data set for graph approach
                idx_train_partial_graph = idx_train_subs_partial["sub1"] + idx_train_subs_partial["sub2"] + \
                                          idx_train_subs_partial["sub3"]
                X_train_graph_partial = X_train[idx_train_partial_graph, :]
                y_train_graph_partial = y_train[idx_train_partial_graph]

                # Create partial data set for hybrid approach : variant 2, it's the same as graph
                X_train_hybrid_partial = X_train_graph_partial
                y_train_hybrid_partial = y_train_graph_partial

                if model == "IDW":
                    # Create parametrized bb for naive approach
                    IDW_naive_bb = variant2_bb_wrapper(X_train_naive_partial, y_train_naive_partial,
                                                             X_test_naive, y_test_naive, nb_max_var, nb_var_sub, "naive", "IDW")
                    RMSE_naive[i, j] = IDW_naive_bb(*x_best_naive_IDW)

                    # Create parametrized bb for graph approach
                    IDW_graph_bb = variant2_bb_wrapper(X_train_graph_partial, y_train_graph_partial,
                                                             X_test, y_test, nb_max_var, nb_var_sub, "graph", "IDW")
                    RMSE_graph[i, j] = IDW_graph_bb(*x_best_graph_IDW)

                    # For variant 2, hybrid has the same training and test sets, hence we can reuse X_test and y_test
                    IDW_hybrid_bb_test = variant2_bb_wrapper(X_train_hybrid_partial, y_train_hybrid_partial,
                                                             X_test, y_test, nb_max_var, nb_var_sub, "hybrid", "IDW")
                    RMSE_hybrid[i, j] = IDW_hybrid_bb_test(*x_best_hybrid_IDW)

                elif model == "KNN":
                    continue

                # Determine next draw is amongst which subpbs and check if while-loop is done
                draw = draw_subpbs_candidates_variant1_2(idx_train_subs["sub1"], idx_train_subs["sub2"],
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
        RMSE_hybrid_mean = RMSE_hybrid.mean(axis=1)
        RMSE_hybrid_std = RMSE_hybrid.std(axis=1)

        # Save test in case figures have to be modified
        np.savetxt("selected_parameters_" + variant + "_size" + str(size) + "_" + model + "_" + architecture + ".txt",
                   (RMSE_graph_mean, RMSE_graph_std, RMSE_naive_mean, RMSE_naive_std, RMSE_hybrid_mean, RMSE_hybrid_std))

    # Figure setup
    fig = plt.figure(dpi=100, tight_layout=True)
    plt.rc('axes', labelsize=28)
    plt.rc('legend', fontsize=28)
    plt.rc('xtick', labelsize=26)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=26)  # fontsize of the tick label
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    nb_plot = nb_pts_train - 4
    # Sub
    plt.plot(range(nb_plot), RMSE_naive_mean[0:nb_plot], linestyle='-', marker='s', markersize=2, label='Sub')
    plt.fill_between(range(nb_plot), RMSE_naive_mean[0:nb_plot] - RMSE_naive_std[0:nb_plot],
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
    #plt.ylim((7.5, 50))
    plt.legend(loc="upper right")
    #plt.savefig("instance2_IDW.pdf", bbox_inches='tight', format='pdf')
    plt.show()