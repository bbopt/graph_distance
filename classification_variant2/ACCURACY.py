import sys
import os
import PyNomad
import argparse

from utils.import_data import load_data_fix_optimizer, prep_naive_data
from utils.problems_instances_setup import variant_setup, variant_size_setup
from utils.label_data_files_bins import create_labeled_variant_size
from utils.seed_stats import *

from problems_variant2 import variant2_bb_wrapper

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--budget_per_param', type=int, required=True)
    parser.add_argument('--nb_seeds', type=int, required=True)
    parser.add_argument('--nb_of_classes', type=int, required=True)
    args = parser.parse_args()

    size = args.size
    budget_per_param = args.budget_per_param
    nb_seeds = args.nb_seeds
    nb_of_classes = args.nb_of_classes

    approaches = ["naive", "graph", "hybrid"]
    models = ["KNN"]

    variant = "variant2"
    is_CNN = False
    create_labeled_variant_size(variant, size, "MLP", N=nb_of_classes)

    # Setup for variant 2
    nb_sub = 3
    nb_max_var = 8
    nb_dec_var = 2
    nb_var_sub = [5, 6, 7]
    var_inc_sub = [[1, 4, 5, 6, 7], [1, 2, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]  # included variables in each subproblem
    nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, size)

    seed_setup_list = list(range(nb_seeds))
    accuracy_validation = {(seed, approach): None for seed in seed_setup_list for approach in approaches}
    accuracy_test = {(seed, approach): None for seed in seed_setup_list for approach in approaches}
    for seed_setup in seed_setup_list:
        for approach in approaches:
            print(approach, "on seed", seed_setup)

            for model in models:

                # Output log file name
                if is_CNN:
                    log_file = "log_" + variant + "_" + approach + "_" + model + "_CNN_classification" + ".txt"
                else:
                    log_file = "log_" + variant + "_" + approach + "_" + model + "_MLP_classification" + ".txt"

                # Parameters for NOMAD: parameters must be reloaded
                nb_params = 0
                if approach == "naive":
                    # Number of parameters is the sum of the number of variables per subproblem
                    nb_params = sum(nb_var_sub)
                    # Add one neighborhood parameter per subprobelm
                    if model == "KNN":
                        nb_params = nb_params + nb_sub

                elif approach == "graph":
                    nb_params = nb_max_var + nb_dec_var
                    # Add a single neighborhood parameter
                    if model == "KNN":
                        nb_params = nb_params + 1

                # Hardcoded by variant
                elif approach == "hybrid":
                    nb_params = nb_max_var
                    if model == "KNN":
                        nb_params = nb_params + 1

                budget = budget_per_param * nb_params
                budget_LHS = int(budget * 0.33)
                params = ["BB_OUTPUT_TYPE OBJ", "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ",
                          "STATS_FILE " + log_file + " BBE OBJ SOL",
                          "MAX_BB_EVAL " + str(budget),
                          "LH_SEARCH " + str(budget_LHS) + " 0"]

                # Data file name
                if is_CNN:
                    data_file = "labeled_data_" + variant + "_size" + str(size) + "_CNN" + ".xlsx"
                else:
                    data_file = "labeled_data_" + variant + "_size" + str(size) + "_MLP" + ".xlsx"

                # Load data
                X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_fix_optimizer(data_file,
                                                                                             nb_sub, nb_max_var, nb_pts,
                                                                                             nb_test_pts, nb_valid_pts,
                                                                                             seed_setup=seed_setup)

                if approach == "naive":
                    # Modifiy data for naive approach
                    X_train, y_train, X_valid, y_valid, X_test, y_test = \
                        prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, variant)

                # Create blackbox parametrized by the training set
                bb = variant2_bb_wrapper(X_train, y_train, X_valid, y_valid, nb_max_var, nb_var_sub, approach, model)

                if approach == "graph":

                    # Create blackbox for Pynomad
                    def KNN_graph_bb_nomad(x):
                        try:
                            # BB : (theta_u2, theta_u3, w1, w2, w3, w4, w5, w6, w7, w8, K); 2 bounds, 8 variables and K
                            f = -bb(x.get_coord(0), x.get_coord(1),
                                             x.get_coord(2), x.get_coord(3), x.get_coord(4), x.get_coord(5),
                                             x.get_coord(6), x.get_coord(7), x.get_coord(8), x.get_coord(9),
                                             int(x.get_coord(10)))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # Setup for nomad
                    # (theta_u2, theta_u3, w1=l, w2=u1, w3=u2, w4=u3, w5=lr, w6=lambd, w7=alpha, w8=t0, K), 11 variables
                    #x0 = [10, 10, 1, 0, 0, 0, 1, 0, 0, 0, 5]
                    x0 = []
                    #lb = [10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    lb = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # log
                    #ub = [1e6]*10 + [100]
                    ub = [30]*2 + [1]*8 + [100]
                    input_type = "BB_INPUT_TYPE (R R R R R R R R R R I)"
                    params = params + [input_type]
                    result = PyNomad.optimize(KNN_graph_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    x_best[-1] = int(x_best[-1])

                elif approach == "naive":

                    # Construct parametrized bb for nomad
                    def KNN_naive_bb_nomad(x):
                        try:
                            # (w11, w12, w13, w14, w15,
                            #                      w21, w22, w23, w24, w25, w26,
                            #                      w31, w32, w33, w34, w35, w36, w37,
                            #                      K1, K2, K3)
                            f = -bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                             x.get_coord(5), x.get_coord(6), x.get_coord(7), x.get_coord(8), x.get_coord(9),
                                             x.get_coord(10),
                                             x.get_coord(11), x.get_coord(12), x.get_coord(13), x.get_coord(14),
                                             x.get_coord(15), x.get_coord(16), x.get_coord(17),
                                             int(x.get_coord(18)), int(x.get_coord(19)), int(x.get_coord(20)))

                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # Setup for nomad
                    # (w11, w12, w13, w14, w15; (u1, lr, lamdb, alpha, t0)
                    #                      w21, w22, w23, w24, w25, w26; (u1, u2, lr, lamdb, alpha, t0)
                    #                      w31, w32, w33, w34, w35, w36, w37; (u1, u2, u3, lr, lamdb, alpha, t0)
                    #                      K1, K2, K3)
                    #x0 = [0, 1, 0, 0, 0,
                    #      0, 0, 1, 0, 0, 0,
                    #      0, 0, 0, 1, 0, 0, 0,
                    #      25, 25, 25]
                    x0 = []
                    lb = [0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0,
                          1, 1, 1]
                    ub = [1]*18 + [int(x/2) for x in nb_pts]
                    input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R R R R I I I)"
                    params = params + [input_type]
                    result = PyNomad.optimize(KNN_naive_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    for i in range(3):
                        x_best[-(i+1)] = int(x_best[-(i+1)])


                # Construct the optimization problem w.r.t to the approach and the model
                if approach == "hybrid":

                    # Create blackbox for Pynomad
                    def KNN_hybrid_bb_nomad(x):
                        try:
                            # BB : (w1, w2, w3, w4, w5, w6, w7, w8, K)
                            f = -bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3),
                                   x.get_coord(4), x.get_coord(5), x.get_coord(6), x.get_coord(7),
                                   int(x.get_coord(8)))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation


                    # Setup for nomad
                    # (w1=l, w2=u1, w3=u2, w4=u3, w5=lr, K)
                    x0 = []
                    lb = [0, 0, 0, 0, 0, 0, 0, 0, 1]
                    ub = [1] * 8 + [50]
                    input_type = "BB_INPUT_TYPE (R R R R R R R R I)"
                    params = params + [input_type]
                    result = PyNomad.optimize(KNN_hybrid_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    x_best[-1] = int(x_best[-1])

                # Validation final value
                accuracy_validation[(seed_setup, approach)] = bb(*x_best)
                print(bb(*x_best))

                # Test
                bb_test = variant2_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub, approach,
                                              model)
                accuracy_test[(seed_setup, approach)] = bb_test(*x_best)
                print(bb_test(*x_best))


    # Number of seeds
    num_seeds = len(seed_setup_list)

    # Compute averages and standard deviations
    avg_validation = compute_average(accuracy_validation)
    std_validation = compute_std(accuracy_validation, avg_validation)

    avg_test = compute_average(accuracy_test)
    std_test = compute_std(accuracy_test, avg_test)

    # Display results
    print("Validation Averages:", avg_validation)
    print("Validation Standard Deviations:", std_validation)
    print("Test Averages:", avg_test)
    print("Test Standard Deviations:", std_test)

    # Generate filename dynamically
    filename = f"results_avg_std_size{size}.txt"

    # Save results to the dynamically named text file
    with open(filename, "w") as f:
        f.write("Validation Averages: " + str(avg_validation) + "\n")
        f.write("Validation Standard Deviations: " + str(std_validation) + "\n")
        f.write("Test Averages: " + str(avg_test) + "\n")
        f.write("Test Standard Deviations: " + str(std_test) + "\n")