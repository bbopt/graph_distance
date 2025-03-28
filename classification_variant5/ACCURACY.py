import sys
import PyNomad
import argparse

from utils.import_data import load_data_optimizers, prep_naive_data, hybrid_data_optimizers
from utils.problems_instances_setup import variant_size_setup
from utils.label_data_files_bins import create_labeled_variant_size
from utils.seed_stats import *

from problems_variant5 import variant5_bb_wrapper

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

    variant = "variant5"
    is_CNN = False
    create_labeled_variant_size(variant, size, "MLP", N=nb_of_classes)

    # Setup for variant 5
    optimizers = ["ASGD", "ADAM"]
    nb_sub = 5
    nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 3}
    nb_max_var = 13  # add the dropout
    nb_dec_var = 8
    var_inc_sub = {("ASGD", 1): [2, 5, 6, 7, 8, 12], ("ASGD", 2): [2, 3, 5, 6, 7, 8, 12],
                   ("ADAM", 1): [2, 5, 9, 10, 11, 12], ("ADAM", 2): [2, 3, 5, 9, 10, 11, 12],
                   ("ADAM", 3): [2, 3, 4, 5, 9, 10, 11, 12]}
    nb_var_sub = {("ASGD", 1): 6, ("ASGD", 2): 7, ("ADAM", 1): 6, ("ADAM", 2): 7, ("ADAM", 3): 8}
    var_sub_hybrid = {"ASGD": [1, 2, 3, 5, 6, 7, 8, 12],
                      "ADAM": [1, 2, 3, 4, 5, 9, 10, 11, 12]}
    nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, size)

    seed_setup_list = list(range(nb_seeds))
    accuracy_validation = {(seed, approach): None for seed in seed_setup_list for approach in approaches}
    accuracy_test = {(seed, approach): None for seed in seed_setup_list for approach in approaches}
    for seed_setup in seed_setup_list:
        for approach in approaches:
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
                    nb_params = sum(nb_var_sub.values())
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
                    nb_params = sum(len(lst) for lst in var_sub_hybrid.values())
                    if model == "KNN":
                        nb_params = nb_params + 2

                budget = budget_per_param * nb_params
                #budget = 10 * nb_params
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
                X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_optimizers(data_file,
                                                                                          optimizers, nb_sub_per_optimizer,
                                                                                          nb_max_var, nb_pts, nb_test_pts,
                                                                                          nb_valid_pts,
                                                                                          seed_setup=seed_setup)

                if approach == "naive":
                    # Modifiy data for naive approach
                    X_train, y_train, X_valid, y_valid, X_test, y_test = \
                        prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, variant)

                elif approach == "hybrid":
                    X_train, y_train, X_valid, y_valid, X_test, y_test = \
                        hybrid_data_optimizers(X_train, y_train, X_valid, y_valid, X_test, y_test, var_sub_hybrid)

                # Create blackbox parametrized by the training set
                bb = variant5_bb_wrapper(X_train, y_train, X_valid, y_valid, nb_max_var, nb_var_sub, approach, model)

                if approach == "graph":

                    # Create blackbox for Pynomad
                    def KNN_graph_bb_nomad(x):
                        try:
                            # 23 total
                            # 8 bounds : theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                            # 1 cat : o
                            # 13 variables : o, l, u1, u2, u3, lr, hp11, hp12, hp13, hp21, hp22, hp23, p
                            # K
                            f = -bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3),  # bds
                                             x.get_coord(4), x.get_coord(5), x.get_coord(6), x.get_coord(7),  # bds
                                             x.get_coord(8),  # cat
                                             x.get_coord(9), x.get_coord(10), x.get_coord(11), x.get_coord(12),
                                             x.get_coord(13),  # var
                                             x.get_coord(14), x.get_coord(15), x.get_coord(16), x.get_coord(17),
                                             x.get_coord(18),  # var
                                             x.get_coord(19), x.get_coord(20), x.get_coord(21),
                                             int(x.get_coord(22)))  # var
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # 23 total
                    # 8 bounds : theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                    # 1 cat : o
                    # 13 variables : o, l, u1, u2, u3, lr, hp11, hp12, hp13, hp21, hp22, hp23, p
                    # K
                    #x0 = [10, 10, 10, 10, 10, 10, 10, 10,
                    #      10,
                    #      1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    #      5]
                    x0 = []
                    #lb = [10, 10, 2.5, 0.5, 3, 0.2, 0.24995, 1.5,
                    #      0,
                    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #      1]
                    # log
                    lb = [1, 1, 0.4, -0.3, 0.5, -0.7, -0.6, 0.15,
                          -30,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1]

                    #ub = [1e6] * 22 + [100]
                    ub = [30] * (8+1) + [1] * 13 + [300]

                    input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R R R R R R R R I)"
                    params = params + [input_type]
                    result = PyNomad.optimize(KNN_graph_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    x_best[-1] = int(x_best[-1])

                elif approach == "naive":

                    # Construct parametrized bb for nomad
                    def KNN_naive_bb_nomad(x):
                        try:
                            # Total : 39
                            # w11, w12, w13, w14, w15, w16            # 5
                            # w21, w22, w23, w24, w25, w26, w27       # 6
                            # w31, w32, w33, w34, w35, w36            # 5
                            # w41, w42, w43, w44, w45, w46, w47       # 6
                            # w51, w52, w53, w54, w55, w56, w57, w58  # 7
                            # K1, K2, K3, K4, K5                      # 5
                            f = -bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                             x.get_coord(5),
                                             x.get_coord(6), x.get_coord(7), x.get_coord(8), x.get_coord(9),
                                             x.get_coord(10), x.get_coord(11), x.get_coord(12),
                                             x.get_coord(13), x.get_coord(14), x.get_coord(15), x.get_coord(16),
                                             x.get_coord(17), x.get_coord(18),
                                             x.get_coord(19), x.get_coord(20), x.get_coord(21), x.get_coord(22),
                                             x.get_coord(23), x.get_coord(24), x.get_coord(25),
                                             x.get_coord(26), x.get_coord(27), x.get_coord(28), x.get_coord(29),
                                             x.get_coord(30), x.get_coord(31), x.get_coord(32), x.get_coord(33),
                                             int(x.get_coord(34)), int(x.get_coord(35)), int(x.get_coord(36)),
                                             int(x.get_coord(37)), int(x.get_coord(38)))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # Setup : 39 variables
                    # w11, w12, w13, w14, w15, w16           = (u1, lr, lamdb, alpha, t0, p)
                    # w21, w22, w23, w24, w25, w26, w27      = (u1, u2, lr, lamdb, alpha, t0, p)
                    # w31, w32, w33, w34, w35, w36           = (u1, lr, beta1, beta2, eps, p)
                    # w41, w42, w43, w44, w45, w46, w47      = (u1, u2, lr, beta1, beta2, eps, p)
                    # w51, w52, w53, w54, w55, w56, w57, w58 = (u1, u2, u3, lr, beta1, beta2, eps, p)
                    # K1, K2, K3, K4, K5
                    #x0 = [0, 1, 0, 0, 0, 0,
                    #      0, 0, 1, 0, 0, 0, 0,
                    #      0, 1, 0, 0, 0, 0,
                    #      0, 0, 1, 0, 0, 0, 0,
                    #      0, 0, 0, 1, 0, 0, 0, 0,
                    #      5, 5, 5, 5, 5]
                    x0 = []
                    lb = [0] * 34 + [1] * 5
                    ub = [1e6] * 34 + [50] * 5
                    input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R R R R R R R R R R R R R R R R R R R R I I I I I)"
                    params = params + [input_type]
                    result = PyNomad.optimize(KNN_naive_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    for i in range(5):
                        x_best[-(i+1)] = int(x_best[-(i+1)])

                elif approach == "hybrid":

                    # Construct parametrized bb for nomad
                    def KNN_hybrid_bb_nomad(x):
                        try:
                            # w11, w12, w13, w14, w15, w16, w17, w18      = (l, u1, u2, lr, lamdb, alpha, t0, p)
                            # w21, w22, w23, w24, w25, w26, w27, w28, w29 = (l, u1, u2, u3, lr, beta1, beta2, eps, p)
                            f = -bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                   x.get_coord(5), x.get_coord(6), x.get_coord(7),
                                   x.get_coord(8), x.get_coord(9), x.get_coord(10), x.get_coord(11), x.get_coord(12),
                                   x.get_coord(13), x.get_coord(14), x.get_coord(15), x.get_coord(16),
                                   int(x.get_coord(17)), int(x.get_coord(18)))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation


                    # Setup : 19 variables
                    # w11, w12, w13, w14, w15, w16, w17, w18      = (l, u1, u2, lr, lamdb, alpha, t0, p)
                    # w21, w22, w23, w24, w25, w26, w27, w28, w29 = (l, u1, u2, lr, beta1, beta2, eps, p)
                    # K1, K2
                    x0 = []
                    lb = [0] * 17 + [1] * 2
                    ub = [1e6] * 17 + [50] * 2
                    input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R R R I I)"
                    params = params + [input_type]
                    result = PyNomad.optimize(KNN_hybrid_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    for i in range(2):
                        x_best[-(i + 1)] = int(x_best[-(i + 1)])

                # Validation final value
                accuracy_validation[(seed_setup, approach)] = bb(*x_best)
                print(bb(*x_best))

                # Test
                bb_test = variant5_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub,
                                              approach,
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