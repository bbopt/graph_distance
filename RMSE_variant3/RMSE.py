import sys
import argparse
import PyNomad

from utils.import_data import load_data_optimizers, prep_naive_data, hybrid_data_optimizers
from utils.problems_instances_setup import variant_size_setup
from problems_variant3 import variant3_bb_wrapper


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, required=True, choices=["MLP", "CNN"])
    parser.add_argument("--seed_setup", type=int, required=True)
    parser.add_argument("--budget_per_param", type=int, required=True)
    args = parser.parse_args()

    architecture = args.architecture
    seed_setup = args.seed_setup
    budget_per_param = args.budget_per_param

    approaches = ["naive", "graph", "hybrid"]
    # models = ["IDW", "KNN"]
    models = ["IDW"]

    # Setup for variant 3
    optimizers = ["ASGD", "ADAM"]
    nb_sub = 4
    nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 2}  # number are to determine nb pts in a subproblems
    nb_max_var = 11  # 12 - 1, with u3 removed
    nb_dec_var = 7
    var_inc_sub = {("ASGD", 1): [2, 4, 5, 6, 7], ("ASGD", 2): [2, 3, 4, 5, 6, 7],
                   ("ADAM", 1): [2, 4, 8, 9, 10], ("ADAM", 2): [2, 3, 4, 8, 9, 10]}
    nb_var_sub = {("ASGD", 1): 5, ("ASGD", 2): 6, ("ADAM", 1): 5, ("ADAM", 2): 6}
    var_sub_hybrid = {"ASGD": [1, 2, 3, 4, 5, 6, 7],
                      "ADAM": [1, 2, 3, 4, 8, 9, 10]}
    variant = "variant3"

    # Setup for size amongst 1=verysmall, 2=small, 3=medium and 4=large
    for size in [1, 2, 3, 4]:
        nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, size)

        for approach in approaches:
            for model in models:
                # Status
                print(approach, "with", model, "for size", size)

                # Output log file name
                log_file = "log_" + variant + "_" + "size" + str(size) + "_" + approach + "_" + model + "_" + architecture + ".txt"

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

                #budget = 200 * nb_params
                budget = budget_per_param * nb_params
                budget_LHS = int(budget * 0.33)
                params = ["BB_OUTPUT_TYPE OBJ", "DISPLAY_DEGREE 2",
                          "DISPLAY_ALL_EVAL false",
                          "DISPLAY_STATS BBE OBJ",
                          "STATS_FILE " + log_file + " BBE OBJ SOL",
                          "MAX_BB_EVAL " + str(budget),
                          "LH_SEARCH " + str(budget_LHS) + " 0",
                          "SEED " + str(seed_setup)]

                # Data file name
                data_file = "data_" + variant + "_size" + str(size) + "_" + architecture + ".xlsx"

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
                bb = variant3_bb_wrapper(X_train, y_train, X_valid, y_valid, nb_max_var, nb_var_sub, approach, model)

                if approach == "graph":
                    if model == "IDW":

                        # Create blackbox for Pynomad
                        def IDW_graph_bb_nomad(x):
                            try:
                                # 19 total
                                # 7 bounds : theta_u2, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                                # 1 cat : o
                                # 11 variables : o, l, u1, u2, lr, hp11, hp12, hp13, hp21, hp22, hp23
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                                 # bds
                                                 x.get_coord(5), x.get_coord(6),  # bds
                                                 x.get_coord(7),  # cat
                                                 x.get_coord(8), x.get_coord(9), x.get_coord(10), x.get_coord(11),
                                                 x.get_coord(12),  # var
                                                 x.get_coord(13), x.get_coord(14), x.get_coord(15), x.get_coord(16),
                                                 x.get_coord(17),  # var
                                                 x.get_coord(18))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation

                        # Setup : 19 total
                        # 7 bounds : theta_u2, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                        # 1 cat : o
                        # 11 variables : o, l, u1, u2, lr, hp11, hp12, hp13, hp21, hp22, hp233
                        #x0 = [10, 10, 10, 10, 10, 10, 10,
                        #      10,
                        #      1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                        x0 = []
                        #lb = [10, 2.5, 0.5, 3, 0.2, 0.24995, 1.5,
                        #      0,
                        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        #ub = [1e6] * 19
                        # log
                        lb = [1, 0.4, -0.3, 0.5, -0.7, -0.6, 0.15,
                              -30,  # o
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        ub = [30] * (7 + 1) + [1] * 11

                        result = PyNomad.optimize(IDW_graph_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']

                    elif model == "KNN":

                        # Create blackbox for Pynomad
                        def KNN_graph_bb_nomad(x):
                            try:
                                # 20 total
                                # 7 bounds : theta_u2, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                                # 1 cat : o
                                # 11 variables : o, l, u1, u2, lr, hp11, hp12, hp13, hp21, hp22, hp23
                                # 1 K
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                                 # bds
                                                 x.get_coord(5), x.get_coord(6),  # bds
                                                 x.get_coord(7),  # cat
                                                 x.get_coord(8), x.get_coord(9), x.get_coord(10), x.get_coord(11),
                                                 x.get_coord(12),  # var
                                                 x.get_coord(13), x.get_coord(14), x.get_coord(15), x.get_coord(16),
                                                 x.get_coord(17),  # var
                                                 x.get_coord(18),
                                                 int(x.get_coord(19)))  # K
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation


                        # Setup : 20 total
                        # 7 bounds : theta_u2, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                        # 1 cat : o
                        # 11 variables : o, l, u1, u2, lr, hp11, hp12, hp13, hp21, hp22, hp23
                        # 1 K
                        #x0 = [10, 10, 10, 10, 10, 10, 10,
                        #      10,
                        #      1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                        #      5]
                        x0 = []
                        #lb = [10, 2.5, 0.5, 3, 0.2, 0.24995, 1.5,
                        #      0,
                        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        #      1]
                        #ub = [1e6] * 19 + [100]
                        # log
                        lb = [1, 0.4, -0.3, 0.5, -0.7, -0.6, 0.15,
                              -30,  # o
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              1]
                        ub = [30] * (7 + 1) + [1] * 11 + [300]

                        input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R R R R R I)"
                        params = params + [input_type]
                        result = PyNomad.optimize(KNN_graph_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']
                        x_best[-1] = int(x_best[-1])

                elif approach == "naive":
                    if model == "IDW":

                        # Construct parametrized bb for nomad
                        def IDW_naive_bb_nomad(x):
                            try:
                                # w11, w12, w13, w14, w15,
                                # w21, w22, w23, w24, w25, w26,
                                # w31, w32, w33, w34, w35,
                                # w41, w42, w43, w44, w45, w46
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                                 x.get_coord(5), x.get_coord(6), x.get_coord(7), x.get_coord(8), x.get_coord(9),
                                                 x.get_coord(10),
                                                 x.get_coord(11), x.get_coord(12), x.get_coord(13), x.get_coord(14),
                                                 x.get_coord(15),
                                                 x.get_coord(16), x.get_coord(17), x.get_coord(18), x.get_coord(19),
                                                 x.get_coord(20), x.get_coord(21))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation


                        # Setup : 22 variables
                        # w11, w12, w13, w14, w15      = (u1, lr, lamdb, alpha, t0)
                        # w21, w22, w23, w24, w25, w26 = (u1, u2, lr, lamdb, alpha, t0)
                        # w31, w32, w33, w34, w35      = (u1, lr, beta1, beta2, eps)
                        # w41, w42, w43, w44, w45, w46 = (u1, u2, lr, beta1, beta2, eps)
                        #x0 = [0, 1, 0, 0, 0,
                        #      0, 0, 1, 0, 0, 0,
                        #      0, 1, 0, 0, 0,
                        #      0, 0, 1, 0, 0, 0]
                        x0 = []
                        lb = [0] * 22
                        ub = [1] * 22
                        result = PyNomad.optimize(IDW_naive_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']

                    elif model == "KNN":

                        # Construct parametrized bb for nomad
                        def KNN_naive_bb_nomad(x):
                            try:
                                # w11, w12, w13, w14, w15,        5
                                # w21, w22, w23, w24, w25, w26,   6
                                # w31, w32, w33, w34, w35,        5
                                # w41, w42, w43, w44, w45, w46    6
                                # K1, K2, K3, K4                  4
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                                 x.get_coord(5), x.get_coord(6), x.get_coord(7), x.get_coord(8), x.get_coord(9),
                                                 x.get_coord(10),
                                                 x.get_coord(11), x.get_coord(12), x.get_coord(13), x.get_coord(14),
                                                 x.get_coord(15),
                                                 x.get_coord(16), x.get_coord(17), x.get_coord(18), x.get_coord(19),
                                                 x.get_coord(20), x.get_coord(21),
                                                 int(x.get_coord(22)), int(x.get_coord(23)), int(x.get_coord(24)),
                                                 int(x.get_coord(25)))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation

                        # Setup : 22 variables
                        # w11, w12, w13, w14, w15      = (u1, lr, lamdb, alpha, t0)
                        # w21, w22, w23, w24, w25, w26 = (u1, u2, lr, lamdb, alpha, t0)
                        # w31, w32, w33, w34, w35      = (u1, lr, beta1, beta2, eps)
                        # w41, w42, w43, w44, w45, w46 = (u1, u2, lr, beta1, beta2, eps)
                        # K1, K2, K3, K4
                        #x0 = [0, 1, 0, 0, 0,
                        #      0, 0, 1, 0, 0, 0,
                        #      0, 1, 0, 0, 0,
                        #      0, 0, 1, 0, 0, 0,
                        #      25, 25, 25, 25]
                        x0 = []
                        lb = [0] * 22 + [1] * 4
                        ub = [1] * 22 + [50] * 4
                        input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R R R R R R R R I I I I)"
                        params = params + [input_type]
                        result = PyNomad.optimize(KNN_naive_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']
                        for i in range(4):
                            x_best[-(i+1)] = int(x_best[-(i+1)])

                elif approach == "hybrid":
                    if model == "IDW":

                        # Construct parametrized bb for nomad
                        def IDW_hybrid_bb_nomad(x):
                            try:
                                # w11, w12, w13, w14, w15, w16, w17 = (l, u1, u2, lr, lamdb, alpha, t0)
                                # w21, w22, w23, w24, w25, w26, w27 = (l, u1, u2, lr, beta1, beta2, eps)
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                       x.get_coord(5), x.get_coord(6),
                                       x.get_coord(7), x.get_coord(8), x.get_coord(9), x.get_coord(10), x.get_coord(11),
                                       x.get_coord(12), x.get_coord(13))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation


                        # Setup : 14 variables
                        # w11, w12, w13, w14, w15, w16, w17 = (l, u1, u2, lr, lamdb, alpha, t0)
                        # w21, w22, w23, w24, w25, w26, w27 = (l, u1, u2, lr, beta1, beta2, eps)
                        x0 = []
                        lb = [0] * 14
                        ub = [1] * 14
                        result = PyNomad.optimize(IDW_hybrid_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']

                    elif model == "KNN":

                        # Construct parametrized bb for nomad
                        def KNN_hybrid_bb_nomad(x):
                            try:
                                # w11, w12, w13, w14, w15, w16, w17 = (l, u1, u2, lr, lamdb, alpha, t0)
                                # w21, w22, w23, w24, w25, w26, w27 = (l, u1, u2, lr, beta1, beta2, eps)
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                       x.get_coord(5), x.get_coord(6),
                                       x.get_coord(7), x.get_coord(8), x.get_coord(9), x.get_coord(10), x.get_coord(11),
                                       x.get_coord(12), x.get_coord(13),
                                       int(x.get_coord(14)), int(x.get_coord(15)))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation


                        # Setup : 16 variables
                        # w11, w12, w13, w14, w15, w16, w17 = (l, u1, u2, lr, lamdb, alpha, t0)
                        # w21, w22, w23, w24, w25, w26, w27 = (l, u1, u2, lr, beta1, beta2, eps)
                        # K1, K2
                        x0 = []
                        lb = [0] * 14 + [1] * 2
                        ub = [1] * 14 + [50] * 2
                        input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R I I)"
                        params = params + [input_type]
                        result = PyNomad.optimize(KNN_hybrid_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']
                        for i in range(2):
                            x_best[-(i + 1)] = int(x_best[-(i + 1)])


            # Solution is found
            # Validation final value
            print(bb(*x_best))

            # Test
            bb_test = variant3_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub, approach, model)
            print(bb_test(*x_best))

    # Old
    # Best solutions instance 3
    #x_best_graph_IDW = [30.1889, 2.50003, 999.999, 242.704, 998.835, 1000, 999.999, 884.139, 0, 18.3991, 0, 0, 999.998, 1.33758e-07, 0.202883, 0.00711024, 0.000318189, 23.8456, 7.33904e-06]
    #x_best_graph_KNN = [900.009, 202.222, 79.6939, 43.1, 0.74478, 0.30695, 570.879, 1.5827, 999.98, 690.908, 2.25, 0.4295, 922.115, 907.74, 897.772, 5.383, 0.0082, 400.771, 24.01, 2]
    #x_best_naive_IDW = [2.61066e-05, 2.71553, 0.000484097, 0.000229857, 9.24545e-08, 0, 0, 2.65616, 1.24037e-07, 3.06573e-07, 0, 92.154, 1.39001, 1.17084e-07, 5.74629e-05, 5.95153e-08, 0, 1.38136e-07, 0.47074, 9.74467e-07, 1.26138e-06, 7.09046e-08]
    #x_best_naive_KNN = [0.00217033, 400.252, 1.68182e-06, 3.90824e-05, 9.21491e-06, 7.7606e-06, 7.01465e-07, 0.624878, 0.000917052, 4.38702e-05, 8.43823e-05, 903.918, 10.3381, 103.512, 93.5844, 196.627, 0.000197105, 0.000615527, 0.400975, 0.00732583, 0.00401179, 7.70802e-06, 8, 1, 7, 6]