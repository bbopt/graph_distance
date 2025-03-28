import sys
import argparse
import PyNomad

from utils.import_data import load_data_optimizers, prep_naive_data, hybrid_data_optimizers
from utils.problems_instances_setup import variant_size_setup
from problems_variant4 import variant4_bb_wrapper


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

    # Setup for variant 4
    optimizers = ["ASGD", "ADAM"]
    nb_sub = 5
    nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 3}
    nb_max_var = 12
    nb_dec_var = 8
    var_inc_sub = {("ASGD", 1): [2, 5, 6, 7, 8], ("ASGD", 2): [2, 3, 5, 6, 7, 8],
                   ("ADAM", 1): [2, 5, 9, 10, 11], ("ADAM", 2): [2, 3, 5, 9, 10, 11],
                   ("ADAM", 3): [2, 3, 4, 5, 9, 10, 11]}
    nb_var_sub = {("ASGD", 1): 5, ("ASGD", 2): 6, ("ADAM", 1): 5, ("ADAM", 2): 6, ("ADAM", 3): 7}
    var_sub_hybrid = {"ASGD": [1, 2, 3, 5, 6, 7, 8],
                      "ADAM": [1, 2, 3, 4, 5, 9, 10, 11]}
    variant = "variant4"

    # Setup for size amongst 1=verysmall, 2=small, 3=medium and 4=large
    for size in [1,2,3,4]:
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
                bb = variant4_bb_wrapper(X_train, y_train, X_valid, y_valid, nb_max_var, nb_var_sub, approach, model)

                if approach == "graph":
                    if model == "IDW":

                        # Create blackbox for Pynomad
                        def IDW_graph_bb_nomad(x):
                            try:
                                # 21 total
                                # 8 bounds : theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                                # 1 cat : o
                                # 12 variables : o, l, u1, u2, u3, lr, hp11, hp12, hp13, hp21, hp22, hp23
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3),  # bds
                                                 x.get_coord(4), x.get_coord(5), x.get_coord(6), x.get_coord(7),  # bds
                                                 x.get_coord(8),  # cat
                                                 x.get_coord(9), x.get_coord(10), x.get_coord(11), x.get_coord(12),
                                                 x.get_coord(13),  # var
                                                 x.get_coord(14), x.get_coord(15), x.get_coord(16), x.get_coord(17),
                                                 x.get_coord(18),  # var
                                                 x.get_coord(19), x.get_coord(20))  # var
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation

                        # 21 total
                        # 8 bounds : theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                        # 1 cat : o
                        # 12 variables : o, l, u1, u2, u3, lr, hp11, hp12, hp13, hp21, hp22, hp23
                        #x0 = [10, 10, 10, 10, 10, 10, 10, 10,
                        #      10,
                        #      1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                        x0 = []
                        #lb = [10, 10, 2.5, 0.5, 3, 0.2, 0.24995, 1.5,
                        #      0,
                        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        #ub = [1e6] * 21
                        lb = [1, 1, 0.4, -0.3, 0.5, -0.7, -0.6, 0.15,
                              -30,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        ub = [30] * (8 + 1) + [1] * 12
                        result = PyNomad.optimize(IDW_graph_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']

                    elif model == "KNN":

                        # Create blackbox for Pynomad
                        def KNN_graph_bb_nomad(x):
                            try:
                                # 22 total
                                # 8 bounds : theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                                # 1 cat : o
                                # 12 variables : o, l, u1, u2, u3, lr, hp11, hp12, hp13, hp21, hp22, hp23
                                # K
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3),  # bds
                                                 x.get_coord(4), x.get_coord(5), x.get_coord(6), x.get_coord(7),  # bds
                                                 x.get_coord(8),  # cat
                                                 x.get_coord(9), x.get_coord(10), x.get_coord(11), x.get_coord(12),  # var
                                                 x.get_coord(13),x.get_coord(14), x.get_coord(15), x.get_coord(16),  # var
                                                 x.get_coord(17), x.get_coord(18), x.get_coord(19), x.get_coord(20), # var
                                                 int(x.get_coord(21)))  # K-neighbors
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation

                        # 22 total
                        # 8 bounds : theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                        # 1 cat : o
                        # 12 variables : o, l, u1, u2, u3, lr, hp11, hp12, hp13, hp21, hp22, hp23
                        # K
                        #x0 = [10, 10, 10, 10, 10, 10, 10, 10,
                        #      10,
                        #      1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                        #      5]
                        x0 = []
                        #lb = [10, 10, 2.5, 0.5, 3, 0.2, 0.24995, 1.5,
                        #      0,
                        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        #      1]
                        #ub = [1e6] * 21 + [100]
                        lb = [1, 1, 0.4, -0.3, 0.5, -0.7, -0.6, 0.15,
                              -30,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              1]
                        ub = [30] * (8 + 1) + [1] * 12 + [300]
                        input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R R R R R R R I)"
                        params = params + [input_type]
                        result = PyNomad.optimize(KNN_graph_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']
                        x_best[-1] = int(x_best[-1])

                elif approach == "naive":
                    if model == "IDW":

                        # Construct parametrized bb for nomad
                        def IDW_naive_bb_nomad(x):
                            try:
                                # Setup : 29 variables
                                # w11, w12, w13, w14, w15,
                                # w21, w22, w23, w24, w25, w26,
                                # w31, w32, w33, w34, w35,
                                # w41, w42, w43, w44, w45, w46,
                                # w51, w52, w53, w54, w55, w56, w57
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                                 x.get_coord(5), x.get_coord(6), x.get_coord(7), x.get_coord(8), x.get_coord(9),
                                                 x.get_coord(10),
                                                 x.get_coord(11), x.get_coord(12), x.get_coord(13), x.get_coord(14),
                                                 x.get_coord(15),
                                                 x.get_coord(16), x.get_coord(17), x.get_coord(18), x.get_coord(19),
                                                 x.get_coord(20), x.get_coord(21),
                                                 x.get_coord(22), x.get_coord(23), x.get_coord(24), x.get_coord(25),
                                                 x.get_coord(26), x.get_coord(27), x.get_coord(28))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation

                        # Setup : 29 variables
                        # w11, w12, w13, w14, w15           = (u1, lr, lamdb, alpha, t0)
                        # w21, w22, w23, w24, w25, w26      = (u1, u2, lr, lamdb, alpha, t0)
                        # w31, w32, w33, w34, w35           = (u1, lr, beta1, beta2, eps)
                        # w41, w42, w43, w44, w45, w46      = (u1, u2, lr, beta1, beta2, eps)
                        # w51, w52, w53, w54, w55, w56, w57 = (u1, u2, u3, lr, beta1, beta2, eps)
                        #x0 = [0, 1, 0, 0, 0,
                        #      0, 0, 1, 0, 0, 0,
                        #      0, 1, 0, 0, 0,
                        #      0, 0, 1, 0, 0, 0,
                        #      0, 0, 0, 1, 0, 0, 0]
                        x0 = []
                        lb = [0] * 29
                        ub = [1] * 29
                        result = PyNomad.optimize(IDW_naive_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']

                    elif model == "KNN":

                        # Construct parametrized bb for nomad
                        def KNN_naive_bb_nomad(x):
                            try:
                                # Total : 34
                                # w11, w12, w13, w14, w15,           # 5
                                # w21, w22, w23, w24, w25, w26,      # 6
                                # w31, w32, w33, w34, w35,           # 5
                                # w41, w42, w43, w44, w45, w46       # 6
                                # w51, w52, w53, w54, w55, w56, w57  # 7
                                # K1, K2, K3, K4, K5                 # 5
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                                 x.get_coord(5), x.get_coord(6), x.get_coord(7), x.get_coord(8), x.get_coord(9),
                                                 x.get_coord(10),
                                                 x.get_coord(11), x.get_coord(12), x.get_coord(13), x.get_coord(14),
                                                 x.get_coord(15),
                                                 x.get_coord(16), x.get_coord(17), x.get_coord(18), x.get_coord(19),
                                                 x.get_coord(20), x.get_coord(21),
                                                 x.get_coord(22), x.get_coord(23), x.get_coord(24), x.get_coord(25),
                                                 x.get_coord(26), x.get_coord(27), x.get_coord(28),
                                                 int(x.get_coord(29)), int(x.get_coord(30)), int(x.get_coord(31)),
                                                 int(x.get_coord(32)), int(x.get_coord(33)))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation


                        # Setup : 34 variables
                        # w11, w12, w13, w14, w15           = (u1, lr, lamdb, alpha, t0)
                        # w21, w22, w23, w24, w25, w26      = (u1, u2, lr, lamdb, alpha, t0)
                        # w31, w32, w33, w34, w35           = (u1, lr, beta1, beta2, eps)
                        # w41, w42, w43, w44, w45, w46      = (u1, u2, lr, beta1, beta2, eps)
                        # w51, w52, w53, w54, w55, w56, w57 = (u1, u2, u3, lr, beta1, beta2, eps)
                        # K1, K2, K3, K4, K5
                        #x0 = [0, 1, 0, 0, 0,
                        #      0, 0, 1, 0, 0, 0,
                        #      0, 1, 0, 0, 0,
                        #      0, 0, 1, 0, 0, 0,
                        #      0, 0, 0, 1, 0, 0, 0,
                        #      5, 5, 5, 5, 5]
                        x0 = []
                        lb = [0] * 29 + [1] * 5
                        ub = [1] * 29 + [50] * 5
                        input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R R R R R R R R R R R R R R R I I I I I)"
                        params = params + [input_type]
                        result = PyNomad.optimize(KNN_naive_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']
                        for i in range(5):
                            x_best[-(i+1)] = int(x_best[-(i+1)])

                elif approach == "hybrid":
                    if model == "IDW":

                        # Construct parametrized bb for nomad
                        def IDW_hybrid_bb_nomad(x):
                            try:
                                # w11, w12, w13, w14, w15, w16, w17 = (l, u1, u2, lr, lamdb, alpha, t0)
                                # w21, w22, w23, w24, w25, w26, w27, w28 = (l, u1, u2, u3, lr, beta1, beta2, eps)
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                       x.get_coord(5), x.get_coord(6),
                                       x.get_coord(7), x.get_coord(8), x.get_coord(9), x.get_coord(10), x.get_coord(11),
                                       x.get_coord(12), x.get_coord(13), x.get_coord(14))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation


                        # Setup : 15 variables
                        # w11, w12, w13, w14, w15, w16, w17      = (l, u1, u2, lr, lamdb, alpha, t0)
                        # w21, w22, w23, w24, w25, w26, w27, w28 = (l, u1, u2, u3, lr, beta1, beta2, eps)
                        x0 = []
                        lb = [0] * 15
                        ub = [1] * 15
                        result = PyNomad.optimize(IDW_hybrid_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']

                    elif model == "KNN":

                        # Construct parametrized bb for nomad
                        def KNN_hybrid_bb_nomad(x):
                            try:
                                # w11, w12, w13, w14, w15, w16, w17      = (l, u1, u2, lr, lamdb, alpha, t0)
                                # w21, w22, w23, w24, w25, w26, w27, w28 = (l, u1, u2, u3, lr, beta1, beta2, eps)
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                       x.get_coord(5), x.get_coord(6),
                                       x.get_coord(7), x.get_coord(8), x.get_coord(9), x.get_coord(10), x.get_coord(11),
                                       x.get_coord(12), x.get_coord(13), x.get_coord(14),
                                       int(x.get_coord(15)), int(x.get_coord(16)))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation


                        # Setup : 17 variables
                        # w11, w12, w13, w14, w15, w16, w17      = (l, u1, u2, lr, lamdb, alpha, t0)
                        # w21, w22, w23, w24, w25, w26, w27, w28 = (l, u1, u2, lr, beta1, beta2, eps)
                        # K1, K2
                        x0 = []
                        lb = [0] * 15 + [1] * 2
                        ub = [1] * 15 + [50] * 2
                        input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R I I)"
                        params = params + [input_type]
                        result = PyNomad.optimize(KNN_hybrid_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']
                        for i in range(2):
                            x_best[-(i + 1)] = int(x_best[-(i + 1)])

                # Solution is found
                # Validation final value
                print(bb(*x_best))

                # Test
                bb_test = variant4_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub, approach, model)
                print(bb_test(*x_best))

    # Old
    # Best solutions instance 4

    #x_best_graph_IDW = [960.544, 10, 916.86, 1000, 3.06424, 999.888, 1000, 1.5, 1.25636e-05, 721.373, 0.775113, 0.000210357, 1.27347e-06, 2.04072e-06, 999.999, 3.0895e-07, 0.360407, 0.00299164, 0.00569467, 37.9814, 1.7398e-05]
    #x_best_graph_KNN = [29.9627, 10.2, 10.9385, 18.9958, 3.26902, 6.72139, 9.16873, 1.50077, 306.161, 9.04198, 12.765, 0.186892, 4.837e-05, 0.0193345, 95.0474, 7.70068, 0.979647, 0.0536753, 102.423, 281.719, 2.52636, 5]
    #x_best_naive_IDW = [6.24743e-06, 1.81619, 0.000340702, 5.776e-06, 0.000121916, 0, 7.12439e-09, 0.844217, 1.00119e-07, 1.15241e-05, 3.06912e-08, 97.0525, 1.37657, 8.13872e-07, 4.28585e-06, 3.73813e-07, 0, 2.76438e-07, 1.32248, 1.09992e-07, 2.7641e-07, 9.1327e-08, 6.33001e-06, 1.5348e-07, 0, 1.15008, 4.35709e-08, 1.21737e-06, 0]
    #x_best_naive_KNN = [0.00330341, 1.32845, 0.00439633, 0.0011544, 0, 0.000429643, 3.15247e-05, 1.08098, 3.59739, 0.00313888, 0.000294585, 9.17391, 1.28799, 0.000114471, 3.24964e-05, 0.00460077, 0.000331102, 0.000300558, 0.956119, 0.00150471, 1.05646e-06, 0.00269876, 0.00191142, 4.72403e-05, 2.73454e-06, 0.73586, 0.0513265, 0.014077, 0.000160786, 7, 1, 25, 5, 7]