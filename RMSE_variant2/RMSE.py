import sys
import argparse
import PyNomad

from utils.import_data import load_data_fix_optimizer, prep_naive_data
from utils.problems_instances_setup import variant_size_setup
from problems_variant2 import variant2_bb_wrapper


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

    # Setup for variant 2
    nb_sub = 3
    nb_max_var = 8
    nb_dec_var = 2
    nb_var_sub = [5, 6, 7]
    var_inc_sub = [[1, 4, 5, 6, 7], [1, 2, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]  # included variables in each subproblem
    variant = "variant2"

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
                    if model == "IDW":

                        # Create blackbox for Pynomad
                        def IDW_graph_bb_nomad(x):
                            try:
                                # BB : theta_u2, theta_u3, w1, w2, w3, w4, w5, w6, w7, w8 ; 2 bounds and 8 variables
                                f = bb(x.get_coord(0), x.get_coord(1),
                                                 x.get_coord(2), x.get_coord(3), x.get_coord(4), x.get_coord(5),
                                                 x.get_coord(6), x.get_coord(7), x.get_coord(8), x.get_coord(9))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation


                        # Setup for nomad
                        # (theta_u2, theta_u3, w1=l, w2=u1, w3=u2, w4=u3, w5=lr, w6=lambd, w7=alpha, w8=t0), 10 variables
                        #x0 = [10, 10, 1, 0, 0, 0, 1, 0, 0, 0]
                        x0 = []
                        # lb = [10, 10, 0, 0, 0, 0, 0, 0, 0, 0]
                        lb = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # log
                        # ub = [1e6]*10
                        ub = [30] * 2 + [1] * 8
                        result = PyNomad.optimize(IDW_graph_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']

                    elif model == "KNN":

                        # Create blackbox for Pynomad
                        def KNN_graph_bb_nomad(x):
                            try:
                                # BB : (theta_u2, theta_u3, w1, w2, w3, w4, w5, w6, w7, w8, K); 2 bounds, 8 variables and K
                                f = bb(x.get_coord(0), x.get_coord(1),
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
                        # lb = [10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                        lb = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # log
                        # ub = [1e6]*10 + [100]
                        ub = [30] * 2 + [1] * 8 + [100]
                        input_type = "BB_INPUT_TYPE (R R R R R R R R R R I)"
                        params = params + [input_type]
                        result = PyNomad.optimize(KNN_graph_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']
                        x_best[-1] = int(x_best[-1])

                elif approach == "naive":
                    if model == "IDW":

                        # Construct parametrized bb for nomad
                        def IDW_naive_bb_nomad(x):
                            try:
                                # (w11, w12, w13, w14, w15,
                                #                      w21, w22, w23, w24, w25, w26,
                                #                      w31, w32, w33, w34, w35, w36, w37)
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                                 x.get_coord(5), x.get_coord(6), x.get_coord(7), x.get_coord(8), x.get_coord(9),
                                                 x.get_coord(10),
                                                 x.get_coord(11), x.get_coord(12), x.get_coord(13), x.get_coord(14),
                                                 x.get_coord(15), x.get_coord(16), x.get_coord(17))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation

                        # Setup for nomad
                        # (w11, w12, w13, w14, w15; (u1, lr, lamdb, alpha, t0)
                        #                      w21, w22, w23, w24, w25, w26; (u1, u2, lr, lamdb, alpha, t0)
                        #                      w31, w32, w33, w34, w35, w36, w37; (u1, u2, u3, lr, lamdb, alpha, t0))
                        #x0 = [0, 1, 0, 0, 0,
                        #      0, 0, 1, 0, 0, 0,
                        #      0, 0, 0, 1, 0, 0, 0]
                        x0 = []
                        lb = [0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0]
                        ub = [1]*18
                        result = PyNomad.optimize(IDW_naive_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']

                    elif model == "KNN":

                        # Construct parametrized bb for nomad
                        def KNN_naive_bb_nomad(x):
                            try:
                                # (w11, w12, w13, w14, w15,
                                #                      w21, w22, w23, w24, w25, w26,
                                #                      w31, w32, w33, w34, w35, w36, w37,
                                #                      K1, K2, K3)
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
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
                    if model == "IDW":

                        # Create blackbox for Pynomad with graph approach
                        def IDW_hybrid_bb_nomad(x):
                            try:
                                # BB : (w1, w2, w3, w4, w5, w6, w7, w8)
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3),
                                       x.get_coord(4), x.get_coord(5), x.get_coord(6), x.get_coord(7))
                                x.setBBO(str(f).encode("UTF-8"))
                            except:
                                print("Unexpected eval error", sys.exc_info()[0])
                                return 0
                            return 1  # 1: success 0: failed evaluation


                        # Setup for nomad
                        # (w1=l, w2=u1, w3=u2, w4=u3, w5=lr)
                        x0 = []
                        lb = [0, 0, 0, 0, 0, 0, 0, 0]
                        ub = [1] * 8
                        result = PyNomad.optimize(IDW_hybrid_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']

                    elif model == "KNN":

                        # Create blackbox for Pynomad
                        def KNN_graph_bb_nomad(x):
                            try:
                                # BB : (w1, w2, w3, w4, w5, w6, w7, w8, K)
                                f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3),
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
                        result = PyNomad.optimize(KNN_graph_bb_nomad, x0, lb, ub, params)
                        x_best = result['x_best']
                        x_best[-1] = int(x_best[-1])


                # Solution is found
                # Validation final value
                print(bb(*x_best))

                # Test
                bb_test = variant2_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub, approach, model)
                print(bb_test(*x_best))

    # Old
    # Best solutions instance 3
    #x_best_graph_IDW = [10.0001, 10.0007, 999.489, 3.49128e-09, 0, 0, 0.00328981, 2.95877e-07, 7.87017e-07, 0]
    #x_best_graph_KNN = [11.0898, 12.058, 7.9969, 0, 0.00771, 6.19327, 22.2049, 10.607, 0.38868, 0.00408, 3]
    #x_best_naive_IDW = [0.000118041, 11.0095, 0.00203427, 0.000454869, 2.7655e-06, 0, 0, 0.989989, 0, 2.74047e-05, 0, 0, 0, 3.498e-09, 1.02804, 0, 8.15094e-06, 1.28858e-08]
    #x_best_naive_KNN = [0.00084898, 1.05695, 0.0978699, 0.122922, 0.0094295, 0, 0.00983246, 489.794, 0.051403, 102.097, 0.006341, 1.0012, 0.080818, 0.017435, 206.008, 50.996, 0.526982, 0.060274, 7, 1, 24]