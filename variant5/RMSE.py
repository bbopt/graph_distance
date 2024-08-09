import sys
import PyNomad

from utils.import_data import load_data_optimizers, prep_naive_data
from utils.problems_instances_setup import variant_size_setup
from problems_variant5 import variant5_bb_wrapper


if __name__ == '__main__':

    approaches = ["naive", "graph"]
    #models = ["IDW", "KNN"]
    models = ["KNN"]

    # Setup for variant 5
    optimizers = ["ASGD", "ADAM"]
    nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 3}
    nb_max_var = 13  # add the dropout
    var_inc_sub = {("ASGD", 1): [2, 5, 6, 7, 8, 12], ("ASGD", 2): [2, 3, 5, 6, 7, 8, 12],
                   ("ADAM", 1): [2, 5, 9, 10, 11, 12], ("ADAM", 2): [2, 3, 5, 9, 10, 11, 12],
                   ("ADAM", 3): [2, 3, 4, 5, 9, 10, 11, 12]}
    nb_var_sub = {("ASGD", 1): 6, ("ASGD", 2): 7, ("ADAM", 1): 6, ("ADAM", 2): 7, ("ADAM", 3): 8}
    variant = "variant5"

    # Setup for size amongst 1=verysmall, 2=small, 3=medium and 4=large
    size = 4
    nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, size)

    for approach in approaches:
        for model in models:

            # Output log file name
            log_file = "log_" + variant + "_" + "size" + str(size) + "_" + approach + "_" + model + ".txt"

            # Parameters for NOMAD
            params = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 3500", "LH_SEARCH 350 10",
                      "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ",
                      "STATS_FILE " + log_file + " BBE OBJ SOL"]

            # Data file name
            data_file = "data_" + variant + "_" + "size" + str(size) + ".xlsx"

            # Load data
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_optimizers(data_file,
                                                                                      optimizers, nb_sub_per_optimizer,
                                                                                      nb_max_var, nb_pts, nb_test_pts,
                                                                                      nb_valid_pts)

            if approach == "naive":
                # Modifiy data for naive approach
                X_train, y_train, X_valid, y_valid, X_test, y_test = \
                    prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, variant)

            # Create blackbox parametrized by the training set
            bb = variant5_bb_wrapper(X_train, y_train, X_valid, y_valid, nb_max_var, nb_var_sub, approach, model)

            if approach == "graph":
                if model == "IDW":

                    # Create blackbox for Pynomad
                    def IDW_graph_bb_nomad(x):
                        try:
                            # 22 total
                            # 8 bounds : theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                            # 1 cat : o
                            # 13 variables : o, l, u1, u2, u3, lr, hp11, hp12, hp13, hp21, hp22, hp23, p
                            f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3),  # bds
                                             x.get_coord(4), x.get_coord(5), x.get_coord(6), x.get_coord(7),  # bds
                                             x.get_coord(8),  # cat
                                             x.get_coord(9), x.get_coord(10), x.get_coord(11), x.get_coord(12),
                                             x.get_coord(13),  # var
                                             x.get_coord(14), x.get_coord(15), x.get_coord(16), x.get_coord(17),
                                             x.get_coord(18),  # var
                                             x.get_coord(19), x.get_coord(20), x.get_coord(21))  # var
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # 22 total
                    # 8 bounds : theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                    # 1 cat : o
                    # 13 variables : o, l, u1, u2, u3, lr, hp11, hp12, hp13, hp21, hp22, hp23, p
                    #x0 = [10, 10, 10, 10, 10, 10, 10, 10,
                    #      10,
                    #      1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                    x0 = []
                    lb = [10, 10, 2.5, 0.5, 3, 0.2, 0.24995, 1.5,
                          0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    ub = [1e6] * 22
                    result = PyNomad.optimize(IDW_graph_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']


                elif model == "KNN":

                    # Create blackbox for Pynomad
                    def KNN_graph_bb_nomad(x):
                        try:
                            # 23 total
                            # 8 bounds : theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23
                            # 1 cat : o
                            # 13 variables : o, l, u1, u2, u3, lr, hp11, hp12, hp13, hp21, hp22, hp23, p
                            # K
                            f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3),  # bds
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
                    # 12 variables : o, l, u1, u2, u3, lr, hp11, hp12, hp13, hp21, hp22, hp23, p
                    # K
                    #x0 = [10, 10, 10, 10, 10, 10, 10, 10,
                    #      10,
                    #      1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    #      5]
                    x0 = []
                    lb = [10, 10, 2.5, 0.5, 3, 0.2, 0.24995, 1.5,
                          0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1]
                    ub = [1e6] * 22 + [100]
                    input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R R R R R R R R I)"
                    params = params + [input_type]
                    result = PyNomad.optimize(KNN_graph_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    x_best[-1] = int(x_best[-1])

            elif approach == "naive":
                if model == "IDW":
                    # Construct parametrized bb for nomad
                    def IDW_naive_bb_nomad(x):
                        try:
                            # 34 variables
                            # w11, w12, w13, w14, w15, w16,
                            # w21, w22, w23, w24, w25, w26, w27,
                            # w31, w32, w33, w34, w35, w36,
                            # w41, w42, w43, w44, w45, w46, w47,
                            # w51, w52, w53, w54, w55, w56, w57, w58
                            f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                             x.get_coord(5),
                                             x.get_coord(6), x.get_coord(7), x.get_coord(8), x.get_coord(9),
                                             x.get_coord(10), x.get_coord(11), x.get_coord(12),
                                             x.get_coord(13), x.get_coord(14), x.get_coord(15), x.get_coord(16),
                                             x.get_coord(17), x.get_coord(18),
                                             x.get_coord(19), x.get_coord(20), x.get_coord(21), x.get_coord(22),
                                             x.get_coord(23), x.get_coord(24), x.get_coord(25),
                                             x.get_coord(26), x.get_coord(27), x.get_coord(28), x.get_coord(29),
                                             x.get_coord(30), x.get_coord(31), x.get_coord(32), x.get_coord(33))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # Setup : 34 variables
                    # w11, w12, w13, w14, w15, w16           = (u1, lr, lamdb, alpha, t0, p)
                    # w21, w22, w23, w24, w25, w26, w27      = (u1, u2, lr, lamdb, alpha, t0, p)
                    # w31, w32, w33, w34, w35, w36           = (u1, lr, beta1, beta2, eps, p)
                    # w41, w42, w43, w44, w45, w46, w47      = (u1, u2, lr, beta1, beta2, eps, p)
                    # w51, w52, w53, w54, w55, w56, w57, w58 = (u1, u2, u3, lr, beta1, beta2, eps, p)
                    #x0 = [0, 1, 0, 0, 0, 0,
                    #      0, 0, 1, 0, 0, 0, 0,
                    #      0, 1, 0, 0, 0, 0,
                    #      0, 0, 1, 0, 0, 0, 0,
                    #      0, 0, 0, 1, 0, 0, 0, 0]
                    x0 = []
                    lb = [0] * 34
                    ub = [1e6] * 34
                    result = PyNomad.optimize(IDW_naive_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']

                elif model == "KNN":

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
                            f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
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

    # Solution is found
    # Validation final value
    print(bb(*x_best))

    # Test
    bb_test = variant5_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub, approach, model)
    print(bb_test(*x_best))

    # Old
    # Best solutions instance 3
    #x_best_graph_IDW = [13.9619, 27.134, 999.999, 599.987, 11.3778, 1000, 660.003, 992.007, 0, 958.484, 1000, 0, 0, 0, 74.0753, 0.00589456, 0, 0, 0.161019, 0, 0, 0.238256]
    #x_best_graph_KNN = [10, 980, 12.5, 10.5, 283, 0.2, 180, 551.5, 601, 999, 905, 1, 0, 8, 1000, 485, 1000, 98, 500, 904, 399, 822, 6]
    #x_best_naive_IDW = [9.88644e-06, 115.112, 1.15111e-05, 57.2049, 0, 2.14856e-07, 0, 1.51894e-08, 6.30149, 2.11e-12, 6.89e-12, 2.57e-12, 1.89485e-09, 3.72e-09, 9.96627, 2.79252e-09, 1.5375e-10, 0, 1.13584e-09, -9e-14, 4.09518e-06, 1.1049, 1.835e-11, 1.363e-11, 3.3132e-10, 0, 3e-14, 1.8228e-10, 1.44395e-05, 6.7535, 2.75411e-08, 6.78755e-09, 0, 0]
    #x_best_naive_KNN = [1.60063e-05, 1.03939, 0.109924, 300.198, 0.00134277, 99.9831, 0.00707337, 0.000104207, 1.002, 0.0188928, 0.000494609, 0.0502207, 0.00254688, 4.79437e-05, 1.04246, 0.0370575, 0.00815143, 0.0026641, 0.000391668, 0.000371317, 0.00115088, 10.8504, 0.00984824, 0.0931623, 11.9758, 99.9888, 0.00104944, 1.64384e-05, 0.00117252, 0.987888, 0.00508727, 0.032226, 0.0174028, 0.00196256, 4, 5, 3, 4, 5]