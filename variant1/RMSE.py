import sys
import PyNomad

from utils.import_data import load_data_fix_optimizer, prep_naive_data
from utils.problems_instances_setup import variant_size_setup
from problems_variant1 import variant1_bb_wrapper


if __name__ == '__main__':

    approaches = ["naive", "graph"]
    #models = ["IDW", "KNN"]
    models = ["KNN"]

    # Setup for variant 1
    nb_sub = 3
    nb_max_var = 5  # l, u1, u2, u3, lr
    nb_var_sub = [2, 3, 4]
    var_inc_sub = [[1, 4], [1, 2, 4], [1, 2, 3, 4]]  # included variables in each subproblem
    variant = "variant1"

    # Setup for size amongst 1=verysmall, 2=small, 3=medium and 4=large
    size = 4
    nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, size)

    for approach in approaches:
        for model in models:

            # Output log file name
            log_file = "log_" + variant + "_" + "size" + str(size) + "_" + approach + "_" + model + ".txt"

            # Parameters for NOMAD: parameters must be reloaded
            params = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 1500", "LH_SEARCH 150 5",
                      "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ",
                      "STATS_FILE " + log_file + " BBE OBJ SOL"]

            # Data file name
            data_file = "data_" + variant + "_" + "size" + str(size) + ".xlsx"

            # Load data
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_fix_optimizer(data_file,
                                                                                         nb_sub, nb_max_var, nb_pts,
                                                                                         nb_test_pts, nb_valid_pts)
            if approach == "naive":
                # Modifiy data for naive approach
                X_train, y_train, X_valid, y_valid, X_test, y_test = \
                    prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, variant)

            # Create blackbox parametrized by the training set
            bb = variant1_bb_wrapper(X_train, y_train, X_valid, y_valid, nb_max_var, nb_var_sub, approach, model)

            # Construct the optimization problem w.r.t to the approach and the model
            if approach == "graph":
                if model == "IDW":

                    # Create blackbox for Pynomad with graph approach
                    def IDW_graph_bb_nomad(x):
                        try:
                            # BB : (theta_u2, theta_u3, w1, w2, w3, w4, w5)
                            f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                   x.get_coord(5), x.get_coord(6))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # Setup for nomad
                    # (theta_u2, theta_u3, w1=l, w2=u1, w3=u2, w4=u3, w5=lr)
                    #x0 = [10, 10, 1, 0, 0, 0, 1]
                    x0 = []
                    lb = [10, 10, 0, 0, 0, 0, 0]
                    ub = [1e6]*7
                    result = PyNomad.optimize(IDW_graph_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']

                elif model == "KNN":

                    # Create blackbox for Pynomad
                    def KNN_graph_bb_nomad(x):
                        try:
                            # BB : (theta_u2, theta_u3, w1, w2, w3, w4, w5, K)
                            f = bb(x.get_coord(0), x.get_coord(1), x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                             x.get_coord(5), x.get_coord(6), int(x.get_coord(7)))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # Setup for nomad
                    # (theta_u2, theta_u3, w1=l, w2=u1, w3=u2, w4=u3, w5=lr, K)
                    #x0 = [10, 10, 1, 0, 0, 0, 1, 5]
                    x0 = []
                    lb = [10, 10, 0, 0, 0, 0, 0, 1]
                    ub = [1e6]*7 + [50]
                    input_type = "BB_INPUT_TYPE (R R R R R R R I)"
                    params = params + [input_type]
                    result = PyNomad.optimize(KNN_graph_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    x_best[-1] = int(x_best[-1])

            elif approach == "naive":
                if model == "IDW":

                    # Construct parametrized bb for nomad
                    def IDW_naive_bb_nomad(x):
                        try:
                            # (w11, w12, w21, w22, w23, w31, w32, w33, w34)
                            # (u1, lr; u1, u2, lr; u1, u2, u3, lr)
                            f = bb(x.get_coord(0), x.get_coord(1),
                                   x.get_coord(2), x.get_coord(3), x.get_coord(4),
                                   x.get_coord(5), x.get_coord(6), x.get_coord(7), x.get_coord(8))
                            x.setBBO(str(f).encode("UTF-8"))
                        except:
                            print("Unexpected eval error", sys.exc_info()[0])
                            return 0
                        return 1  # 1: success 0: failed evaluation

                    # Setup for nomad
                    # (w11, w12; w21, w22, w23; w31, w32, w33, w34) = (u1, lr; u1, u2, lr; u1, u2, u3, lr)
                    #x0 = [0, 1, 0, 0, 1, 0, 0, 0, 1]
                    x0 = []
                    lb = [0, 1e-7, 0, 0, 1e-7, 0, 0, 0, 1e-7]  # fix division by zero
                    ub = [1e6]*9
                    result = PyNomad.optimize(IDW_naive_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']

                elif model == "KNN":

                    # Construct parametrized bb for nomad
                    def KNN_naive_bb_nomad(x):
                        try:
                            # (w11, w12, w21, w22, w23, w31, w32, w33, w34, K1, K2, K3)
                            # (u1, lr, u1, u2, lr, u1, u2, u3, lr, K1, K2, K3)
                            f = bb(x.get_coord(0), x.get_coord(1),
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
                    #x0 = [0, 1, 0, 0, 1, 0, 0, 0, 1, 5, 5, 5]
                    x0 = []
                    lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
                    ub = [1e6]*9 + [int(x/2) for x in nb_pts]
                    input_type = "BB_INPUT_TYPE (R R R R R R R R R I I I)"
                    params = params + [input_type]
                    result = PyNomad.optimize(KNN_naive_bb_nomad, x0, lb, ub, params)
                    x_best = result['x_best']
                    for i in range(3):
                        x_best[-(i+1)] = int(x_best[-(i+1)])

        # Solution is found
        # Validation final value
        print(bb(*x_best))

        # Test
        bb_test = variant1_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub, approach, model)
        print(bb_test(*x_best))


    # -------- Old --------------- #
    # X_train_valid = np.vstack((X_train, X_valid))
    # y_train_valid = np.concatenate((y_train, y_valid))

    # Best solutions instance 3
    # x_best_graph_IDW = [10.034, 140.366, 985.392, 6.16e-12, -2e-14, 2.82614e-09, 2.91153e-05]
    # x_best_graph_KNN = [10, 10.1923, 100.243, 0.234, 0.2857, 0, 205.2, 6]
    # x_best_naive_IDW = [0, 0.834192, 4.07945e-07, 0, 1.15362, 0, 2.667e-11, 0.00981489, 100.862]
    # x_best_naive_KNN = [0.0046632, 0.99748, 0.0672663, 0.0259468, 100.879, 0.000781851, 0.00685504, 0.00642707, 1.02407, 4, 7, 6]