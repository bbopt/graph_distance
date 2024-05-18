import sys
import PyNomad

from utils.import_data import load_data_fix_optimizer, prep_naive_data
from problems_instance2 import instance2_bb_wrapper


if __name__ == '__main__':

    # Approach: choose between "graph" and "naive"
    approach = "graph"
    # Model: choose between "IDW" and "KNN"
    model = "IDW"
    # Best solutions: if best solution is not already founded, set variable as None
    x_best_graph_IDW = [10.0001, 10.0007, 999.489, 3.49128e-09, 0, 0, 0.00328981, 2.95877e-07, 7.87017e-07, 0]
    x_best_graph_KNN = [11.0898, 12.058, 7.9969, 0, 0.00771, 6.19327, 22.2049, 10.607, 0.38868, 0.00408, 3]
    x_best_naive_IDW = [0.000118041, 11.0095, 0.00203427, 0.000454869, 2.7655e-06, 0, 0, 0.989989, 0, 2.74047e-05, 0, 0, 0, 3.498e-09, 1.02804, 0, 8.15094e-06, 1.28858e-08]
    x_best_naive_KNN = [0.00084898, 1.05695, 0.0978699, 0.122922, 0.0094295, 0, 0.00983246, 489.794, 0.051403, 102.097, 0.006341, 1.0012, 0.080818, 0.017435, 206.008, 50.996, 0.526982, 0.060274, 7, 1, 24]

    # Setup
    nb_pts = [100, 120, 140]
    nb_test_pts = [25, 30, 35]
    nb_valid_pts = [25, 30, 35]
    nb_sub = 3
    nb_max_var = 8
    nb_var_sub = [5, 6, 7]
    var_inc_sub = [[1, 4, 5, 6, 7], [1, 2, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]  # included variables in each subproblem

    # Parameters for NOMAD
    params = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 2000", "LH_SEARCH 150 10",
              "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ"]

    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_fix_optimizer('data_instance2.xlsx', nb_sub,
                                                                                 nb_max_var, nb_pts, nb_test_pts,
                                                                                 nb_valid_pts)

    if approach == "naive":
        # Modifiy data for naive approach
        X_train, y_train, X_valid, y_valid, X_test, y_test = \
            prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, 'instance2')

    # Create blackbox parametrized by the training set
    bb = instance2_bb_wrapper(X_train, y_train, X_valid, y_valid, nb_max_var, nb_var_sub, approach, model)

    if approach == "graph":
        if model == "IDW":
            if x_best_graph_IDW is None:
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
                x0 = [10, 10, 1, 0, 0, 0, 1, 0, 0, 0]
                lb = [10, 10, 0, 0, 0, 0, 0, 0, 0, 0]
                ub = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
                result = PyNomad.optimize(IDW_graph_bb_nomad, x0, lb, ub, params)
                x_best = result['x_best']

            # Solution already find
            else:
                x_best = x_best_graph_IDW

        elif model == "KNN":
            if x_best_graph_KNN is None:

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
                x0 = [10, 10, 1, 0, 0, 0, 1, 0, 0, 0, 5]
                lb = [10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                ub = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 100]
                input_type = "BB_INPUT_TYPE (R R R R R R R R R R I)"
                result = PyNomad.optimize(KNN_graph_bb_nomad, x0, lb, ub, params)
                x_best = result['x_best']
                x_best[-1] = int(x_best[-1])

            # Solution already find
            else:
                x_best = x_best_graph_KNN

    elif approach == "naive":
        if model == "IDW":
            if x_best_naive_IDW is None:
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
                x0 = [0, 1, 0, 0, 0,
                      0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0]
                lb = [0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0]
                ub = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                      1000, 1000]
                result = PyNomad.optimize(IDW_naive_bb_nomad, x0, lb, ub, params)
                x_best = result['x_best']

            # Solution already find
            else:
                x_best = x_best_naive_IDW

        elif model == "KNN":
            if x_best_naive_KNN is None:
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
                x0 = [0, 1, 0, 0, 0,
                      0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      25, 25, 25]
                lb = [0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1]
                ub = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                      1000, 1000, 50, 50, 50]
                input_type = "BB_INPUT_TYPE (R R R R R R R R R R R R R R R R R R I I I)"
                result = PyNomad.optimize(KNN_naive_bb_nomad, x0, lb, ub, params)
                x_best = result['x_best']
                for i in range(3):
                    x_best[-(i+1)] = int(x_best[-(i+1)])

            # Solution already find
            else:
                x_best = x_best_naive_KNN

    # Solution is found
    # Validation final value
    print(bb(*x_best))

    # Test
    IDW_bb_test = instance2_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub, approach, model)
    print(IDW_bb_test(*x_best))
