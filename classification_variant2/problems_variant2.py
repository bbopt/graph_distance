import sys
import numpy as np
import pandas as pd

from models.KNN_class_classification import KNN
from models.IDW_class import IDW

from utils.distances import graph_structured_distance, general_distance, hybrid_distance_variant2


def variant2_bb_wrapper(X_train, y_train, X_valid, y_valid, nb_max_var, nb_var_sub, approach, model):

    if approach == "graph":

        def KNN_graph_variant2_bb(theta_u2, theta_u3, w1, w2, w3, w4, w5, w6, w7, w8, K):

            # Variable type, thetas = [l, u1, u2, u3, lr, lamdb, alpha, t0], 8 variables
            thetas = [-1, -1, 10**theta_u2, 10**theta_u3, -1, -1, -1, -1]
            var_types = ["quant"] * nb_max_var
            cat_params = [""] * nb_max_var

            distance = graph_structured_distance(nb_max_var, var_types, thetas, cat_params,
                                                 [w1, w2, w3, w4, w5, w6, w7, w8], p=2)

            model_KNN = KNN(X_train, y_train, distance, K)
            return model_KNN.accuracy(X_valid, y_valid)

        # Return a parametrized (w.r.t partial dataset) blackbox
        return KNN_graph_variant2_bb

    elif approach == "naive":

        def KNN_naive_variant2_bb(w11, w12, w13, w14, w15,
                                     w21, w22, w23, w24, w25, w26,
                                     w31, w32, w33, w34, w35, w36, w37,
                                     K1, K2, K3):

            # sub1 : (u1, lr, lamdb, alpha, t0, K1), parameters are weight w11, w12, w13, w14, w15 and K1
            distance_sub1 = general_distance(nb_var_sub[0], [w11, w12, w13, w14, w15], p=2)
            model_KNN_sub1 = KNN(X_train["sub1"], y_train["sub1"], distance_sub1, K1)
            acc_sub1 = model_KNN_sub1.accuracy(X_valid["sub1"], y_valid["sub1"])
            nb_pts_sub1 = X_valid["sub1"].shape[0]

            # sub 2 : (u1, u2, u3, lr, lamdb, alpha, t0, K2), parameters are weight w21, w22, w23, w24, w25, w26 and K2
            distance_sub2 = general_distance(nb_var_sub[1], [w21, w22, w23, w24, w25, w26], p=2)
            model_KNN_sub2 = KNN(X_train["sub2"], y_train["sub2"], distance_sub2, K2)
            acc_sub2 = model_KNN_sub2.accuracy(X_valid["sub2"], y_valid["sub2"])
            nb_pts_sub2 = X_valid["sub2"].shape[0]

            # sub 3 : (u1, u2, u3, lr, lamdb, alpha, t0, K3, parameters are weight w31, w32, w33, w34, w35, w36, w37 and K3
            distance_sub3 = general_distance(nb_var_sub[2], [w31, w32, w33, w34, w35, w36, w37], p=2)
            model_KNN_sub3 = KNN(X_train["sub3"], y_train["sub3"], distance_sub3, K3)
            acc_sub3 = model_KNN_sub3.accuracy(X_valid["sub3"], y_valid["sub3"])
            nb_pts_sub3 = X_valid["sub3"].shape[0]

            nb_pts = nb_pts_sub1 + nb_pts_sub2 + nb_pts_sub3
            return (nb_pts_sub1 * acc_sub1 + nb_pts_sub2 * acc_sub2 + nb_pts_sub3 * acc_sub3) / nb_pts

            # Return a parametrized (w.r.t partial dataset) blackbox

        return KNN_naive_variant2_bb

    elif approach == "hybrid":

        def KNN_hybrid_variant2_bb(w1, w2, w3, w4, w5, w6, w7, w8, K):

            distance = hybrid_distance_variant2([w1, w2, w3, w4, w5, w6, w7, w8], p=2)

            model_KNN = KNN(X_train, y_train, distance, K)
            return model_KNN.accuracy(X_valid, y_valid)

        # Return a parametrized (w.r.t partial dataset) blackbox
        return KNN_hybrid_variant2_bb