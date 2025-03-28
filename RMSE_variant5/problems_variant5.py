import sys
import numpy as np
import pandas as pd

from models.KNN_class import KNN
from models.IDW_class import IDW

from utils.distances import graph_structured_distance, general_distance, hybrid_distance_variant5


def variant5_bb_wrapper(X_train, y_train, X_valid, y_valid, nb_max_var, nb_var_sub, approach, model):

    if approach == "graph":

        if model == "IDW":
            def IDW_graph_variant5_bb(theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23,
                                       cat_o,
                                       w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13):

                # Variable type [o=-1, l=-1, u1=-1, u2=theta, u3=theta, lr=-1,
                # hp11=-1, hp12=-1, hp13=-1 hp21=-1, hp22=-1, hp23=-1, p=-1], 12 variables
                # 12 variables, 8 bounds and 1 cat distance
                thetas = [-1, -1, -1, 10**theta_u2, 10**theta_u3, -1,
                          10**theta_hp11, 10**theta_hp12, 10**theta_hp13,
                          10**theta_hp21, 10**theta_hp22, 10**theta_hp23, -1]
                var_types = ["cat"] + ["quant"] * (nb_max_var - 1)
                cat_params = [10**cat_o] + [""] * (nb_max_var - 1)

                distance = graph_structured_distance(nb_max_var, var_types, thetas, cat_params,
                                                     [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13], p=2)

                model_IDW = IDW(X_train, y_train, distance)
                return model_IDW.RMSE(X_valid, y_valid)

            # Return a parametrized (w.r.t partial dataset) blackbox
            return IDW_graph_variant5_bb


        elif model == "KNN":
            def KNN_graph_variant5_bb(theta_u2, theta_u3, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23,
                                       cat_o,
                                       w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13,
                                       K):
                # Variable type [o=-1, l=-1, u1=-1, u2=theta, u3=theta, lr=-1,
                # hp11=-1, hp12=-1, hp13=-1 hp21=-1, hp22=-1, hp23=-1], 12 variables
                # 12 variables, 8 bounds and 1 cat distance
                thetas = [-1, -1, -1, 10**theta_u2, 10**theta_u3, -1,
                          10**theta_hp11, 10**theta_hp12, 10**theta_hp13,
                          10**theta_hp21, 10**theta_hp22, 10**theta_hp23, -1]
                var_types = ["cat"] + ["quant"] * (nb_max_var - 1)
                cat_params = [10**cat_o] + [""] * (nb_max_var - 1)

                distance = graph_structured_distance(nb_max_var, var_types, thetas, cat_params,
                                                     [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13], p=2)

                model_KNN = KNN(X_train, y_train, distance, K)
                return model_KNN.RMSE(X_valid, y_valid)

                # Return a parametrized (w.r.t partial dataset) blackbox

            return KNN_graph_variant5_bb


    elif approach == "naive":

        if model == "IDW":
            def IDW_naive_variant5_bb(w11, w12, w13, w14, w15, w16,
                                       w21, w22, w23, w24, w25, w26, w27,
                                       w31, w32, w33, w34, w35, w36,
                                       w41, w42, w43, w44, w45, w46, w47,
                                       w51, w52, w53, w54, w55, w56, w57, w58):

                # "sub_ASGD_l1" : (u1, lr, lamdb, alpha, t0), parameters are weight w11, w12, w13, w14, w15, w16
                distance_ASGD_l1 = general_distance(nb_var_sub["ASGD", 1], [w11, w12, w13, w14, w15, w16], p=2)
                model_IDW_ASGD_l1 = IDW(X_train["ASGD", 1], y_train["ASGD", 1], distance_ASGD_l1)
                RMSE_ASGD_l1 = model_IDW_ASGD_l1.RMSE(X_valid["ASGD", 1], y_valid["ASGD", 1])
                nb_pts_ASGD_l1 = X_valid["ASGD", 1].shape[0]

                # "sub_ASGD_l2" : (u1, u2, u3, lr, lamdb, alpha, t0), parameters are weight w21, w22, w23, w24, w25, w26, w27
                distance_ASGD_l2 = general_distance(nb_var_sub["ASGD", 2], [w21, w22, w23, w24, w25, w26, w27], p=2)
                model_IDW_ASGD_l2 = IDW(X_train["ASGD", 2], y_train["ASGD", 2], distance_ASGD_l2)
                RMSE_ASGD_l2 = model_IDW_ASGD_l2.RMSE(X_valid["ASGD", 2], y_valid["ASGD", 2])
                nb_pts_ASGD_l2 = X_valid["ASGD", 2].shape[0]

                # "sub_ADAM_l1": (u1, lr, beta1, beta1, eps), parameters are weight w31, w32, w33, w34, w35
                distance_ADAM_l1 = general_distance(nb_var_sub["ADAM", 1], [w31, w32, w33, w34, w35, w36], p=2)
                model_IDW_ADAM_l1 = IDW(X_train["ADAM", 1], y_train["ADAM", 1], distance_ADAM_l1)
                RMSE_ADAM_l1 = model_IDW_ADAM_l1.RMSE(X_valid["ADAM", 1], y_valid["ADAM", 1])
                nb_pts_ADAM_l1 = X_valid["ADAM", 1].shape[0]

                # "sub_ADAM_l2" : (u1, u2, u3, lr, beta1, beta2, eps), parameters are weight w41, w42, w43, w44, w45, w46
                distance_ADAM_l2 = general_distance(nb_var_sub["ADAM", 2], [w41, w42, w43, w44, w45, w46, w47], p=2)
                model_IDW_ADAM_l2 = IDW(X_train["ADAM", 2], y_train["ADAM", 2], distance_ADAM_l2)
                RMSE_ADAM_l2 = model_IDW_ADAM_l2.RMSE(X_valid["ADAM", 2], y_valid["ADAM", 2])
                nb_pts_ADAM_l2 = X_valid["ADAM", 2].shape[0]

                # "sub_ADAM_l3" : (u1, u2, u3, lr, beta1, beta2, eps), parameters are weight w51, w52, w53, w54, w55, w56, w56, w58
                distance_ADAM_l3 = general_distance(nb_var_sub["ADAM", 3], [w51, w52, w53, w54, w55, w56, w57, w58], p=2)
                model_IDW_ADAM_l3 = IDW(X_train["ADAM", 3], y_train["ADAM", 3], distance_ADAM_l3)
                RMSE_ADAM_l3 = model_IDW_ADAM_l3.RMSE(X_valid["ADAM", 3], y_valid["ADAM", 3])
                nb_pts_ADAM_l3 = X_valid["ADAM", 3].shape[0]

                nb_pts = nb_pts_ASGD_l1 + nb_pts_ASGD_l2 + nb_pts_ADAM_l1 + nb_pts_ADAM_l2 + nb_pts_ADAM_l3
                return np.sqrt((nb_pts_ASGD_l1 * (RMSE_ASGD_l1 ** 2) + nb_pts_ASGD_l2 * (RMSE_ASGD_l2 ** 2) +
                                nb_pts_ADAM_l1 * (RMSE_ADAM_l1 ** 2) + nb_pts_ADAM_l2 * (RMSE_ADAM_l2 ** 2) +
                                nb_pts_ADAM_l3 * (RMSE_ADAM_l3 ** 2)) / nb_pts)

            # Return a parametrized (w.r.t partial dataset) blackbox
            return IDW_naive_variant5_bb

        elif model == "KNN":
            def KNN_naive_variant5_bb(w11, w12, w13, w14, w15, w16,
                                       w21, w22, w23, w24, w25, w26, w27,
                                       w31, w32, w33, w34, w35, w36,
                                       w41, w42, w43, w44, w45, w46, w47,
                                       w51, w52, w53, w54, w55, w56, w57, w58,
                                       K1, K2, K3, K4, K5):

                # "sub_ASGD_l1" : (u1, lr, lamdb, alpha, t0), parameters are weight w11, w12, w13, w14, w15, w16
                distance_ASGD_l1 = general_distance(nb_var_sub["ASGD", 1], [w11, w12, w13, w14, w15, w16], p=2)
                model_KNN_ASGD_l1 = KNN(X_train["ASGD", 1], y_train["ASGD", 1], distance_ASGD_l1, K1)
                RMSE_ASGD_l1 = model_KNN_ASGD_l1.RMSE(X_valid["ASGD", 1], y_valid["ASGD", 1])
                nb_pts_ASGD_l1 = X_valid["ASGD", 1].shape[0]

                # "sub_ASGD_l2" : (u1, u2, u3, lr, lamdb, alpha, t0), parameters are weight w21, w22, w23, w24, w25, w26, w27
                distance_ASGD_l2 = general_distance(nb_var_sub["ASGD", 2], [w21, w22, w23, w24, w25, w26, w27], p=2)
                model_KNN_ASGD_l2 = KNN(X_train["ASGD", 2], y_train["ASGD", 2], distance_ASGD_l2, K2)
                RMSE_ASGD_l2 = model_KNN_ASGD_l2.RMSE(X_valid["ASGD", 2], y_valid["ASGD", 2])
                nb_pts_ASGD_l2 = X_valid["ASGD", 2].shape[0]

                # "sub_ADAM_l1": (u1, lr, beta1, beta1, eps), parameters are weight w31, w32, w33, w34, w35, w36
                distance_ADAM_l1 = general_distance(nb_var_sub["ADAM", 1], [w31, w32, w33, w34, w35, w36], p=2)
                model_KNN_ADAM_l1 = KNN(X_train["ADAM", 1], y_train["ADAM", 1], distance_ADAM_l1, K3)
                RMSE_ADAM_l1 = model_KNN_ADAM_l1.RMSE(X_valid["ADAM", 1], y_valid["ADAM", 1])
                nb_pts_ADAM_l1 = X_valid["ADAM", 1].shape[0]

                # "sub_ADAM_l2" : (u1, u2, u3, lr, beta1, beta2, eps), parameters are weight w41, w42, w43, w44, w45, w46, w47
                distance_ADAM_l2 = general_distance(nb_var_sub["ADAM", 2], [w41, w42, w43, w44, w45, w46, w47], p=2)
                model_KNN_ADAM_l2 = KNN(X_train["ADAM", 2], y_train["ADAM", 2], distance_ADAM_l2, K4)
                RMSE_ADAM_l2 = model_KNN_ADAM_l2.RMSE(X_valid["ADAM", 2], y_valid["ADAM", 2])
                nb_pts_ADAM_l2 = X_valid["ADAM", 2].shape[0]

                # "sub_ADAM_l3" : (u1, u2, u3, lr, beta1, beta2, eps), parameters are weight w51, w52, w53, w54, w55, w56, w56, w58
                distance_ADAM_l3 = general_distance(nb_var_sub["ADAM", 3], [w51, w52, w53, w54, w55, w56, w57, w58], p=2)
                model_KNN_ADAM_l3 = KNN(X_train["ADAM", 3], y_train["ADAM", 3], distance_ADAM_l3, K5)
                RMSE_ADAM_l3 = model_KNN_ADAM_l3.RMSE(X_valid["ADAM", 3], y_valid["ADAM", 3])
                nb_pts_ADAM_l3 = X_valid["ADAM", 3].shape[0]

                nb_pts = nb_pts_ASGD_l1 + nb_pts_ASGD_l2 + nb_pts_ADAM_l1 + nb_pts_ADAM_l2 + nb_pts_ADAM_l3
                return np.sqrt((nb_pts_ASGD_l1 * (RMSE_ASGD_l1 ** 2) + nb_pts_ASGD_l2 * (RMSE_ASGD_l2 ** 2) +
                                nb_pts_ADAM_l1 * (RMSE_ADAM_l1 ** 2) + nb_pts_ADAM_l2 * (RMSE_ADAM_l2 ** 2) +
                                nb_pts_ADAM_l3 * (RMSE_ADAM_l3 ** 2)) / nb_pts)

            # Return a parametrized (w.r.t partial dataset) blackbox
            return KNN_naive_variant5_bb

    elif approach == "hybrid":

        if model == "IDW":
            def IDW_hybrid_variant4_bb(w11, w12, w13, w14, w15, w16, w17, w18,
                                       w21, w22, w23, w24, w25, w26, w27, w28, w29):
                # "ASGD" : (l, u1, u2, lr, a1, a2, a3), parameters are weight w11, w12, w13, w14, w15, w16, w17, w18
                distance_ASGD = hybrid_distance_variant5([w11, w12, w13, w14, w15, w16, w17, w18], "ASGD", p=2)
                model_IDW_ASGD = IDW(X_train["ASGD"], y_train["ASGD"], distance_ASGD)
                RMSE_ASGD = model_IDW_ASGD.RMSE(X_valid["ASGD"], y_valid["ASGD"])
                nb_pts_ASGD = X_valid["ASGD"].shape[0]

                # "ADAM" : (l, u1, u2, u3, lr, b1, b2, b3), parameters are weight w21, w22, w23, w24, w25, w26, w27, w28, w29
                distance_ADAM = hybrid_distance_variant5([w21, w22, w23, w24, w25, w26, w27, w28, w29], "ADAM", p=2)
                model_IDW_ADAM = IDW(X_train["ADAM"], y_train["ADAM"], distance_ADAM)
                RMSE_ADAM = model_IDW_ADAM.RMSE(X_valid["ADAM"], y_valid["ADAM"])
                nb_pts_ADAM = X_valid["ADAM"].shape[0]

                nb_pts = nb_pts_ASGD + nb_pts_ADAM
                return np.sqrt( (nb_pts_ASGD * (RMSE_ASGD ** 2) + nb_pts_ADAM * (RMSE_ADAM ** 2) )/nb_pts)

            # Return a parametrized (w.r.t partial dataset) blackbox
            return IDW_hybrid_variant4_bb

        elif model == "KNN":
            def KNN_hybrid_variant4_bb(w11, w12, w13, w14, w15, w16, w17, w18,
                                       w21, w22, w23, w24, w25, w26, w27, w28, w29, K1, K2):
                # "ASGD" : (l, u1, u2, lr, a1, a2, a3, K), parameters are weight w11, w12, w13, w14, w15, w16, w17, w18, K1
                distance_ASGD = hybrid_distance_variant5([w11, w12, w13, w14, w15, w16, w17, w18], "ASGD", p=2)
                model_KNN_ASGD = KNN(X_train["ASGD"], y_train["ASGD"], distance_ASGD, K1)
                RMSE_ASGD = model_KNN_ASGD.RMSE(X_valid["ASGD"], y_valid["ASGD"])
                nb_pts_ASGD = X_valid["ASGD"].shape[0]

                # "ADAM" : (l, u1, u2, u3, lr, b1, b2, b3, K), parameters are weight w21, w22, w23, w24, w25, w26, w27, w28, w29, K2
                distance_ADAM = hybrid_distance_variant5([w21, w22, w23, w24, w25, w26, w27, w28, w29], "ADAM", p=2)
                model_KNN_ADAM = KNN(X_train["ADAM"], y_train["ADAM"], distance_ADAM, K2)
                RMSE_ADAM = model_KNN_ADAM.RMSE(X_valid["ADAM"], y_valid["ADAM"])
                nb_pts_ADAM = X_valid["ADAM"].shape[0]

                nb_pts = nb_pts_ASGD + nb_pts_ADAM
                return np.sqrt((nb_pts_ASGD * (RMSE_ASGD ** 2) + nb_pts_ADAM * (RMSE_ADAM ** 2)) / nb_pts)

            # Return a parametrized (w.r.t partial dataset) blackbox
            return KNN_hybrid_variant4_bb