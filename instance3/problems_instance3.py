import sys
import numpy as np
import pandas as pd

from models.KNN_class import KNN
from models.IDW_class import IDW

from utils.distances import graph_structured_distance
from utils.distances import general_distance


def instance3_bb_wrapper(X_train, y_train, X_valid, y_valid, nb_max_var, nb_var_sub, approach, model):

    if approach == "graph":

        if model == "IDW":
            def IDW_graph_instance3_bb(theta_u2, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23,
                                         cat_o,
                                         w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11):

                # Variable type [o=-1, l=-1, u1=-1, u2=theta, lr=-1,
                # hp11=-1, hp12=-1, hp13=-1 hp21=-1, hp22=-1, hp23=-1], 11 variables
                # 11 variables, 7 bounds and 1 cat distance
                thetas = [-1, -1, -1, theta_u2, -1,
                          theta_hp11, theta_hp12, theta_hp13,
                          theta_hp21, theta_hp22, theta_hp23]
                var_types = ["cat"] + ["quant"] * (nb_max_var - 1)
                cat_params = [cat_o] + [""] * (nb_max_var - 1)

                distance = graph_structured_distance(nb_max_var, var_types, thetas, cat_params,
                                                     [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11], p=2)

                model_IDW = IDW(X_train, y_train, distance)
                return model_IDW.RMSE(X_valid, y_valid)

            # Return a parametrized (w.r.t partial dataset) blackbox
            return IDW_graph_instance3_bb


        elif model == "KNN":
            def KNN_graph_instance3_bb(theta_u2, theta_hp11, theta_hp12, theta_hp13, theta_hp21, theta_hp22, theta_hp23,
                                         cat_o,
                                         w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11,
                                         K):

                # Variable type [o=-1, l=-1, u1=-1, u2=theta, lr=-1,
                # hp11=-1, hp12=-1, hp13=-1 hp21=-1, hp22=-1, hp23=-1], 11 variables
                # 11 variables, 7 bounds and 1 cat distance
                thetas = [-1, -1, -1, theta_u2, -1,
                          theta_hp11, theta_hp12, theta_hp13,
                          theta_hp21, theta_hp22, theta_hp23]
                var_types = ["cat"] + ["quant"] * (nb_max_var - 1)
                cat_params = [cat_o] + [""] * (nb_max_var - 1)

                distance = graph_structured_distance(nb_max_var, var_types, thetas, cat_params,
                                                     [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11], p=2)

                model_KNN = KNN(X_train, y_train, distance, K)
                return model_KNN.RMSE(X_valid, y_valid)

            # Return a parametrized (w.r.t partial dataset) blackbox
            return KNN_graph_instance3_bb

    elif approach == "naive":

        if model == "IDW":
            def IDW_naive_instance3_bb(w11, w12, w13, w14, w15,
                                         w21, w22, w23, w24, w25, w26,
                                         w31, w32, w33, w34, w35,
                                         w41, w42, w43, w44, w45, w46):

                # "sub_ASGD_l1" : (u1, lr, lamdb, alpha, t0), parameters are weight w11, w12, w13, w14, w15
                distance_ASGD_l1 = general_distance(nb_var_sub["ASGD", 1], [w11, w12, w13, w14, w15], p=2)
                model_IDW_ASGD_l1 = IDW(X_train["ASGD", 1], y_train["ASGD", 1], distance_ASGD_l1)
                RMSE_ASGD_l1 = model_IDW_ASGD_l1.RMSE(X_valid["ASGD", 1], y_valid["ASGD", 1])
                nb_pts_ASGD_l1 = X_valid["ASGD", 1].shape[0]

                # "sub_ASGD_l2" : (u1, u2, u3, lr, lamdb, alpha, t0), parameters are weight w21, w22, w23, w24, w25, w26
                distance_ASGD_l2 = general_distance(nb_var_sub["ASGD", 2], [w21, w22, w23, w24, w25, w26], p=2)
                model_IDW_ASGD_l2 = IDW(X_train["ASGD", 2], y_train["ASGD", 2], distance_ASGD_l2)
                RMSE_ASGD_l2 = model_IDW_ASGD_l2.RMSE(X_valid["ASGD", 2], y_valid["ASGD", 2])
                nb_pts_ASGD_l2 = X_valid["ASGD", 2].shape[0]

                # "sub_ADAM_l1": (u1, lr, beta1, beta1, eps), parameters are weight w31, w32, w33, w34, w35
                distance_ADAM_l1 = general_distance(nb_var_sub["ADAM", 1], [w31, w32, w33, w34, w35], p=2)
                model_IDW_ADAM_l1 = IDW(X_train["ADAM", 1], y_train["ADAM", 1], distance_ADAM_l1)
                RMSE_ADAM_l1 = model_IDW_ADAM_l1.RMSE(X_valid["ADAM", 1], y_valid["ADAM", 1])
                nb_pts_ADAM_l1 = X_valid["ADAM", 1].shape[0]

                # "sub_ADAM_l2" : (u1, u2, u3, lr, beta1, beta2, eps), parameters are weight w41, w42, w43, w44, w45, w46
                distance_ADAM_l2 = general_distance(nb_var_sub["ADAM", 2], [w41, w42, w43, w44, w45, w46], p=2)
                model_IDW_ADAM_l2 = IDW(X_train["ADAM", 2], y_train["ADAM", 2], distance_ADAM_l2)
                RMSE_ADAM_l2 = model_IDW_ADAM_l2.RMSE(X_valid["ADAM", 2], y_valid["ADAM", 2])
                nb_pts_ADAM_l2 = X_valid["ADAM", 2].shape[0]

                nb_pts = nb_pts_ASGD_l1 + nb_pts_ASGD_l2 + nb_pts_ADAM_l1 + nb_pts_ADAM_l2
                return np.sqrt((nb_pts_ASGD_l1 * (RMSE_ASGD_l1 ** 2) + nb_pts_ASGD_l2 * (RMSE_ASGD_l2 ** 2) +
                                nb_pts_ADAM_l1 * (RMSE_ADAM_l1 ** 2) + nb_pts_ADAM_l2 * (RMSE_ADAM_l2 ** 2)) / nb_pts)

            # Return a parametrized (w.r.t partial dataset) blackbox
            return IDW_naive_instance3_bb


        elif model == "KNN":

            def KNN_naive_instance3_bb(w11, w12, w13, w14, w15,
                                         w21, w22, w23, w24, w25, w26,
                                         w31, w32, w33, w34, w35,
                                         w41, w42, w43, w44, w45, w46,
                                         K1, K2, K3, K4):

                # "sub_ASGD_l1" : (u1, lr, lamdb, alpha, t0), parameters are weight w11, w12, w13, w14, w15
                distance_ASGD_l1 = general_distance(nb_var_sub["ASGD", 1], [w11, w12, w13, w14, w15], p=2)
                model_KNN_ASGD_l1 = KNN(X_train["ASGD", 1], y_train["ASGD", 1], distance_ASGD_l1, K1)
                RMSE_ASGD_l1 = model_KNN_ASGD_l1.RMSE(X_valid["ASGD", 1], y_valid["ASGD", 1])
                nb_pts_ASGD_l1 = X_valid["ASGD", 1].shape[0]

                # "sub_ASGD_l2" : (u1, u2, u3, lr, lamdb, alpha, t0), parameters are weight w21, w22, w23, w24, w25, w26
                distance_ASGD_l2 = general_distance(nb_var_sub["ASGD", 2], [w21, w22, w23, w24, w25, w26], p=2)
                model_KNN_ASGD_l2 = KNN(X_train["ASGD", 2], y_train["ASGD", 2], distance_ASGD_l2, K2)
                RMSE_ASGD_l2 = model_KNN_ASGD_l2.RMSE(X_valid["ASGD", 2], y_valid["ASGD", 2])
                nb_pts_ASGD_l2 = X_valid["ASGD", 2].shape[0]

                # "sub_ADAM_l1": (u1, lr, beta1, beta1, eps), parameters are weight w31, w32, w33, w34, w35
                distance_ADAM_l1 = general_distance(nb_var_sub["ADAM", 1], [w31, w32, w33, w34, w35], p=2)
                model_KNN_ADAM_l1 = KNN(X_train["ADAM", 1], y_train["ADAM", 1], distance_ADAM_l1, K3)
                RMSE_ADAM_l1 = model_KNN_ADAM_l1.RMSE(X_valid["ADAM", 1], y_valid["ADAM", 1])
                nb_pts_ADAM_l1 = X_valid["ADAM", 1].shape[0]

                # "sub_ADAM_l2" : (u1, u2, u3, lr, beta1, beta2, eps), parameters are weight w41, w42, w43, w44, w45, w46
                distance_ADAM_l2 = general_distance(nb_var_sub["ADAM", 2], [w41, w42, w43, w44, w45, w46], p=2)
                model_KNN_ADAM_l2 = KNN(X_train["ADAM", 2], y_train["ADAM", 2], distance_ADAM_l2, K4)
                RMSE_ADAM_l2 = model_KNN_ADAM_l2.RMSE(X_valid["ADAM", 2], y_valid["ADAM", 2])
                nb_pts_ADAM_l2 = X_valid["ADAM", 2].shape[0]

                nb_pts = nb_pts_ASGD_l1 + nb_pts_ASGD_l2 + nb_pts_ADAM_l1 + nb_pts_ADAM_l2
                return np.sqrt((nb_pts_ASGD_l1 * (RMSE_ASGD_l1 ** 2) + nb_pts_ASGD_l2 * (RMSE_ASGD_l2 ** 2) +
                                nb_pts_ADAM_l1 * (RMSE_ADAM_l1 ** 2) + nb_pts_ADAM_l2 * (RMSE_ADAM_l2 ** 2)) / nb_pts)

                # Return a parametrized (w.r.t partial dataset) blackbox

            return KNN_naive_instance3_bb

