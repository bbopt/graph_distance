import numpy as np


def inc_exc_distance(xi, yi, theta, var_type, cat_param):

    if xi != 'EXC' and yi != 'EXC':
        if var_type == "cat":
            return cat_param
        elif var_type == "quant":
            return abs(xi - yi)

    elif xi == 'EXC' and yi == 'EXC':
        return 0

    elif (xi == 'EXC' and yi != 'EXC') or (xi != 'EXC' and yi == 'EXC'):
        return theta




def graph_structured_distance(nb_var, types, thetas, cat_params, weights, p=2):

    def compute(x, y):
        dist = 0
        for i in range(nb_var):
            dist = dist + abs(weights[i]*(inc_exc_distance(x[i], y[i], thetas[i], types[i], cat_params[i]))**p)

        return np.power(dist, 1/p)

    return compute


# For naive approach with multiple models
def general_distance(nb_var, weights, p=2):

    def compute(x, y):
        dist = 0
        for i in range(nb_var):
            dist = dist + weights[i]*abs(x[i]-y[i])**p

        return np.power(dist, 1/p)

    return compute