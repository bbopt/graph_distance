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


def hybrid_distance_variant1(weights, p=2):

    # x = (l (0), u1 (1), u2 (2), u3 (3), lr (4))
    # x[0] = 1 <-> l=1
    # x[0] = 2 <-> l=2
    # x[0] = 3 <-> l=3
    var_shared = None
    def compute(x, y):
        if x[0] in {1, 2, 3} and y[0] == 1:
            # (l, u1, lr)
            var_shared = [0, 1, 4]

        elif x[0] == 1 and y[0] in {2, 3}:
            # (l, u1, lr)
            var_shared = [0, 1, 4]

        elif x[0] == 2 and y[0] in {2, 3}:
            # (l, u1, u2, lr)
            var_shared = [0, 1, 2, 4]

        elif x[0] == 3 and y[0] == 2:
            # (l, u1, u2, lr)
            var_shared = [0, 1, 2, 4]

        elif x[0] == 3 and y[0] == 3:
            # (l, u1, u2, u3, lr)
            var_shared = [0, 1, 2, 3, 4]

        else:
            raise ValueError(f"Invalid (x[0], y[0]) combination: ({x[0]}, {y[0]})")

        dist = 0
        for i in var_shared:
            dist = dist + weights[i] * abs(x[i] - y[i]) ** p

        return np.power(dist, 1 / p)

    return compute


def hybrid_distance_variant2(weights, p=2):

    # for hybrid a point is passed as x = (l (0), u1 (1), u2 (2), u3 (3), lr (4), hp1 (5), hp2 (6), hp3 (7))
    # x[0] = 1 <-> l=1
    # x[0] = 2 <-> l=2
    # x[0] = 3 <-> l=3

    # Find the variables shared for a given subproblem:
    # For variant 1-2, there is a single subproblem
    var_shared = None
    def compute(x, y):
        if x[0] in {1, 2, 3} and y[0] == 1:
            # (l, u1, lr, a1, a2, a3)
            var_shared = [0, 1, 4, 5, 6, 7]

        elif x[0] == 1 and y[0] in {2, 3}:
            # (l, u1, lr, a1, a2, a3)
            var_shared = [0, 1, 4, 5, 6, 7]

        elif x[0] == 2 and y[0] in {2, 3}:
            # (l, u1, u2, lr, a1, a2, a3)
            var_shared = [0, 1, 2, 4, 5, 6, 7]

        elif x[0] == 3 and y[0] == 2:
            # (l, u1, u2, lr, a1, a2, a3)
            var_shared = [0, 1, 2, 4, 5, 6, 7]

        elif x[0] == 3 and y[0] == 3:
            # (l, u1, u2, u3, lr, a1, a2, a3)
            var_shared = [0, 1, 2, 3, 4, 5, 6, 7]

        else:
            raise ValueError(f"Invalid (x[0], y[0]) combination: ({x[0]}, {y[0]})")

        dist = 0
        for i in var_shared:
            dist = dist + weights[i] * abs(x[i] - y[i]) ** p

        return np.power(dist, 1 / p)

    return compute


def hybrid_distance_variant3(weights, p=2):
    # For variant 3-4-5, there is two subproblem, one per optimizer: here, we are working directly in these subproblems
    # A point is passed as x = (l (0), u1 (1), u2 (2), lr (3), hp1 (4), hp2 (5), hp3 (6))
    # x[0] = 1 <-> l=1
    # x[0] = 2 <-> l=2

    # For variant 3-4-5, there is two subproblem, one per optimizer: here, we are working directly in these subproblems
    var_shared = None
    def compute(x, y):
        # Grouped conditions based on comment patterns
        if x[0] == 1 or y[0] == 1:
            # (l, u1, lr, a1, a2, a3)
            var_shared = [0, 1, 3, 4, 5, 6]

        else:
            # (l, u1, u2, lr, a1, a2, a3)
            var_shared = [0, 1, 2, 3, 4, 5, 6]

        # Define compute for ADAM
        dist = 0
        for i in var_shared:
            dist = dist + weights[i] * abs(x[i] - y[i]) ** p

        return np.power(dist, 1 / p)

    return compute


def hybrid_distance_variant4(weights, opt, p=2):
    # For variant 3-4-5, there is two subproblem, one per optimizer: here, we are working directly in these subproblems
    # ASGD: a point is passed as x = (l (0), u1 (1), u2 (2), lr (3), hp1 (4), hp2 (5), hp3 (6))
    # x[0] = 1 <-> l=1
    # x[0] = 2 <-> l=2

    # ADAM: a point is passed as x = (l (0), u1 (1), u2 (2), u(3), lr (4), hp1 (5), hp2 (6), hp3 (7))
    # x[0] = 1 <-> l=1
    # x[0] = 2 <-> l=2
    # x[0] = 3 <-> l=3

    # IMPORTANT: ASGD has NO unit u3, but ADAM does => check indices carefully for each subproblem

    # Find the variables shared for a given subproblem
    var_shared = None
    def compute(x, y):

        if opt == "ASGD":
            if x[0] == 1 or y[0] == 1:
                # (l, u1, lr, a1, a2, a3)
                var_shared = [0, 1, 3, 4, 5, 6]

            else:
                # (l, u1, u2, lr, a1, a2, a3)
                var_shared = [0, 1, 2, 3, 4, 5, 6]

        elif opt == "ADAM":
            if x[0] in {1, 2, 3} and y[0] == 1:
                # (l, u1, lr, b1, b2, b3)
                var_shared = [0, 1, 4, 5, 6, 7]

            elif x[0] == 1 and y[0] in {2, 3}:
                # (l, u1, lr, b1, b2, b3)
                var_shared = [0, 1, 4, 5, 6, 7]

            elif x[0] == 2 and y[0] in {2, 3}:
                # (l, u1, u2, lr, b1, b2, b3)
                var_shared = [0, 1, 2, 4, 5, 6, 7]

            elif x[0] == 3 and y[0] == 2:
                # (l, u1, u2, lr, b1, b2, b3)
                var_shared = [0, 1, 2, 4, 5, 6, 7]

            elif x[0] == 3 and y[0] == 3:
                # (l, u1, u2, u3, lr, b1, b2 , b3)
                var_shared = [0, 1, 2, 3, 4, 5, 6, 7]

        # Define compute for ADAM
        dist = 0
        for i in var_shared:
            dist = dist + weights[i] * abs(x[i] - y[i]) ** p

        return np.power(dist, 1 / p)

    return compute


def hybrid_distance_variant5(weights, opt, p=2):
    # For variant 3-4-5, there is two subproblem, one per optimizer: here, we are working directly in these subproblems
    # ASGD: a point is passed as x = (l (0), u1 (1), u2 (2), lr (3), hp1 (4), hp2 (5), hp3 (6), drop (7))
    # x[0] = 1 <-> l=1
    # x[0] = 2 <-> l=2

    # ADAM: a point is passed as x = (l (0), u1 (1), u2 (2), u(3), lr (4), hp1 (5), hp2 (6), hp3 (7), drop (8))
    # x[0] = 1 <-> l=1
    # x[0] = 2 <-> l=2
    # x[0] = 3 <-> l=3

    # IMPORTANT: ASGD has NO unit u3, but ADAM does => check indices carefully for each subproblem

    # Find the variables shared for a given subproblem
    var_shared = None
    def compute(x, y):

        if opt == "ASGD":
            if x[0] == 1 or y[0] == 1:
                # (l, u1, lr, a1, a2, a3)
                var_shared = [0, 1, 3, 4, 5, 6, 7]

            else:
                # (l, u1, u2, lr, a1, a2, a3)
                var_shared = [0, 1, 2, 3, 4, 5, 6, 7]

        elif opt == "ADAM":
            if x[0] in {1, 2, 3} and y[0] == 1:
                # (l, u1, lr, b1, b2, b3)
                var_shared = [0, 1, 4, 5, 6, 7, 8]

            elif x[0] == 1 and y[0] in {2, 3}:
                # (l, u1, lr, b1, b2, b3)
                var_shared = [0, 1, 4, 5, 6, 7, 8]

            elif x[0] == 2 and y[0] in {2, 3}:
                # (l, u1, u2, lr, b1, b2, b3)
                var_shared = [0, 1, 2, 4, 5, 6, 7, 8]

            elif x[0] == 3 and y[0] == 2:
                # (l, u1, u2, lr, b1, b2, b3)
                var_shared = [0, 1, 2, 4, 5, 6, 7, 8]

            elif x[0] == 3 and y[0] == 3:
                # (l, u1, u2, u3, lr, b1, b2 , b3)
                var_shared = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Define compute for ADAM
        dist = 0
        for i in var_shared:
            dist = dist + weights[i] * abs(x[i] - y[i]) ** p

        return np.power(dist, 1 / p)

    return compute