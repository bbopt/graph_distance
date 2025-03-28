import sys
import PyNomad
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import MixedVariableGA

from utils.import_data import load_data_optimizers, prep_naive_data, hybrid_data_optimizers
from utils.problems_instances_setup import variant_setup, variant_size_setup
from problems_variant4 import variant4_bb_wrapper
from collections import defaultdict


# Define PyMoo problem class with mixed variables
def create_pymoo_problem(bb_func, n_var, lower_bounds, upper_bounds, int_indices=[]):
    # Create a dictionary where each variable has a name and its type
    vars = {
        f"v{i}": Integer(bounds=(lower_bounds[i], upper_bounds[i])) if i in int_indices
        else Real(bounds=(lower_bounds[i], upper_bounds[i]))
        for i in range(n_var)
    }

    class OptimizationProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(vars=vars, n_obj=1)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = -bb_func(*X.values())  # Pass dictionary values directly to bb_func

    return OptimizationProblem()


if __name__ == '__main__':

    #approaches = ["naive", "graph", "hybrid"]
    approaches = ["naive", "graph", "hybrid"]
    #approaches = ["graph"]
    models = ["KNN"]

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
    is_CNN = False

    #nb_pts, nb_test_pts, nb_valid_pts = variant_setup(variant)
    nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, 1)

    seed_setup_list = list(range(5))  # 20
    accuracy_validation = {(seed, approach): None for seed in seed_setup_list for approach in approaches}
    accuracy_test = {(seed, approach): None for seed in seed_setup_list for approach in approaches}
    for seed_setup in seed_setup_list:
        for approach in approaches:
            print(approach, " on seed ", seed_setup)
            for model in models:

                # Output log file name
                if is_CNN:
                    log_file = "log_" + variant + "_" + approach + "_" + model + "_CNN_classification" + ".txt"
                else:
                    log_file = "log_" + variant + "_" + approach + "_" + model + "_MLP_classification" + ".txt"

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

                budget = 100 * nb_params


                # Data file name
                if is_CNN:
                    data_file = "labeled_data_" + variant + "_CNN" + ".xlsx"
                else:
                    data_file = "labeled_data_" + variant + "_MLP" + ".xlsx"

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

                    #lb = [10, 10, 2.5, 0.5, 3, 0.2, 0.24995, 1.5,
                    #      0,
                    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #      1]
                    # log
                    lb = [1, 1, 0.4, -0.3, 0.5, -0.7, -0.6, 0.15,
                          -20,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1]

                    #ub = [1e9] * (8+1) + [1] * 12 + [200]
                    # log
                    ub = [20] * (8+1) + [1] * 12 + [300]

                    problem = create_pymoo_problem(bb, 22, lb, ub, [21])

                elif approach == "naive":

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
                    problem = create_pymoo_problem(bb, 34, lb, ub, list(range(29, 34)))

                elif approach == "hybrid":

                    # Setup : 17 variables
                    # w11, w12, w13, w14, w15, w16, w17      = (l, u1, u2, lr, lamdb, alpha, t0)
                    # w21, w22, w23, w24, w25, w26, w27, w28 = (l, u1, u2, lr, beta1, beta2, eps)
                    # K1, K2
                    x0 = []
                    lb = [0] * 15 + [1] * 2
                    #ub = [1e6] * 15 + [50] * 2
                    ub = [1] * 15 + [50]*2
                    problem = create_pymoo_problem(bb, 17, lb, ub, [15, 16])

                # Use MixedVariableGA instead of GA
                algorithm = MixedVariableGA()

                # Run optimization
                res = minimize(
                    problem,
                    algorithm,
                    termination=("n_evals", budget),  # Use evaluation budget
                    seed=seed_setup,
                    verbose=True
                )

                # Extract best solution
                x_best = list(res.X.values())  # Extract values from dictionary

                # Evaluate final solution
                accuracy_validation[(seed_setup, approach)] = bb(*x_best)
                print(bb(*x_best))

                # Test the best solution
                bb_test = variant4_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub, approach, model)
                accuracy_test[(seed_setup, approach)] = bb_test(*x_best)
                print(bb_test(*x_best))


    # Compute sum of accuracies for each approach
    num_seeds = len(seed_setup_list)


    def compute_average(accuracy_dict):
        """ Compute average accuracy for each approach over seeds """
        approach_sums = defaultdict(float)

        for (seed, approach), value in accuracy_dict.items():
            approach_sums[approach] += value

        return {approach: approach_sums[approach] / num_seeds for approach in approach_sums}


    avg_validation = compute_average(accuracy_validation)
    avg_test = compute_average(accuracy_test)

    # Display results
    print("Validation Averages:", avg_validation)
    print("Test Averages:", avg_test)

