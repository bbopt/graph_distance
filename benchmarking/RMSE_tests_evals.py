import pandas as pd
from utils.import_data import load_data_fix_optimizer, load_data_optimizers, prep_naive_data
from utils.problems_instances_setup import variant_size_setup
from benchmarking.problems.problems_variant1 import *
from benchmarking.problems.problems_variant2 import *
from benchmarking.problems.problems_variant3 import *
from benchmarking.problems.problems_variant4 import *
from benchmarking.problems.problems_variant5 import *


# Setup
nb_variants = 5
nb_instances_per_variant = 4  # in total 5*4=20 pbs
nb_pbs = nb_variants * nb_instances_per_variant
approaches = ["graph", "naive"]
models = ["KNN", "IDW"]


def output_file(filename, it, fct_value):

    with open("logs_test/"+filename, "a") as f:
        f.write(str(it) + " " + str(fct_value) + "\n")


if __name__ == '__main__':

    for i in range(nb_variants):
        for j in range(nb_instances_per_variant):

            if j == 0:
                size = 1
            elif j == 1:
                size = 2
            elif j == 2:
                size = 3
            elif j == 3:
                size = 4

            for approach in approaches:
                for model in models:
                    data_file = "log_variant" + str(i + 1) + "_size" + str(j + 1) + "_" + approach + "_" + model + ".txt"
                    data = pd.read_csv("logs/" + data_file, delim_whitespace=True, header=None)

                    iterations = data[0].tolist()
                    sols_array = data.iloc[:, 2:].to_numpy(dtype=object)  # 2D array of all columns except first two columns
                    nb_evals, nb_var = sols_array.shape

                    for eval_idx in range(nb_evals):
                        point = sols_array[eval_idx]

                        # --------- Problem depends on the problem (variant-instance) and solver (approach-model) --- #
                        # Variant 1
                        if i == 0:
                            variant = "variant1"

                            # Setup for variant
                            nb_sub = 3
                            nb_max_var = 5  # l, u1, u2, u3, lr
                            nb_var_sub = [2, 3, 4]
                            var_inc_sub = [[1, 4], [1, 2, 4], [1, 2, 3, 4]]  # included variables in each subproblem

                            # Setup for instance

                            nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, size)

                            # Data file name: variant variable is set once at the beginning of the index-variant loop
                            data_file = "data_files/" + "data_" + variant + "_" + "size" + str(size) + ".xlsx"

                            # Load data: instance 1 has a fixed optimizer
                            X_train, y_train, X_valid, y_valid, X_test, y_test =\
                                load_data_fix_optimizer(data_file, nb_sub, nb_max_var, nb_pts, nb_test_pts, nb_valid_pts)

                            if approach == "naive":
                                # Modifiy data for naive approach
                                X_train, y_train, X_valid, y_valid, X_test, y_test =\
                                    prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, variant)

                            # Create bb_test
                            bb_test = variant1_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub, approach, model)

                            # Eval bb_test with last solution on the validation
                            fct_test_value = bb_test(*point)

                            # Print evaluation and test fonction value
                            output_filename = "log_variant" + str(i + 1) + "_" + "size" + str(size) + "_" + approach + "_" + model + ".txt"
                            output_file(output_filename, iterations[eval_idx], fct_test_value)


                        # Variant 2
                        elif i == 1:
                            variant = "variant2"

                            # Setup for variant
                            nb_sub = 3
                            nb_max_var = 8
                            nb_var_sub = [5, 6, 7]
                            var_inc_sub = [[1, 4, 5, 6, 7], [1, 2, 4, 5, 6, 7],
                                           [1, 2, 3, 4, 5, 6, 7]]  # included variables in each subproblem

                            # Setup for instance
                            nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, size)

                            # Data file name: variant variable is set once at the beginning of the index-variant loop
                            data_file = "data_files/" + "data_" + variant + "_" + "size" + str(size) + ".xlsx"

                            # Load data: instance 1 has a fixed optimizer
                            X_train, y_train, X_valid, y_valid, X_test, y_test = \
                                load_data_fix_optimizer(data_file, nb_sub, nb_max_var, nb_pts, nb_test_pts,
                                                        nb_valid_pts)

                            if approach == "naive":
                                # Modifiy data for naive approach
                                X_train, y_train, X_valid, y_valid, X_test, y_test = \
                                    prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub,
                                                    variant)

                            # Create bb_test
                            bb_test = variant2_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub,
                                                          approach, model)

                            # Eval bb_test with last solution on the validation
                            fct_test_value = bb_test(*point)

                            # Print evaluation and test fonction value
                            output_filename = "log_variant" + str(i + 1) + "_" + "size" + str(size) + "_" + approach + "_" + model + ".txt"
                            output_file(output_filename, iterations[eval_idx], fct_test_value)

                        # Variant 3
                        elif i == 2:
                            variant = "variant3"

                            # Setup for variant
                            optimizers = ["ASGD", "ADAM"]
                            nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 2}
                            nb_max_var = 11  # 12 - 1, with u3 removed
                            var_inc_sub = {("ASGD", 1): [2, 4, 5, 6, 7], ("ASGD", 2): [2, 3, 4, 5, 6, 7],
                                           ("ADAM", 1): [2, 4, 8, 9, 10], ("ADAM", 2): [2, 3, 4, 8, 9, 10]}
                            nb_var_sub = {("ASGD", 1): 5, ("ASGD", 2): 6, ("ADAM", 1): 5, ("ADAM", 2): 6}

                            # Setup for instance
                            nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, size)

                            # Data file name: variant variable is set once at the beginning of the index-variant loop
                            data_file = "data_files/" + "data_" + variant + "_" + "size" + str(size) + ".xlsx"

                            # Load data
                            X_train, y_train, X_valid, y_valid, X_test, y_test =\
                                load_data_optimizers(data_file, optimizers, nb_sub_per_optimizer, nb_max_var,
                                                     nb_pts, nb_test_pts, nb_valid_pts)

                            if approach == "naive":
                                # Modifiy data for naive approach
                                X_train, y_train, X_valid, y_valid, X_test, y_test = \
                                    prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub,
                                                    variant)

                            # Create bb_test
                            bb_test = variant3_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub,
                                                          approach, model)

                            # Eval bb_test with last solution on the validation
                            fct_test_value = bb_test(*point)

                            # Print evaluation and test fonction value
                            output_filename = "log_variant" + str(i + 1) + "_" + "size" + str(size) + "_" + approach + "_" + model + ".txt"
                            output_file(output_filename, iterations[eval_idx], fct_test_value)

                        # Variant 4
                        elif i == 3:
                            variant = "variant4"

                            # Setup for variant
                            optimizers = ["ASGD", "ADAM"]
                            nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 3}
                            nb_max_var = 12
                            var_inc_sub = {("ASGD", 1): [2, 5, 6, 7, 8], ("ASGD", 2): [2, 3, 5, 6, 7, 8],
                                           ("ADAM", 1): [2, 5, 9, 10, 11], ("ADAM", 2): [2, 3, 5, 9, 10, 11],
                                           ("ADAM", 3): [2, 3, 4, 5, 9, 10, 11]}
                            nb_var_sub = {("ASGD", 1): 5, ("ASGD", 2): 6, ("ADAM", 1): 5, ("ADAM", 2): 6, ("ADAM", 3): 7}

                            # Setup for instance
                            nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, size)

                            # Data file name: variant variable is set once at the beginning of the index-variant loop
                            data_file = "data_files/" + "data_" + variant + "_" + "size" + str(size) + ".xlsx"

                            # Load data
                            X_train, y_train, X_valid, y_valid, X_test, y_test = \
                                load_data_optimizers(data_file, optimizers, nb_sub_per_optimizer, nb_max_var,
                                                     nb_pts, nb_test_pts, nb_valid_pts)

                            if approach == "naive":
                                # Modifiy data for naive approach
                                X_train, y_train, X_valid, y_valid, X_test, y_test = \
                                    prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub,
                                                    variant)

                            # Create bb_test
                            bb_test = variant4_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub,
                                                          approach, model)

                            # Eval bb_test with last solution on the validation
                            fct_test_value = bb_test(*point)

                            # Print evaluation and test fonction value
                            output_filename = "log_variant" + str(i + 1) + "_" + "size" + str(size) + "_" + approach + "_" + model + ".txt"
                            output_file(output_filename, iterations[eval_idx], fct_test_value)

                        # Variant 5
                        elif i == 4:
                            variant = "variant5"

                            # Setup for variant
                            optimizers = ["ASGD", "ADAM"]
                            nb_sub_per_optimizer = {"ASGD": 2, "ADAM": 3}
                            nb_max_var = 13  # add the dropout
                            var_inc_sub = {("ASGD", 1): [2, 5, 6, 7, 8, 12], ("ASGD", 2): [2, 3, 5, 6, 7, 8, 12],
                                           ("ADAM", 1): [2, 5, 9, 10, 11, 12], ("ADAM", 2): [2, 3, 5, 9, 10, 11, 12],
                                           ("ADAM", 3): [2, 3, 4, 5, 9, 10, 11, 12]}
                            nb_var_sub = {("ASGD", 1): 6, ("ASGD", 2): 7, ("ADAM", 1): 6, ("ADAM", 2): 7, ("ADAM", 3): 8}

                            # Setup for instance
                            nb_pts, nb_test_pts, nb_valid_pts = variant_size_setup(variant, size)

                            # Data file name: variant variable is set once at the beginning of the index-variant loop
                            data_file = "data_files/" + "data_" + variant + "_" + "size" + str(size) + ".xlsx"

                            # Load data
                            X_train, y_train, X_valid, y_valid, X_test, y_test = \
                                load_data_optimizers(data_file, optimizers, nb_sub_per_optimizer, nb_max_var,
                                                     nb_pts, nb_test_pts, nb_valid_pts)

                            if approach == "naive":
                                # Modifiy data for naive approach
                                X_train, y_train, X_valid, y_valid, X_test, y_test = \
                                    prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub,
                                                    variant)

                            # Create bb_test
                            bb_test = variant5_bb_wrapper(X_train, y_train, X_test, y_test, nb_max_var, nb_var_sub,
                                                          approach, model)

                            # Eval bb_test with last solution on the validation
                            fct_test_value = bb_test(*point)

                            # Print evaluation and test fonction value
                            output_filename = "log_variant" + str(i + 1) + "_" + "size" + str(size) + "_" + approach + "_" + model + ".txt"
                            output_file(output_filename, iterations[eval_idx], fct_test_value)

