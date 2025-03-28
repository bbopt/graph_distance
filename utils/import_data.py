import numpy as np
import pandas as pd


def load_data_fix_optimizer(file_data, nb_sub, nb_max_var, nb_pts, nb_test_pts, nb_valid_pts, seed_setup=0):

    np.random.seed(seed_setup)

    # Load excel tabs, each tab related to a subproblem k+1
    data_dict = {k: pd.read_excel(file_data, 'sub' + str(k+1)).to_numpy() for k in range(nb_sub)}

    # Shuffle each dataset of each subproblem: ensure it is shuffle for the subproblems
    if seed_setup != 0:
        for k in range(nb_sub):
            np.random.shuffle(data_dict[k])

    # Separate into test, train, validation for each subproblem
    test_dict = {k: data_dict[k][0:nb_test_pts[k]] for k in range(nb_sub)}
    valid_dict = {k: data_dict[k][nb_test_pts[k]:(nb_test_pts[k]+nb_valid_pts[k])] for k in range(nb_sub)}
    train_dict = {k: data_dict[k][(nb_test_pts[k]+nb_valid_pts[k]):nb_pts[k]] for k in range(nb_sub)}

    # Stack the data from the different subproblems
    test = np.empty((0, nb_max_var+1), dtype=object)  # +1 to include the accuracies
    valid = np.empty((0, nb_max_var+1), dtype=object)
    train = np.empty((0, nb_max_var+1), dtype=object)

    for k in range(nb_sub):
        test = np.vstack((test, test_dict[k]))
        valid = np.vstack((valid, valid_dict[k]))
        train = np.vstack((train, train_dict[k]))

    # Separate into X and y
    X_train, y_train = train[:, :-1], train[:, -1]
    X_valid, y_valid = valid[:, :-1], valid[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# optimizers = ["ASGD", "ADAM"]
def load_data_optimizers(file_data, optimizers, nb_sub_per_optimizer, nb_max_var, nb_pts, nb_test_pts, nb_valid_pts, seed_setup=0):

    np.random.seed(seed_setup)

    # Load excel tabs, each tab related to a subproblem k+1
    data_dict = {(opt, k): pd.read_excel(file_data, 'sub' + str(k + 1) + "_" + opt).to_numpy()
                 for opt in optimizers for k in range(nb_sub_per_optimizer[opt])}

    # Shuffle each dataset of each subproblem: ensure it is shuffle for the subproblems
    if seed_setup != 0:
        for opt in optimizers:
            for k in range(nb_sub_per_optimizer[opt]):
                np.random.shuffle(data_dict[(opt, k)])

    # Separate into test, train, validation for each subproblem
    test_dict = {(opt, k): data_dict[(opt, k)][0:nb_test_pts[k]]
                 for opt in optimizers for k in range(nb_sub_per_optimizer[opt])}
    valid_dict = {(opt, k): data_dict[(opt, k)][nb_test_pts[k]:(nb_test_pts[k] + nb_valid_pts[k])]
                  for opt in optimizers for k in range(nb_sub_per_optimizer[opt])}
    train_dict = {(opt, k): data_dict[(opt, k)][(nb_test_pts[k] + nb_valid_pts[k]):nb_pts[k]]
                  for opt in optimizers for k in range(nb_sub_per_optimizer[opt])}

    # Stack the data from the different subproblems
    test = np.empty((0, nb_max_var + 1), dtype=object)  # +1 to include the accuracies
    valid = np.empty((0, nb_max_var + 1), dtype=object)
    train = np.empty((0, nb_max_var + 1), dtype=object)

    for opt in optimizers:
        for k in range(nb_sub_per_optimizer[opt]):
            test = np.vstack((test, test_dict[(opt, k)]))
            valid = np.vstack((valid, valid_dict[(opt, k)]))
            train = np.vstack((train, train_dict[(opt, k)]))

    # Shuffle the arrays in unison: extra layer of shuffle (already done on the subproblems)
    #np.random.shuffle(test), np.random.shuffle(valid), np.random.shuffle(train)

    # Separate into X and y
    X_train, y_train = train[:, :-1], train[:, -1]
    X_valid, y_valid = valid[:, :-1], valid[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def prep_naive_data(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub, variant):

    if variant == "variant1" or variant == "variant2":
        return naive_data_fixed_optimizer(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub)
    elif variant == "variant3":
        return naive_data_optimizers_twosubpbs(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub)
    elif variant == "variant4" or variant == "variant5":
        return naive_data_optimizers_twothreesubpbs(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub)


# variant 1 and 2
def naive_data_fixed_optimizer(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub):
    # Validation data for naive
    idx_valid_sub1 = (np.where(X_valid[:, 0] == 1)[0]).tolist()
    idx_valid_sub2 = (np.where(X_valid[:, 0] == 2)[0]).tolist()
    idx_valid_sub3 = (np.where(X_valid[:, 0] == 3)[0]).tolist()

    X_valid_naive = {"sub1": X_valid[np.ix_(idx_valid_sub1, var_inc_sub[0])],
                     "sub2": X_valid[np.ix_(idx_valid_sub2, var_inc_sub[1])],
                     "sub3": X_valid[np.ix_(idx_valid_sub3, var_inc_sub[2])]}
    y_valid_naive = {"sub1": y_valid[idx_valid_sub1], "sub2": y_valid[idx_valid_sub2], "sub3": y_valid[idx_valid_sub3]}

    # Test data for naive
    idx_test_sub1 = (np.where(X_test[:, 0] == 1)[0]).tolist()
    idx_test_sub2 = (np.where(X_test[:, 0] == 2)[0]).tolist()
    idx_test_sub3 = (np.where(X_test[:, 0] == 3)[0]).tolist()

    X_test_naive = {"sub1": X_test[np.ix_(idx_test_sub1, var_inc_sub[0])],
                    "sub2": X_test[np.ix_(idx_test_sub2, var_inc_sub[1])],
                    "sub3": X_test[np.ix_(idx_test_sub3, var_inc_sub[2])]}
    y_test_naive = {"sub1": y_test[idx_test_sub1], "sub2": y_test[idx_test_sub2], "sub3": y_test[idx_test_sub3]}

    # Training data for naive
    idx_train_sub1 = (np.where(X_train[:, 0] == 1)[0]).tolist()
    idx_train_sub2 = (np.where(X_train[:, 0] == 2)[0]).tolist()
    idx_train_sub3 = (np.where(X_train[:, 0] == 3)[0]).tolist()

    X_train_naive = {"sub1": X_train[np.ix_(idx_train_sub1, var_inc_sub[0])],
                     "sub2": X_train[np.ix_(idx_train_sub2, var_inc_sub[1])],
                     "sub3": X_train[np.ix_(idx_train_sub3, var_inc_sub[2])]}
    y_train_naive = {"sub1": y_train[idx_train_sub1], "sub2": y_train[idx_train_sub2], "sub3": y_train[idx_train_sub3]}

    return X_train_naive, y_train_naive, X_valid_naive, y_valid_naive, X_test_naive, y_test_naive


# variant 3
def naive_data_optimizers_twosubpbs(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub):
    # Validation data for naive
    idx_valid_ASGD_l1 = (np.where((X_valid[:, 0] == "ASGD") & (X_valid[:, 1] == 1))[0]).tolist()
    idx_valid_ASGD_l2 = (np.where((X_valid[:, 0] == "ASGD") & (X_valid[:, 1] == 2))[0]).tolist()
    idx_valid_Adam_l1 = (np.where((X_valid[:, 0] == "ADAM") & (X_valid[:, 1] == 1))[0]).tolist()
    idx_valid_Adam_l2 = (np.where((X_valid[:, 0] == "ADAM") & (X_valid[:, 1] == 2))[0]).tolist()

    # var_inc_sub is a dictionnary with two entries (opt, nb_hidden_layer) -> (str, int)
    X_valid_naive = {("ASGD", 1): X_valid[np.ix_(idx_valid_ASGD_l1, var_inc_sub["ASGD", 1])],
                     ("ASGD", 2): X_valid[np.ix_(idx_valid_ASGD_l2, var_inc_sub["ASGD", 2])],
                     ("ADAM", 1): X_valid[np.ix_(idx_valid_Adam_l1, var_inc_sub["ADAM", 1])],
                     ("ADAM", 2): X_valid[np.ix_(idx_valid_Adam_l2, var_inc_sub["ADAM", 2])]}
    y_valid_naive = {("ASGD", 1): y_valid[idx_valid_ASGD_l1], ("ASGD", 2): y_valid[idx_valid_ASGD_l2],
                     ("ADAM", 1): y_valid[idx_valid_Adam_l1], ("ADAM", 2): y_valid[idx_valid_Adam_l2]}

    # Test data for naive
    idx_test_ASGD_l1 = (np.where((X_test[:, 0] == "ASGD") & (X_test[:, 1] == 1))[0]).tolist()
    idx_test_ASGD_l2 = (np.where((X_test[:, 0] == "ASGD") & (X_test[:, 1] == 2))[0]).tolist()
    idx_test_Adam_l1 = (np.where((X_test[:, 0] == "ADAM") & (X_test[:, 1] == 1))[0]).tolist()
    idx_test_Adam_l2 = (np.where((X_test[:, 0] == "ADAM") & (X_test[:, 1] == 2))[0]).tolist()

    # var_inc_sub is a dictionnary with two entries (opt, nb_hidden_layer) -> (str, int)
    X_test_naive = {("ASGD", 1): X_test[np.ix_(idx_test_ASGD_l1, var_inc_sub["ASGD", 1])],
                    ("ASGD", 2): X_test[np.ix_(idx_test_ASGD_l2, var_inc_sub["ASGD", 2])],
                    ("ADAM", 1): X_test[np.ix_(idx_test_Adam_l1, var_inc_sub["ADAM", 1])],
                    ("ADAM", 2): X_test[np.ix_(idx_test_Adam_l2, var_inc_sub["ADAM", 2])]}
    y_test_naive = {("ASGD", 1): y_test[idx_test_ASGD_l1], ("ASGD", 2): y_test[idx_test_ASGD_l2],
                    ("ADAM", 1): y_test[idx_test_Adam_l1], ("ADAM", 2): y_test[idx_test_Adam_l2]}

    # Train data for naive
    idx_train_ASGD_l1 = (np.where((X_train[:, 0] == "ASGD") & (X_train[:, 1] == 1))[0]).tolist()
    idx_train_ASGD_l2 = (np.where((X_train[:, 0] == "ASGD") & (X_train[:, 1] == 2))[0]).tolist()
    idx_train_Adam_l1 = (np.where((X_train[:, 0] == "ADAM") & (X_train[:, 1] == 1))[0]).tolist()
    idx_train_Adam_l2 = (np.where((X_train[:, 0] == "ADAM") & (X_train[:, 1] == 2))[0]).tolist()

    # var_inc_sub is a dictionnary with two entries (opt, nb_hidden_layer) -> (str, int)
    X_train_naive = {("ASGD", 1): X_train[np.ix_(idx_train_ASGD_l1, var_inc_sub["ASGD", 1])],
                     ("ASGD", 2): X_train[np.ix_(idx_train_ASGD_l2, var_inc_sub["ASGD", 2])],
                     ("ADAM", 1): X_train[np.ix_(idx_train_Adam_l1, var_inc_sub["ADAM", 1])],
                     ("ADAM", 2): X_train[np.ix_(idx_train_Adam_l2, var_inc_sub["ADAM", 2])]}
    y_train_naive = {("ASGD", 1): y_train[idx_train_ASGD_l1], ("ASGD", 2): y_train[idx_train_ASGD_l2],
                     ("ADAM", 1): y_train[idx_train_Adam_l1], ("ADAM", 2): y_train[idx_train_Adam_l2]}

    return X_train_naive, y_train_naive, X_valid_naive, y_valid_naive, X_test_naive, y_test_naive


# variant 4 and 5
def naive_data_optimizers_twothreesubpbs(X_train, y_train, X_valid, y_valid, X_test, y_test, var_inc_sub):
    # Validation data for naive
    idx_valid_ASGD_l1 = (np.where((X_valid[:, 0] == "ASGD") & (X_valid[:, 1] == 1))[0]).tolist()
    idx_valid_ASGD_l2 = (np.where((X_valid[:, 0] == "ASGD") & (X_valid[:, 1] == 2))[0]).tolist()
    idx_valid_Adam_l1 = (np.where((X_valid[:, 0] == "ADAM") & (X_valid[:, 1] == 1))[0]).tolist()
    idx_valid_Adam_l2 = (np.where((X_valid[:, 0] == "ADAM") & (X_valid[:, 1] == 2))[0]).tolist()
    idx_valid_Adam_l3 = (np.where((X_valid[:, 0] == "ADAM") & (X_valid[:, 1] == 3))[0]).tolist()

    # var_inc_sub is a dictionnary with two entries (opt, nb_hidden_layer) -> (str, int)
    X_valid_naive = {("ASGD", 1): X_valid[np.ix_(idx_valid_ASGD_l1, var_inc_sub["ASGD", 1])],
                     ("ASGD", 2): X_valid[np.ix_(idx_valid_ASGD_l2, var_inc_sub["ASGD", 2])],
                     ("ADAM", 1): X_valid[np.ix_(idx_valid_Adam_l1, var_inc_sub["ADAM", 1])],
                     ("ADAM", 2): X_valid[np.ix_(idx_valid_Adam_l2, var_inc_sub["ADAM", 2])],
                     ("ADAM", 3): X_valid[np.ix_(idx_valid_Adam_l3, var_inc_sub["ADAM", 3])]}
    y_valid_naive = {("ASGD", 1): y_valid[idx_valid_ASGD_l1], ("ASGD", 2): y_valid[idx_valid_ASGD_l2],
                     ("ADAM", 1): y_valid[idx_valid_Adam_l1], ("ADAM", 2): y_valid[idx_valid_Adam_l2],
                     ("ADAM", 3): y_valid[idx_valid_Adam_l3]}

    # Test data for naive
    idx_test_ASGD_l1 = (np.where((X_test[:, 0] == "ASGD") & (X_test[:, 1] == 1))[0]).tolist()
    idx_test_ASGD_l2 = (np.where((X_test[:, 0] == "ASGD") & (X_test[:, 1] == 2))[0]).tolist()
    idx_test_Adam_l1 = (np.where((X_test[:, 0] == "ADAM") & (X_test[:, 1] == 1))[0]).tolist()
    idx_test_Adam_l2 = (np.where((X_test[:, 0] == "ADAM") & (X_test[:, 1] == 2))[0]).tolist()
    idx_test_Adam_l3 = (np.where((X_test[:, 0] == "ADAM") & (X_test[:, 1] == 3))[0]).tolist()

    # var_inc_sub is a dictionnary with two entries (opt, nb_hidden_layer) -> (str, int)
    X_test_naive = {("ASGD", 1): X_test[np.ix_(idx_test_ASGD_l1, var_inc_sub["ASGD", 1])],
                    ("ASGD", 2): X_test[np.ix_(idx_test_ASGD_l2, var_inc_sub["ASGD", 2])],
                    ("ADAM", 1): X_test[np.ix_(idx_test_Adam_l1, var_inc_sub["ADAM", 1])],
                    ("ADAM", 2): X_test[np.ix_(idx_test_Adam_l2, var_inc_sub["ADAM", 2])],
                    ("ADAM", 3): X_test[np.ix_(idx_test_Adam_l3, var_inc_sub["ADAM", 3])]}
    y_test_naive = {("ASGD", 1): y_test[idx_test_ASGD_l1], ("ASGD", 2): y_test[idx_test_ASGD_l2],
                    ("ADAM", 1): y_test[idx_test_Adam_l1], ("ADAM", 2): y_test[idx_test_Adam_l2],
                    ("ADAM", 3): y_test[idx_test_Adam_l3]}

    # Train data for naive
    idx_train_ASGD_l1 = (np.where((X_train[:, 0] == "ASGD") & (X_train[:, 1] == 1))[0]).tolist()
    idx_train_ASGD_l2 = (np.where((X_train[:, 0] == "ASGD") & (X_train[:, 1] == 2))[0]).tolist()
    idx_train_Adam_l1 = (np.where((X_train[:, 0] == "ADAM") & (X_train[:, 1] == 1))[0]).tolist()
    idx_train_Adam_l2 = (np.where((X_train[:, 0] == "ADAM") & (X_train[:, 1] == 2))[0]).tolist()
    idx_train_Adam_l3 = (np.where((X_train[:, 0] == "ADAM") & (X_train[:, 1] == 3))[0]).tolist()

    # var_inc_sub is a dictionnary with two entries (opt, nb_hidden_layer) -> (str, int)
    X_train_naive = {("ASGD", 1): X_train[np.ix_(idx_train_ASGD_l1, var_inc_sub["ASGD", 1])],
                     ("ASGD", 2): X_train[np.ix_(idx_train_ASGD_l2, var_inc_sub["ASGD", 2])],
                     ("ADAM", 1): X_train[np.ix_(idx_train_Adam_l1, var_inc_sub["ADAM", 1])],
                     ("ADAM", 2): X_train[np.ix_(idx_train_Adam_l2, var_inc_sub["ADAM", 2])],
                     ("ADAM", 3): X_train[np.ix_(idx_train_Adam_l3, var_inc_sub["ADAM", 3])]}

    y_train_naive = {("ASGD", 1): y_train[idx_train_ASGD_l1], ("ASGD", 2): y_train[idx_train_ASGD_l2],
                     ("ADAM", 1): y_train[idx_train_Adam_l1], ("ADAM", 2): y_train[idx_train_Adam_l2],
                     ("ADAM", 3): y_train[idx_train_Adam_l3]}

    return X_train_naive, y_train_naive, X_valid_naive, y_valid_naive, X_test_naive, y_test_naive


# variant 3,4 and 5 for hybrid where there's a subproblem per optimizer
def hybrid_data_optimizers(X_train, y_train, X_valid, y_valid, X_test, y_test, var_sub):
    # Validation data for hybrid
    idx_valid_ASGD = (np.where(X_valid[:, 0] == "ASGD")[0]).tolist()
    idx_valid_ADAM = (np.where(X_valid[:, 0] == "ADAM")[0]).tolist()

    X_valid_hybrid = {"ASGD": X_valid[np.ix_(idx_valid_ASGD, var_sub["ASGD"])],
                      "ADAM": X_valid[np.ix_(idx_valid_ADAM, var_sub["ADAM"])]}
    y_valid_hybrid = {"ASGD": y_valid[idx_valid_ASGD],
                      "ADAM": y_valid[idx_valid_ADAM]}

    # Test data for hybrid
    idx_test_ASGD = (np.where(X_test[:, 0] == "ASGD")[0]).tolist()
    idx_test_ADAM = (np.where(X_test[:, 0] == "ADAM")[0]).tolist()

    X_test_hybrid = {"ASGD": X_test[np.ix_(idx_test_ASGD, var_sub["ASGD"])],
                     "ADAM": X_test[np.ix_(idx_test_ADAM, var_sub["ADAM"])]}
    y_test_hybrid = {"ASGD": y_test[idx_test_ASGD],
                     "ADAM": y_test[idx_test_ADAM]}

    # Train data for naive
    idx_train_ASGD = (np.where(X_train[:, 0] == "ASGD")[0]).tolist()
    idx_train_ADAM = (np.where(X_train[:, 0] == "ADAM")[0]).tolist()

    X_train_hybrid = {"ASGD": X_train[np.ix_(idx_train_ASGD, var_sub["ASGD"])],
                      "ADAM": X_train[np.ix_(idx_train_ADAM, var_sub["ADAM"])]}
    y_train_hybrid = {"ASGD": y_train[idx_train_ASGD],
                      "ADAM": y_train[idx_train_ADAM]}

    return X_train_hybrid, y_train_hybrid, X_valid_hybrid, y_valid_hybrid, X_test_hybrid, y_test_hybrid

