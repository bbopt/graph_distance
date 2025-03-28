import numpy as np


def draw_subpbs_candidates_variant1_2(idx_train_subs1, idx_train_subs2, idx_train_subs3):

    sub1_nempty = np.any(idx_train_subs1)
    sub2_nempty = np.any(idx_train_subs2)
    sub3_nempty = np.any(idx_train_subs3)
    list_sub_nempty = [sub1_nempty, sub2_nempty, sub3_nempty]

    draw = []
    for i, sub_empty in enumerate(list_sub_nempty):
        if sub_empty:
            draw.append(i+1)

    return draw


# variant 3
def draw_subpbs_candidates_variant3(idx_train_ASGD_l1, idx_train_ASGD_l2, idx_train_ADAM_l1, idx_train_ADAM_l2):

    sub_ASGD_l1_nempty = np.any(idx_train_ASGD_l1)
    sub_ASGD_l2_nempty = np.any(idx_train_ASGD_l2)
    sub_ADAM_l1_nempty = np.any(idx_train_ADAM_l1)
    sub_ADAM_l2_nempty = np.any(idx_train_ADAM_l2)
    list_sub_nempty = [sub_ASGD_l1_nempty, sub_ASGD_l2_nempty, sub_ADAM_l1_nempty, sub_ADAM_l2_nempty]

    draw = []
    for i, sub_empty in enumerate(list_sub_nempty):
        if sub_empty:
            draw.append(i+1)

    return draw


def draw_subpbs_candidates_variant4_5(idx_train_ASGD_l1, idx_train_ASGD_l2, idx_train_ADAM_l1, idx_train_ADAM_l2, idx_train_ADAM_l3):

    sub_ASGD_l1_nempty = np.any(idx_train_ASGD_l1)
    sub_ASGD_l2_nempty = np.any(idx_train_ASGD_l2)
    sub_ADAM_l1_nempty = np.any(idx_train_ADAM_l1)
    sub_ADAM_l2_nempty = np.any(idx_train_ADAM_l2)
    sub_ADAM_l3_nempty = np.any(idx_train_ADAM_l3)
    list_sub_nempty = [sub_ASGD_l1_nempty, sub_ASGD_l2_nempty, sub_ADAM_l1_nempty, sub_ADAM_l2_nempty, sub_ADAM_l3_nempty]

    draw = []
    for i, sub_empty in enumerate(list_sub_nempty):
        if sub_empty:
            draw.append(i+1)

    return draw