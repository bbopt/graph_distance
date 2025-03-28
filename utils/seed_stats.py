from collections import defaultdict
import numpy as np


def compute_average(accuracy_dict):
    """ Compute average accuracy for each approach over seeds """
    approach_sums = defaultdict(float)
    approach_counts = defaultdict(int)

    for (seed, approach), value in accuracy_dict.items():
        approach_sums[approach] += value
        approach_counts[approach] += 1

    return {approach: approach_sums[approach] / approach_counts[approach] for approach in approach_sums}


def compute_std(accuracy_dict, avg_dict):
    """ Compute standard deviation for each approach over seeds """
    approach_sq_diff = defaultdict(float)
    approach_counts = defaultdict(int)

    for (seed, approach), value in accuracy_dict.items():
        approach_sq_diff[approach] += (value - avg_dict[approach]) ** 2
        approach_counts[approach] += 1

    return {approach: np.sqrt(approach_sq_diff[approach] / approach_counts[approach]) for approach in
            approach_sq_diff}