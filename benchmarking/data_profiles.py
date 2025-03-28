import numpy as np
import pandas as pd
import sys, os
import math
from matplotlib import pyplot as plt


def nb_evals_convergence(tau, fct_min_pb, evals, fct_values):

    nb_evals = math.inf
    for idx, eval in enumerate(evals):
        if abs((fct_values[idx] - fct_min_pb)/fct_min_pb) <= tau:
            nb_evals = eval
            break  # end forloop if convergence is met

    return nb_evals


def normalizing_constant(idx_variant, approach, model):

    constant = None
    if idx_variant == 0:  # variant 1
        if approach == "graph" and model == "IDW":
            constant = 7
        elif approach == "graph" and model == "KNN":
            constant = 8
        elif approach == "naive" and model == "IDW":
            constant = 9
        elif approach == "naive" and model == "KNN":
            constant = 12
        elif approach == "hybrid" and model == "IDW":
            constant = 7
        elif approach == "hybrid" and model == "KNN":
            constant = 8

    elif idx_variant == 1:  # variant 2
        if approach == "graph" and model == "IDW":
            constant = 10
        elif approach == "graph" and model == "KNN":
            constant = 11
        elif approach == "naive" and model == "IDW":
            constant = 18
        elif approach == "naive" and model == "KNN":
            constant = 21
        elif approach == "hybrid" and model == "IDW":
            constant = 10
        elif approach == "hybrid" and model == "KNN":
            constant = 11

    elif idx_variant == 2:  # variant 3
        if approach == "graph" and model == "IDW":
            constant = 19
        elif approach == "graph" and model == "KNN":
            constant = 20
        elif approach == "naive" and model == "IDW":
            constant = 22
        elif approach == "naive" and model == "KNN":
            constant = 26
        elif approach == "hybrid" and model == "IDW":
            constant = 14
        elif approach == "hybrid" and model == "KNN":
            constant = 16

    elif idx_variant == 3:  # variant 4
        if approach == "graph" and model == "IDW":
            constant = 21
        elif approach == "graph" and model == "KNN":
            constant = 22
        elif approach == "naive" and model == "IDW":
            constant = 29
        elif approach == "naive" and model == "KNN":
            constant = 34
        elif approach == "hybrid" and model == "IDW":
            constant = 15
        elif approach == "hybrid" and model == "KNN":
            constant = 17

    elif idx_variant == 4:
        if approach == "graph" and model == "IDW":
            constant = 22
        elif approach == "graph" and model == "KNN":
            constant = 23
        elif approach == "naive" and model == "IDW":
            constant = 34
        elif approach == "naive" and model == "KNN":
            constant = 39
        elif approach == "hybrid" and model == "IDW":
            constant = 16
        elif approach == "hybrid" and model == "KNN":
            constant = 18

    return constant


# Setup
nb_variants = 5
nb_instances_per_variant = 4
seeds = range(5)
dir = "logs"  # Choose "logs_test" for data profiles of RMSE tests and choose "logs" for validation
budget_per_params = 200

architectures = ["MLP", "CNN"]
approaches = ["naive", "hybrid", "graph"]
models = ["IDW"]

taus = [0.1, 0.025, 0.005]
kappas = np.linspace(1, budget_per_params, 40)

if __name__ == '__main__':

    fx_min_per_pbs = {(seed, arch, i, j): None
                      for seed in seeds
                      for arch in architectures
                      for i in range(nb_variants)
                      for j in range(nb_instances_per_variant)}


    # Step 1: import optimization logs by creating dictionnaries
    dict_evals = dict()
    dict_fct_values = dict()

    # -----------------------------------------------  #
    nb_instances = 0  # cpt
    for seed in seeds:
        for arch in architectures:
            for i in range(nb_variants):
                for j in range(nb_instances_per_variant):

                    nb_instances = nb_instances + 1
                    min_fx = math.inf
                    for approach in approaches:
                        for model in models:

                            data_file = "log_variant" + str(i+1) + "_size" + str(j+1) + "_" +\
                                        approach + "_" + model + "_" + arch + "." + str(seed) + ".txt"
                            data = pd.read_csv(dir + "/" + data_file, delim_whitespace=True, header=None)
                            evals = data[0].tolist()
                            fct_values = data[1].tolist()

                            # Store in dictionnaries
                            dict_evals[(seed, arch, i, j, approach, model)] = evals
                            dict_fct_values[(seed, arch, i, j, approach, model)] = fct_values

                            # Check if mininum obtained by solver is less than current min
                            #min_fx_solver = fct_values[-1]
                            min_fx_solver = min(fct_values)
                            if min_fx_solver < min_fx:
                                min_fx = min_fx_solver

                    # For given problem, store minimum amongst the solvers (approach-model)
                    fx_min_per_pbs[(seed, arch, i, j)] = min_fx
    # -----------------------------------------------  #

    # Step 2: find number of minimal evaluation to achieve convergence t_{p,s}
    perf_measure = dict()
    perf_normalized_measure = dict()
    rho_graph = dict()
    # -----------------------------------------------  #
    for tau in taus:

        # For each instance find the performance measure of each solver (approach, model)
        for seed in seeds:
            for arch in architectures:
                for i in range(nb_variants):
                    for j in range(nb_instances_per_variant):


                        perf_min_pb = math.inf
                        for approach in approaches:
                            for model in models:
                                perf_solver = nb_evals_convergence(tau, fx_min_per_pbs[(seed, arch, i, j)],
                                                                   dict_evals[(seed, arch, i, j, approach, model)],
                                                                   dict_fct_values[(seed, arch, i, j, approach, model)])

                                perf_measure[(tau, seed, arch, i, j, approach, model)] = perf_solver

                                if perf_solver < perf_min_pb:
                                    perf_min_pb = perf_solver

                        # After performance measure are determined for each solver and performance
                        for approach in approaches:
                            for model in models:
                                # The constant depends on the variant i, and the solver (approach-model)
                                constant_pb_solver = normalizing_constant(i, approach, model)

                                perf_normalized_measure[(tau, seed, arch, i, j, approach, model)] =\
                                    perf_measure[(tau, seed, arch, i, j, approach, model)]/(constant_pb_solver + 1)

        # Step 3: compute number of problems solved with kappas, for a given tau
        for kappa in kappas:

            for approach in approaches:
                for model in models:
                    rho_graph[(tau, kappa, approach, model)] = 0

                    for seed in seeds:
                        for arch in architectures:
                            for i in range(nb_variants):
                                for j in range(nb_instances_per_variant):

                                    if perf_normalized_measure[(tau, seed, arch, i, j, approach, model)] <= kappa:
                                        rho_graph[(tau, kappa, approach, model)] =\
                                            (rho_graph[(tau, kappa, approach, model)]+1/nb_instances)

    # Format rho values into lists
    graph_values = {(tau, approach, model): [rho_graph[(tau, kappa, approach, model)] for kappa in kappas]
                    for tau in taus for approach in approaches for model in models}

    # Setup plot appearance
    plt_size = 40
    plt.rc('axes', labelsize=1.25 * plt_size)
    plt.rc('legend', fontsize=1.25 * plt_size)
    plt.rc('xtick', labelsize=1.25 * plt_size)
    plt.rc('ytick', labelsize=1.25 * plt_size)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Subplots
    fig, ax = plt.subplots(nrows=1, ncols=len(taus), dpi=500, figsize=(35, 10))

    # Marker and label mapping
    # Order of approaches: ["naive", "hybrid", "graph"]
    markers = ['^', '+', '*', 'x', 'o', 's', 'p']
    colors = ['tab:red', 'tab:red', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    approach_model = {
        ("naive", "IDW"): 'Sub-IDW',
        ("hybrid", "IDW"): 'Hybrid-IDW',
        ("graph", "IDW"): 'Graph-IDW'}
    approach_model_colors = {
        ("naive", "IDW"): 'tab:red',
        ("hybrid", "IDW"): 'tab:blue',
        ("graph", "IDW"): 'tab:green'}
    approach_model_markers = {
        ("naive", "IDW"): '^',
        ("hybrid", "IDW"): '+',
        ("graph", "IDW"): '*'}

    taus_fig = [f"{tau * 100:.0f}\\%" if tau * 100 % 1 == 0 else f"{tau * 100:.1f}\\%" for tau in taus]

    # Plotting
    for idx_tau, tau in enumerate(taus):
        cpt = 0
        for approach in approaches:
            for model in models:
                ax[idx_tau].step(kappas, graph_values[tau, approach, model], where='post', linestyle='-', alpha=0.7,
                                 color=approach_model_colors[(approach, model)])
                ax[idx_tau].scatter(kappas, graph_values[tau, approach, model],
                                    s=7*plt_size,
                                    marker=approach_model_markers[(approach, model)],
                                    label=approach_model[(approach, model)],
                                    color=approach_model_colors[(approach, model)]
                                    )
                cpt += 1

        ax[idx_tau].set_ylim([0, 1.05])
        ax[idx_tau].set_xlim([kappas[0], kappas[-1]])
        ax[idx_tau].set_title(r'$\tau=~$' + taus_fig[idx_tau], fontsize=1.25 * plt_size)

        if idx_tau == 0:
            ax[idx_tau].set_ylabel(r'Portion of $\tau$-solved instances', fontsize=1.25*plt_size)

        if idx_tau == 1:
            ax[idx_tau].set_xlabel(r'Groups of $n_{p,s}+1$ evaluations $\kappa$', fontsize=1.25 * plt_size)

        if idx_tau >= 1:
            ax[idx_tau].set_yticklabels([])

    # Save as PDF
    plt.savefig("dataprofiles.pdf", bbox_inches='tight', dpi=500, transparent=True)