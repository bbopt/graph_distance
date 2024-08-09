import numpy as np
import pandas as pd
import sys
import math
from matplotlib import pyplot as plt


def nb_evals_convergence(tau, fct_min_pb, evals, fct_values):

    nb_evals = math.inf
    for idx, eval in enumerate(evals):
        if abs((fct_values[idx] - fct_min_pb)/fct_min_pb) <= tau:
            nb_evals = eval
            break  # end forloop if convergence is met

    return nb_evals


# Setup
nb_variants = 5
nb_instances_per_variant = 4  # in total 5*4=20 pbs
nb_pbs = nb_variants * nb_instances_per_variant
nb_budget_variant = [1500, 2000, 2500, 3000, 3500]
approaches = ["graph", "naive"]
models = ["KNN", "IDW"]

#taus = [1/2, 1/4, 1/16]
taus = [0.25, 0.1, 0.01]

alphas = np.linspace(1, 100, 100)

if __name__ == '__main__':

    fx_min_per_pbs = {(i, j): None for i in range(nb_variants) for j in range(nb_instances_per_variant)}

    # Step 1: import optimization logs by creating dictionnaries
    dict_evals = dict()
    dict_fct_values = dict()

    # --------------------------------- #
    for i in range(nb_variants):
        for j in range(nb_instances_per_variant):
            min_fx = math.inf
            for approach in approaches:
                for model in models:

                    data_file = "log_variant" + str(i+1) + "_size" + str(j+1) + "_" + approach + "_" + model + ".txt"
                    data = pd.read_csv("logs/"+data_file, delim_whitespace=True, header=None)
                    evals = data[0].tolist()
                    fct_values = data[1].tolist()

                    # Store in dictionnaries
                    dict_evals[(i, j, approach, model)] = evals
                    dict_fct_values[(i, j, approach, model)] = fct_values

                    # Check if mininum obtained by solver is less than current min
                    #min_fx_solver = fct_values[-1]
                    min_fx_solver = min(fct_values)
                    if min_fx_solver < min_fx:
                        min_fx = min_fx_solver

            # For given problem, store minimum amongst the solvers (approach-model)
            fx_min_per_pbs[(i, j)] = min_fx

    # Step 2: find number of minimal evaluation to achieve convergence
    perf_measure = dict()
    perf_ratio_measure = dict()
    rho_graph = dict()
    # --------------------------------- #
    for tau in taus:

        # For each problem (i,j) find the performance measure of each solver (approach, model)
        for i in range(nb_variants):
            for j in range(nb_instances_per_variant):

                perf_min_pb = math.inf
                for approach in approaches:
                    for model in models:
                        perf_solver = nb_evals_convergence(tau, fx_min_per_pbs[(i, j)],
                                                           dict_evals[(i, j, approach, model)],
                                                           dict_fct_values[(i, j, approach, model)])

                        perf_measure[(tau, i, j, approach, model)] = perf_solver

                        if perf_solver < perf_min_pb:
                            perf_min_pb = perf_solver

                # After performance measure are determined for each solver and performance min, find ratio
                for approach in approaches:
                    for model in models:
                        perf_ratio_measure[(tau, i, j, approach, model)] =\
                            perf_measure[(tau, i, j, approach, model)]/perf_min_pb

        # Step 3: compute number of problems solved with alphas

        for alpha in alphas:

            for approach in approaches:
                for model in models:
                    rho_graph[(tau, alpha, approach, model)] = 0

                    for i in range(nb_variants):
                        for j in range(nb_instances_per_variant):
                            if perf_ratio_measure[(tau, i, j, approach, model)] <= alpha:
                                rho_graph[(tau, alpha, approach, model)] =\
                                    (rho_graph[(tau, alpha, approach, model)]+1/nb_pbs)

    # Format rho values into lists
    graph_values = {(tau, approach, model): [rho_graph[(tau, alpha, approach, model)] for alpha in alphas]
                    for tau in taus for approach in approaches for model in models}

    # Setup figure
    plt_size = 32
    plt.rc('axes', labelsize=plt_size / 1.25)
    plt.rc('legend', fontsize=plt_size / 1.5)
    plt.rc('xtick', labelsize=plt_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=plt_size)  # fontsize of the tick label
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Subplots
    fig, ax = plt.subplots(nrows=1, ncols=3, tight_layout=True, dpi=100, figsize=(30, 6.5))
    #fig.suptitle("Performance profiles on RMSE validation (objective function)", fontsize=plt_size)


    for idx_tau, tau in enumerate(taus):

        markers = ['^', '+', 'o', 'x']
        approach_model = ['Graph-KNN', 'Graph-IDW', 'Sub-KNN', 'Sub-IDW']
        taus_fig = [str(int(tau*100)) + "\%" for tau in taus]
        cpt = 0
        for approach in approaches:
            for model in models:
                ax[idx_tau].step(alphas, graph_values[tau, approach, model])
                ax[idx_tau].scatter(alphas, graph_values[tau, approach, model], s=1.5*plt_size, marker=markers[cpt], label=approach_model[cpt])
                cpt = cpt + 1
        if idx_tau == 0:
            ax[idx_tau].set_ylabel(r'Proportion of problems solved', fontsize=plt_size)
            ax[idx_tau].legend(loc="upper right")

        if idx_tau == 1:
            ax[idx_tau].set_xlabel(r'Performance ratio $\alpha$', fontsize=plt_size)

        ax[idx_tau].set_ylim([0, 1.05])
        ax[idx_tau].set_xlim([alphas[0], alphas[-1]])
        ax[idx_tau].set_title(r'$\tau=~$'+taus_fig[idx_tau], fontsize=plt_size)

    plt.show()