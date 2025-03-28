import numpy as np
import matplotlib.pyplot as plt
import os

log_dir = "log_selected_parameters"
size = 3
variants = [1, 2, 3, 4, 5]
models = ["MLP", "CNN"]
stride_per_variant = [6, 12, 16, 16, 20]

# Custom offsets for each variant
offset_start_per_variant = {
    1: {"naive": 0, "hybrid": 2, "graph": 5},
    2: {"naive": 0, "hybrid": 2, "graph": 5},
    3: {"naive": 0, "hybrid": 4, "graph": 8},
    4: {"naive": 0, "hybrid": 4, "graph": 8},
    5: {"naive": 0, "hybrid": 5, "graph": 10},
}

# Approach formatting
approach_model = {
    ("naive", "IDW"): 'Sub-IDW',
    ("hybrid", "IDW"): 'Hybrid-IDW',
    ("graph", "IDW"): 'Meta-IDW'
}
approach_model_colors = {
    ("naive", "IDW"): 'tab:red',
    ("hybrid", "IDW"): 'tab:blue',
    ("graph", "IDW"): 'tab:green'
}
approach_model_markers = {
    ("naive", "IDW"): '^',
    ("hybrid", "IDW"): '+',
    ("graph", "IDW"): '*'
}
approach_order = ["naive", "hybrid", "graph"]



if __name__ == "__main__":
    # Store all data
    data = {}

    for variant in variants:
        for model in models:
            file_path = os.path.join(log_dir, f"selected_parameters_variant{variant}_size{size}_IDW_{model}.txt")
            print(f"Looking for: {file_path}")
            if not os.path.isfile(file_path):
                print(f"❌ File not found: {file_path}")
                continue

            values = np.loadtxt(file_path)
            print(f"✅ Found file. Loaded shape: {values.shape}")

            if values.ndim == 1 or values.shape[0] != 6:
                print(f"⚠️ Skipping malformed file: {file_path}")
                continue

            data[f"variant{variant}_{model}"] = {
                "graph_mean": values[0],
                "graph_std": values[1],
                "naive_mean": values[2],
                "naive_std": values[3],
                "hybrid_mean": values[4],
                "hybrid_std": values[5],
            }

    # Plot setup
    plt.rc('axes', labelsize=28)
    plt.rc('legend', fontsize=28)
    plt.rc('xtick', labelsize=26)
    plt.rc('ytick', labelsize=26)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 20), dpi=100)
    legend_handles = []

    for i, variant in enumerate(variants):
        y_mins, y_maxs = [], []

        # First pass: determine shared y-limits per row
        for j, model in enumerate(models):
            key = f"variant{variant}_{model}"
            if key not in data:
                continue
            d = data[key]
            y_vals = np.concatenate([
                d["naive_mean"] - d["naive_std"],
                d["naive_mean"] + d["naive_std"],
                d["graph_mean"] - d["graph_std"],
                d["graph_mean"] + d["graph_std"],
                d["hybrid_mean"] - d["hybrid_std"],
                d["hybrid_mean"] + d["hybrid_std"],
            ])
            y_mins.append(np.min(y_vals))
            y_maxs.append(np.max(y_vals))

        shared_ylim = [min(y_mins), max(y_maxs)]

        # Second pass: plotting
        for j, model in enumerate(models):
            key = f"variant{variant}_{model}"
            if key not in data:
                continue

            d = data[key]
            nb_pts = len(d["graph_mean"])
            x_range = np.arange(nb_pts)
            current_ax = ax[i, j]
            stride = stride_per_variant[i]

            for approach in approach_order:
                mkey = (approach, "IDW")
                label = approach_model[mkey]
                color = approach_model_colors[mkey]
                marker = approach_model_markers[mkey]
                mean = d[f"{approach}_mean"]
                std = d[f"{approach}_std"]

                offset = offset_start_per_variant[variant][approach]
                scatter_x = np.arange(offset, nb_pts, stride)

                current_ax.plot(x_range, mean, linestyle='-', color=color)
                current_ax.fill_between(x_range, mean - std, mean + std, color=color, alpha=0.20)
                scatter = current_ax.plot(
                    scatter_x, mean[scatter_x],
                    linestyle='None', marker=marker, markersize=8,
                    color=color, label=label
                )
                if i == 0 and j == 0:
                    legend_handles.append(scatter[0])

            # Axis settings
            current_ax.set_ylabel("RMSE test", labelpad=5)
            # current_ax.set_ylim(shared_ylim)
            # current_ax.set_yscale("log")

            if j == 0:
                current_ax.text(-0.35, 0.5, f'Variant \#{variant}',
                                horizontalalignment='center',
                                verticalalignment='center',
                                transform=current_ax.transAxes,
                                bbox=dict(facecolor='none', edgecolor='black', pad=5.0),
                                size=28, rotation=90)
            else:
                current_ax.set_ylabel(None)

            if i == 0:
                current_ax.set_title(model, fontsize=28, pad=20,
                                     bbox=dict(facecolor='none', edgecolor='black', pad=5.0))

            if i == len(variants) - 1:
                current_ax.set_xlabel("Training dataset size")

    # Global legend
    fig.legend(
        handles=legend_handles,
        labels=[approach_model[(a, "IDW")] for a in approach_order],
        loc='upper right',
        bbox_to_anchor=(0.99, 0.96),  # lowered to keep it inside figure
        frameon=True,
        framealpha=0.8,
        edgecolor='black',
        fancybox=True,
        fontsize=22,
        ncol=1,
        markerscale=2.5,
        handlelength=2.5
    )

    fig.tight_layout()
    plt.savefig("RMSE_vs_training_size_all_variants.pdf", format='pdf', bbox_inches='tight')
    plt.show()
