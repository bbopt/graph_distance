from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Setup plot appearance consistent with the main figure
    plt_size = 40
    plt.rc('axes', labelsize=1.25 * plt_size)
    plt.rc('legend', fontsize=1.25 * plt_size)
    plt.rc('xtick', labelsize=1.25 * plt_size)
    plt.rc('ytick', labelsize=1.25 * plt_size)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Define approach-model combinations, display names, colors, and markers
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

    # Create dummy plot
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_axis_off()

    # Create legend handles using explicit mapping
    handles = []
    for key, label in approach_model.items():
        marker = approach_model_markers[key]
        color = approach_model_colors[key]

        handle = plt.Line2D(
            [0], [0], marker=marker, linestyle='None', alpha=1.0,
            markersize=plt_size * 0.3,
            markerfacecolor='none' if marker in ['+', 'x'] else color,
            markeredgecolor=color,
            markeredgewidth=2.5 if marker in ['+', 'x'] else 0,
            label=label
        )
        handles.append(handle)

    # Create the legend with styling
    legend = plt.legend(
        handles=handles,
        loc='center',
        fontsize=0.5 * plt_size,
        ncol=3,
        frameon=True,
        framealpha=0.8,
        edgecolor='black',
        fancybox=True,
        columnspacing=1.5,
        handletextpad=0.5,
        markerscale=1.5
    )

    # Save to file
    plt.savefig("dataprofiles_legend.pdf", bbox_inches='tight', dpi=500, transparent=True)
