import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata  #
from matplotlib import cm


def bo3(log_file, pairs, param3):
    """
    script to generate 3d graph visualising
    the results of the bayesean optimization on the defined pairs
    """

    entries = []

    with open(log_file, 'r') as file:
        for line in file:
            try:
                json_entry = json.loads(line.strip())
                entries.append(json_entry)
            except:
                print("FAILURE at vis_log 1")

    def remove_duplicate_targets(data):
        unique_entries = {}

        for entry in data:
            target = entry["target"]
            if target not in unique_entries:
                unique_entries[target] = entry

        return list(unique_entries.values())

    entries = remove_duplicate_targets(entries)[:80]

    targets = np.array([entry["target"] for entry in entries])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12), subplot_kw={"projection": "3d"})
    axes = axes.flatten()

    cmap = plt.get_cmap('inferno')

    for idx, (param1, param2) in enumerate(pairs):
        param1_data = np.array([entry["params"][param1] for entry in entries])
        param2_data = np.array([entry["params"][param2] for entry in entries])

        try:
            param3_data = np.array([entry["params"][param3] for entry in entries])
        except:
            param3_data = np.array([entry["target"] for entry in entries])

        ax = axes[idx]
        if idx != 3:
            param1_grid, param2_grid = np.meshgrid(
                np.linspace(min(param1_data), max(param1_data), 50),
                np.linspace(min(param2_data), max(param2_data), 50)
            )

            param3_grid = griddata((param1_data, param2_data), param3_data, (param1_grid, param2_grid), method='linear')
            target_grid = griddata((param1_data, param2_data), targets, (param1_grid, param2_grid), method='linear')

            facecolors = cmap(
                (target_grid - np.nanmin(target_grid)) / np.nanmax((target_grid - np.nanmin(target_grid))))
            surf = ax.plot_surface(param1_grid, param2_grid, param3_grid, facecolors=facecolors,
                                   edgecolor='none', rstride=1, cstride=1, antialiased=False)

            ax.set_xlim(right=0, left=200)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_zlabel(param3)
            ax.set_title(f'F1 vs {param1} and {param2}')

            ax.view_init(elev=70, azim=-60)
        else:
            normalized_targets = (targets - np.min(targets)) / (np.max(targets) - np.min(targets))
            ax.scatter3D(param1_data, param1_data, param3_data, 'o', edgecolor='k', s=40, label=param1,
                         c=normalized_targets, cmap=cmap)

            ax.set_xlim([min(param1_data), max(param1_data)])
            ax.set_xlabel(param1)
            ax.set_ylabel(param1)
            ax.set_zlabel(param3)
            ax.set_title(f'F1 vs {param1}')

    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(targets)
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)

    cbar_ax = fig.add_axes([0.87, 0.4, 0.02, 0.2])

    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('F1 score', labelpad=10, rotation=270, horizontalalignment='center')

    plt.savefig("plots/bo3.svg")
    plt.show()


pairs = [["reset", "start_lvl"], ["reset", "ratio"], ["reset", "epochs"], ["reset", "reset"]]
param3 = "target"
log_file = "../../../nn/bayesian/bays_log_TestAC_I.log"

bo3(log_file, pairs, param3)
