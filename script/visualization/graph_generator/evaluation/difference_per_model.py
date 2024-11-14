import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import matplotlib.lines as mlines

from script.configs.configs import TrainingMethod, DataSetsConfig
from script.configs.setup import meta_configDYNBaysSMOTEBays
from script.utils import calculate_gini
from script.visualization.util.utils import parseEntriesByOversampling
from script.visualization.graph_generator.data.visualize_data import set_plot_defaults

color_list = sns.color_palette("deep", 6)
color_map = color_list[-3:]


def visualizeAccLoss(Y, crossfold, modelCrossfold, path="", skip=False):
    """
    function to generate graph depicting
    performance differences of oversampling technique
    between the synthetic datasets
    """
    set_plot_defaults(28)
    configs_model = dict()
    configs_temp = []
    config_old = DataSetsConfig(0, 0, 0, 0, 0, 0)

    for config in Y:
        if config.configDf != config_old:
            if configs_temp:
                configs_model[config_old] = configs_temp
                configs_temp = []
            config_old = config.configDf
        configs_temp.append(config)

    configs_model[config_old] = configs_temp

    max_value = 0
    bar_data = {}
    difference_data = []

    for dataFrame, config in configs_model.items():
        max_value = max(max_value, dataFrame.c, dataFrame.f)
        gini = calculate_gini(dataFrame, "../../../../")
        bar_data[dataFrame] = [dataFrame.c, dataFrame.f, gini]

    num_plots = max(1, len(configs_model))

    for i, (dataFrame, config) in enumerate(configs_model.items()):
        ax_plot = plt.subplot2grid((num_plots, 5), (i, 1), colspan=4)
        ax_bar = plt.subplot2grid((num_plots, 5), (i, 0))

        try:
            crossfold_data_list = parseEntriesByOversampling(config, crossfold, modelCrossfold, path, skip)
        except:
            break

        keys = list(crossfold_data_list.keys())

        reference_test_accs = crossfold_data_list[keys[-1]]["TestAcc"]["mean"] * 100
        reference_test_f1 = crossfold_data_list[keys[-1]]["TestF1"]["mean"] * 100
        reference_test_accsSTD = crossfold_data_list[keys[-1]]["TestAcc"]["std"] * 100
        reference_test_f1STD = crossfold_data_list[keys[-1]]["TestF1"]["std"] * 100

        ax_plot.axhline(0, color='red', linestyle=':', linewidth=3)

        for idx in range(0, 3):
            epochs = crossfold_data_list[keys[idx]]["Epoch"]["mean"]
            test_accs = crossfold_data_list[keys[idx]]["TestAcc"]["mean"] * 100
            test_f1 = crossfold_data_list[keys[idx]]["TestF1"]["mean"] * 100
            test_accsSTD = crossfold_data_list[keys[idx]]["TestAcc"]["std"] * 100
            test_f1STD = crossfold_data_list[keys[idx]]["TestF1"]["std"] * 100

            acc_diff = test_accs - reference_test_accs
            f1_diff = test_f1 - reference_test_f1

            acc_diffSTD = np.sqrt(test_accsSTD ** 2 + reference_test_accsSTD ** 2)
            f1_diffSTD = np.sqrt(test_f1STD ** 2 + reference_test_f1STD ** 2)

            color = color_map[idx]
            ax_plot.set_yscale('symlog')

            def custom_tick_format(value, tick_number):
                if value > 0:
                    return f'{int(value) / 100}'
                elif value < 0:
                    return f'{int(value) / 100}'
                return '0'

            ax_plot.yaxis.set_major_formatter(FuncFormatter(custom_tick_format))

            ax_plot.plot(epochs, acc_diff, linestyle='-', color=color,
                         label=f'{TrainingMethod(idx).name}', linewidth=5)
            ax_plot.fill_between(epochs, acc_diff - acc_diffSTD, acc_diff + acc_diffSTD, alpha=0.05, color=color)

            ax_plot.plot(epochs, f1_diff, linestyle='--', color=color, linewidth=5)
            ax_plot.fill_between(epochs, f1_diff - f1_diffSTD, f1_diff + f1_diffSTD, alpha=0.05, color=color)

            difference_data.append({
                'DataFrame': str(bar_data[dataFrame]),
                'Method': TrainingMethod(idx).name,
                'TestAcc': str(int(test_accs[-1] * 100) / 10000).replace(',', '.'),
                'TestF1': str(int(test_f1[-1] * 100) / 10000).replace(',', '.'),
                'Diff TestAcc': str(int(acc_diff[-1] * 100) / -10000).replace(',', '.'),
                'Diff TestF1': str(int(f1_diff[-1] * 100) / -10000).replace(',', '.')
            })

        manual_legend_entry = mlines.Line2D([], [], color='black', linestyle='--', label='F1')

        handles, labels = ax_plot.get_legend_handles_labels()
        handles.append(manual_legend_entry)
        labels.append('F1')

        ax_plot.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 0.2), fancybox=True,
                       ncol=4)

        bar_labels = ['class', 'feat', 'gini']
        bar_values = bar_data[dataFrame]

        ax_bar.bar(bar_labels[:2], bar_values[:2], color=color_list[:2])

        ax_gini = ax_bar.twinx()
        ax_gini.bar(bar_labels[2], bar_values[2], color=color_list[2])

        ax_bar.set_ylim(0, max_value + 0.1)

        ax_gini.set_ylim(0.1, 0.6)

        line_x_pos = 1.5
        ax_bar.axvline(x=line_x_pos, color='black', linestyle='-', linewidth=0.5)

        ax_bar.set_xticks(range(len(bar_labels)))
        ax_bar.set_xticklabels(bar_labels)

    df_difference = pd.DataFrame(difference_data)
    df_difference.to_excel('difference_data.xlsx', sheet_name='Last Epoch Differences', index=False)

    plt.savefig("plots/tpd.svg")
    plt.tight_layout()
    plt.show()


visualizeAccLoss(meta_configDYNBaysSMOTEBays, 2, 1, "../../../../", skip=False)
