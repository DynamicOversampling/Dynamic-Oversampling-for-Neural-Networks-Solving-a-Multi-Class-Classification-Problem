from matplotlib import pyplot as plt

from script.configs.configs import crf_naming
from script.configs.setup import meta_configDYNBays
from script.visualization.util.utils import parseEntriesByOversampling
from script.visualization.graph_generator.data.visualize_data import set_plot_defaults


def visualizeAccLoss(meta_config, crossfold, modelCrossfold, path="", skip=False, enforce=False):
    """
    function to generate graph depicting
    accuracy and f1 score over epochs
    comparing different dynamic oversampling setups
    """
    set_plot_defaults()
    crossfold_data_list = parseEntriesByOversampling(meta_config, crossfold, modelCrossfold, path, skip, enforce)

    num_plots = 2

    for idx in range(3):
        if idx == 0:
            ax_plot = plt.subplot2grid((num_plots, 2), (0, 0), colspan=2)
            title = "DYNAMIC Non Optimised"
        elif idx == 1:
            ax_plot = plt.subplot2grid((num_plots, 2), (1, 0), colspan=1)
            title = "DYNAMIC Accuracy Optimised"
        else:
            ax_plot = plt.subplot2grid((num_plots, 2), (1, 1), colspan=1)
            title = "DYNAMIC F1 Optimised"

        idx = crf_naming(meta_config[idx].configTr)

        epochs = crossfold_data_list[idx]["Epoch"]["mean"]
        training_accs = crossfold_data_list[idx]["TrainingAcc"]["mean"]
        test_accs = crossfold_data_list[idx]["TestAcc"]["mean"]
        training_f1 = crossfold_data_list[idx]["TrainingF1"]["mean"]
        test_f1 = crossfold_data_list[idx]["TestF1"]["mean"]
        training_accsSTD = crossfold_data_list[idx]["TrainingAcc"]["std"]
        test_accsSTD = crossfold_data_list[idx]["TestAcc"]["std"]

        ax_plot.plot(epochs, training_accs, linestyle='-', color='#FF0000FF', label='Training Accuracy')
        ax_plot.fill_between(epochs, training_accs - training_accsSTD, training_accs + training_accsSTD,
                             color='#FF0000FF', alpha=0.2)

        ax_plot.plot(epochs, training_f1, linestyle='-', color='#ffff00FF', label='Training F1')

        ax_plot.plot(epochs, test_accs, linestyle='-', color='#0000FFFF', label='Test Accuracy')
        ax_plot.fill_between(epochs, test_accs - test_accsSTD, test_accs + test_accsSTD, color='#0000FFFF', alpha=0.2)

        ax_plot.plot(epochs, test_f1, linestyle='-', color='#33cc33FF', label='Test F1')

        ax_plot.set_title(title)
        ax_plot.set_xlabel('Epoch')
        ax_plot.set_xticks(epochs[::20] - 1)
        ax_plot.set_ylabel('Accuracy and F1')

        ax_plot.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig("plots/caeBaysComp.svg")
    plt.show()


visualizeAccLoss(meta_configDYNBays, 4, 4, "../../../../")
