from matplotlib import pyplot as plt

from script.configs.configs import TrainingMethod
from script.visualization.util.utils import parseEntriesByOversampling
from script.visualization.graph_generator.data.visualize_data import set_plot_defaults


def visualize_oversampling_comparison(meta_config, crossfold, modelCrossfold, path="", skip=False, enforce=False):
    """
    function to generate graph depicting accuracy and f1 score over epochs

    Parameters:
    meta_config (list): Configuration for networks need to be trained.
    crossfold (int): Number of repetitions per conf.
    modelCrossfold (int): Number of models per conf
    path (str, optional): Path for saving.
    skip (bool, optional): Flag to skip models
    enforce (bool, optional): Flag to enforce regeneration.
    """
    set_plot_defaults()
    crossfold_data_list = parseEntriesByOversampling(meta_config, crossfold, modelCrossfold, path, skip, enforce)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    for idx in range(4):
        row, col = divmod(idx, 2)
        epochs = crossfold_data_list[idx]["Epoch"]["mean"]
        training_accs = crossfold_data_list[idx]["TrainingAcc"]["mean"]
        test_accs = crossfold_data_list[idx]["TestAcc"]["mean"]
        training_f1 = crossfold_data_list[idx]["TrainingF1"]["mean"]
        test_f1 = crossfold_data_list[idx]["TestF1"]["mean"]
        training_accsSTD = crossfold_data_list[idx]["TrainingAcc"]["std"]
        test_accsSTD = crossfold_data_list[idx]["TestAcc"]["std"]

        axs[row, col].plot(epochs, training_accs, linestyle='-', color='#FF0000FF', label='Training Accuracy')
        axs[row, col].fill_between(epochs, training_accs - training_accsSTD, training_accs + training_accsSTD,
                                   color='#FF0000FF', alpha=0.2)
        axs[row, col].plot(epochs, training_f1, linestyle='-', color='#ffff00FF', label='Training F1')
        axs[row, col].plot(epochs, test_accs, linestyle='-', color='#0000FFFF', label='Test Accuracy')
        axs[row, col].fill_between(epochs, test_accs - test_accsSTD, test_accs + test_accsSTD, color='#0000FFFF',
                                   alpha=0.2)
        axs[row, col].plot(epochs, test_f1, linestyle='-', color='#33cc33FF', label='Test F1')
        axs[row, col].set_title(f'{TrainingMethod(idx).name}')
        axs[row, col].set_xlabel('Epoch')
        axs[row, col].set_xticks(epochs[::20] - 1)
        axs[row, col].set_ylabel('Accuracy and F1')
        axs[row, col].legend()

    plt.savefig("plots/cae.svg")
    plt.tight_layout()
    plt.show()