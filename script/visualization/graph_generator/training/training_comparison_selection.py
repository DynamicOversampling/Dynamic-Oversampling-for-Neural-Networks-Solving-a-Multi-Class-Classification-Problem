from matplotlib import pyplot as plt

import script.configs.setup
from script.configs.configs import nameDf, nameMd, nameTr, MetaConfig
from script.configs.setup import DataFrameConfigs, fatal_health, configMd, configNONE, configSMOTE, configDYN
from script.visualization.util.utils import parseEntries
from script.visualization.graph_generator.data.visualize_data import set_plot_defaults


def visualize_acc_loss(meta_config, crossfold, modelCrossfold, path="", skip=False):
    """
    function to generate graph depicting
    accuracy and f1 score over epochs
    """
    set_plot_defaults()
    parsed_entries_list = []
    crossfold_data_list = []
    names = []
    for e in meta_config:
        configDf, configMd, configTr = e.configDf, script.configs.setup.configMd, e.configTr
        log_list, crossfold_data = parseEntries(configDf, configMd, configTr, crossfold, modelCrossfold, path, skip)
        if log_list:
            parsed_entries_list.append(log_list)
            crossfold_data_list.append(crossfold_data)
        names.append(f"{nameMd(configMd)}_{nameDf(configDf, modelCrossfold)}_{nameTr(configTr)}")

    for idx, parsed_entries in enumerate(parsed_entries_list):
        epochs = crossfold_data_list[idx]["Epoch"]["mean"]
        training_accs = crossfold_data_list[idx]["TrainingAcc"]["mean"]
        test_accs = crossfold_data_list[idx]["TestAcc"]["mean"]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, training_accs, label='Training Accuracy', marker='o')
        plt.plot(epochs, test_accs, label='Test Accuracy', marker='o')

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Test Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()


meta_config = [
    MetaConfig(DataFrameConfigs[5], configMd, configNONE),
    MetaConfig(DataFrameConfigs[5], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[5], configMd, configDYN),
]

visualize_acc_loss([MetaConfig(fatal_health, configMd, configNONE)], 1, 1)
