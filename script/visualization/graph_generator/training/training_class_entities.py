from matplotlib import pyplot as plt, cm
from sympy.physics.control.control_plots import matplotlib

import script.configs.setup
from script.configs.configs import nameDf, nameMd, nameTr, MetaConfig, TrainingConfig, \
    TrainingMethod, DYNSETTING
from script.configs.setup import DataFrameConfigs, configMd
from script.visualization.util.utils import parseEntries
from script.visualization.graph_generator.data.visualize_data import set_plot_defaults


def visualizeAccLoss(meta_config, crossfold, modelCrossfold, path="", skip=False, enforce=False):
    """
    function to generate graph depicting
    the number of entities per class during the training process
    """
    set_plot_defaults()
    parsed_entries_list = []
    crossfold_data_list = []
    names = []
    for e in meta_config:
        configDf, configMd, configTr = e.configDf, script.configs.setup.configMd, e.configTr
        log_list, crossfold_data = parseEntries(configDf, configMd, configTr, crossfold, modelCrossfold, path, skip,
                                                enforce=enforce)
        if log_list:
            parsed_entries_list.append(log_list)
            crossfold_data_list.append(crossfold_data)
        names.append(f"{nameMd(configMd)}_{nameDf(configDf, modelCrossfold)}_{nameTr(configTr)}")

    num_configs = len(parsed_entries_list)

    fig, axs = plt.subplots(num_configs, 2, figsize=(16, 7 * num_configs))

    for idx, parsed_entries in enumerate(parsed_entries_list):
        epochs = crossfold_data_list[idx]["Epoch"]["mean"]
        cat = crossfold_data_list[idx]["ClassNumberTrain"]["mean"]
        acc_class_train = crossfold_data_list[idx]["ClassAccuraciesTrain"]["mean"]

        # 9 3 2 0 1 8 4
        temp = []
        temp.append(cat[13])
        temp.append(cat[14])
        temp.append(cat[18])
        temp.append(cat[4])
        temp.append(cat[0])
        temp.append(cat[12])
        cat = temp

        temp = []
        temp.append(acc_class_train[13])
        temp.append(acc_class_train[14])
        temp.append(acc_class_train[18])
        temp.append(acc_class_train[4])
        temp.append(acc_class_train[0])
        temp.append(acc_class_train[12])
        acc_class_train_orig = acc_class_train
        acc_class_train = temp

        num_classes = len(cat)
        cmap = cm.get_cmap('tab20', num_classes)
        colors = [cmap(i) for i in range(num_classes)]

        for class_id, data in enumerate(cat):
            color = colors[int(class_id) % num_classes]
            axs[0].plot(epochs, data, label=f'Class {class_id + 1}', color=f"{matplotlib.colors.to_hex(color)}FF")

        axs[0].set_title("Number of Class Entities")
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Number of Entities')
        axs[0].legend(loc="lower right")

        num_classes = len(acc_class_train)
        cmap = cm.get_cmap('tab20', num_classes)
        colors = [cmap(i) for i in range(num_classes)]

        for class_id, data in enumerate(acc_class_train):
            color = colors[int(class_id) % num_classes]
            axs[1].plot(epochs, data, label=f'Class {class_id + 1}',
                        color=f"{matplotlib.colors.to_hex(color)}FF")

        average_accuracy = [sum(epoch_data) / len(epoch_data) for epoch_data in zip(*acc_class_train_orig)]
        axs[1].plot(epochs, average_accuracy, linestyle=':', color='red', linewidth=5, label='Average')

        axs[1].set_title("Training Class Accuracy")
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].legend(loc="lower right")

    plt.savefig("plots/cne.svg")
    plt.tight_layout()
    plt.show()


configDYN = TrainingConfig(TrainingMethod.DYNAMIC, DYNSETTING(99, 4, 42, 0.39))

meta_config = [
    MetaConfig(DataFrameConfigs[5], configMd, configDYN),
]

visualizeAccLoss(meta_config, 1, 1, "../../../../", enforce=False)
