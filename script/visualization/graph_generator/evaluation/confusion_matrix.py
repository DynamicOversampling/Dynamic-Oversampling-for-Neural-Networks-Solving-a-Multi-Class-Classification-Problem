import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from script.nn.evaluation.eval_model import eval_model_acc_class_load
from script.visualization.graph_generator.data.visualize_data import set_plot_defaults


def generateConfusionMatrix():
    """
    function to generate graph depicting
    the confusion matrix for the covtype dataset of all oversampling techniques
    """

    set_plot_defaults(14)
    path = "../../../../"
    df = pd.read_csv(f"../../../../datasets/reallife/covtype.csv").iloc[:, 1:]

    for e in ("DYNAMIC_56-3-24-0", "SMOTE", "ADASYN", "DYNAMIC_56-3-24-0"):
        results = eval_model_acc_class_load(f"model_ml2_mn6_md0.25_covtype_ts{e}", df, path)
        results = np.array(results, dtype=float)
        print(results)

        row_sums = results.sum(axis=1, keepdims=True)
        data_normalized = (results / row_sums)

        df_cm_normalized = pd.DataFrame(data_normalized, index=[f'Class {i}' for i in range(1, len(results) + 1)],
                                        columns=[f'Class {i}' for i in range(1, len(results[0]) + 1)])

        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm_normalized, annot=True, fmt=".4f", cmap=plt.get_cmap('YlGnBu'), cbar=True)
        plt.title(f"Confusion Matrix - {e.split('_')[0]}")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")

        plt.savefig(f"../../../plots/cm_{e}.svg")
        plt.tight_layout()
        plt.show()


generateConfusionMatrix()