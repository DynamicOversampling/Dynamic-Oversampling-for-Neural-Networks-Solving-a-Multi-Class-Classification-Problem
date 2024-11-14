import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# General settings for all plots
def set_plot_defaults(legend=16):
    plt.rcParams.update({
        'font.size': legend + 2,
        'axes.titlesize': legend + 2,
        'axes.labelsize': legend + 2,
        'legend.fontsize': legend,
        'xtick.labelsize': legend,
        'ytick.labelsize': legend,
        'figure.figsize': (10, 8),
        'savefig.dpi': 300,
        'savefig.format': 'svg',
        'figure.autolayout': True
    })


def visualizeDF(df, path="", log=False, nr=-1):
    """
    function to visualize any datasets
    hist, histTarget, boxPlot & pairPlot
    """
    feature_columns = [f'Feature {i + 1}' for i in range(df.shape[1])]
    feature_columns[-1] = "Target"
    df.columns = feature_columns
    unique_targets = sorted(df['Target'].unique())
    target_mapping = {val: f'Class {i + 1}' for i, val in enumerate(unique_targets)}
    df['Target'] = df['Target'].map(target_mapping)

    set_plot_defaults()
    hist(df, path, log, nr)
    histTarget(df, path, log, nr)
    boxPlot(df)
    pairPlot(df, "Target")


def hist(df, path, log=False, nr=-1):
    """
    function to visualize histogram
    """
    df = df.iloc[:, :5]
    set_plot_defaults(12)
    plt.figure(figsize=(4, 4))
    edgecolor = 'none'
    sns.histplot(data=df.iloc[:, :-1], kde=False, palette="deep")
    if log:
        plt.yscale('log')
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    plt.grid(True)
    if nr == -1:
        plt.savefig(f"{path}plots/hist.svg")
        plt.show()
    else:
        plt.savefig(f"{path}plots/hist/{nr}.svg")


def histTarget(df, path, log=False, nr=-1):
    """
    function to visualize histogram of target class
    """
    set_plot_defaults(5)
    plt.figure(figsize=(3, 2))

    target_counts = df["Target"].value_counts().sort_index()
    target_classes = range(1, len(target_counts.index) + 1)

    colorblind_palette = sns.color_palette("deep", len(target_classes))

    print(target_counts.values)

    sns.barplot(
        x=target_classes,
        y=target_counts.values,
        palette=colorblind_palette
    )
    if log:
        plt.yscale('log')
    plt.xlabel("Target Class")
    plt.ylabel("Frequency")

    if nr == -1:
        plt.savefig(f"{path}plots/histTarget.svg")
        plt.show()
    else:
        plt.title(f"Dataset 6 | {nr + 1}")
        plt.savefig(f"{path}plots/hist/{nr}.svg")


def boxPlot(df):
    """
    function to visualize boxplot
    """
    df = df.iloc[:, :5]
    set_plot_defaults()
    plt.figure()
    sns.boxplot(data=df.iloc[:, :-1], palette="colorblind")
    plt.title("Boxplot of Features")
    plt.xlabel("Features")
    plt.ylabel("Value Distribution")
    plt.grid(True)
    plt.savefig("plots/box.svg")
    plt.show()


def pairPlot(df, target_column):
    """
    function to visualize pairplot of selected features
    """
    set_plot_defaults()

    df = df.sample(frac=0.08)

    if target_column not in df.columns:
        raise ValueError("Target column miss")

    features = df.columns[df.columns != target_column]
    selected_features = features[:8]

    df_selected = df[list(selected_features) + [target_column]]

    g = sns.pairplot(
        df_selected,
        diag_kind="hist",
        hue=target_column,
        palette="colorblind",
        plot_kws={'s': 80}
    )

    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    plt.savefig("plots/pair.svg")
    plt.show()
