import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from script.configs.configs import nameDf, DataSetsConfig
from script.configs.setup import DataFrameConfigs, configMd
from script.data.data_generation.data_generation import generateSaveData
from script.nn.model import AdaptiveModel


def setup():
    """
    function to generate all needed datasets in case they do not exist yet
    """
    for config in DataFrameConfigs:
        file_name = f"datasets/synthetic/df_{nameDf(config, 0)}.csv"
        try:
            pd.read_csv(file_name)
        except:
            generateSaveData(config, 0)


def loadDF(dataset_key, path="", cross=0, config_model=configMd):
    def load_synthetic_dataset(config, cross):
        """
        Load or generate a synthetic dataset based on the configuration.

        Parameters:
            dataset_key (str): Dataset identifier.
            path (str): Path to datasets
            cross (int): Cross-validation
            config_model (configMd): Model configuration

        Returns:
            DataFrame: Loaded or generated dataset.
        """
        model = AdaptiveModel(config_model)
        model.initialize(config.f, config.c)
        n_samples = max(5000, int(sum(p.numel() for p in model.parameters() if p.requires_grad) / 0.75))
        config_df = DataSetsConfig(n_samples, config.c, config.f, config.co, config.sc, config.leafs)
        file_name = f"{path}datasets/synthetic/df_{nameDf(config_df, cross)}.csv"
        try:
            return pd.read_csv(file_name).iloc[:, 1:]
        except FileNotFoundError:
            return generateSaveData(config_df, cross, path)

    def load_real_life_dataset(name):
        """
        Load a real-life dataset and adjust column names.
        """
        file_name = f"{path}datasets/reallife/{name}.csv"
        df = pd.read_csv(file_name).iloc[:, 1:]
        num_features = df.shape[1] - 1
        df.columns = [f'F{i}' for i in range(num_features)] + ['target']
        df['target'] = df['target'].astype(int) - 1
        if name == "covtype":
            df = df.sample(frac=0.5)
        return df

    # Main function logic
    try:
        config = DataFrameConfigs[dataset_key]
        return load_synthetic_dataset(config, cross)
    except KeyError:
        try:
            # Attempt to use dataset_key as direct configuration
            return load_synthetic_dataset(dataset_key, cross)
        except FileNotFoundError:
            try:
                return load_real_life_dataset(dataset_key)
            except FileNotFoundError:
                # Generate data if all previous attempts fail
                config = DataFrameConfigs.get(dataset_key, dataset_key)
                return load_synthetic_dataset(config, cross)


def simpleSny():
    """
    function to generate a synthetic dataframe using a standard library approach
    """
    X, y = make_classification(
        n_samples=10000,
        n_features=10,
        n_informative=10,
        n_redundant=0,
        class_sep=2,
        n_classes=3,
        weights=[0.7, 0.2, 0.1],
        random_state=42
    )

    feature_columns = [f'F{i}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_columns)

    df['target'] = y

    return df


def calculate_gini(x, path=""):
    """
    function to calculate gini for specific dataset

    Parameters:
    x (str): Dataset identifier.
    path (str, optional): Path to dataset

    Returns:
    g (float): Gini coefficient.
    """
    x = loadDF(x, path).sample(frac=0.05)["target"].to_numpy()
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad / np.mean(x)
    g = 0.5 * rmad
    return g


def generateSynData():
    """
    function to generate all needed datasets even in case they already exist
    """
    for config in DataFrameConfigs:
        generateSaveData(config)
