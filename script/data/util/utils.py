import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from script.configs.setup import DataFrameConfigs
from script.utils import calculate_gini
from script.data.data_preperation.data_prep import set_gen


def scale(df):
    """
    function to normalize dataset
    """
    scaler = MinMaxScaler()
    features = df.drop(columns=['target'])

    features_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df_norm = pd.concat([features_normalized, df['target'].reset_index(drop=True)], axis=1)

    return set_gen(df_norm, 0.2)


def get_class_n(train_dataset):
    """
    function to return the size of a class in dataset
    """
    try:
        return train_dataset['target'].value_counts().to_dict()
    except:
        train_dataset = train_dataset[1]
        position_counts = {i: 0 for i in range(train_dataset.shape[1])}

        for row in train_dataset:
            for index, value in enumerate(row):
                if value == 1.0:
                    position_counts[index] += 1
    return position_counts


def gini_data():
    """
    function to output all gini values of datasets
    """
    for e in enumerate(DataFrameConfigs):
        print(e[0])
        print(calculate_gini(e[0], "../../../"))
