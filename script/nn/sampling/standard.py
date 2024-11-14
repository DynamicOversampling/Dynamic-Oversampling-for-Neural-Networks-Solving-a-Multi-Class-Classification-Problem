import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN


def smoteDf(df, lvl=1):
    """
    function to oversample the dataset using SMOTE with the option to only partially level out

    Parameters:
        df (DataFrame): Dataset.
        lvl (float, optional): Level of oversampling (default=1 -> SMOTE).

    Returns:
        df_resampled (DataFrame): Resampled dataset
    """
    X = df.drop('target', axis=1)
    y = df['target']

    print(y.value_counts())
    print(max(y.value_counts()))

    dic = y.value_counts()
    max_value = max(y.value_counts())
    transformed_target = {k: int(v + (max_value - v) * lvl) for k, v in dic.items()}
    print(transformed_target)

    smote = SMOTE(sampling_strategy=transformed_target, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['target'] = y_resampled
    return df_resampled


def adasynDf(df):
    """
    function to oversample the dataset using ADASYN

    Parameters:
        df (DataFrame): Dataset

    Returns:
        df_resampled (DataFrame): Resampled dataset
    """
    X = df.drop('target', axis=1)
    y = df['target']

    ada = ADASYN(random_state=42)
    X_resampled, y_resampled = ada.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['target'] = y_resampled
    return df_resampled
