import numpy as np
import torch
import torch.utils.data as data_utils


def set_gen(df, test_frac: float):
    """
    function to split the dataset using random sample method

    Parameters:
        df (DataFrame): Input dataset.
        test_frac (float): Fraction of data to be used for testing

    Returns:
        tuple: Training and testing datasets.
    """
    test_set = df.sample(frac=test_frac)
    train_set = df.drop(test_set.index)

    return train_set, test_set


def set_gen_ai(df):
    """
    function convert the target class encoding into one-hot encoding and truncate to work in batches

    Parameters:
        df (DataFrame/tuple): Input data features/targets.

    Returns:
        TensorDataset: Processed dataset for batch.
    """
    try:
        data = df.drop(df.columns[0], axis=1)
        f = data.drop('target', axis=1).values
        t = data['target'].values
        t = np.array(list(map(lambda x: string_to_array(x, t.max()), t)))
    except:
        f, t = df

    f = np.array(f)
    t = np.array(t)

    length = len(f)
    truncate_length = length - (length % 32)

    if truncate_length < length:
        f = f[:truncate_length]
        t = t[:truncate_length]

    f_tensor = torch.tensor(f, dtype=torch.float32)
    t_tensor = torch.tensor(t, dtype=torch.float32)

    dataset = data_utils.TensorDataset(f_tensor, t_tensor)

    return dataset


def string_to_array(value, max):
    """
    function for one-hot encoding

    Parameters:
        value (str): Target class value.
        max (int): Maximum class index

    Returns:
        list: One-hot encoded array.
    """
    arr = [0] * (int(max) + 1)
    arr[int(value)] = 1
    return arr
