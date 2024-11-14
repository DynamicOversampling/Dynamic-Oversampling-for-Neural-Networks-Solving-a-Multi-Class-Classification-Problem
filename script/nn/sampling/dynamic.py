import sys
from math import log

import numpy as np
from sklearn.neighbors import NearestNeighbors

from script.configs.configs import DYNSETTING
from script.data.data_preperation.data_prep import string_to_array


def dynamic_oversampling(t, f, conf: DYNSETTING, epochs, k=5, accuracy_dict=None):
    """
    function to oversample the trainings data during training process using the k-neighbors procedure

    Parameters:
        t (array): Target labels.
        f (array): Feature values
        conf (DYNSETTING): Configuration settings.
        epochs (int): Number of epochs.
        k (int, optional): Neighbors count
        accuracy_dict (dict): Accuracy dictionary

    Returns:
        X_resampled (array): Resampled feature data.
        y_resampled_one_hot (array): One-hot encoded labels.
    """
    class_indices = np.argmax(f, axis=1)

    X_resampled = t.copy()
    y_resampled = class_indices.copy()

    if accuracy_dict is None:
        accuracy_dict = {cls: 100 for cls in np.unique(class_indices)}

    total_percentage = sum(accuracy_dict.values())
    num_entries = len(accuracy_dict)
    average_value = total_percentage / num_entries

    unique_classes = np.unique(class_indices)
    minority_classes = [cls for cls in unique_classes if
                        np.sum(class_indices == cls) <= np.max(np.bincount(class_indices))]

    acc_min = min(accuracy_dict.values())

    print(f"Unique classes in dataset: {unique_classes}")
    print(f"Identified minority classes for oversampling: {minority_classes}")

    for cls in minority_classes:
        X_min = t[class_indices == cls]
        n_samples = X_min.shape[0]

        print(average_value)
        print(accuracy_dict[cls])

        if cls in accuracy_dict:
            n_synthetic, squeezed_value = calc_n_syn(np.max(np.bincount(class_indices)), n_samples, average_value,
                                                     accuracy_dict[cls], acc_min, epochs, conf.ratio)
        else:
            n_synthetic = 0
            squeezed_value = 0

        print(
            f"Class {cls}: {n_samples} samples, oversampling ratio: {squeezed_value:.2f}, synthetic samples to generate: {n_synthetic}")

        neigh = NearestNeighbors(n_neighbors=k + 1)
        neigh.fit(X_min)

        for i in range(n_synthetic):
            idx = np.random.randint(0, n_samples)
            sample = X_min[idx]

            nn_indices = neigh.kneighbors([sample], return_distance=False)[0][1:]

            nn_idx = np.random.choice(nn_indices)
            nn = X_min[nn_idx]

            alpha = np.random.random()
            synthetic_sample = sample + alpha * (nn - sample)

            X_resampled = np.vstack((X_resampled, synthetic_sample))
            y_resampled = np.append(y_resampled, cls)

    num_classes = f.shape[1]
    y_resampled_one_hot = np.zeros((y_resampled.size, num_classes))
    y_resampled_one_hot[np.arange(y_resampled.size), y_resampled] = 1

    print("Oversampling completed.")
    return X_resampled, y_resampled_one_hot


def calc_n_syn(n_max, n_class, acc_av, acc_class, acc_min, epochs, lvl):
    """
    function to calculate the number of new synthetics per class based primarily on class accuracy and size

    Parameters:
        n_max (int): Max class entries.
        n_class (int): Current number of entries.
        acc_av (float): Average accuracy
        acc_class (float): Class accuracy.
        acc_min (float): Minimum accuracy
        epochs (int): Number of training epochs.
        lvl (float): Scaling facto

    Returns:
        int: Number of synthetic samples to generate.
        float: Normalized value used in calculation.
    """
    diff = max(0, (acc_av - acc_class) if (acc_av - acc_class) <= 0 else (acc_av - (acc_class - acc_min)))
    normalized_value = max(0, (diff - acc_min) / (acc_av - acc_min + sys.float_info.min))
    squeezed_value = (np.sqrt(normalized_value) + sys.float_info.min) / (epochs * 0.01)
    return int((n_max - n_class) * 1 / log(n_class + 1, 2.3) * squeezed_value * lvl / 25), squeezed_value


def data_oversample_wheight(train_dataset, class_accuracies, param, dynsettings, epochs):
    """
    function to prepare dataset for dynamic oversampling splitting features and targets

    Parameters:
        train_dataset (DataFrame): Dataset to oversample.
        class_accuracies (dict): Accuracy per class.
        param (float): Oversampling parameter.
        dynsettings (dict): Settings for dynamic oversampling.
        epochs (int): Number of training epochs.

    Returns:
        array: Oversampled features and targets.
    """
    try:
        data = train_dataset.drop(train_dataset.columns[0], axis=1)
        f = data.drop('target', axis=1).values
        t = data['target'].values
        t = np.array(list(map(lambda x: string_to_array(x, t.max()), t)))
    except:
        f, t = train_dataset

    return dynamic_oversampling(f, t, dynsettings, epochs, param, class_accuracies)
