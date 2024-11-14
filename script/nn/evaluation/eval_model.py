from collections import defaultdict

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from script.data.data_preperation.data_prep import set_gen_ai


def eval_model_f1(targets, predictions, num_classes):
    """
    function to calculate the macro f1 score on the gpu during training process

    Parameters:
    targets (Tensor): Target labels.
    predictions (Tensor): Predicted labels.
    num_classes (int): Number of classes.

    Returns:
    float: Macro F1 score.
    """
    tp = torch.zeros(num_classes, device=targets.device)
    fp = torch.zeros(num_classes, device=targets.device)
    fn = torch.zeros(num_classes, device=targets.device)

    for i in range(num_classes):
        tp[i] = ((predictions == i) & (targets == i)).sum()
        fp[i] = ((predictions == i) & (targets != i)).sum()
        fn[i] = ((predictions != i) & (targets == i)).sum()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_macro = f1.mean()

    return f1_macro


def eval_model_acc(model, data_loader):
    """
    function to calculate the accuracy on the gpu during training process

    Parameters:
    model (nn.Module): Trained model.
    data_loader (DataLoader): DataLoader for test set

    Returns:
    tuple: Accuracy and F1 score.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    correct = 0
    total = 0

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            if targets.dim() > 1:
                targets = torch.max(targets, 1)[1]

            all_targets.append(targets)
            all_predictions.append(predicted)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    all_targets = torch.cat(all_targets).to(device)
    all_predictions = torch.cat(all_predictions).to(device)

    f1_macro = eval_model_f1(all_targets, all_predictions, num_classes=len(torch.unique(all_targets)))

    return round(100 * correct / total, 2), round(f1_macro.item(), 4)


def eval_model_acc_class(model, data_loader):
    """
    function to calculate the accuracy per class on the gpu during training process

    Parameters:
    model (nn.Module): Trained model.
    data_loader (DataLoader): DataLoader for the test set.

    Returns:
    tuple: Dictionary of class accuracies, confusion matrix, confusion counts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.eval()
    correct_by_class = defaultdict(int)
    total_by_class = defaultdict(int)
    class_accuracies = {}

    class_matrix = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for inputs, targets in data_loader:
            if inputs.size(0) < 512 or inputs.size(1) < 52:
                continue
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            target = torch.argmax(targets, dim=1)

            for t, p in zip(target, predicted):
                t_id = t.item()
                p_id = p.item()
                total_by_class[t_id] += 1
                class_matrix[t_id][p_id] += 1
                if t == p:
                    correct_by_class[t_id] += 1

    for class_id in total_by_class:
        total = total_by_class[class_id]
        correct = correct_by_class[class_id]
        class_accuracies[class_id] = round(100 * correct / total, 2)

    sorted_class_accuracies = dict(sorted(class_accuracies.items()))

    result_list = []

    for class_id, accuracy in sorted_class_accuracies.items():
        accuracy = sorted_class_accuracies[class_id]
        formatted_percent = f"{accuracy:05.2f}%"
        confusion_counts = [class_matrix[class_id][i] for i in range(len(total_by_class))]
        formatted_counts = [f"{count:05d}" for count in confusion_counts]
        print(f"{class_id}: {formatted_percent}% | {', '.join(map(str, formatted_counts))}")
        result_list.append(formatted_counts)

    return sorted_class_accuracies, class_matrix, result_list


def eval_model_acc_class_load(modelName, df, path):
    """
    function to re-evaluate model on dataset covtype

    Parameters:
        modelName (str): Name of model
        df (DataFrame): DataFrame of the dataset.
        path (str): Path to the model

    Returns:
        list: Confusion matrix by class.
    """

    model = torch.load(f"{path}models/"+modelName+".pt")

    num_features = df.shape[1] - 1
    feature_columns = [f'F{i}' for i in range(num_features)]
    df.columns = feature_columns + ['target']

    df['target'] = df['target'] - 1
    df['target'] = df['target'].astype(int)

    print(df.head())
    print(df['target'])

    df = df.sample(frac=0.5)

    scaler = MinMaxScaler()
    features = df.drop(columns=['target'])

    features_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df_norm = pd.concat([features_normalized, df['target'].reset_index(drop=True)], axis=1)

    dataloader = DataLoader(set_gen_ai(df_norm), batch_size=512, shuffle=False, pin_memory=False)
    _, _, results = eval_model_acc_class(model, dataloader)
    return results
