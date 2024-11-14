import json
import random

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from script.configs.configs import nameTr, nameMd, nameDf, TrainingMethod, nameCr
from script.data.util.utils import get_class_n
from script.nn.evaluation.eval_model import eval_model_acc, eval_model_acc_class
from script.nn.model import AdaptiveModel

from script.data.data_preperation.data_prep import set_gen, set_gen_ai
from script.nn.sampling.standard import smoteDf, adasynDf
from script.nn.sampling.dynamic import data_oversample_wheight


def trainModel(df, model_conf, dataset, trainingConfig, crossfold=1, modelCr=0, path="", epochs=150):
    """
    function to train the neural network based on the configurations and dataset
    Generating detailed logs and saving them

    Parameters:
        df (DataFrame): dataset to train model.
        model_conf (ModelConfig): Configuration for model
        dataset (DataSetsConfig): Configuration for dataset
        trainingConfig (TrainingConfig): Configuration for training parameters.
        crossfold (int): The current repitions number
        modelCr (int): Model creation version
        path (str): Path to save
        epochs (int): Number of epochs
    """
    log_file = f'{path}logs/log_{nameMd(model_conf)}_{nameDf(dataset, modelCr)}_{nameTr(trainingConfig)}_{nameCr(crossfold)}.log'
    class_number_train_old = dict()
    try:
        with open(log_file, 'w') as file:
            file.write('')
    except:
        pass

    logs = []

    scaler = MinMaxScaler()
    features = df.drop(columns=['target'])

    constant_columns = features.columns[features.nunique() <= 1]
    features = features.drop(columns=constant_columns)
    features_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df_norm = pd.concat([features_normalized, df['target'].reset_index(drop=True)], axis=1)

    train_dataset, test_dataset = set_gen(df_norm, 0.2)

    if trainingConfig.oversampling == TrainingMethod.SMOTE:
        train_dataset = smoteDf(train_dataset)
    elif trainingConfig.oversampling == TrainingMethod.ADASYN:
        try:
            train_dataset = adasynDf(train_dataset)
        except:
            train_dataset = smoteDf(train_dataset)
    elif trainingConfig.oversampling == TrainingMethod.DYNAMIC and trainingConfig.dynsetting.start_lvl != 0:
        train_dataset = smoteDf(train_dataset, trainingConfig.dynsetting.start_lvl)

    test_dataset_tensor = set_gen_ai(test_dataset)
    train_dataset_tensor = set_gen_ai(train_dataset)
    train_dataset_init = train_dataset

    f_cnt = len(test_dataset_tensor[1][0])
    t_nr = len(test_dataset_tensor[1][1])

    model = AdaptiveModel(model_conf)

    model.initialize(f_cnt, t_nr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())
    test_loader = DataLoader(test_dataset_tensor, batch_size=512, shuffle=False, pin_memory=True)
    train_loader = DataLoader(train_dataset_tensor, batch_size=512, shuffle=True, pin_memory=True)

    class_accuracies = None

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        if trainingConfig.oversampling == TrainingMethod.DYNAMIC:
            if epoch % trainingConfig.dynsetting.reset == 0:
                train_dataset = train_dataset_init
            if epoch % trainingConfig.dynsetting.epochs == 0 and epoch > 1:
                train_dataset = data_oversample_wheight(train_dataset, class_accuracies, 3, trainingConfig.dynsetting,
                                                        epochs)
            train_dataset_tensor = set_gen_ai(train_dataset)

            train_loader = DataLoader(train_dataset_tensor, batch_size=512, shuffle=True, pin_memory=True)

        for inputs, targets in train_loader:
            model.train()

            inputs, targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loader_small = DataLoader(Subset(train_dataset_tensor, random.sample(range(len(train_dataset_tensor)),
                                                                                   min(5000,
                                                                                       len(train_dataset_tensor)))),
                                        batch_size=512, shuffle=True, pin_memory=True)

        training_acc, training_f1 = eval_model_acc(model, train_loader_small)
        test_acc, test_f1 = eval_model_acc(model, test_loader)

        string = f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, ' \
                 f'TrainingAcc: {training_acc}%, ' \
                 f'TestAcc: {test_acc}%, ' \
                 f'TrainingF1: {training_f1}, ' \
                 f'TestF1: {test_f1}, ' \
                 f'lr: 0, '
        print(string)

        class_accuracies_train = "undefined"
        class_accuracies_test = "undefined"

        if ((epoch + 1) % 5 == 0) or epochs - 1 == epoch or epoch == 0:
            print("train_loader:")
            class_accuracies_train, _ = eval_model_acc_class(model, train_loader_small)
            class_accuracies = class_accuracies_train

        if ((epoch + 1) % 5 == 0) or epochs - 1 == epoch or epoch == 0:
            print("test_loader:")
            class_accuracies_test, _ = eval_model_acc_class(model, test_loader)

        class_number_train = get_class_n(train_dataset)
        if class_number_train != class_number_train_old:
            class_number_train_old = class_number_train
        else:
            class_number_train = "undefined"

        log_data = {
            'Epoch': f'{(epoch + 1)}',
            'Loss': f'{loss}',
            'TrainingAcc': f'{training_acc / 100}',
            'TrainingF1': f'{training_f1}',
            'TestAcc': f'{test_acc / 100}',
            'TestF1': f'{test_f1}',
            'lr': f'0',
            'ClassNumberTrain': class_number_train,
            'ClassAccuraciesTrain': class_accuracies_train,
            'ClassAccuraciesTest': class_accuracies_test
        }

        logs.append(log_data)

    try:
        with open(log_file, 'w') as file:
            file.write(json.dumps(logs))
    except:
        pass

    torch.save(model, f"{path}models/model_{nameMd(model_conf)}_{nameDf(dataset, modelCr)}_{nameTr(trainingConfig)}.pt")
