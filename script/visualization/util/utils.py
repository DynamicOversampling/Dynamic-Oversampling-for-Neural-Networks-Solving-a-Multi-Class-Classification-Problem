import json
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy import swapaxes

import script.configs.setup
from script.configs.configs import nameMd, nameDf, nameTr, nameCr, DataSetsConfig, crf_naming
from script.data.data_generation.data_generation import generateSaveData
from script.nn.model import AdaptiveModel
from script.nn.train import trainModel


def save_crossfold_data_to_excel(crossfold_data, filename, path=""):
    """
    function to save aggregated data to excel

    Parameters:
        crossfold_data (dict): Dictionary of data
        filename (str): file name.
        path (str): Directory path.
    """
    all_data = pd.DataFrame()

    for key, value in crossfold_data.items():
        try:
            df_mean = pd.DataFrame(value['mean'], columns=[f'{key}_mean'])
            df_std = pd.DataFrame(value['std'], columns=[f'{key}_std'])

            df_combined = pd.concat([df_mean, df_std], axis=1)
        except:
            df_combined = pd.DataFrame()

            for i in range(len(value['mean'])):
                df_mean = pd.DataFrame(value['mean'][i], columns=[f'{key}:{i}_mean'])
                df_std = pd.DataFrame(value['std'][i], columns=[f'{key}:{i}_std'])

                df_temp = pd.concat([df_mean, df_std], axis=1)

                df_combined = pd.concat([df_combined, df_temp], axis=1)

        all_data = pd.concat([all_data, df_combined], axis=1)

    with pd.ExcelWriter(f"{path}excel/{filename}.xlsx") as writer:
        all_data.to_excel(writer, sheet_name='CrossfoldData')


def parseEntries(configDf, configMd, configTr, crossfold, modelCrossfold, path="", skip=False, enforce=False,
                 epochs=150):
    """
    function to aggregate data by either collecting existing logs or start training procedure

    Parameters:
        configDf (DataSetsConfig): Dataset configuration.
        configMd (ModelConfig): Model configuration.
        configTr (TrainingConfig): Training configuration.
        crossfold (int): Repition count.
        modelCrossfold (int): Model number count.
        path (str): Directory path.
        skip (bool): Skip flag.
        enforce (bool): Enforce flag.
        epochs (int): Number of epochs.
    """

    def prepare_dataset(configDf, modelCr):
        """Prepare or generate the required dataset based on configuration."""
        if configDf.n == 0:
            model = AdaptiveModel(configMd)
            model.initialize(configDf.f, configDf.c)
            n = max(5000, int(sum(p.numel() for p in model.parameters() if p.requires_grad) / 0.75))
            configDf = DataSetsConfig(n, configDf.c, configDf.f, configDf.co, configDf.sc, configDf.leafs)
        return f"{path}datasets/synthetic/df_{nameDf(configDf, modelCr)}.csv"

    def load_log_data(log_file):
        """Load JSON entries from a log file."""
        entries = []
        with open(log_file, 'r') as file:
            for line in file:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print("Error decoding JSON log entry.")
        return entries

    def start_training(configDf, modelCr, cross):
        """Run training and log results if file is missing or enforce is True."""
        file_name = prepare_dataset(configDf, modelCr)
        if enforce or not pd.read_csv(file_name, on_bad_lines='skip').empty:
            trainModel(generateSaveData(configDf, modelCr, path), configMd, configDf, configTr, cross, modelCr, path,
                       epochs)

    # Main loop to handle each model crossfold entry
    crossfold_entries = []
    for modelCr in range(modelCrossfold):
        for cross in range(crossfold):
            log_file = f'{path}logs/log_{nameMd(configMd)}_{nameDf(configDf, modelCr)}_{nameTr(configTr)}_{nameCr(cross)}.log'
            if enforce or not skip:
                start_training(configDf, modelCr, cross)
            else:
                print(f"Skipping missing log file: {log_file}")
                continue

            try:
                crossfold_entries.extend(load_log_data(log_file))
            except FileNotFoundError:
                print(f"Log file not found: {log_file}")

    # Parse log entries into structured data
    data_bundle = defaultdict(list)
    for log in crossfold_entries:
        for key in ['Epoch', 'Loss', 'TrainingAcc', 'TestAcc', 'TrainingF1', 'TestF1']:
            data_bundle[key].append([float(entry[key]) for entry in log])

        # Class-level metrics processing
        class_counts, acc_train, acc_test = [], [], []
        for entry in log:
            class_counts.append(entry.get('ClassNumberTrain', {}))
            acc_train.append(entry.get('ClassAccuraciesTrain', {}))
            acc_test.append(entry.get('ClassAccuraciesTest', {}))

        data_bundle['ClassNumberTrain'].append(class_counts)
        data_bundle['ClassAccuraciesTrain'].append(acc_train)
        data_bundle['ClassAccuraciesTest'].append(acc_test)

    # Calculate mean and std for each metric
    crossfold_data = {key: {
        'mean': np.mean(values, axis=0),
        'std': np.std(values, axis=0)
    } for key, values in data_bundle.items()}

    # Save results
    excel_name = f"{crossfold}-{modelCrossfold}-{nameMd(configMd)}_{nameDf(configDf, modelCrossfold)}_{nameTr(configTr)}_{nameCr(crossfold)}"
    save_crossfold_data_to_excel(crossfold_data, excel_name, path)

    return crossfold_entries, crossfold_data


def parseEntriesByOversampling(meta_config, crossfold, modelCrossfold, path="", skip=False, enforce=False):
    """
    function to aggregate data that are grouped by oversampling method

    Parameters:
        meta_config (list): List of configurations.
        crossfold (int): Cross-validation fold number
        modelCrossfold (int): Model crossfold number
        path (str): Directory path.
        skip (bool): Skip flag.
        enforce (bool): Enforce flag.
    """
    oversampling_data = defaultdict(list)
    keys_to_append = ["Epoch", "TrainingAcc", "TestAcc", "TrainingF1", "TestF1", "Loss"]

    for e in meta_config:
        configDf, configMd, configTr = e.configDf, script.configs.setup.configMd, e.configTr
        _, crossfold_data = parseEntries(configDf, configMd, configTr, crossfold, modelCrossfold, path, skip, enforce)
        filtered_data = {key: crossfold_data[key] for key in keys_to_append if key in crossfold_data}
        oversampling_data[crf_naming(configTr)].append(filtered_data)
    print(oversampling_data.items())

    crossfold_data_list = {}
    for oversampling_type, data_list in oversampling_data.items():
        if data_list:
            averaged_data = {
                key: {
                    'mean': np.mean([data[key]['mean'] for data in data_list], axis=0),
                    'std': np.mean([data[key]['std'] for data in data_list], axis=0)
                }
                for key in data_list[0]
            }

            def format_value(mean, std):
                return f"{round(mean[-1], 4)} ({round(std[-1], 4)})" if mean is not None and std is not None else None

            table_data = {
                'Sampling Method': [oversampling_type],
                'Training Accuracy': [format_value(averaged_data.get('TrainingAcc', {}).get('mean'),
                                                   averaged_data.get('TrainingAcc', {}).get('std'))],
                'Training F1 – Score': [format_value(averaged_data.get('TrainingF1', {}).get('mean'),
                                                     averaged_data.get('TrainingF1', {}).get('std'))],
                'Test Accuracy': [format_value(averaged_data.get('TestAcc', {}).get('mean'),
                                               averaged_data.get('TestAcc', {}).get('std'))],
                'Test F1 – Score': [format_value(averaged_data.get('TestF1', {}).get('mean'),
                                                 averaged_data.get('TestF1', {}).get('std'))],
                'Loss': [
                    format_value(averaged_data.get('Loss', {}).get('mean'), averaged_data.get('Loss', {}).get('std'))]
            }

            df = pd.DataFrame(table_data)
            excel_name = f"CAESUM_{crossfold}-{modelCrossfold}-{nameMd(configMd)}_{nameDf(configDf, modelCrossfold)}_{nameTr(configTr)}_{nameCr(crossfold)}"
            excel_path = f"{path}excel/{excel_name}.xlsx"

            try:
                existing_df = pd.read_excel(excel_path)
                final_df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                final_df = df

            final_df.to_excel(excel_path, index=False)

            excel_name = f"CAE-{oversampling_type}_{crossfold}-{modelCrossfold}-{nameMd(configMd)}_{nameDf(configDf, modelCrossfold)}_{nameTr(configTr)}_{nameCr(crossfold)}"
            save_crossfold_data_to_excel(averaged_data, excel_name, path)
            crossfold_data_list[oversampling_type] = averaged_data

    return crossfold_data_list
