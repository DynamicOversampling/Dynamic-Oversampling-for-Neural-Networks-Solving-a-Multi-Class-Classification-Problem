import json

import numpy as np


def bo3HT(log_file):
    """
    script to find the maximum value and standard deviation of
    the results of the bayesean optimization
    """

    entries = []

    with open(log_file, 'r') as file:
        for line in file:
            try:
                json_entry = json.loads(line.strip())
                entries.append(json_entry)
            except:
                print("FAILURE at vis_log 1")

    top = sorted(entries, key=lambda x: x['target'], reverse=True)
    unique_targets = set()
    top_unique = []

    for entry in top:
        if entry['target'] not in unique_targets:
            top_unique.append({'target': entry['target'], 'params': entry['params']})
            unique_targets.add(entry['target'])

        if len(top_unique) == 5:
            break

    print(top_unique)

    top = [{'target': entry['target'], 'params': entry['params']} for entry in top_unique]

    targets = np.array([entry['target'] for entry in top])
    epochs = np.array([int(entry['params']['epochs']) for entry in top])
    ratios = np.array([entry['params']['ratio'] / 25 for entry in top])
    resets = np.array([int(entry['params']['reset']) for entry in top])
    start_lvls = np.array([entry['params']['start_lvl'] / 80 for entry in top])

    std_target = np.std(targets)
    std_epochs = np.std(epochs)
    std_ratios = np.std(ratios)
    std_resets = np.std(resets)
    std_start_lvls = np.std(start_lvls)

    print(top)
    print(f"  Target: {std_target}")
    print(f"  Epochs: {std_epochs}")
    print(f"  Ratio: {std_ratios}")
    print(f"  Reset: {std_resets}")
    print(f"  Start Level: {std_start_lvls}")


log_file = "../../../nn/bayesian/bays_log_TestF1.log"
bo3HT(log_file)
