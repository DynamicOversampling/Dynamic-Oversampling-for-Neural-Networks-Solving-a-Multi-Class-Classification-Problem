from script.visualization.graph_generator.data.visualize_data import set_plot_defaults
import numpy as np
import matplotlib.pyplot as plt


def plot_function():
    """
    function to generate graph depicting the n synthetics ratio in dependency of class accuracy
    normalized by acc_min
    """
    set_plot_defaults()

    def function(acc_class: float):
        acc_av = 0.95
        acc_min = 0.87
        epochs = 100
        if acc_class < 0.87:
            return None

        if acc_class > 0.95:
            return None

        diff = max(0, (acc_av - acc_class) if (acc_av - acc_class) <= 0 else (acc_av - (acc_class - acc_min)))

        normalized_value = max(0, (diff - acc_min) / (acc_av - acc_min))
        squeezed_value = (np.sqrt(normalized_value) + 0.001) / (epochs * 0.01)
        return squeezed_value

    data = np.arange(0, 1000)
    learning_rates = [function(data / 1000) for data in data]

    plt.figure(figsize=(10, 6))
    plt.plot(data / 1000, learning_rates, label='Ratio', color='b')
    plt.axvline(0.95, color='r', linestyle='--', label='Average Test Accuracy')
    plt.axvline(0.87, color='g', linestyle='--', label='Minimum Test Accuracy')
    plt.xlabel('Test Class Accuracy')
    plt.ylabel('New Synthetics Ratio')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig("plots/qlrAdap.svg")
    plt.show()


plot_function()
