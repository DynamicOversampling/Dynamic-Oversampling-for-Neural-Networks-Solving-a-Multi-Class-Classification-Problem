from script.visualization.graph_generator.data.visualize_data import set_plot_defaults


def plot_function():
    """
    function to generate graph depicting
    the n synthetics ratio in dependency of class accuracy
    """
    import numpy as np
    import matplotlib.pyplot as plt

    set_plot_defaults()
    def function(acc_class: float):
        acc_av = 0.8
        acc_min = 0.0
        epochs = 100

        diff = max(0, (acc_av - acc_class) if (acc_av - acc_class) <= 0 else (acc_av - (acc_class - acc_min)))

        normalized_value = max(0, (diff - acc_min) / (acc_av - acc_min))
        print(normalized_value)
        squeezed_value = (np.sqrt(normalized_value) + 0.001) / (epochs * 0.01)

        return squeezed_value

    data = np.arange(0, 1000)
    learning_rates = [function(data/1000) for data in data]

    plt.figure(figsize=(10, 6))
    plt.plot(data/1000, learning_rates, label='Ratio', color='b')
    plt.axvline(0.8, color='r', linestyle='--', label='Average Test Accuracy')
    plt.xlabel('Test Class Accuracy')
    plt.ylabel('New Synthetics Ratio')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig("plots/qlr.svg")
    plt.show()

plot_function()
