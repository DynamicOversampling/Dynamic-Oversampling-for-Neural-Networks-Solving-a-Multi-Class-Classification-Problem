from cmath import log
import numpy as np
import matplotlib.pyplot as plt

from script.visualization.graph_generator.data.visualize_data import set_plot_defaults


def plot_function(learning_rate: float, learning_rate_goal: float, num_epochs_lr: int, total_epochs: int):
    """
    function to generate graph depicting
    the n synthetics in dependency of class size
    """
    set_plot_defaults()

    def function(n_max: float, n_classa: float, num_epochs_lr: int, n_class: int):
        return (n_max - n_class) * 1 / log(n_class + 1, 2.3)

    epochs = np.arange(2, total_epochs)
    learning_rates = [function(learning_rate, learning_rate_goal, num_epochs_lr, epoch) for epoch in epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, label='New Synthetics', color='b')
    plt.axvline(1000, color='r', linestyle='--', label='N Majority Class')
    plt.xlabel('N minority class')
    plt.ylabel('New Synthetics')
    plt.title('Oversampling N Schedule')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/ons.svg")
    plt.show()


plot_function(learning_rate=1000, learning_rate_goal=0.005, num_epochs_lr=100, total_epochs=1000)
