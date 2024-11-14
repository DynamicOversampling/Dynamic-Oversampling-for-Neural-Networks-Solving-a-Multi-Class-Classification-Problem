import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from script.visualization.graph_generator.data.visualize_data import set_plot_defaults


def regression(data):
    """
    funciton for regression analyzes for
    the neural network setup layer, neuron_factor vs target
    """
    X = np.array([[int(d["params"]["layer"]), int(d["params"]["neuron_factor"])] for d in data])
    y = np.array([d["target"] for d in data])

    regressor = LinearRegression()
    regressor.fit(X, y)

    layer_coef, neuron_factor_coef = regressor.coef_
    intercept = regressor.intercept_

    print(f"target = {intercept:.4f} + {layer_coef:.4f} * layer + {neuron_factor_coef:.4f} * neuron_factor")

    neuron_factor_mean = np.mean(X[:, 1])
    layer_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    predicted_target_layer = regressor.predict(
        np.column_stack((layer_values, np.full_like(layer_values, neuron_factor_mean))))

    layer_mean = np.mean(X[:, 0])
    neuron_factor_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    predicted_target_neuron = regressor.predict(
        np.column_stack((np.full_like(neuron_factor_values, layer_mean), neuron_factor_values)))

    set_plot_defaults(14)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[1].scatter(X[:, 0], y, color='blue')
    axs[1].plot(layer_values, predicted_target_layer, color='red')
    axs[1].set_xlabel("Layer")
    axs[1].set_ylabel("Accuracy")

    axs[0].scatter(X[:, 1], y, color='green')
    axs[0].plot(neuron_factor_values, predicted_target_neuron, color='red')
    axs[0].set_xlabel("Neuron Factor")
    axs[0].set_ylabel("Accuracy")

    plt.tight_layout()

    plt.savefig("plots/baysReg.svg")

    plt.show()


data = [
    {"target": 0.9129266666666667, "params": {"layer": 3.08511002351287, "neuron_factor": 6.762595947537265},
     "datetime": {"datetime": "2024-10-20 00:20:10", "elapsed": 0.0, "delta": 0.0}},
    {"target": 0.8124266666666666, "params": {"layer": 1.0005718740867244, "neuron_factor": 3.418660581054718},
     "datetime": {"datetime": "2024-10-20 00:20:11", "elapsed": 0.439096, "delta": 0.439096}},
    {"target": 0.9129266666666667, "params": {"layer": 3.155953832547476, "neuron_factor": 6.581823400427613},
     "datetime": {"datetime": "2024-10-20 00:20:11", "elapsed": 1.035228, "delta": 0.596132}},
    {"target": 0.8968733333333333, "params": {"layer": 6.0, "neuron_factor": 7.800105617053243},
     "datetime": {"datetime": "2024-10-20 00:20:12", "elapsed": 1.553343, "delta": 0.518115}},
    {"target": 0.8545733333333334, "params": {"layer": 6.0, "neuron_factor": 3.91266688804262},
     "datetime": {"datetime": "2024-10-20 00:20:13", "elapsed": 2.259497, "delta": 0.706154}},
    {"target": 0.8519466666666665, "params": {"layer": 1.0, "neuron_factor": 9.0},
     "datetime": {"datetime": "2024-10-20 00:20:13", "elapsed": 2.855628, "delta": 0.596131}},
    {"target": 0.9163466666666669, "params": {"layer": 4.2706058702082546, "neuron_factor": 9.0},
     "datetime": {"datetime": "2024-10-20 00:20:14", "elapsed": 3.482768, "delta": 0.62714}},
    {"target": 0.8796133333333332, "params": {"layer": 4.359816263914438, "neuron_factor": 7.259524259901938},
     "datetime": {"datetime": "2024-10-20 00:20:15", "elapsed": 4.232932, "delta": 0.750164}},
    {"target": 0.9298733333333333, "params": {"layer": 5.695597596893604, "neuron_factor": 9.0},
     "datetime": {"datetime": "2024-10-20 00:28:17", "elapsed": 486.332189, "delta": 482.099257}},
    {"target": 0.8448666666666667, "params": {"layer": 6.0, "neuron_factor": 1.0},
     "datetime": {"datetime": "2024-10-20 00:31:10", "elapsed": 659.544368, "delta": 173.212179}},
    {"target": 0.8247333333333334, "params": {"layer": 1.5761200901305512, "neuron_factor": 6.349548051646141},
     "datetime": {"datetime": "2024-10-20 00:33:21", "elapsed": 790.216167, "delta": 130.671799}},
    {"target": 0.9170200000000001, "params": {"layer": 3.0097296853456816, "neuron_factor": 9.0},
     "datetime": {"datetime": "2024-10-20 00:39:56", "elapsed": 1185.546302, "delta": 395.330135}},
    {"target": 0.8763333333333333, "params": {"layer": 2.8954899907696285, "neuron_factor": 1.0},
     "datetime": {"datetime": "2024-10-20 00:42:03", "elapsed": 1313.176432, "delta": 127.63013}},
    {"target": 0.8570266666666666, "params": {"layer": 3.927945685707461, "neuron_factor": 4.928655242072012},
     "datetime": {"datetime": "2024-10-20 00:44:40", "elapsed": 1469.452869, "delta": 156.276437}},
    {"target": 0.8859066666666668, "params": {"layer": 3.269642546604944, "neuron_factor": 8.137987230058311},
     "datetime": {"datetime": "2024-10-20 00:50:07", "elapsed": 1796.203854, "delta": 326.750985}},
    {"target": 0.9080466666666667, "params": {"layer": 5.133770261180666, "neuron_factor": 8.609154160362328},
     "datetime": {"datetime": "2024-10-20 01:01:06", "elapsed": 2456.159755, "delta": 659.955901}},
    {"target": 0.8424666666666666, "params": {"layer": 1.0, "neuron_factor": 1.0},
     "datetime": {"datetime": "2024-10-20 01:03:17", "elapsed": 2586.238411, "delta": 130.078656}},
    {"target": 0.8823533333333333, "params": {"layer": 6.0, "neuron_factor": 6.073650440688892},
     "datetime": {"datetime": "2024-10-20 01:10:23", "elapsed": 3012.942417, "delta": 426.704006}},
    {"target": 0.95112, "params": {"layer": 6.0, "neuron_factor": 9.0},
     "datetime": {"datetime": "2024-10-20 01:26:05", "elapsed": 3954.643881, "delta": 941.701464}},
    {"target": 0.9114866666666668, "params": {"layer": 6.0, "neuron_factor": 8.625155783522597},
     "datetime": {"datetime": "2024-10-20 01:37:45", "elapsed": 4655.070187, "delta": 700.426306}},
    {"target": 0.8179333333333333, "params": {"layer": 3.8889868632603593, "neuron_factor": 2.5525457682273993},
     "datetime": {"datetime": "2024-10-20 01:40:23", "elapsed": 4812.792937, "delta": 157.72275}},
    {"target": 0.9002266666666665, "params": {"layer": 2.2769081692916515, "neuron_factor": 9.0},
     "datetime": {"datetime": "2024-10-20 01:43:59", "elapsed": 5028.602479, "delta": 215.809542}}
]

regression(data)
