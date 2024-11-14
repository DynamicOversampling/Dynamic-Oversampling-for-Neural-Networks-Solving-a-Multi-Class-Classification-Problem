import torch
from torch import nn
from torchinfo import summary

from script.configs.configs import ModelConfig


class AdaptiveModel(nn.Module):
    """
    class that defines the neural network
    depending on the network settings and adapting to the datasets characteristics
    """

    def __init__(self, config: ModelConfig):
        super(AdaptiveModel, self).__init__()
        if config.factor != 0:
            temp = config.factor
        else:
            temp = 2
        self.model_neurons_factor = temp
        self.model_depth = config.layers
        self.drop = config.drop_out
        self.layers = nn.ModuleList()
        self.output_layer = None
        self.model_neurons = 0
        self.softmax = nn.Softmax(dim=1)

    def initialize(self, input_size, output_size):
        self.layers = nn.ModuleList()
        self.model_neurons = max(30, input_size * self.model_neurons_factor)
        current_size = input_size

        for i in range(self.model_depth):
            self.layers.append(nn.Linear(current_size, self.model_neurons))
            self.layers.append(nn.BatchNorm1d(self.model_neurons))
            self.layers.append(nn.Dropout(self.drop))
            current_size = self.model_neurons

        self.output_layer = nn.Linear(current_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = torch.relu(layer(x))
            elif isinstance(layer, (nn.BatchNorm1d, nn.Dropout)):
                x = layer(x)

        x = self.output_layer(x)
        x = self.softmax(x)
        return x


def analyze():
    """
    function to print the neural network characteristics
    """
    model = AdaptiveModel(ModelConfig(3, 2, 0.25))
    f_cnt, t_nr = 20, 10
    model.initialize(f_cnt, t_nr)
    summary(model, input_size=(512, f_cnt))
