import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events

import script.configs.setup
from script.configs.configs import ModelConfig
from script.visualization.util.utils import parseEntries


def run_bays(layer, neuron_factor):
    """
    function to use bayesian optimization
    to fine-tune the neural network setup

    @Misc{,
        author = {Fernando Nogueira},
        title = {{Bayesian Optimization}: Open source constrained global optimization tool for {Python}},
        year = {2014--},
        url = " https://github.com/bayesian-optimization/BayesianOptimization"
    }
    """
    crossfold_data_list = []
    for e in script.bays_config_model:
        configDf, configMd, configTr = e.configDf, script.configs.setup.configMd, e.configTr
        log_list, crossfold_data = parseEntries(configDf, ModelConfig(int(neuron_factor), int(layer), 0.25), configTr,
                                                1, 1, "../../../", epochss=80)
        crossfold_data_list.append(crossfold_data["TestAcc"]["mean"][-5:])
    return np.mean(np.array(crossfold_data_list))


pbounds = {'layer': (1, 6), 'neuron_factor': (1, 9)}

optimizer = BayesianOptimization(
    f=run_bays,
    pbounds=pbounds,
    random_state=1,
)

logger = JSONLogger(path="bays_log_model.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=2,
    n_iter=20,
)
