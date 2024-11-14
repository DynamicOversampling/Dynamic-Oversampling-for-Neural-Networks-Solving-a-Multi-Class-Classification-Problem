import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events

import script.configs.setup
from script.configs.configs import TrainingConfig, TrainingMethod, DYNSETTING
from script.visualization.util.utils import parseEntries

def run_bays(ratio, epochs, reset, start_lvl):
    """
    function to use bayesian optimization
    to fine-tune the dynamic oversampling algorithm

    @Misc{,
        author = {Fernando Nogueira},
        title = {{Bayesian Optimization}: Open source constrained global optimization tool for {Python}},
        year = {2014--},
        url = " https://github.com/bayesian-optimization/BayesianOptimization"
    }
    """
    crossfold_data_list = []
    for e in script.bays_config:
        configDf, configMd, configTr = e.configDf, script.configs.setup.configMd, e.configTr
        log_list, crossfold_data = parseEntries(configDf, configMd, TrainingConfig(TrainingMethod.DYNAMIC,
                                                                                   DYNSETTING(int(ratio), int(epochs),
                                                                                              int(reset),
                                                                                              start_lvl / 80)),
                                                script.bays_cross, script.bays_modelCross, "../../../", epochss=150,
                                                enforce=True)
        crossfold_data_list.append(crossfold_data["TestAcc"]["mean"][-5:])
    return np.mean(np.array(crossfold_data_list))


def bayesian_optimization():
    """
    function performing the bayesian optimization
    """
    pbounds = {'ratio': (5, 100), 'epochs': (3, 10), 'reset': (20, 200), 'start_lvl': (0, 100)}

    optimizer = BayesianOptimization(
        f=run_bays,
        pbounds=pbounds,
        random_state=42,
    )

    logger = JSONLogger(path="bays_log_TestAC_I.log")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=2,
        n_iter=100,
    )
