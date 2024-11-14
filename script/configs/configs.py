from dataclasses import dataclass
from enum import Enum
from typing import Any

"""
file declaring all configuration data classes
to configure the synthetic datasets, neural networks, training process and dynamic oversampling setup
"""


class DataSetsConfig:
    def __init__(self, n: int, c: int, f: int, co: float, sc: int, leafs: int):
        self.n = n
        self.c = c
        self.f = f
        self.co = co
        self.sc = sc
        self.leafs = leafs


class TrainingMethod:
    pass


class DYNSETTING:
    pass


@dataclass
class TrainingConfig:
    oversampling: TrainingMethod
    dynsetting: DYNSETTING = DYNSETTING


class TrainingMethod(Enum):
    NONE = 0
    SMOTE = 1
    ADASYN = 2
    DYNAMIC = 3


@dataclass
class DYNSETTING:
    ratio: float = 50  # 0-100
    epochs: int = 5  # each x epochs
    reset: int = 20
    start_lvl: float = 0  # 0-1


@dataclass
class ModelConfig:
    factor: int
    layers: int
    drop_out: float


@dataclass
class MetaConfig:
    configDf: Any
    configMd: ModelConfig
    configTr: TrainingConfig


"""
functions to generate unqie name for
datasets, models and training log
"""


def nameDf(configDf, modelCr):
    try:
        return f"n_{configDf.n}_c{configDf.c}_f{configDf.f}_co{configDf.co}_sc{configDf.sc}_l{configDf.leafs}_mc{modelCr}"
    except:
        return f"{configDf}"


def nameMd(model_conf, modelCr=None):
    if modelCr:
        return f"ml{model_conf.layers}_mn{model_conf.factor}_md{model_conf.drop_out}"
    else:
        return f"ml{model_conf.layers}_mn{model_conf.factor}_md{model_conf.drop_out}"


def nameTr(trainingConfig):
    if trainingConfig.oversampling.name != TrainingMethod.DYNAMIC.name:
        return f"ts{trainingConfig.oversampling.name}"
    else:
        return f"ts{trainingConfig.oversampling.name}_{int(trainingConfig.dynsetting.ratio)}-{int(trainingConfig.dynsetting.epochs)}-{int(trainingConfig.dynsetting.reset)}-{int(trainingConfig.dynsetting.start_lvl)}"


def nameCr(crossfold):
    return f"cr{crossfold}"


def crf_naming(configTr):
    try:
        return f"{configTr.oversampling.value}-{configTr.dynsetting.start_lvl}"
    except:
        return configTr.oversampling.value
