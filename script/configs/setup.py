from script.configs.configs import DataSetsConfig, ModelConfig, TrainingConfig, TrainingMethod, DYNSETTING, MetaConfig

"""
file implementing all needed configuration data classes
to configure the synthetic datasets, neural networks, training process and dynamic oversampling setup
"""

DataFrameConfigs = [
    DataSetsConfig(n=0, c=3, f=5, co=1.25, sc=1, leafs=5),
    DataSetsConfig(n=0, c=3, f=10, co=1.25, sc=2, leafs=10),
    DataSetsConfig(n=0, c=10, f=10, co=1.25, sc=3, leafs=5),
    DataSetsConfig(n=0, c=10, f=20, co=1.25, sc=1, leafs=10),
    DataSetsConfig(n=0, c=20, f=20, co=1.25, sc=2, leafs=5),
    DataSetsConfig(n=0, c=20, f=30, co=1.25, sc=2, leafs=10),
    DataSetsConfig(n=0, c=30, f=20, co=1.25, sc=1, leafs=5),
    DataSetsConfig(n=0, c=30, f=30, co=1.25, sc=2, leafs=10),
]

cov_type = "covtype"
fatal_health = "fetal_health"

configMd = ModelConfig(6, 2, 0.25)
configNONE = TrainingConfig(TrainingMethod.NONE)
configSMOTE = TrainingConfig(TrainingMethod.SMOTE)
configADASYN = TrainingConfig(TrainingMethod.ADASYN)
configDYN = TrainingConfig(TrainingMethod.DYNAMIC, DYNSETTING(25, 5, 99999, 0))
configDYNBaysF1 = TrainingConfig(TrainingMethod.DYNAMIC, DYNSETTING(56, 3, 24, 0.02))
configDYNBaysAcc = TrainingConfig(TrainingMethod.DYNAMIC, DYNSETTING(99, 4, 42, 0.39))

meta_config = [
    MetaConfig(DataFrameConfigs[0], configMd, configNONE),
    MetaConfig(DataFrameConfigs[0], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[0], configMd, configADASYN),
    MetaConfig(DataFrameConfigs[0], configMd, configDYN),
    MetaConfig(DataFrameConfigs[1], configMd, configNONE),
    MetaConfig(DataFrameConfigs[1], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[1], configMd, configADASYN),
    MetaConfig(DataFrameConfigs[1], configMd, configDYN),
    MetaConfig(DataFrameConfigs[2], configMd, configNONE),
    MetaConfig(DataFrameConfigs[2], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[2], configMd, configADASYN),
    MetaConfig(DataFrameConfigs[2], configMd, configDYN),
    MetaConfig(DataFrameConfigs[3], configMd, configNONE),
    MetaConfig(DataFrameConfigs[3], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[3], configMd, configADASYN),
    MetaConfig(DataFrameConfigs[3], configMd, configDYN),
    MetaConfig(DataFrameConfigs[4], configMd, configNONE),
    MetaConfig(DataFrameConfigs[4], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[4], configMd, configADASYN),
    MetaConfig(DataFrameConfigs[4], configMd, configDYN),
    MetaConfig(DataFrameConfigs[5], configMd, configNONE),
    MetaConfig(DataFrameConfigs[5], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[5], configMd, configADASYN),
    MetaConfig(DataFrameConfigs[5], configMd, configDYN),
    MetaConfig(DataFrameConfigs[6], configMd, configNONE),
    MetaConfig(DataFrameConfigs[6], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[6], configMd, configADASYN),
    MetaConfig(DataFrameConfigs[6], configMd, configDYN),
    MetaConfig(DataFrameConfigs[7], configMd, configNONE),
    MetaConfig(DataFrameConfigs[7], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[7], configMd, configADASYN),
    MetaConfig(DataFrameConfigs[7], configMd, configDYN),
]

meta_configDYNBays = [
    MetaConfig(DataFrameConfigs[0], configMd, configDYN),
    MetaConfig(DataFrameConfigs[0], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[0], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[1], configMd, configDYN),
    MetaConfig(DataFrameConfigs[1], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[1], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[2], configMd, configDYN),
    MetaConfig(DataFrameConfigs[2], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[2], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[3], configMd, configDYN),
    MetaConfig(DataFrameConfigs[3], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[3], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[4], configMd, configDYN),
    MetaConfig(DataFrameConfigs[4], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[4], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[5], configMd, configDYN),
    MetaConfig(DataFrameConfigs[5], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[5], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[6], configMd, configDYN),
    MetaConfig(DataFrameConfigs[6], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[6], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[7], configMd, configDYN),
    MetaConfig(DataFrameConfigs[7], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[7], configMd, configDYNBaysF1),
]

meta_configDYNBaysF1 = [
    MetaConfig(DataFrameConfigs[0], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[0], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[0], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[0], configMd, configDYN),
    MetaConfig(DataFrameConfigs[1], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[1], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[1], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[1], configMd, configDYN),
    MetaConfig(DataFrameConfigs[2], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[2], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[2], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[2], configMd, configDYN),
    MetaConfig(DataFrameConfigs[3], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[3], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[3], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[3], configMd, configDYN),
    MetaConfig(DataFrameConfigs[4], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[4], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[4], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[4], configMd, configDYN),
    MetaConfig(DataFrameConfigs[5], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[5], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[5], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[5], configMd, configDYN),
    MetaConfig(DataFrameConfigs[6], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[6], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[6], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[6], configMd, configDYN),
    MetaConfig(DataFrameConfigs[7], configMd, configSMOTE),
    MetaConfig(DataFrameConfigs[7], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[7], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[7], configMd, configDYN),
]

meta_configDYNBaysSMOTEBays = [
    MetaConfig(DataFrameConfigs[0], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[0], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[0], configMd, configDYN),
    MetaConfig(DataFrameConfigs[2], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[2], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[2], configMd, configDYN),
    MetaConfig(DataFrameConfigs[6], configMd, configDYNBaysAcc),
    MetaConfig(DataFrameConfigs[6], configMd, configDYNBaysF1),
    MetaConfig(DataFrameConfigs[6], configMd, configDYN),
]

meta_config_fata = [
    MetaConfig(fatal_health, ModelConfig(3, 2, 0.25), configNONE),
    MetaConfig(fatal_health, ModelConfig(3, 2, 0.25), configSMOTE),
    MetaConfig(fatal_health, ModelConfig(3, 2, 0.25), configADASYN),
    MetaConfig(fatal_health, ModelConfig(3, 2, 0.25), configDYNBaysF1),
]

meta_config_cov = [
    MetaConfig(cov_type, configMd, configNONE),
    MetaConfig(cov_type, configMd, configSMOTE),
    MetaConfig(cov_type, configMd, configADASYN),
    MetaConfig(cov_type, configMd, configDYNBaysF1),
]

bays_config = [
    MetaConfig(DataFrameConfigs[0], configMd,
               TrainingConfig(TrainingMethod.DYNAMIC, DYNSETTING(20, 5, 20))),
    MetaConfig(DataFrameConfigs[2], configMd,
               TrainingConfig(TrainingMethod.DYNAMIC, DYNSETTING(20, 5, 20))),
    MetaConfig(DataFrameConfigs[6], configMd,
               TrainingConfig(TrainingMethod.DYNAMIC, DYNSETTING(20, 5, 20))),
]

bays_config_model = [
    MetaConfig(DataFrameConfigs[0], configMd,
               TrainingConfig(TrainingMethod.DYNAMIC, DYNSETTING(25, 5, 99999, 0))),
    MetaConfig(DataFrameConfigs[1], configMd,
               TrainingConfig(TrainingMethod.DYNAMIC, DYNSETTING(25, 5, 99999, 0))),
    MetaConfig(DataFrameConfigs[2], configMd,
               TrainingConfig(TrainingMethod.DYNAMIC, DYNSETTING(25, 5, 99999, 0))),
]

bays_cross = 2
bays_modelCross = 1
