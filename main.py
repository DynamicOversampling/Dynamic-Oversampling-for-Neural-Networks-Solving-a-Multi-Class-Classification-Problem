from script.configs.setup import meta_config, meta_configDYNBays, meta_configDYNBaysF1, meta_config_fata, meta_config_cov
from script.nn.bayesian.bays_dynamic import bayesian_optimization
from script.visualization.graph_generator.training.training_comparison import visualize_oversampling_comparison

if __name__ == '__main__':
    """
    generating the primary comparisons
    Note: this does not reflect the entire script.
    Find in visualization/graph_generator additional important methods 
    """

    # Comparison of the 8 synthetic datasets between NONE, SMOTE, ADASYN & DYNAMIC
    # with 4 repetitions on 4 models per dataset configuration
    visualize_oversampling_comparison(meta_config, 4, 4)

    # Bayesian optimization for the DYNAMIC oversampling algorithm
    bayesian_optimization()

    # Comparison of the 8 synthetic datasets between NON-OPTIMISED, ACCURACY-OPTIMISED & F1-OPTIMISED
    # with 4 repetitions on 4 models per dataset configuration
    visualize_oversampling_comparison(meta_configDYNBays, 4, 4)

    # Comparison of the 8 synthetic datasets between NONE, SMOTE, ADASYN & DYNAMIC F1-OPTIMISED
    # with 4 repetitions on 4 models per dataset configuration
    visualize_oversampling_comparison(meta_configDYNBaysF1, 4, 4)

    # Comparison of the real-life datasets between NONE, SMOTE, ADASYN & DYNAMIC
    # with 4 repetitions per dataset
    visualize_oversampling_comparison(meta_config_fata, 4, 1)
    visualize_oversampling_comparison(meta_config_cov, 4, 1)


