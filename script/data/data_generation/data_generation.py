import random

from anytree import RenderTree

from script.configs.configs import nameDf, DataSetsConfig
from script.data.data_generation.DataGenerator import Generator
from script.data.data_generation.Taxonomy import Node


def generate_tree(target, n_instances_total, features, total_n_classes):
    """
    function to generate tree structure which is used to create synthetic datasets

    Parameters:
    target (int): Target number of nodes.
    n_instances_total (int): Total number of instances.
    features (int): Total number of features.
    total_n_classes (int): Total number of classes.

    Returns:
    Node: Root node of the generated tree.
    """
    node = Node(node_id="0-0-0", n_instances=n_instances_total, feature_set=features, n_classes=total_n_classes,
                classes=(0, total_n_classes))
    nodes = [[node]]
    num_children = 1

    while num_children < target:
        num_children = 0
        new_depth = []

        for parent_node in range(len(nodes[-1])):
            new_children = random.choice([1, 1, 2, 2, 2, 2, 3, 3])
            num_children += max(new_children, 1)

            for child_index in range(new_children):
                new_node_id = f"{len(nodes)}-{parent_node}-{child_index}"
                new_node = Node(node_id=new_node_id, parent=nodes[-1][parent_node])
                new_depth.append(new_node)

        nodes.append(new_depth)

    print(RenderTree(node))
    return node


def generateSaveData(config: DataSetsConfig, cr, path=""):
    """
    function to generate synthetic datasets and save them according its attributes

    Parameters:
        config (DataSetsConfig): Dataset configuration.
        cr (str): Run identifier.
        path (str, optional): Save path.

    Returns:
        DataFrame: Generated dataset.
    """
    print(config.leafs)
    root = generate_tree(config.leafs, config.n, config.f, config.c)

    generator = Generator(n=config.n, n_features=config.f, c=config.c, class_overlap=config.co,
                          sC=Generator.imbalance_degrees[config.sc], root=root,
                          random_state=int(random.random() * 1000))
    df = generator.generate_data_from_taxonomy()
    df_sliced = df.iloc[:, :config.f + 1]
    file_name = f"{path}datasets/synthetic/df_{nameDf(config, cr)}.csv"
    df_sliced.to_csv(file_name)

    return df_sliced
