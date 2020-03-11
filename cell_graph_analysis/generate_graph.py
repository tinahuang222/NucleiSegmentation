""" Functions used to create input data sets for pytorch geometric

Author: Samir Akre
"""
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from typing import List
import ast
import os
from tqdm import tqdm


def create_data_set(
    node_feature_folder: str,
    tile_label_file_path: str,
    regression_coefficient_file: str,
    n_features: int = 64,
    max_neighbours: int = 8,
    radius: float = 50
) -> List[Data]:
    """ Creates data set for input to model

    Arguments:
        node_feature_folder {str} -- Path to folder containing csv's of
            nuclear features files in folder expected to have file name
            "[tile_id].csv"
        tile_label_file_path {str} -- Path to csv labelling the 
            Cancer/no-cancer for tiles
        regression_coefficient_file {str} -- Path to file containing
            regression coefficients for feature selection

    Keyword Arguments:
        n_features {int} -- Number of node features to extract (default: {64})
        max_neighbours {int} --  max neighbors to use in graph generation
            (default: {8})
        radius {float} -- Max pixel radius for edge generation in graph
            construction (default: {50})

    Returns:
        List[Data] -- List pytorch input graphs, 1 item per tile
    """
    data_sets = []
    labels = pd.read_csv(tile_label_file_path)
    feature_files = os.listdir(node_feature_folder)
    feature_files = [f for f in feature_files if f[0] != '.']
    feature_cols = None

    for feature_file_path in tqdm(feature_files):
        tile_id = feature_file_path.split('/')[-1].split('.')[0]
        label = labels[labels.tile_name == tile_id].label.values[0]

        feature_path = os.path.join(node_feature_folder, feature_file_path)
        if feature_cols is None:
            feature_cols = select_node_feature_columns(
                feature_path,
                regression_coefficient_file,
                n_features,
            )

        data = gen_graph(
            feature_path,
            feature_cols,
            label,
            max_neighbours,
            radius,
        )
        data_sets.append(data)

    return data_sets


def gen_graph(
    node_feature_path: str,
    node_feature_cols: List[str],
    label: int,
    max_neighbours: int = 8,
    radius: float = 50,
) -> Data:
    """ Generates graph from node features as pytorch object

    Arguments:
        node_feature_path {str} -- Path to node features for tile
        node_feature_cols {List[str]} -- list of features to use
        label {int} -- 1, or 0 for cancer/no-cancer

    Keyword Arguments:
        max_neighbours {int} -- k-max neighbors for graph edge (default: {8})
        radius {float} -- max pixel radius for graph edge (default: {50})

    Returns:
        Data object for use with pytorch geometric graph convolution
    """
    features = pd.read_csv(
        node_feature_path,
        converters={
            "diagnostics_Mask-original_CenterOfMass": ast.literal_eval
        }
    )

    # Normalize Features
    f_df = features[node_feature_cols]
    f_norm_df = (f_df - f_df.mean()) / (f_df.max() - f_df.min())
    f_norm_nafilled = f_norm_df.fillna(0)

    # Set centroid coordinates from node features
    coordinates = torch.tensor(
        features['diagnostics_Mask-original_CenterOfMass'].tolist()
    )

    # Set node features from normalized features
    node_features = torch.tensor(
        f_norm_nafilled.astype(float).values,
        dtype=torch.float32
    )

    # Initialize data for pytorch
    y = torch.tensor([label], dtype=torch.long)
    data = Data(
        x=node_features,
        pos=coordinates,
        y=y,
        num_classes=2
    )

    # Create Graph
    data.edge_index = radius_graph(
        data.pos,
        radius,
        None,
        True,
        max_neighbours
    )

    return data


def select_node_feature_columns(
        nuclear_feature_file: str,
        regression_coefficient_file: str,
        n: int
) -> List:
    """ Finds the top set of features from linear regression
    to use as nucleus features

    Arguments:
        nuclear_feature_file {str} -- Any nuclear feature file
            (Tina's csv output from pyradiomics)
        regression_coefficient_file {str} -- csv with tile level
            aggregated features ranked by coefficient in linear regression
        n {int} - Number of features

    Returns:
        List -- all features ranked by regression coefficients
    """
    features = pd.read_csv(nuclear_feature_file)
    feat_rank = pd.read_csv(regression_coefficient_file)
    feat_rank = feat_rank.rename(
        columns={feat_rank.columns[0]: 'feature'}
    )

    # Removes the 'aggregation' suffix
    feat_rank['base_feat'] = feat_rank['feature']\
        .str.split('_').str[:-1].str.join('_')

    # Find top features based on coefficient
    nuc_feat = pd.DataFrame(
        feat_rank.groupby('base_feat').abs_coef.max()
    ).reset_index()
    variable_cols = [
        col for col in features.columns
        if ('diagnostics_Image-original' not in col)
        and (col != 'diagnostics_Mask-original_VolumeNum')
    ]
    nuc_feat = nuc_feat[nuc_feat.base_feat.isin(variable_cols)]
    return nuc_feat.base_feat.tolist()[:n]
