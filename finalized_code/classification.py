import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import ast
from typing import List
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import torch.nn.functional as F
import torch_geometric.nn as nn
from torch_geometric.nn import BatchNorm, SAGPooling, SAGEConv
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
import pickle

"""
Classification with random forest
Author:Minh
"""

# Training
def prepare_data(feature_df, feature_ranking_path, top_feature):
    rank_df = pd.read_csv(feature_ranking_path)
    top_feature_names = rank_df['name'][:top_feature]

    df = feature_df.filter(top_feature_names)
    X = df[top_feature_names].values

    return X


def train_rf_model(feature_df, feature_ranking_path, top_feature=15):
    X = prepare_data(feature_df, feature_ranking_path, top_feature)
    y = feature_df['label'].values
    rf = RandomForestRegressor(n_estimators=200,
                               max_depth=50,
                               min_samples_leaf=4,
                               min_samples_split=4,
                               bootstrap=True,
                               max_features=6,
                               n_jobs=-1)
    rf.fit(X, y)
    return rf

# predict
def predict_from_model(model, feature_df, feature_ranking_path, top_feature=15):
    X = prepare_data(feature_df, feature_ranking_path, top_feature)
    y_pred = model.predict(X)
    return y_pred

# k_fold test
def k_fold_test(feature_df, feature_ranking_path, fold=10, top_feature=15):
    X = prepare_data(feature_df, feature_ranking_path, top_feature)
    y = feature_df['label'].values

    y_pred_list = []
    y_test_list = []

    skf = StratifiedKFold(n_splits=fold, random_state=None, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf = RandomForestRegressor(n_estimators=200,
                                   max_depth=50,
                                   min_samples_leaf=4,
                                   min_samples_split=4,
                                   bootstrap=True,
                                   max_features=6,
                                   n_jobs=-1,
                                   verbose=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test.astype(float))

    y_test_all = np.concatenate(y_test_list)
    y_pred_all = np.concatenate(y_pred_list)

    fpr, tpr, _ = roc_curve(y_test_all, y_pred_all)
    print('Auc:', auc(fpr, tpr))
    plt.plot(fpr, tpr, label='RF')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.show()
    plt.clf()

"""
Classification with graph convolution network
Author:Samir
"""

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

        feature_path = node_feature_folder / feature_file_path
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
    return nuc_feat.base_feat.drop_duplicates().tolist()[:n]

class Net(torch.nn.Module):
    def __init__(self, in_feats):
        super(Net, self).__init__()

        hs_1 = in_feats * 2
        self.conv1 = SAGEConv(in_feats, hs_1)
        self.bn1 = BatchNorm(hs_1)
        self.pool1 = SAGPooling(hs_1, ratio=0.5)

        hs_2 = int(hs_1 * 2)
        self.conv2 = SAGEConv(hs_1, hs_2)
        self.bn2 = BatchNorm(hs_2)
        self.pool2 = SAGPooling(hs_2, ratio=0.5)

        num_classes = 2
        self.lin1 = Linear(hs_2, num_classes).cuda()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, perm, score = self.pool1(
          x, edge_index, batch=data.batch
        )

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, perm, score = self.pool2(
          x, edge_index, batch=batch
        )

        x = nn.global_mean_pool(x, batch)
        x = F.relu(x)
        x = self.lin1(x)

        return F.softmax(x, dim=1)

# train
def train_graph_conv(dataset, save_path=None, epochs=500, batch_size=128, lr=0.0001, weight_decay=5e-4,
                     top_feature=64, device_type='cuda'):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    device = torch.device(device_type)

    loss_function = torch.nn.CrossEntropyLoss()
    model = Net(top_feature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    for epoch in tqdm(range(epochs), desc='epochs'):
        for batch in train_loader:
            data = batch.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_function(out, data.y)
            loss.backward()
            optimizer.step()
            if torch.isnan(loss):
                break

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return model

# predict
def predict_graph_conv(dataset, model=None, model_path=None, top_feature=64, batch_size=1, device_type='cuda'):
    test_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=False)
    device = torch.device(device_type)

    if model_path is not None:
        model = Net(top_feature).to(device)
        model.load_state_dict(torch.load(model_path))
    model.eval()
    result = []
    for data in test_loader:
        data = data.to(device)
        result.append(model(data)[0][1].item())

    return result

# k_fold_test
def k_fold_test_graph_conv(dataset, fold=10, epochs=500, batch_size=128, lr=0.0001, weight_decay=5e-4,
                           top_feature=64, device_type='cuda'):
    labels = [x.y.item() for x in dataset]
    y_pred = []
    y_true = []

    skf = StratifiedKFold(n_splits=fold, random_state=None, shuffle=True)

    for train_index, test_index in tqdm(skf.split(dataset, labels), total=fold):
        X_train = [dataset[i] for i in train_index]
        X_test = [dataset[i] for i in test_index]

        y_test = [x.y.item() for x in X_test]

        model = train_graph_conv(X_train, None, epochs, batch_size, lr, weight_decay,
                                 top_feature, device_type)
        result = predict_graph_conv(X_test, model=model)

        y_pred += result
        y_true += y_test

    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    target_names = ['no_cancer', 'cancer']

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    print('Auc:', auc(fpr, tpr))
    plt.plot(fpr, tpr, label='RF')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.show()
    plt.clf()