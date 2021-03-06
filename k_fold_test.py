from pathlib import Path
import os
from finalized_code import classification as clf
import pandas as pd
import pickle
"""
K_fold test with random forest
# Author:Minh
# """
data_path = Path('./data') / 'train'
mix_feature_ranking_path = data_path / 'graph_feature' /  'mix_feature_ranking.csv'
mix_feature_path = data_path / 'graph_feature'/ 'mix_feature.csv'
#
# pred_results_dir = data_path / 'results'
# os.makedirs(pred_results_dir, exist_ok=True)
#
# feature_df = pd.read_csv(mix_feature_path, index_col='tile_name')
# clf.k_fold_test(feature_df, mix_feature_ranking_path, fold=10, top_feature=20)

"""
K_fold test with graph convolution network
Author: Samir
"""

graph_dir = data_path / 'graph_feature'
node_feature_dir = data_path / 'masks_features'
tile_label_path = data_path / 'tiles_rois' / 'dataset.csv'
coef_file = data_path / 'cell_feature' / 'feature_ranking.csv'

dataset = clf.create_data_set(node_feature_dir, tile_label_path, coef_file)
with open(graph_dir / 'graph_data.pkl', 'wb') as fp:
    pickle.dump(dataset, fp)

dataset = pickle.load(open(graph_dir / 'graph_data.pkl', 'rb'))
clf.k_fold_test_graph_conv(dataset)