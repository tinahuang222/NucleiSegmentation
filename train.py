from pathlib import Path
import os
import pickle
from finalized_code import classification as clf
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Process train or test data.')
parser.add_argument('--test', dest='test', default=False,
                   help='processing test data (False by default)')

args = parser.parse_args()
if args.test:
    data_path = Path('.') / 'data' / 'test'
else:
    data_path = Path('.') / 'data' / 'train'

"""
Train random forest classifier
Author:Minh
"""
mix_feature_ranking_path = data_path / 'graph_feature' / 'mix_feature_ranking.csv'
mix_feature_path = data_path / 'graph_feature'/ 'mix_feature.csv'

model_dir = Path('./data') / 'models' / 'rf'
os.makedirs(model_dir, exist_ok=True)
save_path = model_dir / 'rf.pkl'

feature_df = pd.read_csv(mix_feature_path, index_col='tile_name')
rf = clf.train_rf_model(feature_df, mix_feature_ranking_path, model_dir / 'rf.pkl')
pickle.dump(rf, open(save_path, 'wb'))

"""
Train graph convolution classifier
Author:Samir
"""
graph_dir = data_path / 'graph_feature'
model_dir = Path('./data') / 'models' / 'graph_conv'
os.makedirs(model_dir, exist_ok=True)

node_feature_dir = data_path / 'masks_features'
tile_label_path = data_path / 'tiles_rois' / 'dataset.csv'
coef_file = data_path / 'cell_feature' / 'feature_ranking.csv'

dataset = clf.create_data_set(node_feature_dir, tile_label_path, coef_file)
with open(graph_dir / 'graph_data.pkl', 'wb') as fp:
    pickle.dump(dataset, fp)

dataset = pickle.load(open(graph_dir / 'graph_data.pkl', 'rb'))

model= clf.train_graph_conv(dataset, save_path= model_dir/ 'graph_conv.pth')
