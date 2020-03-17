from pathlib import Path
import os
import pickle
from finalized_code import classification as clf
import pandas as pd
"""
Train random forest classifier
Author:Minh
"""
data_path = Path('./data')
mix_feature_ranking_path = data_path / 'graph_feature' /  'mix_feature_ranking.csv'
mix_feature_path = data_path / 'graph_feature'/ 'mix_feature.csv'

model_dir = data_path / 'models'
os.makedirs(model_dir, exist_ok=True)
save_path = model_dir / 'rf.pkl'

feature_df = pd.read_csv(mix_feature_path, index_col='tile_name')
rf = clf.train_rf_model(feature_df, mix_feature_ranking_path, model_dir / 'rf.pkl')
pickle.dump(rf, open(save_path, 'wb'))

"""
Train graph convolution classifier
Author:Samir
"""
data_path = Path('./data')
graph_dir = data_path / 'graph_feature'
model_path = data_path / 'models' / 'graph_conv'
os.makedirs(model_path, exist_ok=True)

node_feature_dir = data_path / 'masks_features'
tile_label_path = data_path / 'tiles_rois' / 'dataset.csv'
coef_file = data_path / 'cell_feature' / 'feature_ranking.csv'

dataset = clf.create_data_set(node_feature_dir, tile_label_path, coef_file)
with open(graph_dir / 'graph_data.pkl', 'wb') as fp:
    pickle.dump(dataset, fp)

dataset = pickle.load(open(graph_dir / 'graph_data.pkl', 'rb'))

model= clf.train_graph_conv(dataset, save_path=model_path / 'graph_conv.pth')
