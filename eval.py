from pathlib import Path
from finalized_code import classification as clf
import pickle
import os
import pandas as pd
import numpy as np

data_path = Path('./data') / 'test'
result_path = Path('./data') / 'results'
os.makedirs(result_path, exist_ok=True)
"""
Predict result with random forest
"""

mix_feature_ranking_path = data_path / 'graph_feature' /  'mix_feature_ranking.csv'
mix_feature_path = data_path / 'graph_feature'/ 'mix_feature.csv'
model_dir = Path('./data') / 'models'
model_path = model_dir / 'rf.pkl'

pred_results_dir = data_path / 'results'
os.makedirs(pred_results_dir, exist_ok=True)

feature_df = pd.read_csv(mix_feature_path, index_col='tile_name')
model = pickle.load(open(model_path, 'rb'))
rf_result = clf.predict_from_model(model, feature_df, mix_feature_ranking_path)

df = pd.DataFrame()
df['tile_name'] = rf_result[1]
df['y_pred'] = rf_result[0]
df.to_csv(result_path / 'rf_results.csv', index=False)

"""
Predict result with graph conv model
"""
graph_dir = data_path / 'graph_feature'

dataset = pickle.load(open(graph_dir / 'graph_data.pkl', 'rb'))

model_path = Path('./data') / 'models' / 'graph_conv' / 'graph_conv.pth'
gcn_result = clf.predict_graph_conv(dataset, model_path=model_path)

df = pd.DataFrame()
df['tile_name'] = gcn_result[1]
df['y_pred'] = gcn_result[0]
df.to_csv(result_path / 'graph_conv_results.csv', index=False)

"""
Predict result with weight_avg
"""

weights = [0.82, 0.18]
rf_result = sorted(rf_result, key=lambda x:x[1])
gcn_result = sorted(gcn_result, key=lambda  x:x[1])

tile_name = np.array(rf_result[1])
avg_result = weights[0] * np.array(rf_result[0]) + weights[1] * np.array(gcn_result[0])

df = pd.DataFrame()
df['tile_name'] = tile_name
df['y_pred'] = avg_result
df.to_csv(result_path / 'weight_avg_results.csv', index=False)



