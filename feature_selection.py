from finalized_code import feature_selection as fs
from pathlib import Path
import pandas as pd

"""
Feature selection for both graph feature and cell feature
using random forest
Author:Minh
"""
data_path = Path('./data')
cellgraph_feature_path = data_path / 'graph_feature' / 'graph_feature.csv'
aggr_cell_feature_path = data_path / 'cell_feature'/ 'cell_features_aggr.csv'
label_path = data_path / 'tiles_rois' / 'dataset.csv'
mix_feature_ranking_path = data_path / 'graph_feature' / 'mix_feature_ranking.csv'
mix_feature_path = data_path / 'graph_feature'/ 'mix_feature.csv'

label_df = pd.read_csv(label_path, index_col='tile_name')
label_df = label_df.drop(columns=['slide_id'])

aggr_cell_feature_df = pd.read_csv(aggr_cell_feature_path, index_col='tile_name')
cell_graph_feature_df = pd.read_csv(cellgraph_feature_path, index_col='tile_name')

df = label_df.merge(cell_graph_feature_df, on='tile_name')
df = df.merge(aggr_cell_feature_df, on='tile_name')
df.to_csv(mix_feature_path)
print('Finish load data')

sorted_features = fs.evaluate_features_rf(df)
print('Finish evaluating feature')
ranking_df = pd.DataFrame(columns=['name', 'coef'])
ranking_df['name'] = sorted_features[0]
ranking_df['coef'] = sorted_features[1]
ranking_df.to_csv(mix_feature_ranking_path, index=False)
print('Save feature ranking to', mix_feature_ranking_path)

"""
Feature selection for cell feature using logistic regression
Author:Tina
"""
data_path = Path('./data')
aggr_path = data_path / 'cell_feature' / 'cell_features_aggr.csv'
save_dir = data_path / 'cell_feature'
fs.get_logreg_coef(aggr_path, label_path, save_dir)
