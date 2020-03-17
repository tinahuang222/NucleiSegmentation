from finalized_code import pyradiomics_feature_generation as pfg
from finalized_code import fps_graph_feature_generation as fgfg
from pathlib import Path
import os
from tqdm import tqdm
import h5py
import pandas as pd
import numpy as np
"""
Generate Pyradomics feature
Author:Tina
"""
data_path = Path('./data')
mask_dir = data_path / 'mask'
mask_file_names = os.listdir(mask_dir)
img_arr_path = data_path / 'img_arr' / 'img_arr.h5'

cell_feature_dir = data_path / 'cell_feature'
os.makedirs(cell_feature_dir, exist_ok=True)

hema_cell_feature_dir = cell_feature_dir / 'hema'
os.makedirs(hema_cell_feature_dir, exist_ok=True)

eosin_cell_feature_dir = cell_feature_dir / 'eosin'
os.makedirs(eosin_cell_feature_dir, exist_ok=True)

rgb_cell_feature_dir = cell_feature_dir / 'rgb'
os.makedirs(rgb_cell_feature_dir, exist_ok=True)

# Only extract hema feature
img_types = ['h']
with h5py.File(img_arr_path, 'r') as file:
    for i in tqdm(range(len(mask_file_names))):
        mask_path = mask_dir / mask_file_names[i]
        mask_name = mask_file_names[i].split('.')[0]
        for img_type in img_types:
            arr = file['{}/{}'.format(mask_name, img_type)]
            df = pfg.extract_features_from_masks(arr, mask_path)
            df.index.name = 'mask_num'

            if img_type == 'h':
                df.to_csv(hema_cell_feature_dir / mask_file_names[i])
            elif img_type == 'e':
                df.to_csv(eosin_cell_feature_dir / mask_file_names[i])
            else:
                df.to_csv(rgb_cell_feature_dir / mask_file_names[i])

"""
Generate Histogram of Pyradiomics feature
Author:Tina
"""
data_path = Path('./data')
cell_feature_dir = data_path / 'cell_feature'
hema_cell_feature_dir = cell_feature_dir / 'hema'
hema_cell_feature_file_names = os.listdir(hema_cell_feature_dir)

acc = pd.DataFrame()

for i in tqdm(range(len(hema_cell_feature_file_names))):
    df = pfg.features_aggregator(hema_cell_feature_dir / hema_cell_feature_file_names[i])
    acc = pd.concat([acc, df])

acc.index.name = 'tile_name'
acc.to_csv(cell_feature_dir / 'cell_features_aggr.csv')

"""
Generate Furthest point sampling graph feature
Author:Minh
"""
data_path = Path('./data')
centroid_dir = data_path / 'centroid'
cellgraph_feature_dir = data_path / 'graph_feature'
os.makedirs(cellgraph_feature_dir, exist_ok=True)
lap_matrix_dir = cellgraph_feature_dir / 'lap_matrix'
os.makedirs(lap_matrix_dir, exist_ok=True)

centroid_names = os.listdir(centroid_dir)
features = []
feature_name = None
for i in tqdm(range(len(centroid_names))):
    name = centroid_names[i].split('.')[0]
    data, feature_name = fgfg.generate_cell_graph_statistic(centroid_dir / centroid_names[i], lap_matrix_dir)
    features.append([name] + data)

features = np.array(features)

col_names = ['tile_name'] + feature_name
df = pd.DataFrame(columns=col_names)

for i in tqdm(range(0, len(col_names))):
    df[col_names[i]] = features[:, i]

df.to_csv(cellgraph_feature_dir / 'graph_feature.csv', index=False)

