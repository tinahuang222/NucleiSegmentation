from finalized_code import pyradiomics_feature_generation as pfg
from finalized_code import fps_graph_feature_generation as fgfg
from pathlib import Path
import os
from tqdm import tqdm
from p_tqdm import p_map
import h5py
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool
NUM_OF_WORKERS = cpu_count() - 1  # Number of processors to use, keep one processor free for other work

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
Generate Pyradomics feature
Author:Tina
"""
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

print('Load mask data ...')
img_type = 'h'
data = []
with h5py.File(img_arr_path, 'r') as file:
    for i in range(len(mask_file_names)):
        mask_path = mask_dir / mask_file_names[i]
        mask_name = mask_path.name.split('.')[0]
        arr = file['{}/{}'.format(mask_name, img_type)][:]
        data.append([mask_path, arr])

def process_single_file(mask_data):
    mask_path = mask_data[0]
    arr = mask_data[1]
    df = pfg.extract_features_from_masks(arr, mask_path)
    df.index.name = 'mask_num'

    if img_type == 'h':
        df.to_csv(hema_cell_feature_dir / mask_path.name)
    elif img_type == 'e':
        df.to_csv(eosin_cell_feature_dir / mask_path.name)
    else:
        df.to_csv(rgb_cell_feature_dir / mask_path.name)

# Only extract hema feature
print('Generating radiomic feature with {} workers ...'.format(NUM_OF_WORKERS))
p_map(process_single_file, data, num_cpus=NUM_OF_WORKERS)
print('Finish generating radiomic feature')

"""
Generate Histogram of Pyradiomics feature
Author:Tina
"""
cell_feature_dir = data_path / 'cell_feature'
hema_cell_feature_dir = cell_feature_dir / 'hema'
hema_cell_feature_file_names = os.listdir(hema_cell_feature_dir)

acc = pd.DataFrame()

print('Aggregating radiomic feature ...')

for i in tqdm(range(len(hema_cell_feature_file_names))):
    df = pfg.features_aggregator(hema_cell_feature_dir / hema_cell_feature_file_names[i])
    acc = pd.concat([acc, df])

acc.index.name = 'tile_name'
acc.to_csv(cell_feature_dir / 'cell_features_aggr.csv')
print('Finsh aggregating radiomic feature')
print('Save to', cell_feature_dir/ 'cell_features_aggr.csv')

"""
Generate Furthest point sampling graph feature
Author:Minh
"""
centroid_dir = data_path / 'centroid'
cellgraph_feature_dir = data_path / 'graph_feature'
os.makedirs(cellgraph_feature_dir, exist_ok=True)
lap_matrix_dir = cellgraph_feature_dir / 'lap_matrix'
os.makedirs(lap_matrix_dir, exist_ok=True)

centroid_names = os.listdir(centroid_dir)
features = []
feature_name = None

print('Generating graph feature ...')

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
print('Finish generating graph feature')
print('Save to', cellgraph_feature_dir / 'graph_feature.csv')

