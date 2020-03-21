from finalized_code import basic_feature_generation


from pathlib import Path
import os
from tqdm import tqdm
import h5py

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
Generate HE image
Author:Samir
"""
tiles_path = data_path / 'tiles_rois' / 'original'
model_weight_path = str(Path('./mask_rcnn_weights.h5').resolve())

if not os.path.exists(model_weight_path):
    basic_feature_generation.download_file_from_google_drive('1pvySgKD53NLpDHtAJvoaj26SZWFIYIYR', Path('./mask_rcnn_weights.h5'))

mask_output_path = data_path / 'mask'
os.makedirs(mask_output_path, exist_ok=True)

file_names = os.listdir(tiles_path)
model = basic_feature_generation.load_model(model_weight_path)
for i in tqdm(range(len(file_names))):
    masks = basic_feature_generation.gen_nuclear_masks_from_tile(model, tiles_path / file_names[i])
    csv_name = file_names[i].split('.')[0] + '.csv'
    masks.to_csv(mask_output_path / csv_name, index=False)

"""
Generate HE image
Author:Keane
"""
img_arr_dir = data_path / 'img_arr'
file_names = os.listdir(tiles_path)
os.makedirs(img_arr_dir, exist_ok=True)
with h5py.File(img_arr_dir / 'img_arr.h5', 'w') as file:
    for i in tqdm(range(len(file_names))):
        h, e, rgb = basic_feature_generation.generate_he_image(tiles_path / file_names[i])
        name = file_names[i].split('.')[0]
        file.create_dataset('{}/h'.format(name), data=h)
        file.create_dataset('{}/e'.format(name), data=e)
        file.create_dataset('{}/rgb'.format(name), data=rgb)

"""
Generate Centroids
Author:Samir
"""
mask_dir = data_path / 'mask'
centroid_dir = data_path / 'centroid'
os.makedirs(centroid_dir, exist_ok=True)

mask_file_names = os.listdir(mask_dir)
for i in tqdm(range(len(mask_file_names))):
    mask_path = mask_dir / mask_file_names[i]
    mask_name = str(mask_path.name).split('.')[0]
    df = basic_feature_generation.generate_centroid(mask_path)
    df.to_csv(centroid_dir / '{}.csv'.format(mask_name), index=False)