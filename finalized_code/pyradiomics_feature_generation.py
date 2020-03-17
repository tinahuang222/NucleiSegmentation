from pathlib import Path
from radiomics import glcm, imageoperations, shape, glrlm, glszm, getTestCase
from radiomics import featureextractor
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
import h5py
from tqdm import trange, tqdm
import radiomics
import logging

radiomics.logger.setLevel(logging.CRITICAL)

"""
Generate Pyradiomics feature
Author:Tina
"""

# =====================================
# Global error constants
SITKERROR = 999
MASKERROR = 990
SAVEERROR = 980
EXTRACTERROR = 970
CONCATERROR = 960


# ======================================

# this function will reconstruct the mask with size of 512x512 from an array of coordinates [(r,c)]
#
def reconstruct_mask(coord, row=512, col=512):
    # list of row coordinate, list of col coordinates
    coord = np.array(coord)
    rc = coord[:, 0]
    cc = coord[:, 1]

    mask = np.zeros((row, col))
    mask[rc, cc] = 1
    np.ma.make_mask(mask)
    return mask


def extract_features_from_mask_helper(img, mk):
    image = sitk.GetImageFromArray(img)
    mask = sitk.GetImageFromArray(mk)

    # by default, it's 3d; so we need to change it
    extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)

    # Enables all feature classes
    extractor.enableAllFeatures()
    featureVector = extractor.execute(image, mask)

    # remove unnecessary parameter settings

    keys = list(featureVector.keys())
    for k in keys[:10]:
        del featureVector[k]
    return featureVector


def extract_features_from_masks(image, mask_path):
    # read in masks
    masks_csv = pd.read_csv(mask_path, header=0)
    masks_csv.drop(columns=['tile'], axis=1, inplace=True)
    mask_nums = masks_csv.mask_id.unique()
    mask_groups = masks_csv.groupby(['mask_id'])

    # acc will accumulate all the mask information
    acc = pd.DataFrame()
    dim_error = []
    other_error = []
    for mask in mask_nums:
        row = mask_groups.get_group(mask).x
        col = mask_groups.get_group(mask).y
        coord = list(zip(row, col))

        # reconstruct mask from coordinates
        reconstructed_mask = reconstruct_mask(coord)

        # TASK -> READ THE IMAGE
        try:
            featureVector = extract_features_from_mask_helper(image, reconstructed_mask)

            values = np.array(list(featureVector.values())).reshape((1, -1))
            header = featureVector.keys()

            # since some mask has 1D array, thus we need to neglect those
            df1 = pd.DataFrame(values, index=[mask], columns=header)

            # accumulate the mask with previous one
            acc = pd.concat([acc, df1])
        except:
            # 1D mask
            pass
    return acc

"""
Generate Histogram of Pyradiomics feature
Author:Tina
"""
def mean_extract_helper(tile):
    # clean data by removing the object_type columns
    tile = remove_categorical(tile)

    # run through all means
    mean_tile = pd.DataFrame(tile.mean(), columns=['tile_name']).transpose()
    total_nuclei = tile.count()[0]
    mean_tile['nuclei_counts'] = total_nuclei
    return mean_tile


def extract_histogram_features_helper(file_name, tile):
    description = tile.describe(percentiles=[.05, 0.50, 0.95])
    # cell_count
    cell_count = tile.index.nunique()

    # dropping unwanted features
    unwanted_labels = ['count']

    description.drop(labels=unwanted_labels, axis=0, inplace=True)

    # for each features, generate histogram features
    description = description.unstack(level=-1).to_frame().transpose()
    description.set_axis([f"{x + '_'}{y}" for x, y in description.columns], axis=1, inplace=True)
    description.index = [file_name]
    description['cell_count'] = cell_count
    return description


def remove_categorical(tile):
    # remove the object_type columns
    types = tile.dtypes
    obj_types_bool = list(map(lambda col: np.object == types[col], tile.columns))
    result = list(tile.columns[obj_types_bool])

    # remove categorical features in the tile, and preserving the numerical only
    tile.drop(columns=result, inplace=True)
    return tile


def features_aggregator(file_path):
    # read_data
    tile = pd.read_csv(file_path, index_col=0)

    # remove_categorical variables/ clean data
    tile = remove_categorical(tile)

    return extract_histogram_features_helper(file_path.name.split('.')[0], tile)

