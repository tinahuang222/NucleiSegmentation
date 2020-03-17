import os
import numpy as np
from PIL import Image
import histomicstk as htk
from pathlib import Path
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import nucleus
import tensorflow as tf # 1.14.0
import os
import skimage
import pandas as pd
import numba
import keras # 2.1.0
import numpy as np
import requests
from tqdm import tqdm

"""
Generate HE image
Author:Samir
"""
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def load_model(model_weight_path):
    config = nucleus.NucleusInferenceConfig()
    config.display()
    device = "/cpu:0"

    model = None
    with tf.device(device):
        model = modellib.MaskRCNN(mode="inference", model_dir=os.getcwd(), config=config)

    model.load_weights(model_weight_path, by_name=True)
    return model

def gen_nuclear_masks_from_tile(
        nmodel: modellib.MaskRCNN,
        tile_file: str,
) -> pd.DataFrame:
    """ Runs MaskRCNN model on tile, returns data frame of true mask coordinates

    Arguments:
        nmodel {modellib.MaskRCNN} -- MaskRCNN model with weights loaded
        tile_file {str} -- path to tile image

    Returns:
        pd.DataFrame -- Data frame with tile_id, mask_id, and x,y coordinates of
            true values from mask
    """
    tile_id = tile_file.name

    tile = skimage.io.imread(tile_file)
    scaled_tile = skimage.transform.rescale(tile, 2, multichannel=True) * 255
    patch_size = scaled_tile.shape[0]
    results = nmodel.detect([scaled_tile], verbose=0)
    r = results[0]
    individual_nuclei = r['masks']
    mask_dfs = []

    for i in range(individual_nuclei.shape[2]):
        descale_mask = skimage.transform.rescale(individual_nuclei[:, :, i], 0.5)
        y, x = np.where(descale_mask)
        y_mean, x_mean = np.mean((y, x), axis=1)
        mask_dict = {
            'tile': tile_id,
            'mask_id': i,
            'x': x,
            'y': y,
        }
        mask_dfs.append(pd.DataFrame.from_dict(mask_dict))

    return pd.concat(mask_dfs)


"""
Generate HE image
Author:Keane
"""
def generate_he_image(image_path):
    im = Image.open(image_path)
    im_arr = np.array(im)
    # create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',  # cytoplasm stain
              'null']  # set to null if input contains only two stains

    W = np.array([stain_color_map[st] for st in stains]).T

    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(im_arr, W)

    deconvolved_image_h = imDeconvolved.Stains[:, :, 0]
    deconvolved_image_e = imDeconvolved.Stains[:, :, 1]

    return deconvolved_image_h, deconvolved_image_e, im_arr

"""
Generate HE image
Author:Samir
"""
def generate_centroid(mask_path):
    masks_csv = pd.read_csv(mask_path, header=0)

    masks_csv.drop(columns=['tile'], axis=1, inplace=True)
    mask_nums = masks_csv.mask_id.unique()
    mask_groups = masks_csv.groupby(['mask_id'])

    x = []
    y = []
    for mask in mask_nums:
        x.append(mask_groups.get_group(mask).x.mean())
        y.append(mask_groups.get_group(mask).y.mean())

    df = pd.DataFrame(columns=['x', 'y'])
    df['x'] = np.array(x)
    df['y'] = np.array(y)
    return df

