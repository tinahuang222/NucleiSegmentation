""" Generate and save masks from tile input folder.
Edit configuration file to run on your own device,
example file is 'generate_masks_config.json',

For python environment view 'requirements.txt'

Author: Samir Akre
"""
import os
import argparse
import numpy as np
import skimage
import mrcnn.model as modellib
import nucleus 
import tensorflow as tf
from scipy.io import savemat
from tqdm import tqdm
import pandas as pd
import json
from typing import Dict


def load_config(config_path: str) -> Dict:
    """ Opens configuration JSON as a dictionary
    
    Arguments:
        config_path {str} -- configuration JSON file path
    
    Returns:
        Dict -- dictionary of configuration variables
    """
    with open(config_path, 'r') as fp:
        return json.load(fp)



def config_model(
        weights_path: str,
        device = '/cpu:0'
    ) -> modellib.MaskRCNN:
    """ Initialize MaskRCNN model with pretrained weights
    
    Arguments:
        weights_path {str} -- path to h5 file containing model weights
    
    Keyword Arguments:
        device {str} -- where to run model (default: {'/cpu:0'})
    
    Returns:
        modellib.MaskRCNN -- pre-trained MaskRCNN model
    """
    config = nucleus.NucleusInferenceConfig()
    config.display()
    with tf.device(device):
        model = modellib.MaskRCNN(mode="inference",
            model_dir=os.getcwd(),
            config=config
        )
    model.load_weights(weights_path, by_name=True)  
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
    tile_id = tile_file.split('/')[-1].split('.')[0]
    
    tile = skimage.io.imread(tile_file)
    scaled_tile = skimage.transform.rescale(tile, 2, multichannel=True) * 255
    patch_size = scaled_tile.shape[0]
    results = nmodel.detect([scaled_tile], verbose=0)
    r = results[0]
    individual_nuclei = r['masks']
    mask_dfs = []

    for i in range(individual_nuclei.shape[2]):
        descale_mask = skimage.transform.rescale(individual_nuclei[:,:,i], 0.5)
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


def main(config_path: str) -> None:
    """ Generate Nuclear Masks for all tile images
    
    Arguments:
        config_path {str} -- Path to configuration JSON file
    """
    # Get file path variables
    config = load_config(config_path)
    tile_folder = config['tile_folder']
    masks_out_folder = config['masks_out_folder']
    weights_path = config['weights_path']

    if not os.path.isdir(masks_out_folder):
        print('Creating mask output directory:', masks_out_folder)
        os.mkdir(masks_out_folder)

    # Load MaskRCNN Model
    model = config_model(weights_path, config['device'])

    # Get all available tiles
    tile_files = os.listdir(tile_folder)
    tile_file_paths = [os.path.join(tile_folder, tf) for tf in tile_files if tf[0] != '.']

    # Save masks for each tile file
    for tile_file in tqdm(tile_file_paths):
        tile_id = tile_file.split('/')[-1].split('.')[0]
        out_path = os.path.join(
                masks_out_folder,
                tile_id + '.csv',
        )
        masks = gen_nuclear_masks_from_tile(model, tile_file)
        masks.to_csv(out_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to the json format configuration file')
    args = parser.parse_args()

    main(args.config)
        