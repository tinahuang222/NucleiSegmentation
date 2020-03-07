import matplotlib.pyplot as plt
from skimage import feature
from scipy import ndimage as ndi 
import pandas as pd
import numpy as np
import cv2 as cv
import SimpleITK as sitk
import skimage, imageio, importlib, pickle, time, os, statistics, sys,six

#This is a function that we will import from the current directory,
#make sure pickle_helper.py is under current working_directory
import pickle_helper
import nuclei_feature_extraction as extract

#try if mask has problems
error_file = []
sitkError_files = []
loadmaskError_files = []
saveError_files = []
extractError_files = []
concatError_files = []

#img_ext = '' #need to specify img_ext
#mask_ext = '.csv'
#save_dir= os.path.join(root_dir,'masks_features') #save to directory
def only_appropriate_file_name (x):
    if os.path.splitext(x)[1] == '.csv':
        return os.path.splitext(x)[0]
    
""" 
feature_generation takes in 3 parameters,
    1. he_image_pickle: pickle object from Keane's work
    2. mask_dir: containing masks for all tiles. This program will read all the .csv files within this directory
    3. save_dir: all the aggregated_features will be stored in this save_dir automatically with respect to their tile's names
    4. bug_dir: it will log all bugs into 'bug_file_logs' under bug_dir directory 
and it has a void output. Features will be automatically generated with the same tile_name as filename under save_dir.

"""
def feature_generation(he_image_pickle, mask_dir, save_dir, bug_dir, mask_ext = '.csv'):
    log_error = pd.DataFrame()
    he_image = he_image_pickle
    #Run this by default
    mask_files = os.listdir(mask_dir)
    
    #created save_dir if needed
    if os.path.exists(save_dir) == False:
        print("created_directory: ", save_dir)
        os.mkdir(save_dir)
        
    #get all the filenames with .csv extensions only
    mask_name = list(map(only_appropriate_file_name, mask_files))
    mask_name = list(filter(None, mask_name)) 
    
    for file in mask_name:
        print(file)
        #he_image is a matrix that represents the image for that mask
        #mask is just a file name, not necessarily a mask matrix 
        he_image = pickle_helper.lookup(normalized_img, file)   
        try:    
            dim_error_files, other_error_files = extract.extract_features_from_masks(he_image, masks_dir, mask_ext, file, save_dir)
            #print(dim_error_files, other_error_files)
            error_division = pd.DataFrame(data={'dim_error': np.NaN, 'other_error': np.NaN}, index = [file], dtype = object)
            if len(dim_error_files) > 0:
                error_division.at[file, 'dim_error'] = np.array(dim_error_files)
            if len(other_error_files) > 0:
                error_division.at[file, 'other_error'] = np.array(other_error_files)
            log_error = pd.concat([log_error, error_division])

        except BaseException as e:
            error_type = str(e)
            print(file, " has an error type of ", error_type)
            if error_type == SITKERROR:
                sitkError_files.append(file)
            elif error_type == MASKERROR:
                loadmaskError_files.append(file)
            elif error_type == SAVEERROR: 
                saveError_files.append(file)
            elif error_type == EXTRACTERROR: 
                extractError_files.append(file)
            elif error_type == CONCATERROR:
                concatError_files.append(file)
            else: 
                error_file.append(file)
    error_division = pd.DataFrame({'dim_error': dim_error_files, 'other_error': other_error_files}, index = [file])
    log_error = pd.concat([log_error, error_division])
    log_error.to_csv(os.path.join(bug_dir, 'bug_file_logs' + '.csv'))