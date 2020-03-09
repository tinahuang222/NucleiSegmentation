import matplotlib.pyplot as plt
from skimage import feature
from scipy import ndimage as ndi 
import os
import pandas as pd
import numpy as np
import time
import statistics
import cv2 as cv
import skimage 
import imageio
from radiomics import glcm, imageoperations, shape, glrlm, glszm, getTestCase
from radiomics import featureextractor
import SimpleITK as sitk
import warnings
import logging
import sys

#=====================================
#set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
#=====================================

#=====================================
#Global error constants
SITKERROR = 999
MASKERROR = 990
SAVEERROR = 980
EXTRACTERROR = 970
CONCATERROR = 960
#======================================


#this function will reconstruct the mask with size of 512x512 from an array of coordinates [(r,c)]
#
def reconstruct_mask (coord, row=512, col = 512):
	#list of row coordinate, list of col coordinates
	coord = np.array(coord)
	rc = coord[:,0]
	cc = coord[:,1]

	mask = np.zeros((row,col))
	mask[rc,cc] = 1
	np.ma.make_mask(mask)
	#print("mask being\n ",mask)
	return mask

#test cases 
def extract_features_from_mask_helper (img, mk):
#    image = sitk.ReadImage(imageName)
#    mask = sitk.ReadImage(maskName)
#not sure whether this has to be sitk object
    image =  sitk.GetImageFromArray(img)
    try:
        image =  sitk.GetImageFromArray(img)
        mask = sitk.GetImageFromArray(mk)
    except:    
        #raise SitkError("Failed to convert array into Sitk Object")
        sys.exit(SITKERROR)
    
    #error in extracting vectors?
    #try:
    extractor = featureextractor.RadiomicsFeatureExtractor(force2D = True) #by default, it's 3d; so we need to change it
    # Enables all feature classes
    extractor.enableAllFeatures()  
        # Enables all feature classes
        # Alternative: only first order
    #extractor.disableAllFeatures  # All features enabled by default
    #extractor.enableFeatureClassByName('firstorder')
    featureVector = extractor.execute(image, mask)
    #except:
    #    sys.exit(EXTRACTERROR)         
    #for (key,val) in six.iteritems(featureVector):
    #  print("\t%s: %s" % (key, val))
    
    #remove unnecessary parameter settings
    
    keys = list(featureVector.keys())
    for k in keys[:10]:
        del featureVector[k]
    return featureVector

#this function should be able to handle a list of masks coordinates 
#and append all features onto a csv 
def extract_features_from_masks (image, maskDir, maskExt, file_name, saveDir):
    
    #because we changed the format of image into a matrix already, no need to read image from path
    """
    read in normalized_image(gray_scale)
    img_path = os.path.join(imgDir, filename+imageExt)
    try:
        image = imageio.imread(img_path)
    except:
        print("failed to load image: ", image)
    """    
    start = time.time()
    
    #read in masks
    mask_path = os.path.join(maskDir, file_name + maskExt)
    try:
        masks_csv = pd.read_csv(mask_path, header = 0)
    except:
        sys.exit(MASKERROR)

    masks_csv.drop(columns = ['tile'], axis=1, inplace = True)
    mask_nums = masks_csv.mask_id.unique()
    mask_groups = masks_csv.groupby(['mask_id'])
    
   
    #acc will accumulate all the mask information
    acc = pd.DataFrame() 
    dim_error = []
    other_error = []
    for mask in mask_nums:
        row = mask_groups.get_group(mask).x
        col = mask_groups.get_group(mask).y
        coord = list(zip(row,col))
        # print("coord: \n", coord)

        # reconstruct mask from coordinates 
        reconstructed_mask = reconstruct_mask(coord)

        # TASK -> READ THE IMAGE
        try:
            featureVector =  extract_features_from_mask_helper(image, reconstructed_mask)
       
            values = np.array(list(featureVector.values())).reshape((1,-1))
            header = featureVector.keys()
            
            #since some mask has 1D array, thus we need to neglect those 
            df1 = pd.DataFrame(values, index = [mask], columns = header)
            
            #accumulate the mask with previous one
            acc = pd.concat([acc,df1])
            
        except ValueError:
            dim_error.append(mask)
        except:
            other_error.append(mask)

    #changing the name for the index so every one represents an unique mask
    
    #save file to saveDir with mask_name the same
    store_file = os.path.join(saveDir, file_name + '.csv')
    try:
        acc.to_csv(store_file)
    except:
        #raise saveError("Failed to load mask")
        #print("Failed to save file at ", store_file)
        sys.exit(SAVEERROR)
    #append features to csv file 
    
    print("Finished extracting features for ", file_name, ". \nTotal time lapsed = ", time.time() - start )
    #print("dim_error: \n", dim_error)
    #print("other_error: \n", other_error)
    return (dim_error, other_error)
