#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Mar 12 16:16:56 2020
ColorDeconv.ColorDeconv( '/home/kgonzalez/BE223B_2020/TEST_TILES/normalized/','/home/kgonzalez/BE223B_2020/TEST_TILES/HE_OUTPUT_IMAGES/')

@author: keane
'''

def ColorDeconv(input_image_dir='/home/kgonzalez/BE223B_2020',output_image_dir = '/home/kgonzalez/BE223B_2020'):
   # sys.path.append('ColorDeconv')
    import os
    import sys
    import random
    import math
    import re
    import time
    import numpy as np
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image  
    
    
    original_image_folder = input_image_dir 
    outputs_dir = output_image_dir
     
    #
    #pickle file used to store full image sets, which take more than a minute to run
    #
    he_pickle_file = os.path.join(output_image_dir,'HE_IMAGES_PICKLE.pck')

    print('Starting Color Deconvolution')
    resized_images,number_resized_images,image_files = get_image_files(original_image_folder)
    deconvolved_image_h, deconvolved_image_e = deconvolution(resized_images, number_resized_images,image_files,he_pickle_file)
    print('HE Images created')
    return deconvolved_image_h, deconvolved_image_e, image_files


def get_image_files(original_image_folder,NX=512,NY=512):
    '''
    Get image files and load them into memory
    '''
    import os
    from PIL import Image
    import matplotlib
    import matplotlib.pyplot as plt
    import pickle
    
    print('original image folder = ', original_image_folder)
    
    #get listing of image files in directory
    image_files = os.listdir(original_image_folder)
    
    image_names =[]
    #store filenames without extension for output purposes later
    for filename in image_files:
        root_ext = os.path.splitext(filename) #will return two parts, name and ext
        image_names.append(root_ext[0])
    
    #open the images
    image_data = {}
    for fcounter,filename in enumerate(image_files):
        full_filename = os.path.join(original_image_folder,filename)
    
        if ((fcounter % 100) == 0):
            print('Now at image # ', fcounter)
        
        image_data[fcounter] = Image.open(full_filename)
        image_data[fcounter] = image_data[fcounter].resize((NX, NY), Image.ANTIALIAS)
        
    
    return image_data,fcounter, image_files


def deconvolution(image_data,fcounter,image_files,he_pickle_file):
    '''
    Perform histomicsTK prep
    '''
    import numpy as np
    import histomicstk as htk
    import pickle
    
    
    
    #defaults for HE staining from HTK site
    # create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    print('stain_color_map:', stain_color_map, sep='\n')

    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
            'eosin',        # cytoplasm stain
            'null']         # set to null if input contains only two stains

    # create stain matrix
    W = np.array([stain_color_map[st] for st in stains]).T

    deconvolved_image_h={}
    deconvolved_image_e={}
    for ii,fcounter in enumerate(image_data):

        #current loop status
        if ((fcounter % 100) == 0):
            print('Now Deconvolving image # ', fcounter)
            
        imInput = np.array(image_data[fcounter]) #convert to NP array format first
        # perform standard color deconvolution
        imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, W)

        #store this output set in H and E dictionaries
        deconvolved_image_h[fcounter] = imDeconvolved.Stains[:,:,0]
        deconvolved_image_e[fcounter] = imDeconvolved.Stains[:,:,1]

    print('shape of imDeconvolved is ', np.shape(imDeconvolved))

    #
    # Save Pickle output of data
    #
    he_data = [deconvolved_image_h,deconvolved_image_e, image_files]
    print('pickle file is ', he_pickle_file)
    with open(he_pickle_file, 'wb') as f:
        pickle.dump(he_data, f)
    print('Saved full HE image set to pickle file')

    
    return deconvolved_image_h, deconvolved_image_e




