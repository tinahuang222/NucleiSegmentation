import os
import pandas as pd
import numpy as np

#the lookuptable has 3 arrays
#lookuptable[0]: dict of gray images with blue channels with key=index_position, value=img
#lookuptable[1]: dict of gray images with red channels with key=index_position, value=img
#lookuptable[2]: list of tile names corresponding
#from filename, find the first index that matches filename in normalized_img[2] 
#lookup img-data in normalized_img[0] using the index as a key

def lookup (lookuptable, filename):
    if len(lookuptable) < 3:
        print("The ", normal, " has a wrong array size.")
        exit(-1) 
    
    index = lookuptable[2].index(filename)
    if index == -1:
        print("Error in locating ", filename)
        exit(-2) 
    
    #by default interested in the blue column, which is lookuptable col[0]
    return lookuptable[0].get(index)