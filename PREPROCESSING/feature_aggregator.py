import os
import pandas as pd
import numpy as np
import sys

#this function checks if there's any missing value in the columns
#given matrix in pd data structure form.
def checkMissing (matrix):
    cols = tile.columns[matrix.isna().any()].tolist()
    print(cols)
   # if missing > 0:
    #    nafile.append(file_name)
    
def mean_extract_helper(tile):
    try:
        #clean data by removing the object_type columns 
        tile = remove_categorical(tile)
        
        #run through all means
        mean_tile = pd.DataFrame(tile.mean(), columns =[file_name]).transpose()
        total_nuclei = tile.count()[0]
        mean_tile['nuclei_counts'] = total_nuclei
        return mean_tile
    except:
        print("error")
        sys.exit(-1)
        
def extract_histogram_features_helper(file_name , tile):
    description=tile.describe(percentiles=[.05,0.50,0.95])
    #cell_count 
    cell_count = tile.index.nunique()
    
    #dropping unwanted features
    unwanted_labels = ['count']
    description.drop(labels = unwanted_labels,axis = 0, inplace = True)

    #for each features, generate histogram features  
    description= description.unstack(level=-1).to_frame().transpose()
    description.set_axis([f"{x+'_'}{y}" for x, y in description.columns], axis=1, inplace=True)
    description.index = [file_name]
    description['cell_count'] = cell_count
    return description

def remove_categorical(tile):
    #remove the object_type columns
    types = tile.dtypes
    obj_types_bool = list(map(lambda col: np.object == types[col], tile.columns))
    result = list(tile.columns[obj_types_bool])

    #remove categorical features in the tile, and preserving the numerical only 
    tile.drop(columns = result, inplace = True)
    return tile
""" 
def features_agg(save_mask_dir, mask_names, ext = '.csv'):
    acc = pd.DataFrame()
    for file_name in mask_names:
        file_path = os.path.join(save_mask_dir, file_name+ext) 
        try:
            #read_data
            tile = pd.read_csv(file_path, index_col = 'mask_num') 
            
            #remove_categorical variables/ clean data
            tile = remove_categorical(tile)
        except:
            print("error_in_file_reading: ", file_path)
            sys.exit(-1)
        
        acc = pd.concat([acc, extract_histogram_features_helper(file_name, tile)])
        acc.index.name = 'tile_name'
    return acc
"""