import os
import pandas as pd
import numpy as np
import sys
import helper_functions as hf


__all__ = ['features_aggregator_main']

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

#features_agg should requre 3 input parameters:
# save_mask_dir = directory of where the mask_features are stored
# save_to_dir = directory of where the output should be stored
# save_file_name = output filename that will lists aggregated features for all tiles. It will be in .csv format.
# This function has a void output, but it will outputs aggregated_features_ for allmasks in save_file_name.mask_ext.
#
def features_aggregator_main(save_mask_dir, save_to_dir, save_file_name, mask_ext = '.csv'):
    
    #check whether the save_mask_dir exists
    if os.path.exists(save_mask_dir) == False:
        print(save_mask_dir, " does not exists. Please double check!")
        sys.exit(-1)
        
    #check whether the save_to_dir exists. If not, create one.    
    if os.path.exists(save_to_dir) == False:
        print("creating directory: ", save_to_dir)
        os.mkdir(save_to_dir)
    
    
    base_files = os.listdir(save_mask_dir)
    #get all the filenames with .csv extensions only
    # mask_names = list of mask_features's names (excluding '.csv')
    mask_names = list(map(hf.only_appropriate_file_name, base_files))
    mask_names = list(filter(None, mask_names)) 
    
    print("Features_aggregators detected ", len(mask_names), " files." )
    print("Proceeding in features_aggregating...")
    
    acc = pd.DataFrame()
    for file_name in mask_names:
        file_path = os.path.join(save_mask_dir, file_name+mask_ext) 
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
    save_aggr = save_file_name + mask_ext
    acc.to_csv(os.path.join(save_to_dir, save_aggr))
    print("Completed features_aggregating: ", save_aggr)
