""" TODO: Samir
1. calling function from nuclear_mask_generation (Samir)
this should output masks_dir, masks coordinates for all tiles
"""

""" TODO: Keane
2. color deconvolve from PREPROCESSING (Keane)
this should output pickle file 
"""
#input_image_dir is the full path to the raw TILE images (should all be in one folder, as it will
#   just get a listing of files in that folder
#output_image_dir is where the pickle file will be written out (used to contain debug images, but
#   those are not in this pared down version)
deconvolved_image_h, deconvolved_image_e,image_names = ColorDeconv(input_image_dir,output_image_dir)
#writes a pickle file with the returned items above into output_image_dir. The pickle filename
#   is 'HE_IMAGES_PICKLE.pck'
#   if the return values are not needed, remove them
#--deconvolved_image_h is an array containing the 512x512x1 Haema images
#--deconvolved_image_e is an array containing the 512x512x1 Eosin images
#--image_names is a dictionary of the original image names. The order of names here correspond
#    to the order in the h and e arrays (for cancer labeling purposes)




""" TODO: Tina
3. Nuclear_Feature_Generation(Tina)
feature_generation will take he_image_pickle(Keane), mask_dir(from Samir), and save_dir(output_dir) + bug_dir (will log the bug if occured). This will output tile_features for all tiles in the save_dir.
""" 
#nuclei_feature_generation_main.feature_generation(he_image_pickle, mask_dir, save_dir, bug_dir)



""" TODO: Tina
4. feature_aggregator will take all files with .csv extension under save_mask_dir and output the aggregated_features into path 'save_to_dir'/'save_file_name'.csv
"""
#nuclei_feature_aggregator_main.features_aggregator_main(save_mask_dir, save_to_dir, save_file_name)



""" TODO:  Samir + Minh
5.  Cell_graph_analysis (Samir+Minh):

""" 


""" TODO: Tina?  
6. Feature_selection(Tina/Minh):
   a)  TODO: Tina
       run logistic regression to rank feature_importance (i'm not sure whether this will work on cell_graph_features)  
          
   """
        #get_logreg_coef(aggr_dir, save_aggr,tiles_dir, tiles_names,save_to_dir, save_filename)

   """ 
   b)TODO: Minh
       random_forest to select variable from both cell_level & cell_graph_analysis 
       
   """       
        #get_logreg_coef(aggr_dir, save_aggr,tiles_dir, tiles_names,save_to_dir, save_filename)


"""
5. Run_classifier(Minh)

""" 
