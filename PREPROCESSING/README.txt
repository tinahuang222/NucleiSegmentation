This folder contains the code to do color deconvolutions on tile images. The output data is stored in a GDRIVE folder.

The folder makeup required to run this (where to put input data) is:

On GDRIVE (where the default is \My Drive\:
\FEATURE_EXTRACTION
		  \tiles_rois  *taken directly from input data folders
				\original
				\normalized

Opening the ColorConvole.ipynb file in Gdrive should bring you directly to the Colab interace. Connect and run.

2/11/20 Keane

Aggregate_Features() reads CSV files with 115 columns of features (for each mask in that CSV file) and averages each feature by quadrant, then outputs those 4 averages features to be merged with those from every other mask in every other file.
