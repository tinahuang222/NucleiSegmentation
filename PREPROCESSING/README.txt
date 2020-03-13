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
--------------------------------------------------------------------------------------------------------------------------
3/12/20 Keane
HE_DECONV contains a .py version of the Color Deconvolution code. It requires two input arguments: input folder containing slide images and an output folder to contain the pickle file of split H & E channel arrays. The main function will also return the H and E data outside of the pickle format too.

Requirements.txt added, since the HistomicsTK library has specific version needs. A virtualenv was used to do a clean setup (may require a few tries installing numpy before some of the SK libraries)

To call this function use the following call with your input and output directories replacing these:
ColorDeconv(input_image_dir='/home/BE223B_2020/IN',output_image_dir = '/home/BE223B_2020/OUT'):
-- you will end up with a pickle file containing image arrays of H & E and a file list to indicate what original file they were split from. The same data is also returned from the function (if you call from another script)


