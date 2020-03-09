import os
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


#declaring public functions
__all__ = ['get_logreg_coef']


""" 
index_label_aggregated_features takes in two inputs and it will combined both pd_object based on the tile_name so the labels will align. It will return the X(features) and y(labels) in the same index. It will also drop the original index(which is the 'tile_name' column). 

"""
def index_label_aggregated_features(pd_aggregated_features,pd_label):
    #join based on tile_name and then split it again so index of the labels matches the aggregated fxn
    agg_features_withLabel = pd_aggregated_features.join(pd_label, lsuffix='_caller', rsuffix='_other')

    #normalized to same index 
    X_original = pd.DataFrame(pd_aggregated_features.values, columns = pd_aggregated_features.columns)
    y_original = pd.DataFrame(agg_features_withLabel['label'].values, columns = ['label'])
    return (X_original, y_original)

def indicator_helper(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

"""
output_logistic_regression_coefficient will take 6 inputs and output logistic_regression_coefficient under 
'save_to_dir/save_file.csv' in decomposed form
1. aggr_dir: directory of aggregated features file (in .csv) are stored
2. save_aggr: filename of aggregated features file (just basename)
3. tiles_dir: directory of the tiles_label 
4. tiles_names: filename of tiles_label(just basename) // so this should be "dataset".csv
5. save_to_dir: directory of the output file
6. save_filename: output filename which will store all logistic_regression_coefficients // '' 
"""
def get_logreg_coef(aggr_dir, save_aggr, 
                    tiles_dir, tiles_names, 
                    save_to_dir, save_filename):
    try:
        #read aggregated_features
        agg_features = pd.read_csv(os.path.join(aggr_dir, save_aggr+'.csv'), index_col = 'tile_name')
        labels = pd.read_csv(os.path.join(tiles_dir,tiles_names+ 'csv'), index_col='tile_name')
        labels.drop(columns=['slide_id'],inplace = True)
    except: 
        print("Failed to read files. Please check")
        sys.exit(-1)
  
    X_original, y_original = index_label_aggregated_features(agg_features, labels)
    
    y = y_original
    X = X_original
    min_max_scaler = preprocessing.MinMaxScaler()

    X = min_max_scaler.fit_transform(X)
    X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    y_train = (y_train.values).ravel()
    y_test = (y_test.values).ravel()
    
    #manually default to this
    clf = LogisticRegression(max_iter= 5000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
    
    #get_logistic_regression_coefficients
    abs_coef = list(map(lambda x: abs(x), clf.coef_.flatten()))
    indicator = list(map(lambda x: indicator_helper(x), clf.coef_.flatten()))
    
    #output that into the a csv file with feature importance rank from highest to lowest
    d = {'abs_coef': abs_coef, 'indicator': indicator}
    var_coef = pd.DataFrame(data = d, index = agg_features.columns)
    var_coef.sort_values(by=['abs_coef'], ascending = False, inplace = True)
    
    try:
        output_path = os.path.join(save_to_dir, save_filename+'.csv')
        var_coef.to_csv(output_path)
        print("Successfully outputted coefficients to: ", output_path)
    except:
        print("Failed to get logistic regression")
        sys.exit(-1)