import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

"""
Feature selection for both graph feature and cell feature
using random forest
Author:Minh
"""


def generate_train_test_dataset(features_df, selected_features=None):
    cols = features_df.columns.values
    feature_names = []

    if selected_features is None:
        for name in cols:
            if name == 'tile_name' or name == 'label':
                continue
            feature_names.append(name)
    else:
        feature_names = selected_features

    data = []

    for name in feature_names:
        arr = np.nan_to_num(features_df[name].values)
        data.append(arr)

    data = np.array(data)
    t_data = np.transpose(data)

    labels = features_df['label'].values

    return t_data, labels, feature_names


def evaluate_features_rf(features_df):
    X_train, y_train, feature_names = generate_train_test_dataset(features_df)
    print('Train Random Forest')
    rf = RandomForestClassifier(n_estimators=10000, random_state=35, verbose=1, n_jobs=-1)
    rf.fit(X_train, y_train)

    imp_score = rf.feature_importances_

    high_to_low_index = np.flip(np.argsort(imp_score))
    feature_names = np.array(feature_names)[high_to_low_index]
    imp_score = imp_score[high_to_low_index]
    feature_scores = [feature_names, imp_score]
    return feature_scores

"""
Feature selection for cell feature using logistic regression
Author:Tina
"""

""" 
index_label_aggregated_features takes in two inputs and it will combined both pd_object based on the tile_name so the labels will align. It will return the X(features) and y(labels) in the same index. It will also drop the original index(which is the 'tile_name' column). 
"""
def index_label_aggregated_features(pd_aggregated_features, pd_label):
    # join based on tile_name and then split it again so index of the labels matches the aggregated fxn
    agg_features_withLabel = pd_aggregated_features.join(pd_label, lsuffix='_caller', rsuffix='_other')

    # normalized to same index
    X_original = pd.DataFrame(pd_aggregated_features.values, columns=pd_aggregated_features.columns)
    y_original = pd.DataFrame(agg_features_withLabel['label'].values, columns=['label'])
    return (X_original, y_original)


def indicator_helper(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def get_logreg_coef(aggr_path, label_path, save_dir):
    # read aggregated_features
    agg_features = pd.read_csv(aggr_path, index_col='tile_name')
    labels = pd.read_csv(label_path, index_col='tile_name')
    labels.drop(columns=['slide_id'], inplace=True)

    X_original, y_original = index_label_aggregated_features(agg_features, labels)

    y = y_original
    X = X_original
    min_max_scaler = preprocessing.MinMaxScaler()

    X = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    y_train = (y_train.values).ravel()
    y_test = (y_test.values).ravel()

    # manually default to this
    clf = LogisticRegression(max_iter=5000, verbose=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)

    # get_logistic_regression_coefficients
    abs_coef = list(map(lambda x: abs(x), clf.coef_.flatten()))
    indicator = list(map(lambda x: indicator_helper(x), clf.coef_.flatten()))

    # output that into the a csv file with feature importance rank from highest to lowest
    d = {'abs_coef': abs_coef, 'indicator': indicator}
    var_coef = pd.DataFrame(data=d, index=agg_features.columns)
    var_coef.sort_values(by=['abs_coef'], ascending=False, inplace=True)

    output_path = save_dir / 'feature_ranking.csv'
    var_coef.to_csv(output_path)
    print("Successfully outputted coefficients to: ", output_path)