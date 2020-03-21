# Pathology Cancer image classifier
## Environment setup
### Directory setup (only need to put the tiles rois into data/train and data/test accordingly)
```
.
└── data
    ├── models
    │   ├── graph_conv
    │   ├── rf
    │   └── weight_avg
    ├── test                         (extract test zip data and put it here)
    │   ├── cell_feature
    │   │   ├── eosin
    │   │   ├── hema
    │   │   └── rgb
    │   ├── centroid
    │   ├── graph_feature
    │   │   └── lap_matrix
    │   ├── img_arr
    │   ├── mask
    │   ├── results
    │   └── tiles_rois
    │       ├── normalized
    │       └── original
    └── train
        ├── cell_feature
        │   ├── eosin
        │   ├── hema
        │   └── rgb
        ├── centroid
        ├── graph_feature
        │   └── lap_matrix
        ├── img_arr
        ├── mask
        ├── masks_features
        ├── models
        │   └── graph_conv
        ├── output
        ├── results
        └── tiles_rois            (extract train zip data and put it here)
            ├── normalized
            └── original

```
### Package setup

1. Install package in the requirements.txt
2. Install Nvidia cuda 10.1
3. Install pytorch follow this link: https://pytorch.org/get-started/locally/
4. Install pyradiomics
    Follow the instruction here: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
    Downgrade torch-cluster to torch-cluster=1.4.5

## Pipeline Structure
1. Basic feature generation
    1. Mask generation
    2. Hema image generation
    3. Centroid generation
2. Feature generation
    1. Pyradiomic cell feature
    2. Histogram aggregation of cell feature
    3. Furthest point sampling graph feature
3. Feature Selection
    1. Mix feature ranking
    2. Cell feature ranking for graph convolution
4. Classification
    1. Random forest classifcation
        - train
        - predict
        - k-fold test
    2. Graph convolution classification
        - train
        - predict
        - k-fold test
    3. Weight average the result [best 0.82, 0.18]

### 1. Generate basic feature
`python generate_basic_feature.py --test=False` (if True will generate feature in test folder instead)

The result feature will be save in:

    A. Mask generation: data/(train/test)/mask
    B. Hema image generation: data/(train/test)/img_arr (hdf5 file)
    C. Centroid generation: data/(train/test)/centroid

### 2. Generate advance feature
`python generate_advance_feature.py --test=False` (if True will generate feature in test folder instead)

The result feature will be save in:

    A. Cell feature: data/(train/test)/cell_feature/hema
    B. Histogram aggregation of cell feature: data/(train/test)/cell_feature/cell_features_aggr.csv
    C. Furthest point sampling graph feature: data/(train/test)/graph_feature/graph_feature.csv

### 3. Feature selection
`python feature_selection.py` (will only run in train data with label)

    A. Mix feature ranking: 
        1. data/(train/test)/graph_feature/mix_feature_ranking.csv   (ranking of all feature
        2. data/(train/test)/graph_feature/mix_feature.csv  (combine all feature in one csv file)
    B. Cell feature ranking for graph conv: data/(train/test)/cell_feature/feature_ranking.csv

### 4. Classification
`python train.py`

The trained models will be save in:

    A. Random forest classifier: data/models/rf/rf.pkl
    B. Graph convolution network: data/models/graph_conv/graph_conv.pth
    C. There is no save model for weight_avg_classifier as it just avg the 2 above with ratio 0.82, 0.18
    
`python eval.py`

The prediction results will be save in:

    A. Random forest classifier: data/results/rf_results.csv
    B. Graph convolution network: data/results/graph_conv_results.csv
    C. Weight avg classifier: data/results/weight_avg_results.csv



