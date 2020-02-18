import skimage.io as io
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import progressbar as pb


def kmeans_graph_features_all_slides(
    tiles_path: str,
    centroids_path: str,
  ) -> pd.DataFrame:
  tile_files = os.listdir(tiles_path)
  tile_files = [tf for tf in tile_files if tf[0] != '.']
  metrics_dict = {
    'tile': [],
    'n_nuc': [],
    'max_clust_area': [],
    'max_clust_n_nuc': [],
  }
  for tf in pb.progressbar(tile_files):
    tile_id = tf.split('/')[-1].split('.')[0]
    tile = io.imread(os.path.join(tiles_path, tile_id + '.png'))
    centroid = pd.read_csv(os.path.join(centroids_path, tile_id + '.csv'))
    c_list = list(centroid.itertuples(index=False))
    c_dict = {
        i: {
            'position': (c_list[i].x, c_list[i].y)
        }
        for i in range(len(c_list))
    }
    radius = 50
    edge_dict = {
        'x0': [],
        'y0': [],
        'x1': [],
        'y1': []
    }
    for cd_0, cd_1 in combinations(c_dict.items(), 2):
      c0 = cd_0[1]['position']
      c1 = cd_1[1]['position']
      dist = np.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2)

      if dist < radius:
        edge_dict['x0'].append(c0[0])
        edge_dict['y0'].append(c0[1])
        edge_dict['x1'].append(c1[0])
        edge_dict['y1'].append(c1[1])

    edges = pd.DataFrame.from_dict(edge_dict)
    edges['label'] = KMeans().fit(edges).labels_

    largest_cluster = edges.groupby('label').x0.count().sort_values(ascending=False).index[0]

    big_clust = edges[edges.label == largest_cluster]
    big_clust_0 = big_clust[['x0', 'y0']].rename(columns={'x0': 'x', 'y0': 'y'})
    big_clust_1 = big_clust[['x1', 'y1']].rename(columns={'x1': 'x', 'y1': 'y'})
    big_clust_nodes = pd.concat([big_clust_0, big_clust_1]).drop_duplicates()
    hull = ConvexHull(big_clust_nodes[['x', 'y']])

    metrics_dict['tile'].append(tile_id)
    metrics_dict['n_nuc'].append(centroid.shape[0])
    metrics_dict['max_clust_area'].append(hull.area)
    metrics_dict['max_clust_n_nuc'].append(big_clust_nodes.shape[0])
  return pd.DataFrame.from_dict(metrics_dict)
