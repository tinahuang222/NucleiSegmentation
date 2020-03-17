import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from PIL import Image, ImageDraw
from pathlib import Path
import pandas as pd

"""
Generate Furthest point sampling graph feature
Author:Minh
"""


def calculate_distance_point_to_sets(point, points):
    point_dub = np.empty(points.shape)
    point_dub[:, 0] = point[0]
    point_dub[:, 1] = point[1]
    diff = point_dub - points
    diff_square = np.multiply(diff, diff)
    diff_square_sum = diff_square[:, 0] + diff_square[:, 1]
    eucledian = np.sqrt(diff_square_sum)
    min_distance = np.min(eucledian)
    return min_distance


def find_furthest_points(s_points, d_points):
    distance_array = np.empty(len(s_points))
    for i in range(len(s_points)):
        distance_array[i] = calculate_distance_point_to_sets(s_points[i], d_points)

    return np.argmax(distance_array)


def furthest_point_sampling(points, count, img_shape, graph=False):
    sample_points = []
    temp_points = points.copy()
    if count <= 1:
        return None

    sample_points.append(temp_points[0])
    temp_points = np.delete(temp_points, 0, axis=0)

    excluded_points = []
    excluded_points.append([sample_points[0][0], sample_points[0][1]])
    excluded_points.append([0, 0])
    excluded_points.append([img_shape[0], 0])
    excluded_points.append([0, img_shape[1]])
    excluded_points.append([img_shape[0], img_shape[1]])
    for i in range(count - 1):
        index = find_furthest_points(temp_points, np.array(excluded_points))
        sample_points.append(temp_points[index])
        excluded_points.append(temp_points[index])
        temp_points = np.delete(temp_points, index, axis=0)

    sample_points = np.array(sample_points)

    if graph:
        points = np.array(points)
        plt.scatter(points[:, 0], points[:, 1])
        plt.scatter(sample_points[:, 0], sample_points[:, 1])
        plt.show()
        plt.clf()
    return sample_points


def generate_cell_graph(points, radius_limit, connection_limit, img_size, graph=False):
    neigh = radius_neighbors_graph(points, radius_limit, metric='euclidean')
    cell_graph = neigh.toarray()
    vertices = []
    edges = []
    adj_matrix = np.zeros((len(points), len(points)))
    degree_matrix = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        current_connection = cell_graph[i]
        connect_locs = np.argwhere(current_connection == 1).flatten()

        if len(connect_locs) > connection_limit:
            distance_array = np.array([np.linalg.norm(points[i] - points[x]) for x in connect_locs])
            sort_indices = np.argsort(distance_array)
            connect_locs = connect_locs[sort_indices]
            connect_locs = connect_locs[:connection_limit]

        vertices.append(points[i].tolist())
        edges.append(points[np.array(connect_locs)].tolist())

        degree_matrix[i, i] = len(connect_locs)
        for conn in connect_locs:
            adj_matrix[i, conn] = np.linalg.norm(points[i] - points[conn]) / float(radius_limit)

    laplacian_matrix = degree_matrix - adj_matrix

    if graph:
        size = 2
        blank = Image.new('RGBA', img_size, (0, 0, 0, 0))
        d = ImageDraw.Draw(blank)
        for i in range(len(vertices)):
            d.ellipse([points[i][0] - size, points[i][1] - size,
                       points[i][0] + size, points[i][1] + size], fill=(255, 0, 0))

            start_point = vertices[i]
            end_points = edges[i]
            for j in range(len(end_points)):
                d.line([start_point[0], start_point[1],
                        end_points[j][0], end_points[j][1]],
                       fill=(0, 255, 0), width=2)
        blank.show()

    return vertices, edges, adj_matrix, laplacian_matrix


def calculate_graph_statistic(cell_graph):
    vertices = cell_graph[0]
    edges = cell_graph[1]

    num_vertices = len(vertices)
    total_connections = 0
    total_connection_length = 0
    non_connecting_nodes = 0
    for i in range(num_vertices):
        total_connections += len(edges[i])
        conn_length = [np.linalg.norm(np.array(vertices[i]) - np.array(y)) for y in edges[i]]
        if len(conn_length) > 0:
            for length in conn_length:
                total_connection_length += length
        else:
            non_connecting_nodes += 1

    avg_connections = total_connections / float(num_vertices)
    avg_connection_length = total_connection_length / float(total_connections)
    non_connecting_nodes_fraction = non_connecting_nodes / float(num_vertices)

    data = [num_vertices, total_connections, total_connection_length,
            avg_connections, avg_connection_length, non_connecting_nodes,
            non_connecting_nodes_fraction]

    return data


def generate_cell_graph_statistic(centroid_path, matrix_path):
    alpha_ratio = 0.35
    beta_ratio = 0.15
    img_size = [512, 512]
    radius_limit = 100.
    connection_limit = 8

    df = pd.read_csv(centroid_path)

    x = df['x'].values
    x = x[np.logical_not(np.isnan(x))]

    y = df['y'].values
    y = y[np.logical_not(np.isnan(y))]

    points = [[a, b] for a, b in zip(x, y)]

    num_points = len(points)
    alpha_num_sample_points = int(alpha_ratio * num_points)
    alpha_sample_points = furthest_point_sampling(points, alpha_num_sample_points, img_size, graph=False)

    beta_num_sample_points = int(beta_ratio * num_points)

    beta_sample_points_index = np.random.choice(alpha_num_sample_points, beta_num_sample_points, replace=False)
    beta_sample_points = np.take(alpha_sample_points, beta_sample_points_index, axis=0)

    vertices, edges, adj_matrix, lap_matrix = generate_cell_graph(beta_sample_points, radius_limit, connection_limit,
                                                                  img_size, graph=False)
    np.savetxt(matrix_path / centroid_path.name, lap_matrix, delimiter=',')

    cell_graph = [vertices, edges]

    cols = ['num_vertices', 'total_connections', 'total_connection_length',
            'avg_connections', 'avg_connection_length', 'non_connecting_nodes',
            'non_connecting_nodes_fraction']
    data = calculate_graph_statistic(cell_graph)

    return data, cols