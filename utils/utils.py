import snap
import networkx as nx
import numpy as np

import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def switch_to_snap_format(g):
    """
        parameter:
            g : a networkx graph
        return:
            snap_graph : an undirected snap graph
    """
    network = nx.to_edgelist(g)
    snap_graph = snap.PUNGraph.New()
    for item in network:
        node1 = item[0]
        node2 = item[1]
        if not snap_graph.IsNode(node1):
            snap_graph.AddNode(node1)
        if not snap_graph.IsNode(node2):
            snap_graph.AddNode(node2)
        if not snap_graph.IsEdge(node1, node2):
            snap_graph.AddEdge(node1, node2)
    return snap_graph

def edgelist_switch_to_snap_format(edge_end, edge_end2):
    """
        parameter:
            edge_end & edge_end2 : store the two endpoints of N edges
                e.g. edge_end = [0, 2, 3], edge_end2 = [1, 3, 4]
                it's a graph on vertices [0, 1, 2, 3, 4] with edges [0, 1], [2, 3], [3, 4]
        return:
            snap_graph : an undirected snap graph 
    """
    snap_graph = snap.PUNGraph.New()
    for i in range(len(edge_end)):
        node1 = edge_end[i]
        node2 = edge_end2[i]
        if not snap_graph.IsNode(node1):
            snap_graph.AddNode(node1)
        if not snap_graph.IsNode(node2):
            snap_graph.AddNode(node2)
        if not snap_graph.IsEdge(node1, node2):
            snap_graph.AddEdge(node1, node2)
    return snap_graph

def get_persistence_diagram_list(EH_data):
    pdiagset = [[]] * len(EH_data)
    j = 0
    for EH_node in EH_data:
        eh = []
        for key, value in EH_node.items():
            for _ in range(value):
                eh.append([key[0], key[1]])
        if len(eh) == 0:
            eh.append([1.0, 1.0])
        eh = np.asarray(eh, dtype=np.float64)
        pdiagset[j] = eh
        j += 1
    return pdiagset

def get_codebook_dexPer0(k_ring):
    if k_ring >= 4:
        return 

    codebook_dexPer0 = []

    if k_ring == 1:
        codebook_dexPer0 = np.asarray([[0.0, 1.0], [0.0, 1.5], [1.0, 1.5]], dtype=object)
    elif k_ring == 2:
        codebook_dexPer0 = np.asarray([[0.0, 1.0], [0.0, 1.5], [0.0, 2.0], [0.0, 2.5], [1.0, 1.5], [1.0, 2.0], [1.0, 2.5], [2.0, 2.5]], dtype=object)
    elif k_ring == 3:
        codebook_dexPer0 = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 1.5], [0.0, 2.0], [0.0, 2.5], [0.0, 3.0], [0.0, 3.5], 
                                [1.0, 1.5], [1.0, 2.0], [1.0, 2.5], [1.0, 3.0], [1.0, 3.5], [2.0, 2.5], 
                                [2.0, 3.0], [2.0, 3.5], [3.0, 3.5]], dtype=object)

    return codebook_dexPer0

def get_codebook_dexPer1(k_ring):
    if k_ring >= 4:
        return

    codebook_dexPer1 = []

    if k_ring == 1:
        codebook_dexPer1 = np.asarray([[0.0, 1.5], [1.0, 1.5]], dtype=object)
    elif k_ring == 2:
        codebook_dexPer1 = np.asarray([[0.0, 1.0], [0.0, 1.5], [0.0, 2.0], [0.0, 2.5], [1.0, 1.5], [1.0, 2.0], [1.0, 2.5], [2.0, 2.5]], dtype=object)
    elif k_ring == 3:
        codebook_dexPer1 = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 1.5], [0.0, 2.0], [0.0, 2.5], [0.0, 3.0], [0.0, 3.5], 
                             [1.0, 1.5], [1.0, 2.0], [1.0, 2.5], [1.0, 3.0], [1.0, 3.5], [2.0, 2.5], 
                             [2.0, 3.0], [2.0, 3.5], [3.0, 3.5]], dtype=object)

    return codebook_dexPer1

def getParameterSetting(dataset_name):
    k_ring = 1
    vec_num_clusters_dexPer0 = 50
    vec_num_clusters_dexPer1 = 50
    max_depth = None
    n_estimators = 400

    bio_chemo_datasets = ["MUTAG", "COX2", "DHFR", "PROTEINS", "NCI1", "NCI109", "FRANKENSTEIN"]

    if dataset_name == "IMDB-BINARY" or dataset_name == "IMDB-MULTI":
        max_depth = 10
        return k_ring, vec_num_clusters_dexPer0, vec_num_clusters_dexPer1, n_estimators, max_depth

    elif dataset_name == "COLLAB":
        return k_ring, vec_num_clusters_dexPer0, vec_num_clusters_dexPer1, n_estimators, max_depth

    elif dataset_name == "REDDIT-MULTI-5K" or dataset_name == "REDDIT-MULTI-12K":
        k_ring = 3
        max_depth = 10
        return k_ring, vec_num_clusters_dexPer0, vec_num_clusters_dexPer1, n_estimators, max_depth

    elif dataset_name in bio_chemo_datasets:
        k_ring = 3
        return k_ring, vec_num_clusters_dexPer0, vec_num_clusters_dexPer1, n_estimators, max_depth

def torusUnif(n ,a, c):
    pcd = [[]] * n
    theta = []

    while len(theta) < n:
        xvec = np.random.uniform(low=0, high=2.0 * np.pi)
        yvec = np.random.uniform(low=0, high=1.0 / np.pi)
        fx = (1 + (a / c) * np.cos(xvec)) / (2.0 * np.pi)
        if (yvec < fx):
            theta.append(xvec)

    phi = np.random.uniform(low=0, high=2.0 * np.pi, size=n)
    x = (c + a * np.cos(theta)) * np.cos(phi)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(x,y,z)
    # plt.show()
    
    return np.vstack((x, y, z)).T

def sphereUnif(n, d, r):
    pcd = [[]] * n
    X = np.split(np.random.normal(loc=0.0, scale=1.0, size= n * (d+1)), n)
    for i in range(n):
        while np.linalg.norm(X[i]) == 0:
            X[i] = np.random.normal(loc=0.0, scale=1.0, size=d+1)
        X[i] = r * X[i] / np.linalg.norm(X[i])

    # x = [x[0] for x in X]
    # y = [x[1] for x in X]
    # z = [x[2] for x in X]
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(x,y,z)
    # plt.show()
    
    return X

def nrgg_torus(n, a, c, nbhd_size, p, network_dir):
    
    pcd = torusUnif(n, a, c)

    edge_list_0 = []
    edge_list_1 = [] 

    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pcd[i] - pcd[j]) <= nbhd_size:
                edge_list_0.append(i)
                edge_list_1.append(j)
            else:
                if np.random.uniform(low=0, high=1) <= p:
                    edge_list_0.append(i)
                    edge_list_1.append(j)

    # print(f"Number of edges = {len(edge_list_0)}")

    with open(network_dir, 'w') as handle:
        for i in range(len(edge_list_0)):
            handle.write("%s\t%s\n" % (edge_list_0[i], edge_list_1[i]))

# nrgg_torus(1000, 1.8, 5.0, 1., 0.001, "./test.edgelist")

def nrgg_sphere(n, d, r, nbhd_size, p, network_dir):
    
    pcd = sphereUnif(n, d, r)

    edge_list_0 = []
    edge_list_1 = [] 

    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pcd[i] - pcd[j]) <= nbhd_size:
                edge_list_0.append(i)
                edge_list_1.append(j)
            else:
                if np.random.uniform(low=0, high=1) <= p:
                    edge_list_0.append(i)
                    edge_list_1.append(j)

    # print(f"Number of edges = {len(edge_list_0)}")

    with open(network_dir, 'w') as handle:
        for i in range(len(edge_list_0)):
            handle.write("%s\t%s\n" % (edge_list_0[i], edge_list_1[i]))

# nrgg_sphere(1000, 2, 1, 0.23, 0.001, "./sphere.edgelist")