#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import math
import snap
import random
import os
import argparse


import matplotlib.pyplot as plt

import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import KMeans, MiniBatchKMeans

from utils import dexPer_of_all_vertices_dir
from utils import quantizer
from utils import switch_to_snap_format, get_codebook_dexPer0, get_codebook_dexPer1
from utils import nrgg_sphere, nrgg_torus

from collections import defaultdict

from gudhi.representations.vector_methods import Atol

def compute_LocalPer_vectors(networks_dir, k_ring, budget_0, budget_1):    
    num_networks = len(networks_dir)

    dexPer0_Each = [[]] * len(networks_dir)
    dexPer1_Each = [[]] * len(networks_dir)

    # if k_ring >= 4, then take the top 30 types of points appearing in all the diagrams  
    dexPer0_codebook_tuple = defaultdict(int)
    dexPer1_codebook_tuple = defaultdict(int)

    codebook_dexPer0 = []
    codebook_dexPer1 = []

    for i in range(len(networks_dir)):
        dexPer0_Each[i], dexPer1_Each[i] = dexPer_of_all_vertices_dir(networks_dir[i], k_ring)  

        if k_ring >= 4:
            for items in dexPer0_Each[i]:
                for key, value in items.items():
                    dexPer0_codebook_tuple[key] += value
                
            for items in dexPer1_Each[i]:
                for key, value in items.items():
                    dexPer1_codebook_tuple[key] += value           

    if k_ring >= 4:
        dexPer0_codebook_tuple = {k: v for k, v in sorted(dexPer0_codebook_tuple.items(), key=lambda item: item[1], reverse=True)[:30]}
        dexPer1_codebook_tuple = {k: v for k, v in sorted(dexPer1_codebook_tuple.items(), key=lambda item: item[1], reverse=True)[:30]}

        for key, value in dexPer0_codebook_tuple.items():
            codebook_dexPer0.append([key[0], key[1]])

        for key, value in dexPer1_codebook_tuple.items():
            codebook_dexPer1.append([key[0], key[1]])
    else:
        codebook_dexPer0 = get_codebook_dexPer0(k_ring)
        codebook_dexPer1 = get_codebook_dexPer1(k_ring)

    quantizer_dexPer0 = quantizer(codebook_dexPer0)
    quantizer_dexPer1 = quantizer(codebook_dexPer1)

    vec_dexPer0_Each = [[]] * num_networks
    vec_dexPer1_Each = [[]] * num_networks


    for i in range(num_networks):
        print(f"quantizing network {i+1} / {num_networks}", end="\r", flush=True)
        dexPer0_pairs = [[]] * len(dexPer0_Each[i])
        dexPer0_weights = [[]] * len(dexPer0_Each[i])

        for idx, item in enumerate(dexPer0_Each[i]):
            eh = []
            eh_weights = []
            for key, value in item.items():
                eh.append([key[0], key[1]])
                eh_weights.append(value)
            if len(eh) == 0:
                dexPer0_pairs[idx] = [[0.0, 0.0]]
                dexPer0_weights[idx] = [0]
            else:
                dexPer0_pairs[idx] = eh
                dexPer0_weights[idx] = eh_weights

        vec_dexPer0_Each[i] = quantizer_dexPer0.transform(dexPer0_pairs, dexPer0_weights)

        dexPer1_pairs = [[]] * len(dexPer1_Each[i])
        dexPer1_weights = [[]] * len(dexPer1_Each[i])

        for idx, item in enumerate(dexPer1_Each[i]):
            eh = []
            eh_weights = []
            for key, value in item.items():
                eh.append([key[0], key[1]])
                eh_weights.append(value)
            if len(eh) == 0:
                dexPer1_pairs[idx] = [[0.0, 0.0]]
                dexPer1_weights[idx] = [0]
            else:
                dexPer1_pairs[idx] = eh
                dexPer1_weights[idx] = eh_weights

        vec_dexPer1_Each[i] = quantizer_dexPer1.transform(dexPer1_pairs, dexPer1_weights)
       

    vec_dexPer0_Each = np.asarray(vec_dexPer0_Each, dtype=object)
    vec_dexPer1_Each = np.asarray(vec_dexPer1_Each, dtype=object)

    print("Quantization Done!")

    vec_ATOL_dexPer0 = Atol(quantiser=MiniBatchKMeans(n_clusters=budget_0, random_state=42), weighting_method="iidproba")
    vec_ATOL_dexPer0.fit(X=vec_dexPer0_Each)
    LocalPer_dexPer0 = vec_ATOL_dexPer0.transform(vec_dexPer0_Each)

    vec_ATOL_dexPer1 = Atol(quantiser=MiniBatchKMeans(n_clusters=budget_1, random_state=42), weighting_method="iidproba")
    vec_ATOL_dexPer1.fit(X=vec_dexPer1_Each)
    LocalPer_dexPer1 = vec_ATOL_dexPer1.transform(vec_dexPer1_Each)

    return LocalPer_dexPer0, LocalPer_dexPer1

def localPer_clustering_random_graphs(dataset_scale, num_per_category, ring_number, budget_0, budget_1):
    num_networks_per_category = num_per_category
    leaf_size = 4

    if dataset_scale == "small":
        datasets = ["A_ER", "B_Barabasi_Albert", "C_RGG", "D_WSG", "E_Sphere", "F_Torus"]
    elif dataset_scale == "large":
        datasets = ["A_LARGE_ER", "B_LARGE_Barabasi_Albert", "C_LARGE_RGG", "D_LARGE_WSG", "E_LARGE_Sphere", "F_LARGE_Torus"]
    elif dataset_scale == "sparse":
        datasets = ["A_SPARSE_ER", "B_SPARSE_Barabasi_Albert", "C_SPARSE_RGG", "D_SPARSE_WSG", "E_SPARSE_Sphere", "F_SPARSE_Torus"]
    elif dataset_scale == "small_large":
        datasets = ["A_ER", "A_LARGE_ER", "B_Barabasi_Albert", "B_LARGE_Barabasi_Albert", "C_RGG", "C_LARGE_RGG", "E_Sphere", "E_LARGE_Sphere"]
    elif dataset_scale == "small_sparse":
        datasets = ["A_ER", "A_SPARSE_ER", "B_Barabasi_Albert", "B_SPARSE_Barabasi_Albert", "C_RGG", "C_SPARSE_RGG", "E_Sphere", "E_SPARSE_Sphere"]
    elif dataset_scale == "large_sparse":
        datasets = ["A_LARGE_ER", "A_SPARSE_ER", "B_LARGE_Barabasi_Albert", "B_SPARSE_Barabasi_Albert", "C_LARGE_RGG", "C_SPARSE_RGG", "E_LARGE_Sphere", "E_SPARSE_Sphere"]
    elif dataset_scale == "small_large_sparse":
        leaf_size = 2.5
        datasets = ["A_ER", "A_LARGE_ER", "A_SPARSE_ER", "B_Barabasi_Albert", "B_LARGE_Barabasi_Albert", "B_SPARSE_Barabasi_Albert", "C_RGG", "C_LARGE_RGG", "C_SPARSE_RGG", "E_Sphere", "E_LARGE_Sphere", "E_SPARSE_Sphere"]

    else:
        print("Please set the scale of random graphs from 'small', 'large' and 'combined'.")
        return 
        
    num_categories = len(datasets)
    
    k_ring = ring_number

    b0 = budget_0
    b1 = budget_1
    
    Dir = './syn_data/'
    
    if not os.path.isdir(Dir):
        os.makedirs(Dir)

    networks_dir = []
    networks_names = []
    networks = []

    print("Generating random networks...")
    
    ##################################################################
    
    for name in datasets:
        if name == "A_ER":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                p = 0.01

                network_name = name + "_n_" + str(n) + "_p_" + str(p) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.gnp_random_graph(n, p, seed=np.random)            
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                

        if name == "A_LARGE_ER":
            for i in range(1, num_networks_per_category+1):
                n = 2000 # 1000
                p = 0.01 # 0.005
                       
                network_name = name + "_n_" + str(n) + "_p_" + str(p) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.gnp_random_graph(n, p, seed=np.random)
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)

        if name == "A_SPARSE_ER":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                p = 0.004
                   
                network_name = name + "_n_" + str(n) + "_p_" + str(p) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.gnp_random_graph(n, p, seed=np.random)         
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                


        if name == "B_Barabasi_Albert":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                m = 5
                       
                network_name = name + "_n_" + str(n) + "_m_" + str(m) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.barabasi_albert_graph(n, m, seed=np.random)  
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                

        if name == "B_LARGE_Barabasi_Albert":
            for i in range(1, num_networks_per_category+1):
                n = 2000
                m = 10
                     
                network_name = name + "_n_" + str(n) + "_m_" + str(m) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.barabasi_albert_graph(n, m, seed=np.random)
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                                     

        if name == "B_SPARSE_Barabasi_Albert":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                m = 2
                  
                network_name = name + "_n_" + str(n) + "_m_" + str(m) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.barabasi_albert_graph(n, m, seed=np.random)       
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                
        if name == "C_RGG":
            for i in range(1, num_networks_per_category+1):
                n = 1000 #2000 # 1000
                dim = 3
                r = 0.141
                                
                network_name = name + "_n_" + str(n) + "_dim_" + str(dim) + "_r_" + str(r) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.random_geometric_graph(n, r, dim) 
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                

        if name == "C_LARGE_RGG":
            for i in range(1, num_networks_per_category+1):
                n = 2000
                dim = 3
                r = 0.141
                         
                network_name = name + "_n_" + str(n) + "_dim_" + str(dim) + "_r_" + str(r) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.random_geometric_graph(n, r, dim) 
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                
                
        if name == "C_SPARSE_RGG":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                dim = 3
                r = 0.103
                        
                network_name = name + "_n_" + str(n) + "_dim_" + str(dim) + "_r_" + str(r) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.random_geometric_graph(n, r, dim)  
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                

        if name == "D_WSG":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                k = 10 
                q = 0.1
                      
                network_name = name + "_n_" + str(n) + "_knn_" + str(k) + "_q_" + str(q) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.watts_strogatz_graph(n, k, q) 
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                

        if name == "D_LARGE_WSG":
            for i in range(1, num_networks_per_category+1):
                n = 2000
                k = 20
                q = 0.1
                     
                network_name = name + "_n_" + str(n) + "_knn_" + str(k) + "_q_" + str(q) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir) 

                if not os.path.exists(network_dir):
                    g = nx.watts_strogatz_graph(n, k, q)  
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                


        if name == "D_SPARSE_WSG":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                k = 4 
                q = 0.1
                      
                network_name = name + "_n_" + str(n) + "_knn_" + str(k) + "_q_" + str(q) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    g = nx.watts_strogatz_graph(n, k, q)
                    nx.write_edgelist(g, network_dir, delimiter='\t', data=False)
                

        if name == "E_Sphere":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                d = 2
                r = 1.0
                nbhd = 0.19
                p = 0.001            
                network_name = name + "_n_" + str(n) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    nrgg_sphere(n, d, r, nbhd, p, network_dir)

        if name == "E_LARGE_Sphere":
            for i in range(1, num_networks_per_category+1):
                n = 2000
                d = 2
                r = 1.0
                nbhd = 0.19
                p = 0.001                
                network_name = name + "_LARGE_n_" + str(n) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    nrgg_sphere(n, d, r, nbhd, p, network_dir)
                

        if name == "E_SPARSE_Sphere":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                d = 2
                r = 1.0
                nbhd = 0.11
                p = 0.001            
                network_name = name + "_n_" + str(n) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    nrgg_sphere(n, d, r, nbhd, p, network_dir)


        if name == "F_Torus":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                a = 1.8
                c = 5
                nbhd = 1.0
                p = 0.001            
                network_name = name + "_n_" + str(n) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    nrgg_torus(n, a, c, nbhd, p, network_dir)

        if name == "F_LARGE_Torus":
            for i in range(1, num_networks_per_category+1):
                n = 2000 
                a = 1.8
                c = 5
                nbhd = 1.0
                p = 0.001                
                network_name = name + "_LARGE_n_" + str(n) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    nrgg_torus(n, a, c, nbhd, p, network_dir)

        if name == "F_SPARSE_Torus":
            for i in range(1, num_networks_per_category+1):
                n = 1000
                a = 1.8
                c = 5
                nbhd = 0.58
                p = 0.001            
                network_name = name + "_n_" + str(n) + "-" + str(i) 
                networks_names.append(network_name)
                network_dir = Dir + network_name
                networks_dir.append(network_dir)

                if not os.path.exists(network_dir):
                    nrgg_torus(n, a, c, nbhd, p, network_dir)
    
    print(networks_names)    

    LocalPer_dexPer0_all_networks, LocalPer_dexPer1_all_networks = compute_LocalPer_vectors(networks_dir, k_ring, b0, b1)
    LocalPer_vec_networks = np.hstack((LocalPer_dexPer0_all_networks, LocalPer_dexPer1_all_networks))

    print("Finish LocalPer featurization!")
    
    total_networks = num_categories * num_networks_per_category
    
    distance_matrix_vec = np.zeros([total_networks, total_networks])
    
    for i in range(total_networks):
        for j in range(i):
            distance_matrix_vec[i][j] = np.linalg.norm(LocalPer_vec_networks[i] - LocalPer_vec_networks[j]) # L2 norm
            distance_matrix_vec[j][i] = distance_matrix_vec[i][j]

    fig1 = plt.figure()
    axes = fig1.add_subplot(111)
    caxes = axes.matshow(distance_matrix_vec)
    plt.title(f"LocalPer_{dataset_scale}_Random_graphs_{k_ring}_ring_L2")
    fig1.colorbar(caxes)
    plt.savefig(f"LocalPer_{dataset_scale}_Random_graphs_{k_ring}_ring_L2.pdf")
    plt.show()


    fig2 = plt.figure()    
    distArray = ssd.squareform(distance_matrix_vec)
    linked = linkage(distArray, 'average')
    den = dendrogram(linked, labels=networks_names, orientation='left', distance_sort='descending', show_leaf_counts=False, leaf_font_size=leaf_size)
    plt.savefig(f"Den_LocalPer_{dataset_scale}_Random_graphs_{k_ring}_ring_L2.pdf", bbox_inches='tight')
    plt.show()

           
###############################################################################

def main(dataset_scale, num_per_category, ring_number, budget_0, budget_1):
    localPer_clustering_random_graphs(dataset_scale, num_per_category, ring_number, budget_0, budget_1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Clustering synthetic random networks using LocalPer method, including ER, BA, RGG, WSG, SPHERE and TORUS.')
    parser.add_argument('-S', '--dataset_scale', type=str, default="small", help="Scale of the synthetic datasets. Should be one of 'sparse', 'small', 'large', 'small_large_sparse', 'small_sparse', 'large_sparse' and 'small_large'.")
    parser.add_argument('-N', '--num_per_category', type=int, default="10", help="Number of networks in each category.")
    parser.add_argument('-k', '--ring_number', type=int, default="1", help="The ring number. Should be a small integer.")
    parser.add_argument('-b0', '--budget_0', type=int, default="50", help="The budget for the 0th dimensional features.")
    parser.add_argument('-b1', '--budget_1', type=int, default="50", help="The budget for the 1st dimensional features.")
    args = parser.parse_args()
    main(args.dataset_scale, args.num_per_category, args.ring_number, args.budget_0, args.budget_1)


