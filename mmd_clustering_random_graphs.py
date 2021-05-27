#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import math
import snap
import torch
import random
import os
import argparse

import matplotlib.pyplot as plt

import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage

from utils import dexPer_of_all_vertices_dir
from utils import torch_swd_mmd_kernel

def mmd_clustering_random_graphs(dataset_scale, num_per_category, ring_number):
    if num_per_category > 10:
        print("MMD+SWK is extremely slow when there are too many networks in the datasets. Please choose another num_per_category <= 10.")
        return
    
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
    num_networks_per_category = num_per_category

    k_ring = ring_number
    
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

    ### compute distance-to-root extended persistence diagrams   
    dexPer0_all_networks = [[]] * len(networks_dir)
    dexPer1_all_networks = [[]] * len(networks_dir)

    i = 0
    for network_dir in networks_dir:
        print(f"Computing dexPers diagrams for network {i+1} / {len(networks_dir)}", end="\r", flush=True)
        dexPer0_all_networks[i], dexPer1_all_networks[i] = dexPer_of_all_vertices_dir(network_dir, k_ring)
        i += 1  
        
    
    print("Finish computation of all dexPer diagrams!")

    ### preprocess the diagrams such that dexPer_0pairs (dexPer_1pairs) contains all the types of points appearing in dexPer_0 (dexPer_1) and dexPer0_mul (dexPer1_mul) is a collection of vectors recording the multiplicity.
    # E.g., if dexPer0_all_networks = [[{(0,1): 2, (1, 1.5):10}, {(0,1): 1, (1, 1.5): 10}], [{(0, 1.5): 2}, {(1, 1.5): 2}]], then dexPer0_pairs = [[0,1], [1,1.5], [0,1.5]] and dexPer0_mul = [[[2, 10, 0], [1, 10, 0]], [[0, 0, 2], [0, 2, 0]]]
    dexPer0_mul, dexPer0_pairs = preprocess_all_dexPer(dexPer0_all_networks)
    dexPer1_mul, dexPer1_pairs = preprocess_all_dexPer(dexPer1_all_networks)
        
    total_networks = num_categories * num_networks_per_category
    
    distance_matrix_0 = np.zeros([total_networks, total_networks])
    distance_matrix_1 = np.zeros([total_networks, total_networks])
    
    distance_matrix_combined_arithmetic = np.zeros([total_networks, total_networks])
    distance_matrix_combined_quadratic = np.zeros([total_networks, total_networks])

    # parameters for sliced wasserstein kernel
    num_directions = 6
    swk_bandwidth = 10

    for i in range(total_networks):
        print("Working on " + str(i))
        for j in range(i):
            print("   " + str(j))

            print("      dim:0")            
            distance_matrix_0[i][j] = math.sqrt(max(0, torch_swd_mmd_kernel(dexPer0_pairs, dexPer0_mul[i], dexPer0_mul[j], num_directions, swk_bandwidth)))
            print(distance_matrix_0[i][j])
            distance_matrix_0[j][i] = distance_matrix_0[i][j]
                        
            print("      dim:1")
            distance_matrix_1[i][j] = math.sqrt(max(0, torch_swd_mmd_kernel(dexPer1_pairs, dexPer1_mul[i], dexPer1_mul[j], num_directions, swk_bandwidth)))
            print(distance_matrix_1[i][j])
            distance_matrix_1[j][i] = distance_matrix_1[i][j]

            distance_matrix_combined_arithmetic[i][j] = distance_matrix_0[i][j] + distance_matrix_1[i][j]
            distance_matrix_combined_arithmetic[j][i] = distance_matrix_combined_arithmetic[i][j]
 

    fig1 = plt.figure()
    axes = fig1.add_subplot(111)
    caxes = axes.matshow(distance_matrix_combined_arithmetic)
    fig1.colorbar(caxes)
    heatmap_output = f"MMD_SWK_{dataset_scale}_Random_graphs_{k_ring}_ring_arithmetic"
    plt.title(heatmap_output)
    plt.savefig(f"MMD_SWK_{dataset_scale}_Random_graphs_{k_ring}_ring_arithmetic.pdf")
    plt.show()
        
    fig2 = plt.figure()
    distArray = ssd.squareform(distance_matrix_combined_arithmetic)
    linked = linkage(distArray, 'average')
    den = dendrogram(linked, labels=networks_names, orientation='left', distance_sort='descending', show_leaf_counts=False, leaf_font_size=leaf_size)
    plt.savefig(f"Den_MMD_SWK_{dataset_scale}_random_graphs_{k_ring}_ring_arithmetic.pdf", bbox_inches='tight')
    plt.show()
        
##############################################################################

def preprocess_all_dexPer(dexPer_data):
    '''
        Parameters:
            dexPer_data (list of lists of dictionaries): input collection of collections of persistence diagrams in dictionary form.
            
        Returns:
            pairs (list of 2-dim arrays): list of all types of points present in the persistence diagrams  
            multi (list of lists of len(pairs)-dim arrays): the multiplicities of the corresponding pairs

        e.g., If dexPer_data = [[{(0,1): 2, (1, 1.5): 10}, {(0,1): 1, (1, 1.5): 10}], [{(0, 1.5): 2}, {(1, 1.5): 2}]], then
            pairs = [[0,1], [1,1.5], [0,1.5]]
            multi = [[[2, 10, 0], [1, 10, 0]], [[0, 0, 2], [0, 2, 0]]]
    '''
    multi = [[]] * len(dexPer_data)
    pairs = []
    
    for ehs in dexPer_data:
        for eh in ehs:
            for key, value in eh.items():
                pair = [key[0], key[1]]
                if pair not in pairs:
                    pairs.append(pair)
    
    for row in range(len(dexPer_data)):
        eh = preprocess_dexPer(dexPer_data[row], pairs)
        multi[row] = eh
    return multi, pairs

def preprocess_dexPer(dexPer_data, pairs):
    multi = [[]] * len(dexPer_data)
    for row in range(len(dexPer_data)):
        eh = np.zeros(len(pairs))
        Diagram = dexPer_data[row]
        for key, value in Diagram.items():
            pair = [key[0], key[1]]
            eh[pairs.index(pair)] = value
        multi[row] = eh
    return multi
           
###############################################################################


def main(dataset_scale, num_per_category, ring_number):
    mmd_clustering_random_graphs(dataset_scale, num_per_category, ring_number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Clustering synthetic random networks using MMD+SWK method, including ER, BA, RGG, WSG, SPHERE and TORUS.')
    parser.add_argument('-S', '--dataset_scale', type=str, default="small", help="Scale of the synthetic datasets. Should be one of 'sparse', 'small', 'large', 'small_large_sparse', 'small_sparse', 'large_sparse' and 'small_large'.")
    parser.add_argument('-N', '--num_per_category', type=int, default="4", help="Number of networks in each category. Should be <= 10")
    parser.add_argument('-k', '--ring_number', type=int, default="1", help="The ring number. Should be a small integer.")
    args = parser.parse_args()
    main(args.dataset_scale, args.num_per_category, args.ring_number)


