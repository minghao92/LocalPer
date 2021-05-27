#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import snap
import random
import pickle
import time
import os
import argparse

from sklearn.ensemble import *
from sklearn.preprocessing   import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans

from gudhi.representations.vector_methods import Atol

from collections import defaultdict

from utils import dexPer_of_all_vertices
from utils import quantizer
from utils import edgelist_switch_to_snap_format, get_codebook_dexPer0, get_codebook_dexPer1, getParameterSetting

from TU_data import download_TUDataset 


def single_ten_fold_classification(datasetName):
    Dir = './dexPer_computed/'
    if not os.path.isdir(Dir):
            os.makedirs(Dir)

    TU_datasets = datasetName

    k_ring, vec_num_clusters_dexPer0, vec_num_clusters_dexPer1, n_estimators, max_depth = getParameterSetting(TU_datasets)
    print(f"####Setting of Parameters####")
    print(f"Ring number: {k_ring}")
    print(f"Budget b0: {vec_num_clusters_dexPer0}")
    print(f"Budget b1: {vec_num_clusters_dexPer1}")
    print(f"Number of estimators: {n_estimators}")
    print(f"Max depth: {max_depth}")
    print(f"#############################")

    dexPer_data_filename = Dir + TU_datasets + "_EH_" + str(k_ring) + "_ring.pickle"
    dexPer_data_label_filename = Dir + "label_" + TU_datasets + ".pickle"
    vec_dexPers_filename = Dir + TU_datasets + "_dexPer_" + str(k_ring) + "_ring_temp.pickle"

    labels = []

    if not os.path.exists(vec_dexPers_filename):
        if not os.path.exists(dexPer_data_filename):
            dataset = download_TUDataset(root='./real_data/', name=TU_datasets)
            dataset.download()
            dataset.process()
            dataset.loadData()

            networks = []
            num_networks = dataset.getNumGraphs()

            dexPer0_Each = [[]] * num_networks
            dexPer1_Each = [[]] * num_networks

            print(f"Number of networks: {num_networks}")
            labels = np.empty(num_networks)

            for i in range(num_networks): 
                edge_end = dataset.get(i)['edge_index'][0].tolist()
                edge_end2 = dataset.get(i)['edge_index'][1].tolist()    
                network = edgelist_switch_to_snap_format(edge_end, edge_end2)
                networks.append(network)    
                labels[i] = dataset.get(i)['y'].item()

            # labels = LabelEncoder().fit_transform(labels)

            codebook_dexPer0 = []
            codebook_dexPer1 = []

            # if k_ring >= 4, then take the top 30 persent types of points in all the diagrams for each dimension
            dexPer0_codebook_tuple = defaultdict(int)
            dexPer1_codebook_tuple = defaultdict(int)
            select_top = 30

            i = 0
            for graph in networks:
                print(f"Computing dexPers diagrams for network {i+1} / {num_networks}", end="\r", flush=True)
                dexPer0_Each[i], dexPer1_Each[i] = dexPer_of_all_vertices(graph, k_ring)  
                if k_ring >= 4:
                    for items in dexPer0_Each[i]:
                        for key, value in items.items():
                            dexPer0_codebook_tuple[key] += value
                            
                    for items in dexPer1_Each[i]:
                        for key, value in items.items():
                            dexPer1_codebook_tuple[key] += value           
                i += 1


            dexPers_tmp = {}

            dexPers_tmp["dexPer0"] = dexPer0_Each
            dexPers_tmp["dexPer1"] = dexPer1_Each

            with open(dexPer_data_filename, 'wb') as handle:
                pickle.dump(dexPers_tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(dexPer_data_label_filename, 'wb') as handle:
                pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

            del dexPers_tmp
            print("Preprocessing")

        else:
            dexPers_tmp = {}

            with open(dexPer_data_label_filename, 'rb') as handle:
                labels = pickle.load(handle)

            with open(dexPer_data_filename, 'rb') as handle:
                dexPers_tmp = pickle.load(handle)

            dexPer0_Each = dexPers_tmp["dexPer0"]
            dexPer1_Each = dexPers_tmp["dexPer1"]

            num_networks = len(labels)
            del dexPers_tmp

        if k_ring >= 4:
            dexPer0_codebook_tuple = {k: v for k, v in sorted(dexPer0_codebook_tuple.items(), key=lambda item: item[1], reverse=True)[:select_top]}
            dexPer1_codebook_tuple = {k: v for k, v in sorted(dexPer1_codebook_tuple.items(), key=lambda item: item[1], reverse=True)[:select_top]}

            for key, value in dexPer0_codebook_tuple.items():
                codebook_dexPer0.append([key[0], key[1]])

            for key, value in dexPer1_codebook_tuple.items():
                codebook_dexPer1.append([key[0], key[1]])

        else:
            codebook_dexPer0 = get_codebook_dexPer0(k_ring)
            codebook_dexPer1 = get_codebook_dexPer1(k_ring)


        vectoriser_dexPer0 = quantizer(codebook_dexPer0)
        vectoriser_dexPer1 = quantizer(codebook_dexPer1)

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

            vec_dexPer0_Each[i] = vectoriser_dexPer0.transform(dexPer0_pairs, dexPer0_weights)


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

            vec_dexPer1_Each[i] = vectoriser_dexPer1.transform(dexPer1_pairs, dexPer1_weights)


        vec_dexPer0_Each = np.asarray(vec_dexPer0_Each, dtype=object)
        vec_dexPer1_Each = np.asarray(vec_dexPer1_Each, dtype=object)

        vec_dexPers_tmp = {}
        vec_dexPers_tmp['vec_dexPer0_Each'] = vec_dexPer0_Each
        vec_dexPers_tmp['vec_dexPer1_Each'] = vec_dexPer1_Each
        vec_dexPers_filename = Dir + TU_datasets + "_dexPer_" + str(k_ring) + "_ring_temp.pickle"

        with open(vec_dexPers_filename, 'wb') as handle:
            pickle.dump(vec_dexPers_tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)

        del vec_dexPers_tmp
        print("Quantization Done!")
    else:
        vec_dexPers_tmp = {}
        with open(dexPer_data_label_filename, 'rb') as handle:
            labels = pickle.load(handle)
        with open(vec_dexPers_filename, 'rb') as handle:
            vec_dexPers_tmp = pickle.load(handle)

        vec_dexPer0_Each = vec_dexPers_tmp['vec_dexPer0_Each']
        vec_dexPer1_Each = vec_dexPers_tmp['vec_dexPer1_Each']
        del vec_dexPers_tmp


    num_folds = 10
    test_size = .1

    # folds= ShuffleSplit(n_splits=num_folds, test_size=test_size, random_state=42).split(np.empty([len(labels)]))
    folds = KFold(n_splits=num_folds, random_state=42, shuffle=True).split(np.empty([len(labels)]))
    # folds = StratifiedKFold(n_splits=num_folds, random_state=42, shuffle=True).split(np.empty([len(labels)]), labels)

    te = []
    tr = []

    i = 1
    for (ir, ie) in folds:
        train_labels, test_labels = labels[ir], labels[ie]

        vec_ATOL_dexPer0 = Atol(quantiser=MiniBatchKMeans(n_clusters=vec_num_clusters_dexPer0, random_state=42))
        vec_ATOL_dexPer0.fit(X=vec_dexPer0_Each[ir])
        LocalPer_dexPer0_train = vec_ATOL_dexPer0.transform(vec_dexPer0_Each[ir])
        LocalPer_dexPer0_test = vec_ATOL_dexPer0.transform(vec_dexPer0_Each[ie])

        vec_ATOL_dexPer1 = Atol(quantiser=MiniBatchKMeans(n_clusters=vec_num_clusters_dexPer1, random_state=42))
        vec_ATOL_dexPer1.fit(X=vec_dexPer1_Each[ir])
        LocalPer_dexPer1_train = vec_ATOL_dexPer1.transform(vec_dexPer1_Each[ir])
        LocalPer_dexPer1_test = vec_ATOL_dexPer1.transform(vec_dexPer1_Each[ie])
            
        LocalPer_train = np.hstack((LocalPer_dexPer0_train, LocalPer_dexPer1_train))
        LocalPer_test = np.hstack((LocalPer_dexPer0_test, LocalPer_dexPer1_test))

        classifier = make_pipeline(RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42))
        classifier.fit(LocalPer_train, train_labels)
        sr = classifier.score(LocalPer_train, train_labels)
        se = classifier.score(LocalPer_test,  test_labels)
        print(f"Train accuracy of Fold {i} = {sr}, test accuracy = {se}")
        tr.append(sr)
        te.append(se)
        i += 1

    print("Average train accuracy = " + str(np.mean(tr)) + ", Average test accuracy = " + str(np.mean(te)) + ", test sd = " + str(np.std(te)))


def main(dataset_name):
    single_ten_fold_classification(dataset_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'A single ten-fold graph classification for TUDataset')
    parser.add_argument('-n', '--dataset_name', type=str, default="IMDB-BINARY", help="Default is 'IMDB-BINARY'. Check https://chrsmrrs.github.io/datasets/docs/datasets/ for all the available datasets.")
    args = parser.parse_args()
    main(args.dataset_name)
