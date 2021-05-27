#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
        This is a simplified version of torch_geometric.TUDataset
        We do not use nodes/edges attributes that some datasets have.
"""

import os
import os.path as osp
import shutil
import glob

import numpy as np

import torch
import zipfile

from itertools import repeat, product
from six.moves import urllib
from TU_data import Data

def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)

def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()
    return src

def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, '{}_{}.txt'.format(prefix, name))
    return read_txt_array(path, sep=',', dtype=dtype)

def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}

    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


def remove_self_loops(edge_index):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return edge_index

def maybe_log(path, log=True):
    if log:
        print('Extracting', path)

def download_url(url, folder, log=True):
    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print('Using exist file', filename)
        return path

    if log:
        print('Downloading', url)

    os.makedirs(folder)
    data = urllib.request.urlopen(url)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1
    
    edge_attr = None
    y = None
    if 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1
    edge_index = remove_self_loops(edge_index)

    data = Data(edge_index=edge_index, y=y)
    data, slices = split(data, batch)
    return data, slices



class download_TUDataset(Data):
    r"""
        This is a simplified version of torch_geometric.TUDataset
    .. note::
        We do not use nodes/edges attributes that some datasets have.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets>`_ of the
            dataset.
    """

    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'

    def __init__(self, root, name):
        self.name = name
        self.root = root
        self.raw_dir = osp.join(self.root, self.name, 'raw')
        self.data = None
        self.slices = None
        names = ['A', 'graph_indicator']  
        self.raw_file_names = ['{}_{}.txt'.format(self.name, name) for name in names]

        self.processed_dir = osp.join(self.root, self.name, 'processed')
        self.processed_file_names = 'data.pt'
        

    def download(self):
        url = self.url
        folder = osp.join(self.root, self.name)
        if not os.path.isdir(folder):
            path = download_url('{}/{}.zip'.format(url, self.name), folder)
            extract_zip(path, folder)
            os.unlink(path)
            os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        if not os.path.isdir(self.processed_dir):
            self.data, self.slices = read_tu_data(self.raw_dir, self.name)        
            os.makedirs(self.processed_dir)
            torch.save((self.data, self.slices), osp.join(self.processed_dir, self.processed_file_names))

    def loadData(self):
        self.data, self.slices = torch.load(osp.join(self.processed_dir, self.processed_file_names))

    def get(self, idx):
        edgelist, edge_slices = self.data['edge_index'], self.slices['edge_index']
        edge_start, edge_end = edge_slices[idx].item(), edge_slices[idx + 1].item()
        labellist, label_slices = self.data['y'], self.slices['y']

        data = Data(edge_index=edgelist[:, edge_start:edge_end], y = labellist[label_slices[idx].item(): label_slices[idx+1].item()])
        return data

    def getNumGraphs(self):
        return list(self.data['y'].size())[0]

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

