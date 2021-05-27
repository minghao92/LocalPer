#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch


def torch_swd_mmd_kernel(pairs, X, Y, num_directions, bandwidth):
    XX = torch_swd_kernel(pairs, X, X, num_directions, bandwidth)
    XY = torch_swd_kernel(pairs, X, Y, num_directions, bandwidth)
    YY = torch_swd_kernel(pairs, Y, Y, num_directions, bandwidth)
    return (XX.mean() + YY.mean() - 2 * XY.mean()).item()

def torch_swd_kernel(pairs, X, Y, num_directions, bandwidth):
    SWK = torch_SlicedWassersteinKernel(num_directions, bandwidth)
    SWK.fit(pairs, X)
    return SWK.transform(Y)

#############################################
# Kernel Class ############################
#############################################

class torch_SlicedWassersteinDistance():
    """
    This is a class for computing the sliced Wasserstein distance matrix from a list of (pairs, multi) persistence diagrams. 
    This class a modified version of SlicedWassersteinDistance class in Gudhi http://gudhi.gforge.inria.fr/python/latest/_modules/gudhi/representations/kernel_methods.html#SlicedWassersteinKernel 
    See http://proceedings.mlr.press/v70/carriere17a.html for more details. 
    """
    def __init__(self, num_directions=6):
        """
        Parameters:
            num_directions (int): number of lines evenly sampled from [-pi/2,pi/2] in order to approximate and speed up the distance computation (default 6). 
        """
        
        self.num_directions = num_directions
        thetas = np.linspace(-np.pi/2, np.pi/2, num=self.num_directions+1)[np.newaxis,:-1]
        self.lines_ = np.concatenate([np.cos(thetas), np.sin(thetas)], axis=0)
        
    def fit(self, pairs, multi):
        """
        Parameters:
            pairs (list of m x 2 numpy arrays): all the types of points appearing in persistence diagrams.
            multi (n x m array): the collection of vectors of multiplicities of correponding pairs for all persistence diagrams.
        """
        
        self.diagrams_ = pairs
        self.approx_ = torch.as_tensor(np.matmul(pairs, self.lines_), dtype=torch.float32)        
        self.multiplicity = torch.sparse.CharTensor(multi)
        
        diag_proj = (1./2) * np.ones((2,2))
        self.approx_diag_ = torch.as_tensor(np.matmul(np.matmul(pairs, diag_proj), self.lines_), dtype=torch.float32)

        return self


    def transform(self, multi):
        """
        Parameters:
            multi (list of n x m numpy arrays): input collection of vectors of multiplicities of correponding pairs.

        Returns:
            numpy array of shape (number of diagrams in **diagrams**) x (number of diagrams in X): matrix of pairwise sliced Wasserstein distances.
        """
        device = torch.device('cpu')        
        multiplicity = torch.sparse.CharTensor(multi)
        n_1, n_2 = multiplicity.size(0), self.multiplicity.size(0)
        dim = multiplicity.size(1)
        
        expanded_1 = multiplicity.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = self.multiplicity.unsqueeze(0).expand(n_1, n_2, dim)
        
        Multi = torch.cat([expanded_2, expanded_1], dim=2)
            
        del expanded_1, expanded_2
        
        Sort_A, A_indices = torch.sort(torch.cat([self.approx_, self.approx_diag_], dim=0), dim=0)
        Sort_A = Sort_A.t().to(device)
        A_indices = A_indices.t()
        
        A_ = Multi[:, :, A_indices].to(device)
        
        Sort_B, B_indices = torch.sort(torch.cat([self.approx_diag_, self.approx_], dim=0), dim=0)
        B_indices = B_indices.t()
        
        B_ = Multi[:, :, B_indices].to(device)
        
        del Multi, A_indices, Sort_B, B_indices
        
        dummyMat = [[0 for x in range(2 * dim -1)] for y in range(2*dim)] 
        
        for i in range(2 * dim -1):
            for j in range(i, 2 * dim -1):
                dummyMat[i][j] = 1
        
        _dummyMat = torch.sparse.CharTensor(dummyMat)                    
        diff = torch.abs(torch.matmul(A_ - B_, _dummyMat))
        
        del A_, B_, _dummyMat

        diff_Matrix = [[0 for x in range(2 * dim -1)] for y in range(2*dim)]     
        for i in range(2 * dim-1):
            diff_Matrix[i][i] = -1
            diff_Matrix[i+1][i] = 1
            
        
        diff_Matrix = torch.sparse.FloatTensor(diff_Matrix)
        Diff_matrix = torch.matmul(Sort_A, diff_Matrix)
        
        del Sort_A, diff_Matrix
               
        diff = torch.mul(diff.float(), Diff_matrix)
        
        del Diff_matrix
        
        return torch.mean(torch.sum(diff, dim=3), dim=2)
        


class torch_SlicedWassersteinKernel():
    """
    This is a class for computing the sliced Wasserstein kernel matrix from a list of persistence diagrams. 
    The sliced Wasserstein kernel is computed by exponentiating the corresponding sliced Wasserstein distance with a Gaussian kernel. 
    See http://proceedings.mlr.press/v70/carriere17a.html for more details. 
    """
    def __init__(self, num_directions=6, bandwidth=10):
        """
        Parameters:
            bandwidth (double): bandwidth of the Gaussian kernel applied to the sliced Wasserstein distance (default 10.).
            num_directions (int): number of lines evenly sampled from [-pi/2,pi/2] in order to approximate and speed up the kernel computation (default 6).
        """
        self.bandwidth = bandwidth
        self.sw_ = torch_SlicedWassersteinDistance(num_directions=num_directions)


    def fit(self, pairs, multi):
        self.sw_.fit(pairs, multi)
        return self


    def transform(self, multi):
        return torch.exp(-self.sw_.transform(multi)/self.bandwidth)
