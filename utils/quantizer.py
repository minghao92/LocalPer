#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise

def _normalization(sample_weight):
    if sum(sample_weight) != 0:
        sample_weight = sample_weight / sum(sample_weight)
    return sample_weight 

class quantizer():
    def __init__(self, centers, contrast="gaus"):
        self.contrast = lambda measure, centers, inertias: np.exp(-pairwise.pairwise_distances(measure, Y=centers) / (inertias))
        self.centers = centers

        dist_centers = pairwise.pairwise_distances(self.centers)
        np.fill_diagonal(dist_centers, np.inf)
        self.inertias = np.min(dist_centers, axis=0)/2

    def __call__(self, measure, sample_weight=None, normalize=True):
        sample_weight = np.asarray(sample_weight, dtype=float)
        if normalize == True:
            sample_weight = _normalization(sample_weight)
        return np.sum(sample_weight * self.contrast(measure, self.centers, self.inertias.T).T, axis=1)

    def transform(self, X, sample_weight=None, normalize=True):
        return np.stack([self(measure, sample_weight=weight, normalize=normalize) for measure, weight in zip(X, sample_weight)])
