#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from collections import defaultdict
import snap

r"""
        Computation of dexPer diagrams for 1-ring rooted subgraphs
        Check Appendix D.2 for more details.
"""

class EH_pairs_1_ring():
    def __init__(self, G, NId): # snap graph + rootId
        self.Graph = G
        self.NId = NId           
        self.SH0 = defaultdict(int)
        self.EH1 = defaultdict(int)

        Nbrs = snap.TIntV()
        snap.GetNodesAtHop(self.Graph, self.NId, 1, Nbrs, False)
        one_ring_subgraph = snap.GetSubGraph(self.Graph, Nbrs)

        one_ring_components = snap.TCnComV()
        snap.GetWccs(one_ring_subgraph, one_ring_components)

        self.SH0[0.0, 1.0, 0] = 0
        self.SH0[0.0, 1.5, 0] = 0
        self.SH0[1.0, 1.5, 0] = 0

        self.EH1[0.0, 1.5, 1] = 0
        self.EH1[1.0, 1.5, 1] = 0

        for CnCom in one_ring_components:
            if CnCom.Len() == 1:
                self.SH0[0.0, 1.0, 0] += 1
            else:
                self.SH0[0.0, 1.5, 0] += 1
                CnCom_edges = snap.GetSubGraph(self.Graph, CnCom()).GetEdges()
                self.SH0[1.0, 1.5, 0] +=  CnCom_edges - 1
                self.EH1[1.0, 1.5, 1] +=  CnCom_edges - CnCom.Len() + 1
        
        self.EH1[0.0, 1.5, 1] = one_ring_subgraph.GetEdges() - self.EH1[1.0, 1.5, 1]

    def get_SH0(self):
        return self.SH0     
    
    def get_EH1(self):
        return self.EH1
        


















