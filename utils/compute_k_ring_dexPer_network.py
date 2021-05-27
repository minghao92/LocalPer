#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import snap
from multiprocessing import Pool
import numpy as np

import networkx as nx

from utils import EH_pairs
from utils import EH_pairs_1_ring

from functools import partial

        
def pick_node_compute_dexPer(Graph, k_ring, NId):
    dexPer0 = {}
    dexPer1 = {}
    [BfsTree_size, _, BfsTree_depth] = snap.GetSubTreeSz(Graph, NId, True, True)
    #print "Size %d, Depth %d" % (BfsTree_size, BfsTree_depth)
    
    CnCom = snap.TIntV()
    snap.GetNodeWcc(Graph, NId, CnCom)
    SubGraph = snap.GetSubGraph(Graph, CnCom)

    ############################################################################################################

    # Isolated nodes
    if BfsTree_depth == 0:
        return dexPer0, dexPer1
    ############################################################################################# 
    else:
        heightVal = {}
        NodeVec = snap.TIntV()
        heightVal[NId] = 0
        
        for dist in range (1, min(k_ring + 1, BfsTree_depth + 1)):
            snap.GetNodesAtHop(SubGraph, NId, dist, NodeVec, False)
            for item in NodeVec:
                heightVal[item] = dist
   
        Modified_Graph = snap.TUNGraph.New()
        
        New_Node = SubGraph.GetMxNId()
        for EI in SubGraph.Edges():
            Src_idx = EI.GetSrcNId()
            Dst_idx = EI.GetDstNId()
            
            Src_dist = heightVal.get(Src_idx, -1)
            Dst_dist = heightVal.get(Dst_idx, -1)
            
            if not Modified_Graph.IsNode(Src_idx):
                Modified_Graph.AddNode(Src_idx)
            if not Modified_Graph.IsNode(Dst_idx):
                Modified_Graph.AddNode(Dst_idx)
            
            
            if Src_dist >= 0 and Dst_dist >= 0:
                if Src_dist == Dst_dist:                       
                    #print(New_Nodes)
                    Modified_Graph.AddNode(New_Node)
                    Modified_Graph.AddEdge(EI.GetSrcNId(), New_Node)
                    Modified_Graph.AddEdge(EI.GetDstNId(), New_Node)
                    heightVal[New_Node] = Src_dist + 0.5
                    New_Node += 1
                else:
                    Modified_Graph.AddEdge(EI.GetSrcNId(), EI.GetDstNId())
                    

        EH_computation = EH_pairs(Modified_Graph, heightVal)

        return EH_computation.get_SH0(), EH_computation.get_EH1()     
 

def pick_node_compute_dexPer0(Graph, k_ring, NId):
    dexPer0 = {}
    [BfsTree_size, _, BfsTree_depth] = snap.GetSubTreeSz(Graph, NId, True, True)
    #print "Size %d, Depth %d" % (BfsTree_size, BfsTree_depth)
    
    CnCom = snap.TIntV()
    snap.GetNodeWcc(Graph, NId, CnCom)
    SubGraph = snap.GetSubGraph(Graph, CnCom)

    ############################################################################################################

    # Isolated nodes
    if BfsTree_depth == 0:
        return dexPer0
    ############################################################################################# 
    else:
        heightVal = {}
        NodeVec = snap.TIntV()
        heightVal[NId] = 0
        
        for dist in range (1, min(k_ring + 1, BfsTree_depth + 1)):
            snap.GetNodesAtHop(SubGraph, NId, dist, NodeVec, False)
            for item in NodeVec:
                heightVal[item] = dist
   
        Modified_Graph = snap.TUNGraph.New()
        
        New_Node = SubGraph.GetMxNId()
        for EI in SubGraph.Edges():
            Src_idx = EI.GetSrcNId()
            Dst_idx = EI.GetDstNId()
            
            Src_dist = heightVal.get(Src_idx, -1)
            Dst_dist = heightVal.get(Dst_idx, -1)
            
            if not Modified_Graph.IsNode(Src_idx):
                Modified_Graph.AddNode(Src_idx)
            if not Modified_Graph.IsNode(Dst_idx):
                Modified_Graph.AddNode(Dst_idx)
            
            
            if Src_dist >= 0 and Dst_dist >= 0:
                if Src_dist == Dst_dist:                       
                    #print(New_Nodes)
                    Modified_Graph.AddNode(New_Node)
                    Modified_Graph.AddEdge(EI.GetSrcNId(), New_Node)
                    Modified_Graph.AddEdge(EI.GetDstNId(), New_Node)
                    heightVal[New_Node] = Src_dist + 0.5
                    New_Node += 1
                else:
                    Modified_Graph.AddEdge(EI.GetSrcNId(), EI.GetDstNId())
                    
        EH_computation = EH_pairs(Modified_Graph, heightVal)
        return EH_computation.get_SH0()     


def pick_node_compute_RW_dexPer(Graph, num_steps, sample_rate, NId, max_num_walks_per_node):
    dexPer0 = {}
    dexPer1 = {}

    nbrs = Graph.GetNI(NId).GetDeg()        
    ###########################################################################################################
    # Isolated node
    if nbrs == 0:
        return dexPer0, dexPer1
           
    ###########################################################################################################
    num_walks = min(int(np.ceil(nbrs * sample_rate)), max_num_walks_per_node)

    # print(num_walks)

    visited_nodes = snap.TIntV()
    visited_nodes.Add(NId)
    #print("NodeID: " + str(NId))

    node = NId
    
    for j in range(num_steps):
        NodeVec = snap.TIntV()
        snap.GetNodesAtHop(Graph, node, 1, NodeVec, False)
        new_node_id = np.random.choice(NodeVec, 1).item()
        if new_node_id not in RW_Nodes:
            RW_Nodes.Add(new_node_id)
        node = new_node_id   
  
    SubGraph = snap.GetSubGraph(Graph, visited_nodes)
    [BfsTree_size, _, BfsTree_depth] = snap.GetSubTreeSz(SubGraph, NId, True, True)

    ############################################################################################################

    # Isolated nodes
    if nbrs == 0:
        return dexPer0, dexPer1
    ############################################################################################# 
    else:
        heightVal = {}
        NodeVec = snap.TIntV()
        heightVal[NId] = 0
        
        for dist in range (1, BfsTree_depth + 1):
            snap.GetNodesAtHop(SubGraph, NId, dist, NodeVec, False)
            for item in NodeVec:
                heightVal[item] = dist
   
        Modified_Graph = snap.TUNGraph.New()
        
        New_Node = SubGraph.GetMxNId()
        for EI in SubGraph.Edges():
            Src_idx = EI.GetSrcNId()
            Dst_idx = EI.GetDstNId()
            
            Src_dist = heightVal.get(Src_idx, -1)
            Dst_dist = heightVal.get(Dst_idx, -1)
            
            if not Modified_Graph.IsNode(Src_idx):
                Modified_Graph.AddNode(Src_idx)
            if not Modified_Graph.IsNode(Dst_idx):
                Modified_Graph.AddNode(Dst_idx)
            
            
            if Src_dist >= 0 and Dst_dist >= 0:
                if Src_dist == Dst_dist:                       
                    #print(New_Nodes)
                    Modified_Graph.AddNode(New_Node)
                    Modified_Graph.AddEdge(EI.GetSrcNId(), New_Node)
                    Modified_Graph.AddEdge(EI.GetDstNId(), New_Node)
                    heightVal[New_Node] = Src_dist + 0.5
                    New_Node += 1
                else:
                    Modified_Graph.AddEdge(EI.GetSrcNId(), EI.GetDstNId())
                    
        EH_computation = EH_pairs(Modified_Graph, heightVal)

        return EH_computation.get_SH0(), EH_computation.get_EH1()    


def pick_node_compute_RW_dexPer_in_kring(Graph, num_steps, sample_rate, k_ring, NId):
    dexPer0 = {}
    dexPer1 = {}

    nbrs = Graph.GetNI(NId).GetDeg()        
    ###########################################################################################################
    # Isolated node
    if nbrs == 0:
        return SH0, EH1
           
    ###########################################################################################################
    num_walks = int(np.ceil(nbrs * sample_rate))

    visited_nodes = snap.TIntV()
    visited_nodes.Add(NId)
    #print("NodeID: " + str(NId))

    NodeVec = snap.TIntV()

    [BfsTree_size, _, BfsTree_depth] = snap.GetSubTreeSz(Graph, NId, True, True)
    
    for dist in range (1, min(k_ring + 1, BfsTree_depth + 1)):
        snap.GetNodesAtHop(Graph, NId, dist, NodeVec, False)
        for item in NodeVec:
            visited_nodes.Add(item)
    
    SubGraph = snap.GetSubGraph(Graph, visited_nodes)
    
    RW_Nodes = snap.TIntV()
    RW_Nodes.Add(NId)
    node = NId
    
    for j in range(num_steps):
        NodeVec = snap.TIntV()
        snap.GetNodesAtHop(Graph, node, 1, NodeVec, False)
        new_node_id = np.random.choice(NodeVec, 1).item()
        if new_node_id not in RW_Nodes:
            RW_Nodes.Add(new_node_id)
        node = new_node_id
  
    RW_Graph = snap.GetSubGraph(SubGraph, RW_Nodes)
    

    ############################################################################################################

    heightVal = {}
    NodeVec = snap.TIntV()
    heightVal[NId] = 0
    
    for dist in range (1, min(k_ring + 1, BfsTree_depth + 1)):
        snap.GetNodesAtHop(RW_Graph, NId, dist, NodeVec, False)
        for item in NodeVec:
            heightVal[item] = dist

    Modified_Graph = snap.TUNGraph.New()
    
    New_Node = RW_Graph.GetMxNId()
    for EI in RW_Graph.Edges():
        Src_idx = EI.GetSrcNId()
        Dst_idx = EI.GetDstNId()
        
        Src_dist = heightVal.get(Src_idx, -1)
        Dst_dist = heightVal.get(Dst_idx, -1)
        
        if not Modified_Graph.IsNode(Src_idx):
            Modified_Graph.AddNode(Src_idx)
        if not Modified_Graph.IsNode(Dst_idx):
            Modified_Graph.AddNode(Dst_idx)
        
        
        if Src_dist >= 0 and Dst_dist >= 0:
            if Src_dist == Dst_dist:                       
                #print(New_Nodes)
                Modified_Graph.AddNode(New_Node)
                Modified_Graph.AddEdge(EI.GetSrcNId(), New_Node)
                Modified_Graph.AddEdge(EI.GetDstNId(), New_Node)
                heightVal[New_Node] = Src_dist + 0.5
                New_Node += 1
            else:
                Modified_Graph.AddEdge(EI.GetSrcNId(), EI.GetDstNId())
                        
    EH_computation = EH_pairs(Modified_Graph, heightVal)
    return EH_computation.get_SH0(), EH_computation.get_EH1()    

 
def pick_node_compute_RW_dexPer_steps(Graph, num_steps, NId):
    dexPer0 = {}
    dexPer1 = {}

    nbrs = Graph.GetNI(NId).GetDeg()        
    ###########################################################################################################
    # Isolated node
    if nbrs == 0:
        return dexPer0, dexPer1
           
    ###########################################################################################################
    [BfsTree_size, _, BfsTree_depth] = snap.GetSubTreeSz(Graph, NId, True, True)
    
    RW_Nodes = snap.TIntV()
    RW_Nodes.Add(NId)
    node = NId

    for j in range(num_steps):
        NodeVec = snap.TIntV()
        snap.GetNodesAtHop(Graph, node, 1, NodeVec, False)
        new_node_id = np.random.choice(NodeVec, 1).item()
        if new_node_id not in RW_Nodes:
            RW_Nodes.Add(new_node_id)
        node = new_node_id   
  
    RW_Graph = snap.GetSubGraph(Graph, RW_Nodes)    
    ############################################################################################# 
    
    heightVal = {}
    NodeVec = snap.TIntV()
    heightVal[NId] = 0
    
    for dist in range (1, BfsTree_depth + 1):
        snap.GetNodesAtHop(RW_Graph, NId, dist, NodeVec, False)
        for item in NodeVec:
            heightVal[item] = dist

    Modified_Graph = snap.TUNGraph.New()
    
    New_Node = RW_Graph.GetMxNId()
    for EI in RW_Graph.Edges():
        Src_idx = EI.GetSrcNId()
        Dst_idx = EI.GetDstNId()
        
        Src_dist = heightVal.get(Src_idx, -1)
        Dst_dist = heightVal.get(Dst_idx, -1)
        
        if not Modified_Graph.IsNode(Src_idx):
            Modified_Graph.AddNode(Src_idx)
        if not Modified_Graph.IsNode(Dst_idx):
            Modified_Graph.AddNode(Dst_idx)
        
        
        if Src_dist >= 0 and Dst_dist >= 0:
            if Src_dist == Dst_dist:                       
                #print(New_Nodes)
                Modified_Graph.AddNode(New_Node)
                Modified_Graph.AddEdge(EI.GetSrcNId(), New_Node)
                Modified_Graph.AddEdge(EI.GetDstNId(), New_Node)
                heightVal[New_Node] = Src_dist + 0.5
                New_Node += 1
            else:
                Modified_Graph.AddEdge(EI.GetSrcNId(), EI.GetDstNId())
                        
    EH_computation = EH_pairs(Modified_Graph, heightVal)
    return EH_computation.get_SH0(), EH_computation.get_EH1()    


def pick_node_compute_RW_dexPer_steps_flyback(Graph, num_steps, NId, flyback_prob=0.15):
    dexPer0 = {}
    dexPer1 = {}

    nbrs = Graph.GetNI(NId).GetDeg()        
    ###########################################################################################################
    # Isolated node
    if nbrs == 0:
        return dexPer0, dexPer1
           
    ###########################################################################################################
    [BfsTree_size, _, BfsTree_depth] = snap.GetSubTreeSz(Graph, NId, True, True)
    
    RW_Nodes = snap.TIntV()
    RW_Nodes.Add(NId)
    # node = Graph.GetNI(NId)
    node = NId

    for j in range(num_steps):
        if np.random.uniform(size=1) <= flyback_prob:
            node = NId
            j -= 1
        else:
            NodeVec = snap.TIntV()
            snap.GetNodesAtHop(Graph, node, 1, NodeVec, False)
            new_node_id = np.random.choice(NodeVec, 1).item()
            if new_node_id not in RW_Nodes:
                RW_Nodes.Add(new_node_id)
            node = new_node_id   
  
    RW_Graph = snap.GetSubGraph(Graph, RW_Nodes)
    
    [BfsTree_size, _, BfsTree_depth] = snap.GetSubTreeSz(RW_Graph, NId, True, True)

    ############################################################################################################

    heightVal = {}
    NodeVec = snap.TIntV()
    heightVal[NId] = 0
    
    for dist in range (1, BfsTree_depth + 1):
        snap.GetNodesAtHop(RW_Graph, NId, dist, NodeVec, False)
        for item in NodeVec:
            heightVal[item] = dist

    Modified_Graph = snap.TUNGraph.New()
    
    New_Node = RW_Graph.GetMxNId()
    for EI in RW_Graph.Edges():
        Src_idx = EI.GetSrcNId()
        Dst_idx = EI.GetDstNId()
        
        Src_dist = heightVal.get(Src_idx, -1)
        Dst_dist = heightVal.get(Dst_idx, -1)
        
        if not Modified_Graph.IsNode(Src_idx):
            Modified_Graph.AddNode(Src_idx)
        if not Modified_Graph.IsNode(Dst_idx):
            Modified_Graph.AddNode(Dst_idx)
        
        
        if Src_dist >= 0 and Dst_dist >= 0:
            if Src_dist == Dst_dist:                       
                #print(New_Nodes)
                Modified_Graph.AddNode(New_Node)
                Modified_Graph.AddEdge(EI.GetSrcNId(), New_Node)
                Modified_Graph.AddEdge(EI.GetDstNId(), New_Node)
                heightVal[New_Node] = Src_dist + 0.5
                New_Node += 1
            else:
                Modified_Graph.AddEdge(EI.GetSrcNId(), EI.GetDstNId())
                        
    EH_computation = EH_pairs(Modified_Graph, heightVal)
    return EH_computation.get_SH0(), EH_computation.get_EH1()    
        
def dexPer_of_a_vertex(filename, k_ring, NId):
    Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
    dexPer0, dexPer1 = pick_node_compute_dexPer(Graph, k_ring, NId)
    return [dexPer0, dexPer1]

def RW_dexPer_of_a_vertex(filename, num_steps, rate, NId):
    Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
    SH0, EH1 = pick_node_compute_RW_dexPer(Graph, num_steps, rate, NId)
    return [dexPer0, dexPer1]

def RW_dexPer_of_a_vertex_in_kring(filename, num_steps, rate, k_ring, NId):
    Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
    dexPer0, dexPer1 = pick_node_compute_RW_dexPer_in_kring(Graph, num_steps, rate, k_ring, NId)
    return [dexPer0, dexPer1]

def RW_dexPer_of_a_vertex_steps(filename, num_steps, NId):
    Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
    dexPer0, dexPer1 = pick_node_compute_RW_dexPer_steps(Graph, num_steps, NId)
    return [dexPer0, dexPer1]

##############################################################################################
def dexPer_of_all_vertices(Graph, k_ring):
    dexPer0_all = [[]] * Graph.GetNodes()
    dexPer1_all = [[]] * Graph.GetNodes()
    i = 0
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        if k_ring == 1:
            EH_computation = EH_pairs_1_ring(Graph, NI_Id)
            dexPer0 = EH_computation.get_SH0()
            dexPer1 = EH_computation.get_EH1()
        else:
            dexPer0, dexPer1 = pick_node_compute_dexPer(Graph, k_ring, NI_Id)
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
        i += 1
    return dexPer0_all, dexPer1_all

def dexPer_of_all_vertices_dir(filename, k_ring):    
    Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
    dexPer0_all, dexPer1_all = dexPer_of_all_vertices(Graph, k_ring)
    return dexPer0_all, dexPer1_all

def dexPer0_of_all_vertices(Graph, k_ring):
    dexPer0_all = [[]] * Graph.GetNodes()
    i = 0
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        if k_ring == 1:
            EH_computation = EH_pairs_1_ring(Graph, NI_Id)
            dexPer0 = EH_computation.get_SH0()
        else:
            dexPer0 = pick_node_compute_dexPer0(Graph, k_ring, NI_Id)
        dexPer0_all[i] = dexPer0
        i += 1
    return dexPer0_all

def dexPer_of_vertices_with_large_degree(Graph, k_ring, num_nodes):
    num_nodes = min(Graph.GetNodes(), num_nodes)
    dexPer0_all = [[]] * num_nodes
    dexPer1_all = [[]] * num_nodes
    i = 0

    degree_seq = snap.TIntV()
    snap.GetDegSeqV(Graph, degree_seq)
    degree_seq.Sort()    
    degs = [item for item in degree_seq]
    threshold = degs[-num_nodes]
    for NI in Graph.Nodes():
        if NI.GetDeg() >= threshold and i < num_nodes:
            NI_Id = NI.GetId()
            if k_ring == 1:
                EH_computation = EH_pairs_1_ring(Graph, NI_Id)
                dexPer0 = EH_computation.get_SH0()
                dexPer1 = EH_computation.get_EH1()
            else:
                dexPer0, dexPer1 = pick_node_compute_dexPer(Graph, k_ring, NI_Id)
            dexPer0_all[i] = dexPer0
            dexPer1_all[i] = dexPer1
            i += 1

    return dexPer0_all, dexPer1_all

def dexPer_of_vertices_with_large_degree_percentage(Graph, k_ring, top_percentage):
    num_nodes = int(Graph.GetNodes() * top_percentage)
    dexPer0_all = [[]] * num_nodes
    dexPer1_all = [[]] * num_nodes
    i = 0

    degree_seq = snap.TIntV()
    snap.GetDegSeqV(Graph, degree_seq)
    degree_seq.Sort()    
    degs = [item for item in degree_seq]
    threshold = degs[-num_nodes]
    for NI in Graph.Nodes():
        if NI.GetDeg() >= threshold and i < num_nodes:
            NI_Id = NI.GetId()
            if k_ring == 1:
                EH_computation = EH_pairs_1_ring(Graph, NI_Id)
                dexPer0 = EH_computation.get_SH0()
                dexPer1 = EH_computation.get_EH1()
            else:
                dexPer0, dexPer1 = pick_node_compute_dexPer(Graph, k_ring, NI_Id)
            dexPer0_all[i] = dexPer0
            dexPer1_all[i] = dexPer1
            i += 1

    return dexPer0_all, dexPer1_all

def dexPer_of_vertices_with_large_eigenvector_centrality(Graph, k_ring, num_nodes):
    num_nodes = min(Graph.GetNodes(), num_nodes)
    dexPer0_all = [[]] * num_nodes
    dexPer1_all = [[]] * num_nodes
    i = 0
    eps = 1e-7
    eigen_seq = snap.TIntFltH()
    snap.GetEigenVectorCentr(Graph, eigen_seq)    
    eigens = [eigen_seq[item] for item in eigen_seq]
    eigens = sorted(eigens)
    threshold = eigens[-num_nodes]
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        if eigen_seq[NI_Id] >= (threshold - eps) and i < num_nodes:
            if k_ring == 1:
                EH_computation = EH_pairs_1_ring(Graph, NI_Id)
                dexPer0 = EH_computation.get_SH0()
                dexPer1 = EH_computation.get_EH1()
            else:
                dexPer0, dexPer1 = pick_node_compute_EH(Graph, k_ring, NI_Id)
            dexPer0_all[i] = dexPer0
            dexPer1_all[i] = dexPer1
            i += 1

    return dexPer0_all, dexPer1_all

def dexPer_of_vertices_with_large_eigenvector_centrality_percentage(Graph, k_ring, top_percentage):
    num_nodes = int(Graph.GetNodes() * top_percentage)
    dexPer0_all = [[]] * num_nodes
    dexPer1_all = [[]] * num_nodes
    i = 0
    eps = 1e-7
    eigen_seq = snap.TIntFltH()
    snap.GetEigenVectorCentr(Graph, eigen_seq)    
    eigens = [eigen_seq[item] for item in eigen_seq]
    eigens = sorted(eigens)
    threshold = eigens[-num_nodes]
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        if eigen_seq[NI_Id] >= (threshold - eps) and i < num_nodes:
            if k_ring == 1:
                EH_computation = EH_pairs_1_ring(Graph, NI_Id)
                dexPer0 = EH_computation.get_SH0()
                dexPer1 = EH_computation.get_EH1()
            else:
                dexPer0, dexPer1 = pick_node_compute_EH(Graph, k_ring, NI_Id)
            dexPer0_all[i] = dexPer0
            dexPer1_all[i] = dexPer1
            i += 1

    return dexPer0_all, dexPer1_all

def dexPer_of_all_vertices_sample(Graph, k_ring, sample_rate):
    dexPer0_all = []
    dexPer1_all = []
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    for i in range(int(np.ceil(Graph.GetNodes() * sample_rate))):
        dexPer0, dexPer1 = pick_node_compute_dexPer(Graph, k_ring, Graph.GetRndNId(Rnd))
        dexPer0_all.append(dexPer0)
        dexPer1_all.append(dexPer1)
    return dexPer0_all, dexPer1_all

def RW_dexPer_of_all_vertices(Graph, num_steps, rate):
    dexPer0_all = [[]] * Graph.GetNodes()
    dexPer1_all = [[]] * Graph.GetNodes()
    i = 0
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer(Graph, num_steps, rate, NI_Id)
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
        i += 1
    return dexPer0_all, dexPer1_all

def RW_dexPer_of_all_vertices_in_kring(Graph, num_steps, rate, k_ring):
    dexPer0_all = []
    dexPer1_all = []
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer_in_kring(Graph, num_steps, rate, k_ring, NI_Id)
        dexPer0_all.append(dexPer0)
        dexPer1_all.append(dexPer1)
    return dexPer0_all, dexPer1_all

def RW_dexPer_of_all_vertices_steps(Graph, num_steps):
    dexPer0_all = [[]] * Graph.GetNodes()
    dexPer1_all = [[]] * Graph.GetNodes()
    i = 0
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer_steps(Graph, num_steps, NI_Id)
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
        i += 1
    return dexPer0_all, dexPer1_all

def RW_dexPer_of_all_vertices_steps_flyback(Graph, num_steps, flyback_prob=0.15):
    dexPer0_all = [[]] * Graph.GetNodes()
    dexPer1_all = [[]] * Graph.GetNodes()
    i = 0
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer_steps_flyback(Graph, num_steps, NI_Id, flyback_prob=flyback_prob)
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
        i += 1
    return dexPer0_all, dexPer1_all

def RW_dexPer_of_all_vertices_in_kring_sample(Graph, num_steps, rate, k_ring, sample_rate):
    sample_size = int(np.ceil(Graph.GetNodes() * sample_rate))
    dexPer0_all = [[]] * sample_size
    dexPer1_all = [[]] * sample_size
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    for i in range(sample_size):
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer_in_kring(Graph, num_steps, rate, k_ring, Graph.GetRndNId(Rnd))
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
    return dexPer0_all, dexPer1_all

def RW_dexPer_of_all_vertices_in_kring_fixed_samplesize(Graph, num_steps, rate, k_ring, sample_size):
    dexPer0_all = [[]] * sample_size
    dexPer1_all = [[]] * sample_size
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    for i in range(sample_size):
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer_in_kring(Graph, num_steps, rate, k_ring, Graph.GetRndNId(Rnd))
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
    return dexPer0_all, dexPer1_all


def RW_dexPer_of_all_vertices_steps_sample(Graph, num_steps, sample_rate):
    sample_size = int(np.ceil(Graph.GetNodes() * sample_rate))
    dexPer0_all = [[]] * sample_size
    dexPer1_all = [[]] * sample_size
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    for i in range(sample_size):
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer_steps(Graph, num_steps, Graph.GetRndNId(Rnd))
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
    return dexPer0_all, dexPer1_all

def RW_dexPer_of_all_vertices_steps_fixed_samplesize(Graph, num_steps, sample_size):
    dexPer0_all = [[]] * sample_size
    dexPer1_all = [[]] * sample_size
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    for i in range(sample_size):
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer_steps(Graph, num_steps, Graph.GetRndNId(Rnd))
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
    return dexPer0_all, dexPer1_all

def RW_dexPer_of_all_vertices_steps_fixed_samplesize_flyback(Graph, num_steps, sample_size, flyback_prob):
    dexPer0_all = [[]] * sample_size
    dexPer1_all = [[]] * sample_size
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    for i in range(sample_size):
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer_steps_flyback(Graph, num_steps, Graph.GetRndNId(Rnd), flyback_prob=flyback_prob)
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
    return dexPer0_all, dexPer1_all

def dexPer_of_all_vertices_fixed_samplesize(Graph, k_ring, sample_size):
    dexPer0_all = [[]] * sample_size
    dexPer1_all = [[]] * sample_size
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    for i in range(sample_size):
        dexPer0, dexPer1 = pick_node_compute_dexPer(Graph, k_ring, Graph.GetRndNId(Rnd))
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
    return dexPer0_all, dexPer1_all

def RW_dexPer_of_all_vertices_steps_sample_node_rate(Graph, num_steps, sample_rate, node_rate, max_num_walks_per_node):
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    sample_size = int(np.ceil(Graph.GetNodes() * sample_rate))
    dexPer0_all = [[]] * sample_size
    dexPer1_all = [[]] * sample_size
    for i in range(sample_size):
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer(Graph, num_steps, node_rate, Graph.GetRndNId(Rnd), max_num_walks_per_node)
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
    return dexPer0_all, dexPer1_all

def RW_dexPer_of_all_vertices_steps_fixed_samplesize_node_rate(Graph, num_steps, sample_size, node_rate, max_num_walks_per_node):
    dexPer0_all = [[]] * sample_size
    dexPer1_all = [[]] * sample_size
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    for i in range(sample_size):
        dexPer0, dexPer1 = pick_node_compute_RW_dexPer(Graph, num_steps, node_rate, Graph.GetRndNId(Rnd), max_num_walks_per_node)
        dexPer0_all[i] = dexPer0
        dexPer1_all[i] = dexPer1
    return dexPer0_all, dexPer1_all

def output_dexPer_of_a_vertex(Graph, k_ring, D0_output, D1_output, NId):
    dexPer0, dexPer1 = pick_node_compute_dexPer(Graph, k_ring, NId)

    D0_k_ring_extended = open(D0_output, 'a')
    D1_k_ring_extended = open(D1_output, 'a')
    
    D0_k_ring_extended.write("%d \t %s\n" % (NId, dexPer0))
    dexPer0 = None
    D0_k_ring_extended.close()
    
    D1_k_ring_extended.write("%d \t %s\n" % (NId, dexPer1))
    dexPer1 = None
    D1_k_ring_extended.close()
        
def dexPer_of_all_vertices_parallel(filename, k_ring):    
    Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
    Nodes = []
    
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        Nodes.append(NI_Id)
    
    pool = Pool(8)
    func = partial(dexPer_of_a_vertex, filename, k_ring)
    dexPer_all = pool.map(func, Nodes)
    pool.close()
    pool.join()
    
    return [ item[0] for item in dexPer_all], [ item[1] for item in dexPer_all]

def RW_dexPer_of_all_vertices_parallel(filename, num_steps, rate):    
    Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
    Nodes = []
    
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        Nodes.append(NI_Id)
    
    pool = Pool(8)
    func = partial(RW_dexPer_of_a_vertex, filename, num_steps, rate)
    dexPer_all = pool.map(func, Nodes)
    pool.close()
    pool.join()
    
    return [ item[0] for item in dexPer_all], [ item[1] for item in dexPer_all]

def RW_dexPer_of_all_vertices_in_kring_parallel(filename, num_steps, rate, k_ring):    
    Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
    Nodes = []
    
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        Nodes.append(NI_Id)
    
    pool = Pool(8)
    func = partial(RW_dexPer_of_a_vertex_in_kring, filename, num_steps, rate, k_ring)
    dexPer_all = pool.map(func, Nodes)
    pool.close()
    pool.join()
    
    return [ item[0] for item in dexPer_all], [ item[1] for item in dexPer_all]

def RW_dexPer_of_all_vertices_steps_parallel(filename, num_steps):    
    Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
    Nodes = []
    
    for NI in Graph.Nodes():
        NI_Id = NI.GetId()
        Nodes.append(NI_Id)
    
    pool = Pool(8)
    func = partial(RW_dexPer_of_a_vertex_steps, filename, num_steps)
    dexPer_all = pool.map(func, Nodes)
    pool.close()
    pool.join()
    
    return [ item[0] for item in dexPer_all], [ item[1] for item in dexPer_all]





