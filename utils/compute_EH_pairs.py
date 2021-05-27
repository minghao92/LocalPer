#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    A python implementation of persistence-pair-finding algorithm described in 
    Dey, Tamal K., Dayu Shi, and Yusu Wang. "Comparing Graphs via Persistence Distortion." 
    31st International Symposium on Computational Geometry (SoCG 2015). 
    Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik, 2015.
"""

import math
from collections import defaultdict

class EH_pairs():
    def __init__(self, G, heightVal): # snap graph + (nodeId, value) dict
        self.Graph = G
        self.heightVal = heightVal

        self.Sorted_Vertices = [key for key, value in sorted(self.heightVal.items(), key=lambda item: (item[1], item[0]))]    
        self.Sorted_Vertices_len = len(self.Sorted_Vertices)
        
        self.Index = {} # this dict records the order of all sorted vertices; will be used in vertex comparison
        
        count = 0
        for a in self.Sorted_Vertices:
            self.Index[a] = count
            count += 1
        
        self.usList = {}
        self.dsList = {}
        self.ks = {}
        
        for NI in self.Graph.Nodes():
            i = NI.GetId()
            self.usList[i] = []
            self.dsList[i] = []
            self.ks[i] = 0
            f = self.heightVal.get(i)
            for Id in NI.GetOutEdges():
                g = self.heightVal.get(Id)
                if f < g:
                    self.usList[i].append(Id)
                elif f > g:
                    self.dsList[i].append(Id)
                elif i < Id:
                    self.usList[i].append(Id)
                else:
                    self.dsList[i].append(Id)
         
        ######################################### compute H0 pairs ######################################################            
        self.SH0 = defaultdict(int)

        #initialize the union find set
        ufList = {}
    
        for NI in self.Graph.Nodes():
            NId = NI.GetId()
            ufList[NId] = NId
    
        self.SH0[self.heightVal.get(self.Sorted_Vertices[0]), self.heightVal.get(self.Sorted_Vertices[-1]), 0] = 1
    
        for i in range(1, self.Sorted_Vertices_len):
            k = self.Sorted_Vertices[self.Sorted_Vertices_len - 1 - i]
            if len(self.usList[k]) == 1:
                t = self.usList[k][0]
                while ufList[t] != t:
                    t = ufList[t]
                ufList[k] = t
    
            elif len(self.usList[k]) > 1:
                for m in range (len(self.usList[k])):
                    #j is the upsaddle adjacent vertex to k
                    j = self.usList[k][m]
    
                    while ufList[j] != j:
                        j = ufList[j]
    
                    #first edge in upsaddle
                    if ufList[k] == k:
                        ufList[k] = j
                        continue
    
                    #after first edge in upsaddle
                    l = k
                    while ufList[l] != l:
                        l = ufList[l]
    
                    if j != l:
                        self.ks[k] += 1
                        if self.heightVal.get(j) > self.heightVal.get(l) or math.fabs(self.heightVal.get(j) - self.heightVal.get(l)) < 1e-9 and j > l:    
                            self.SH0[self.heightVal.get(k), self.heightVal.get(l), 0] += 1
                            ufList[k] = j
                            ufList[l] = j
    
                        else:
                            self.SH0[self.heightVal.get(k), self.heightVal.get(j), 0] += 1
                            ufList[j] = l                     
                    
    



    def get_SH0(self):
        return self.SH0

    ######################################### compute H1 pairs ######################################################        
    
    def get_EH1(self):
        EH1 = defaultdict(int)

        for i in range(self.Sorted_Vertices_len - 1):
            s = self.Sorted_Vertices[i]
            # unhandled up-branches
            c = len(self.usList[s]) - self.ks[s] - 1
            if c <= 0:
                continue
    
            # union find set
            uf = {}
            uf[s] = s
    
    
            # add representitives to the union find set
            for k in range (len(self.usList[s])):
                uf[self.usList[s][k]] = self.usList[s][k]
    
            
            for l in range (i+1, self.Sorted_Vertices_len):
                if c > 0:
                    o = self.Sorted_Vertices[l]
                    if len(self.dsList[o]) == 1:
                        if self.Index[self.dsList[o][0]] < self.Index[s]:
                            uf[o] = o
    
                        else:
                            r = self.dsList[o][0]
                            if r == s:
                                for p in range (len(self.usList[s])):
                                    if o == self.usList[s][p]:
                                        r = self.usList[s][p]
                                        break
                            while uf[r] != r:
                                r = uf[r]
                            uf[o] = r
    
    
                    elif len(self.dsList[o]) > 1:
                        r1 = 0
                        r2 = 0
                        temp_i = 0
    
                        for dummy_i in range (len(self.dsList[o])):
                            r1 = self.dsList[o][dummy_i]
                            if self.Index[r1] >= self.Index[s]:
                                break
                            temp_i += 1
        
                        if temp_i < len(self.dsList[o]):
    
                            if r1 == s:
                                for p in range (len(self.usList[s])):
                                    if o == self.usList[s][p]:
                                        r1 = self.usList[s][p]
                                        break
    
                            while uf[r1] != r1:
                                r1 = uf[r1]
                            uf[o] = r1
    
                            for dummy_i in range (temp_i+1, len(self.dsList[o])):
                                r2 = self.dsList[o][dummy_i]
    
                                if self.Index[r2] < self.Index[s]:
                                    r1 = uf[o]
                                    continue
                                else:
                                    if r2 == s:
                                        for p in range (len(self.usList[s])):
                                            if o == self.usList[s][p]:
                                                r2 = self.usList[s][p]
                                                break
    
                                    while uf[r2] != r2:
                                        r2 = uf[r2]
    
                                if r1 in self.usList[s] and r2 in self.usList[s]:
                                    if r1 != r2:
                                        if r1 < r2:
                                            uf[o] = r1
                                            uf[r2] = r1
                                        else:
                                            uf[o] = r2
                                            uf[r1] = r2
                                            r1 = r2

                                        EH1[self.heightVal.get(s), self.heightVal.get(o), 1] += 1
                                        c -= 1
                                        if c <= 0:
                                            break
                                elif r1 in self.usList[s]:
                                    uf[r2] = r1
                                    uf[self.dsList[o][dummy_i]] = r1
                                    uf[o] = r1
                                elif r2 in self.usList[s]:
                                    uf[r1] = r2
                                    uf[self.dsList[o][dummy_i]] = r2
                                    uf[o] = r2
                                    r1 = r2
                                elif r1 < r2:
                                    uf[r2] = r1
                                    uf[self.dsList[o][dummy_i]] = r1
                                    uf[o] = r1
                                else:
                                    uf[r1] = r2
                                    uf[self.dsList[o][dummy_i]] = r2
                                    uf[o] = r2
                                    r1 = r2
    
                        else:
                            uf[o] = o
    
                    else:
                        uf[o] = o
    
    
    
        #print(EH1)
        #print("Finish EH1")
        
        return EH1
        


















