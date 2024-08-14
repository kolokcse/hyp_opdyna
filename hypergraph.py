import numpy as np
import random as rnd
from itertools import combinations
from itertools import permutations 
import bisect
import networkx as nx
import math


class Hypergraph:
    """
    A hypergraph object consisting of nodes and contacts between persons as hyperedges. 
    Each node points to an edgelist which stands from the edges which the node is contained by.
    """
    
    def __init__(self, nodes=[], edges=None):
        self.nodes = nodes
        self.edgelists = [[] for _ in range(len(self.nodes))]
        self.edgelists_indices = [[] for _ in range(len(self.nodes))]
        self.edges = []
        if edges != None:
            self.add_edges(edges)
            
            
    def add_edges(self,edges):
        '''
        Adds edges as a forms of [[a,b,c],[c,d,e],[a,c]] where a,b,c,d,e are nodes.
        '''
        ind=len(self.edges)
        #self.edges+=edges
        for edge in edges:
            edge_to_add=list(edge)
            self.edges+=[edge_to_add]
            for node in edge:
                try:
                    self.edgelists[node].append(edge_to_add)
                    self.edgelists_indices[node].append(ind)
                except:
                    print(node)
            ind+=1
        