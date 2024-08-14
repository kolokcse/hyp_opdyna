import numpy as np
import random as rnd
import math
from scipy.stats import poisson
from scipy.stats import pareto
from scipy.stats import gamma
import bisect
from hypergraph import Hypergraph

'''
The hypergraph consists of individuals as nodes and contacts between them as hyperedges.
'''

def generate_hypergraph(name,args):
    if name=="d_regular":
        return gen_d_regular(args)
    
    
def gen_d_regular(args):
    nodes=np.arange(args['n'])
    hypergraph=Hypergraph(nodes)
    for i in range(args['d']):
        list_of_edges = gen_one_layer_edges(args)
        hypergraph.add_edges(list_of_edges)
        
    return hypergraph
    
    
    
def gen_one_layer_edges(args):
    
    """
    Generate random family hyperedges from hyperedge distribution.
    parameter:
    -------
    _list: dict
    dictionary of the nodes
    
    """
    list_of_edges = []
    if args['distribution']=='uniform':
        nodes=np.arange(args['n'])
        np.random.shuffle(nodes)
        # Split the array into equal parts
        list_of_edges = np.array_split(nodes, int(args['n']/args['edge size']))
        
    return list_of_edges
    