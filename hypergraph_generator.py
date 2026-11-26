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
    if name=="2layer":
        return gen_2layer(args)


def gen_d_regular(args):
    nodes=np.arange(args['n'])
    hypergraph=Hypergraph(nodes)
    for i in range(args['d']):
        list_of_edges = gen_one_layer_edges(args)
        hypergraph.add_edges(list_of_edges)
        
    return hypergraph

def gen_2layer(args):
    nodes=np.arange(args['n'])
    hypergraph=Hypergraph(nodes)
    
    args['edge size']=args['household size']
    list_of_edges = gen_one_layer_edges(args)
    hypergraph.add_edges(list_of_edges)
    
    args['edge size']=args['workplace size']
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
    elif args["distribution"] == "normal":

        remaining_nodes = np.random.permutation(args['n'])
        edges = []

        mean = args["edge size"]
        std = args["std_edge_size"]

        idx = 0

        while idx < args['n']:

            size = int(np.round(np.random.normal(mean, std)))

            # enforce size > 1
            size = max(2, size)

            # last edge gets remaining nodes
            size = min(size, args['n'] - idx)

            edges.append(remaining_nodes[idx:idx + size])
            idx += size

        return edges
    
    elif args["distribution"] == "binomial":
        n = args["n"]
        remaining_nodes = np.random.permutation(n)
        edges = []

        n_bin = args["binomial_n"]
        mean_size = args["edge size"]

        p = mean_size / n_bin

        if not (0 < p <= 1):
            raise ValueError(
                f"mean_edge_size={mean_size} must be <= binomial_n={n_bin}"
            )

        idx = 0

        while idx < n:

            size = np.random.binomial(n_bin, p)

            # enforce size > 1
            size = max(2, size)

            # do not exceed remaining nodes
            size = min(size, n - idx)

            edges.append(remaining_nodes[idx:idx + size])
            idx += size

        return edges

    else:
        raise ValueError(
            f"Unknown distribution: {args['distribution']}"
        )

    return list_of_edges
    