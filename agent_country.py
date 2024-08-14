from hypergraph_generator import generate_hypergraph

import numpy as np
import matplotlib.pyplot as plt
import csv
import multiprocessing
from multiprocessing.pool import Pool


dir_to_save = 'data/'

def run(graph_args, beta = 0.2, q=0.4, endtime=10000,log_freq=1000):
    H = generate_hypergraph(graph_args['name'], graph_args['args'])
    number_of_edges=len(H.edges)

    ### Egyszerű terjedés A=-1, B=1 vélemény fele-fele inicaializálással: ###

    #inicializáljuk 
    N=len(H.nodes) # csúcsok száma
    opinion=-1*np.ones(N) # vélemény vektor
    init_B=np.random.choice(H.nodes,int(N/2),replace=False) #random a fele B véleményre
    opinion[init_B] = np.ones(len(init_B))

    #list of logs
    sum_A_log=[np.sum(opinion == -1)]
    sum_B_log=[np.sum(opinion == 1)]
    
    binned_opinions_home = []
    binned_opinions_wp = []
    binned_edge_sizes_wp=[]
    
    
    
    for i in range(endtime):
        act_node = np.random.choice(H.nodes,1)[0]
        try:
            diff_opinions = np.array([np.sum(opinion[edge]!=opinion[act_node]) for edge in H.edgelists[act_node]])

        except:
            print(act_node,H.edgelists[act_node])
            break

        edge_sizes = np.array([len(edge) for edge in H.edgelists])

        #terjedés : h élek m élek különböző vélemények aránya a háztartás és munkahely élekben: 
        #változás valószínűsége: beta * (h + 0.5 m) / 1.5
        p = np.random.random()
        h=diff_opinions[0]/edge_sizes[0]
        m= diff_opinions[1]/edge_sizes[1]
        if p < beta * np.sum(h + 0.5 * m)/1.5:
            opinion[act_node]=-1*opinion[act_node]

        elif m > 0.5 and np.random.random() < q * (m - 0.5):
            #print('edgelist',H.edgelists[act_node])
            sum_by_edge = np.array([np.sum(opinion[edge]==opinion[act_node]) for edge in H.edges])
            edge_to_move = np.random.choice(np.arange(int(number_of_edges/2))[sum_by_edge[int(number_of_edges/2):]>0.5],1)[0] + int(number_of_edges/2)
            H.edgelists[act_node][1].remove(act_node)
            H.edgelists[act_node].remove(H.edgelists[act_node][1])
            H.edges[edge_to_move].append(act_node)
            H.edgelists[act_node].append(H.edges[edge_to_move])
            #print('act node',act_node)
            #print('edge_to_move', H.edges[edge_to_move])
            #print('edgelist',H.edgelists[act_node])
            #print('edges',H.edges)
        # distribution of workplace edge sizes
        
        
        if i%log_freq==0:
            sum_A_log.append(np.sum(opinion == -1))
            sum_B_log.append(np.sum(opinion == 1))

            # opinion sums
            opinion_sums_home=[np.sum(opinion[edge]==-1) for edge in H.edges[:int(number_of_edges/2)]]
            z=np.histogram(opinion_sums_home, bins=np.arange(6))
            binned_opinions_home.append(z[0])

            opinion_sums_wp=[np.sum(opinion[edge]==-1) for edge in H.edges[int(number_of_edges/2):]]
            z=np.histogram(opinion_sums_wp, bins=np.arange(15))
            binned_opinions_wp.append(z[0])

            # workplace sizes
            edge_sizes_wp=[len(edge) for edge in H.edges[int(number_of_edges/2):]]
            z=np.histogram(edge_sizes_wp, bins=np.arange(15))
            binned_edge_sizes_wp.append(z[0])
        
    return sum_A_log,binned_opinions_home,binned_opinions_wp,binned_edge_sizes_wp


def run_parallel(graph_args, betas, q, endtime,iterations=50, processes=100,log_freq=1000, save=False):
    pool = multiprocessing.Pool(processes=processes)
    results = []
    for beta_i, beta in enumerate(betas):
        results.append([])
        for it in range(iterations):
            results[beta_i].append(pool.apply_async(run, args=(graph_args,beta,q,endtime, log_freq)))
        
        
    pool.close()
    pool.join()
    
    
    row_names = ['sum_opinion_A','binned_opinions_home','binned_opinions_wp','binned_edge_sizes_wp']
    for beta_i, beta in enumerate(betas):
        results_save = []
        for it in range(iterations):
            r_act = results[beta_i][it].get()
            for i in range(len(r_act)):
                results_save.append([it,row_names[i]] + list(r_act[i]))
         
        
        if save:
            # field names 
            cols=['sample id', 'datatype']
            fields = cols + list(np.arange(len(r_act[0])))
            beta_str=str(beta).replace('.', '')
            q_str=str(q).replace('.', '')
            with open(dir_to_save+f'{save}_beta_{beta_str}_q_{q_str}.csv', 'w') as f:

                # using csv.writer method from CSV package
                write = csv.writer(f)

                write.writerow(fields)
                write.writerows(results_save)
        
        
        
    return results_save