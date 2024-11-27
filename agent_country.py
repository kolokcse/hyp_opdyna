from hypergraph_generator import generate_hypergraph

import numpy as np
import matplotlib.pyplot as plt
import csv
import multiprocessing
from multiprocessing.pool import Pool


dir_to_save = 'data/majority_move_nonlin/'

def is_connected(H):
    
    V = list(set(H.edgelists[0][0] + H.edgelists[0][1]))
    seen = np.zeros(len(H.nodes))
    seen[V] = 1
    while V:
        z = V.pop()
        neighs = H.edgelists[z][0] + H.edgelists[z][1]
        for niegh in neighs:
            if seen[niegh] != 1:
                V.append(niegh)
                seen[niegh] = 1
                
    return np.sum(seen) == len(H.nodes)

def run(graph_args, beta = 0.2, q=0.4, r1=0.5,r2=0.5,lambda_=0.5, endtime=10000, move='majority',log_freq=1000, seed=0, spread_type='het'):
    
    np.random.seed(seed)
    H = generate_hypergraph(graph_args['name'], graph_args['args'])
    number_of_edges=len(H.edges)
    
    ### Egyszerű terjedés A=-1, B=1 vélemény fele-fele inicaializálással: ###

    #inicializáljuk 
    N=len(H.nodes) # csúcsok száma
    opinion=-1*np.ones(N) # vélemény vektor
    init_B=np.random.choice(H.nodes,int(N/2),replace=False) #random a fele B véleményre
    opinion[init_B] = np.ones(len(init_B))
    
    ### ToDo
    #beta_het = np.random.choice([beta,beta+0.2], N, p=[0.6,0.4])

    #list of logs
    sum_A_log=[np.sum(opinion == -1)]
    sum_B_log=[np.sum(opinion == 1)]
    
    binned_opinions_home = []
    binned_opinions_wp = []
    binned_opinions_wp_pop_sized = []
    binned_edge_sizes_wp=[]
    
    move_count = []
    op_change_count = []
    
    move_count_act=0
    op_change_count_act =0
    
    for i in range(endtime):
        
        act_node = np.random.choice(H.nodes,1)[0]
        try:
            id_opinions = np.array([np.sum(opinion[edge]==opinion[act_node]) for edge in H.edgelists[act_node]])

        except:
            print(act_node,H.edgelists[act_node])
            break

        edge_sizes = np.array([len(edge) for edge in H.edgelists[act_node]])

        #terjedés : h, m a különböző véleményen lévők aránya a háztartás és munkahely élekben: 
        #változás valószínűsége: beta * (h + w* m) / (1+w)
        #print(edge_sizes)
        #print(id_opinions)
        h = id_opinions[0]/edge_sizes[0]
        w = id_opinions[1]/edge_sizes[1]
        a = (h + lambda_ * w)/(1+lambda_)
        #print(h,w,a,r1)
        if  a < r1  and np.random.random() < beta * (r1 - a):
            opinion[act_node]=-1*opinion[act_node]
            op_change_count_act +=1

        elif w < r2 and np.random.random() < q * (r2-w):
            #print('edgelist',H.edgelists[act_node])
            sum_by_edge = np.array([np.sum(opinion[edge]==opinion[act_node]) for edge in H.edges])
            if move=='majority':
                 edge_to_move = np.random.choice(np.arange(int(number_of_edges/2))[sum_by_edge[int(number_of_edges/2):]>0.5],1)[0] + int(number_of_edges/2)
            elif move=='proportional':
                 edge_to_move = np.random.choice(np.arange(int(number_of_edges/2)),1,p=sum_by_edge[int(number_of_edges/2):]/np.sum(sum_by_edge[int(number_of_edges/2):]))[0] + int(number_of_edges/2)
            
           
            H.edgelists[act_node][1].remove(act_node)
            H.edgelists[act_node].remove(H.edgelists[act_node][1])
            H.edges[edge_to_move].append(act_node)
            H.edgelists[act_node].append(H.edges[edge_to_move])
            move_count_act+=1
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
            z=np.histogram(opinion_sums_home, bins=np.arange(7))
            binned_opinions_home.append(z[0])

            opinion_rates_wp=[np.sum(opinion[edge]==-1)/len(edge) for edge in H.edges[int(number_of_edges/2):]]
            z=np.histogram(opinion_rates_wp, bins=np.arange(11)/10)
            binned_opinions_wp.append(z[0])
            
            popsized=[]
            bins = np.arange(11)/10
            for i in range(10):
                popsized.append(0)
                for edge in H.edges[int(number_of_edges/2):]:
                    if bins[i]<=np.sum(opinion[edge]==-1)/len(edge)<bins[i+1]:
                        popsized[-1] += len(edge)
                    elif i==9 and np.sum(opinion[edge]==-1)/len(edge)==bins[i+1]:
                        popsized[-1] += len(edge)
            binned_opinions_wp_pop_sized.append(np.array(popsized))

            # workplace sizes
            edge_sizes_wp=[len(edge) for edge in H.edges[int(number_of_edges/2):]]
            z=np.histogram(edge_sizes_wp, bins=np.arange(15))
            binned_edge_sizes_wp.append(z[0])
            
            #move,  change count
            move_count.append(move_count_act)
            op_change_count.append(op_change_count_act)
            move_count_act = 0
            op_change_count_act = 0
        
    return sum_A_log,binned_opinions_home,binned_opinions_wp,binned_edge_sizes_wp, binned_opinions_wp_pop_sized, move_count,op_change_count, is_connected(H)

def run_parallel(graph_args, betas, qs, r1s=[0.5],r2s=[0.5],lambdas=[0.5], endtime=10000, move='majority',iterations=50, processes=100,log_freq=1000, save=False):
    pool = multiprocessing.Pool(processes=processes)
    results = []

    for beta_i, beta in enumerate(betas):
        results.append([])
        for q_i, q in enumerate(qs):
            for r1_i, r1 in enumerate(r1s):
                for r2_i, r2 in enumerate(r2s):
                    for lambda_i, lambda_ in enumerate(lambdas):
                        results[beta_i].append([])
                        maxqr = max(q_i,r1_i,r2_i,lambda_i)
                        for it in range(iterations):
                            results[beta_i][maxqr].append(pool.apply_async(run, args=(graph_args,beta,q,r1,r1,lambda_,endtime, move, log_freq, it)))
        
        
    pool.close()
    pool.join()
    
    
    row_names = ['sum_opinion_A','binned_opinions_home','binned_opinions_wp','binned_edge_sizes_wp','binned_opinions_wp_pop_sized', 'move_count', 'op_change_count','is_connected']
    
    data=[]
    for beta_i, beta in enumerate(betas):
        data.append([])
        for q_i, q in enumerate(qs):
            for r1_i, r1 in enumerate(r1s):
                for r2_i, r2 in enumerate(r2s):
                    for lambda_i, lambda_ in enumerate(lambdas):
                        maxqr = max(q_i,r1_i,r2_i,lambda_i)
                        data[beta_i].append([])
                        results_save = []
                        for it in range(iterations):
                            data[beta_i][maxqr].append({})
                            r_act = results[beta_i][maxqr][it].get()
                            for i in range(len(r_act)):
                                data[beta_i][maxqr][it][row_names[i]] = r_act[i]
                                if len(r_act)-3>i>0:
                                    list_to_save = [" ".join(map(str, act)) for act in r_act[i]]
                                    results_save.append([it,row_names[i]] + list_to_save)
                                elif i==len(r_act)-1:
                                    results_save.append([it,row_names[i],r_act])
                                else:
                                    results_save.append([it,row_names[i]] + list(r_act[i]))


                    if save:
                        # field names 
                        cols=['sample id', 'datatype']
                        fields = cols + list(np.arange(len(r_act[0])))
                        beta_str=str(np.round(beta, 2)).replace('.', '')
                        q_str=str(np.round(q, 1)).replace('.', '')
                        r_str=str(np.round(r1, 1)).replace('.', '')
                        with open(dir_to_save+f'{save}_beta_{beta_str}_q_{q_str}_r_{r_str}.csv', 'w') as f:

                            # using csv.writer method from CSV package
                            write = csv.writer(f)

                            write.writerow(fields)
                            write.writerows(results_save)

        
        
    return data
def main():

    graph_args={'name': 'd_regular', 'args': {'n':1000, 'd':2,'edge size':5, 'distribution':'uniform'}}
    data = run_parallel(graph_args, betas=np.arange(0.1,1,0.1), qs=np.arange(0,8,0.1), endtime=10000,move='majority', iterations=20, processes=50,log_freq=1000, save='2_regular_edgesize_5_maj_change')
    
    
if __name__ == "__main__":
    main()