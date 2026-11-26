from hypergraph_generator import generate_hypergraph

import numpy as np
import matplotlib.pyplot as plt
import csv
import multiprocessing
from multiprocessing.pool import Pool
import pickle



def count_components(H, opinion):
    N = len(H.nodes)
    seen = np.zeros(N, dtype=bool)
    
    sizes = []
    sum_as = []
    homos = []   # total size of homogeneous edges in the component

    for start in range(N):
        if not seen[start]:
            # BFS/DFS for component
            stack = [start]
            seen[start] = True
            component = [start]

            while stack:
                node = stack.pop()
                neighs = H.edgelists[node][0] + H.edgelists[node][1]
                for neigh in neighs:
                    if not seen[neigh]:
                        seen[neigh] = True
                        stack.append(neigh)
                        component.append(neigh)

            # 1. Size of component
            sizes.append(len(component))

            # 2. Sum of A-opinion nodes in component
            component = np.array(component)
            sum_as.append(np.sum(opinion[component] == -1))

            # 3. Homophily inside this component:
            #    total size of homogeneous edges incident to it
            #    (each edge counted once)
            edge_indices = set(
                idx
                for node in component
                for idx in H.edgelists_indices[node]
            )

            homo_size = 0
            for ei in edge_indices:
                edge = H.edges[ei]
                numA = np.sum(opinion[edge] == -1)
                if numA == 0 or numA == len(edge):
                    homo_size += len(edge)

            homos.append(homo_size)

    return sizes, sum_as, homos


def run(graph_args, beta = 0.2, q=0.4, r1=0.5,r2=0.5,lambda_=0.5, endtime=10000, move='majority',log_freq=1000, seed=0,x=0.5,log_type=0):
    
    np.random.seed(seed)
    H = generate_hypergraph(graph_args['name'], graph_args['args'])
    number_of_edges=len(H.edges)
    workplace_layer_start = int(graph_args['args']['n']/graph_args['args']['household size'])
    ### Egyszerű terjedés A=-1, B=1 vélemény fele-fele inicaializálással: ###
    if x<0:
        x = np.random.normal(loc=0.5,scale=0.75)
        while x<0 or x>1:
            x = np.random.normal(loc=0.5,scale=0.75)
    #inicializáljuk 
    N=len(H.nodes) # csúcsok száma
    opinion=-1*np.ones(N) # vélemény vektor
    init_B=np.random.choice(H.nodes,int(N*x),replace=False) #random a fele B véleményre
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
    homophily = []
    move_count_act=0
    op_change_count_act =0
    zero_count = 1
    
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
            sum_by_edge = np.array([np.sum(opinion[edge]==opinion[act_node])/len(edge) for edge in H.edges])
            if move=='majority':
                if np.sum(sum_by_edge[workplace_layer_start:]>0.5)>0:
                    edge_to_move = np.random.choice(np.arange(workplace_layer_start, number_of_edges)[sum_by_edge[workplace_layer_start:]>0.5],1)[0]
                    H.edgelists[act_node][1].remove(act_node)
                    H.edgelists[act_node].remove(H.edgelists[act_node][1])
                    H.edgelists_indices[act_node].remove(H.edgelists_indices[act_node][1])
                    
                    H.edges[edge_to_move].append(act_node)
                    H.edgelists[act_node].append(H.edges[edge_to_move])
                    H.edgelists_indices[act_node].append(edge_to_move)
                    move_count_act+=1
                else:
                    pass
            #print('act node',act_node)
            #print('edge_to_move', H.edges[edge_to_move])
            #print('edgelist',H.edgelists[act_node])
            #print('edges',H.edges)
        # distribution of workplace edge sizes
        
        
        if i%log_freq==0 and log_type==0:
            sum_A_log.append(np.sum(opinion == -1))
            sum_B_log.append(np.sum(opinion == 1))

            # opinion sums
            opinion_sums_home=[np.sum(opinion[edge]==-1) for edge in H.edges[:workplace_layer_start]]
            z=np.histogram(opinion_sums_home, bins=np.arange(graph_args['args']['household size']+2))
            binned_opinions_home.append(z[0])

            opinion_rates_wp=[np.sum(opinion[edge]==-1)/len(edge) for edge in H.edges[workplace_layer_start:]]
            z=np.histogram(opinion_rates_wp, bins=np.arange(11)/10)
            binned_opinions_wp.append(z[0])
            
            popsized=[0,0,0,0]
            popsized_num=[1,1,1,1]
            #bins = np.arange(11)/10
            #for i in range(10):
            #    popsized.append(0)
            #    for edge in H.edges[workplace_layer_start:]:
            #        if bins[i]<=np.sum(opinion[edge]==-1)/len(edge)<bins[i+1]:
            #            popsized[-1] += len(edge)
            #        elif i==9 and np.sum(opinion[edge]==-1)/len(edge)==bins[i+1]:
            #            popsized[-1] += len(edge)
            for act_node in H.nodes:
                id_opinions = np.array([np.sum(opinion[edge]==opinion[act_node]) for edge in H.edgelists[act_node]])
                for j in range(1,5):
                    if id_opinions[0]==j:
                        popsized[j-1] += id_opinions[1]
                        popsized_num[j-1] +=1
            
            
            binned_opinions_wp_pop_sized.append(np.array(popsized)/(np.array(popsized_num)))

            # workplace sizes
            edge_sizes_wp=[len(edge) for edge in H.edges[workplace_layer_start:]]
            z=np.histogram(edge_sizes_wp, bins=np.arange(15))
            binned_edge_sizes_wp.append(z[0])
            
            move_count.append(move_count_act)
            op_change_count.append(op_change_count_act)
            move_count_act = 0
            op_change_count_act = 0
            
        if i%log_freq==0 and log_type==-2:
            #move,  change count
            if move_count_act==0 and op_change_count_act==0:
                zero_count += 1
            else: 
                zero_count = 0
            move_count_act = 0
            op_change_count_act = 0
            
        if i%log_freq==0 and log_type==-1:
            # opinion sums
            sum_A_log.append(np.sum(opinion == -1))
            sum_B_log.append(np.sum(opinion == 1))
                
            opinion_sums_home=[np.sum(opinion[edge]==-1) for edge in H.edges[:workplace_layer_start]]
            z=np.histogram(opinion_sums_home, bins=np.arange(graph_args['args']['household size']+2))
            binned_opinions_home=z[0]

            opinion_rates_wp=[np.sum(opinion[edge]==-1)/len(edge) for edge in H.edges[workplace_layer_start:]]
            z=np.histogram(opinion_rates_wp, bins=np.arange(11)/10)
            binned_opinions_wp=z[0]
            
            #homophily
            h_hh = float(graph_args['args']['household size']*(binned_opinions_home[0] + binned_opinions_home[-1])/N)
            
            opinion_sums_wp = np.array([np.sum(opinion[edge]==-1) for edge in H.edges[workplace_layer_start:]])
            opinion_homo_wp = np.array([(np.sum(opinion[edge]==-1) == 0 or np.sum(opinion[edge]==-1) == len(edge)) for edge in H.edges[workplace_layer_start:]])
            sizes=np.array([len(edge) for edge in H.edges[workplace_layer_start:]])
            wp_hh = float(np.sum(sizes[opinion_homo_wp])/N)
            homophily.append([h_hh,wp_hh])
            
            #move,  change count
            if move_count_act==0 and op_change_count_act==0:
                zero_count += 1
            else: 
                zero_count = 0
            move_count.append(move_count_act)
            op_change_count.append(op_change_count_act)
            move_count_act = 0
            op_change_count_act = 0
        try: 
            (float(np.sum(homophily[-1]))==2)
        except:
            print(homophily[-1])
        if (float(np.sum(homophily[-1]))==2) or (int(zero_count) > 100) or i == int(endtime-1):
            #print(((binned_opinions_home[0]+binned_opinions_home[-1] == np.sum(binned_opinions_home)) and (binned_opinions_wp[0]+binned_opinions_wp[-1] == np.sum(binned_opinions_wp))), (zero_count > 100), i==endtime-1)
            
            sum_A_log.append(np.sum(opinion == -1))
            sum_B_log.append(np.sum(opinion == 1))

            # opinion sums
            opinion_sums_home=[np.sum(opinion[edge]==-1) for edge in H.edges[:workplace_layer_start]]
            z=np.histogram(opinion_sums_home, bins=np.arange(graph_args['args']['household size']+2))
            binned_opinions_home=[z[0]]

            opinion_rates_wp=[np.sum(opinion[edge]==-1)/len(edge) for edge in H.edges[workplace_layer_start:]]
            z=np.histogram(opinion_rates_wp, bins=np.arange(11)/10)
            binned_opinions_wp=[z[0]]

            popsized=[0,0,0,0]
            popsized_num=[1,1,1,1]
            #bins = np.arange(11)/10
            #for i in range(10):
            #    popsized.append(0)
            #    for edge in H.edges[workplace_layer_start:]:
            #        if bins[i]<=np.sum(opinion[edge]==-1)/len(edge)<bins[i+1]:
            #            popsized[-1] += len(edge)
            #        elif i==9 and np.sum(opinion[edge]==-1)/len(edge)==bins[i+1]:
            #            popsized[-1] += len(edge)
            for act_node in H.nodes:
                id_opinions = np.array([np.sum(opinion[edge]==opinion[act_node]) for edge in H.edgelists[act_node]])
                for j in range(1,5):
                    if id_opinions[0]==j:
                        popsized[j-1] += id_opinions[1]
                        popsized_num[j-1] +=1


            binned_opinions_wp_pop_sized.append(np.array(popsized)/(np.array(popsized_num)))

            # workplace sizes
            edge_sizes_wp=np.array([len(edge) for edge in H.edges[workplace_layer_start:]])
            z=np.histogram(edge_sizes_wp, bins=np.arange(15))
            binned_edge_sizes_wp.append(z[0])
            
            #homophily
            h_hh = float(graph_args['args']['household size']*(binned_opinions_home[0][0] + binned_opinions_home[0][-1])/N)
            
            opinion_sums_wp = np.array([np.sum(opinion[edge]==-1) for edge in H.edges[workplace_layer_start:]])
            opinion_homo_wp = np.array([(np.sum(opinion[edge]==-1) == 0 or np.sum(opinion[edge]==-1) == len(edge)) for edge in H.edges[workplace_layer_start:]])
            wp_hh = float(np.sum(edge_sizes_wp[opinion_homo_wp])/N)
            homophily.append([h_hh,wp_hh])



            break
    comp_sizes, comp_opinion, comp_homo = count_components(H,opinion)
    return sum_A_log, comp_sizes, homophily

def run_parallel(graph_args, betas, qs, r1s=[0.5],r2s=[0.5],lambdas=[0.5], endtime=10000, move='majority',iterations=50, processes=100,log_freq=1000, dest=0, xs=[0.5],log_type=0, save=0):
    pool = multiprocessing.Pool(processes=processes)
    results = []

    for beta_i, beta in enumerate(betas):
        results.append([])
        for q_i, q in enumerate(qs):
            for r1_i, r1 in enumerate(r1s):
                for r2_i, r2 in enumerate(r2s):
                    for lambda_i, lambda_ in enumerate(lambdas):
                        for x_i,x in enumerate(xs):
                            results[beta_i].append([])
                            maxqr = max(q_i,r1_i,r2_i,lambda_i, x_i)
                            for it in range(iterations):
                                results[beta_i][maxqr].append(pool.apply_async(run, args=(graph_args,beta,q,r1,r2,lambda_,endtime, move, log_freq, it, x,log_type)))
        
        
    pool.close()
    pool.join()
    
    
   # row_names = ['sum_opinion_A','binned_opinions_home','binned_opinions_wp','binned_edge_sizes_wp','move_count', 'op_change_count', 'component_sizes', 'component_opinion', 'component_homo','homophily']
    row_names = ['sum_opinion_A','component_sizes','homophily']

    data=[]
    for beta_i, beta in enumerate(betas):
        data.append([])
        for q_i, q in enumerate(qs):
            for r1_i, r1 in enumerate(r1s):
                for r2_i, r2 in enumerate(r2s):
                    for lambda_i, lambda_ in enumerate(lambdas):
                        for x_i, x in enumerate(xs):
                            maxqr = max(q_i,r1_i,r2_i,lambda_i,x_i)
                            data[beta_i].append([])
                            results_save = []
                            for it in range(iterations):
                                data[beta_i][maxqr].append({})
                                r_act = results[beta_i][maxqr][it].get()
                                for i in range(len(r_act)):
                                    data[beta_i][maxqr][it][row_names[i]] = r_act[i]
                                    if save==1:
                                        if row_names[i] not in ['sum_opinion_A','move_count', 'op_change_count', 'component_sizes', 'component_opinion', 'component_homo','homophily']:
                                            list_to_save = [" ".join(map(str, act)) for act in r_act[i]]
                                            results_save.append([it,row_names[i]] + list_to_save)
                                    #    elif i==len(r_act)-1:
                                    #        results_save.append([it,row_names[i],r_act[i]])
                                        else:
                                            results_save.append([it,row_names[i]] + list(r_act[i]))


                    if save==1:
                        # field names 
                        cols=['sample id', 'datatype']
                        fields = cols + list(np.arange(len(r_act[0])))
                        beta_str=str(np.round(beta, 2)).replace('.', '')
                        q_str=str(np.round(q, 1)).replace('.', '')
                        r_str=str(np.round(r1, 1)).replace('.', '')
                        with open(f'{save}_beta_{beta_str}_q_{q_str}_r_{r_str}.csv', 'w') as f:
                            # using csv.writer method from CSV package
                            write = csv.writer(f)
                            write.writerow(fields)
                            write.writerows(results_save)
                            
                     

    if save==2:
        filename = dest + '.pickle'
        # Save the results_dict in a JSON file
        with open(filename,  'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    return data
def main():
    
    #dir_to_save = 'data/majority_move_lin_long10/'
    #for n in [500,1000,2000]:
    #    graph_args={'name': '2layer', 'args': {'n': n, 'd':2,'household size':5,'workplace size':5, 'distribution':'uniform'}}
    #    data = run_parallel(graph_args, betas=np.arange(0.1,1,0.1), qs=np.arange(0,1,0.1),r1s=[0.5],r2s=[0.5],lambdas=[0.5], endtime=5*10**6,move='majority', iterations=100, processes=100,log_freq = n, dest=f'/scratch/kolokcse/logs_hypergraph/majority_move_nonlin_long_final/long_2_regular_edgesize_5_nonlin_n_{n}',log_type=-1, save=2)

    #for m in [10, 20, 30, 40, 50,60,70,80,90,100]:
    #    graph_args={'name': '2layer', 'args': {'n': 1000, 'd':2,'household size':5,'workplace size':m, 'distribution':'uniform'}}
    #    data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=np.array([0.1,0.5,0.9]),r1s=[0.5],r2s=[0.5],lambdas=[0.5], endtime=5*10**6,move='majority', iterations=100, processes=100,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/majority_move_nonlin_long_final/long_2_regular_n_1000_edgesize_{m}_nonlin',log_type=-1, save=2)
    #    data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=np.array([0.1,0.5,0.9]),r1s=[0.5],r2s=[1],lambdas=[1], endtime=5*10**6,move='majority', iterations=100, processes=100,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/majority_move_lin_long_final/long_2_regular_n_1000_edgesize_{m}_lin',log_type=-1, save=2)

    for q in np.array([0.1,0.5,0.9]):
        graph_args={'name': '2layer', 'args': {'n': 1000, 'd':2,'household size':5,'workplace size':5, 'distribution':'uniform'}}
        data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=[q],r1s=[1],r2s=[1],lambdas=np.arange(0,1,0.2), endtime=5*10**6,move='majority', iterations=100, processes=100,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/lambda/long_2_regular_q_{q}_lin',log_type=-1, save=2)
        data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=[q],r1s=[0.5],r2s=[0.5],lambdas=np.arange(0,1,0.2), endtime=5*10**6,move='majority', iterations=100, processes=100,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/lambda/long_2_regular_q_{q}_nonlin',log_type=-1, save=2)
    for q in np.array([0.1,0.5,0.9]):
        graph_args={'name': '2layer', 'args': {'n': 1000, 'd':2,'household size':5,'workplace size':5, 'distribution':'uniform'}}
        data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=[q],r1s=[1],r2s=[0.25,0.5,0.75],lambdas=[0.5], endtime=5*10**6,move='majority', iterations=100, processes=100,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/r_change/long_2_regular_q_{q}_hhlin',log_type=-1, save=2)
        data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=[q],r1s=[0.5],r2s=[0.25,0.5,0.75],lambdas=[0.5], endtime=5*10**6,move='majority', iterations=100, processes=100,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/r_change/long_2_regular_q_{q}_hhnonlin',log_type=-1, save=2)

    
if __name__ == "__main__":
    main()