from hypergraph_generator import generate_hypergraph

import numpy as np
import matplotlib.pyplot as plt
import csv
import multiprocessing
from multiprocessing.pool import Pool
import pickle


def node_hwa(H, opinion, node, lambda_):
    """
    Compute h, w, a for a given node:
      h = fraction of same-opinion vertices in household
      w = fraction of same-opinion vertices in workplace
      a = weighted average used in opinion change
    """
    id_opinions = np.array(
        [np.sum(opinion[edge] == opinion[node]) for edge in H.edgelists[node]],
        dtype=float
    )
    edge_sizes = np.array([len(edge) for edge in H.edgelists[node]], dtype=float)

    h = id_opinions[0] / edge_sizes[0]
    w = id_opinions[1] / edge_sizes[1]
    a = (h + lambda_ * w) / (1.0 + lambda_)
    return h, w, a


def get_workplace_candidates(H, opinion, node, workplace_layer_start, move='majority',
                             fallback_to_all=False, exclude_current=True):
    """
    Return candidate workplace indices for a workplace move.

    Parameters
    ----------
    fallback_to_all : bool
        If True, then if no majority workplace exists, return all workplaces
        (except optionally the current one). This matches the model in the paper.
        If False, then no candidates are returned when no majority workplace exists.
        This matches your current implementation.
    exclude_current : bool
        If True, exclude the current workplace from the candidate list.
    """
    number_of_edges = len(H.edges)
    workplace_indices = np.arange(workplace_layer_start, number_of_edges)

    current_wp_idx = H.edgelists_indices[node][1]

    # Same-opinion ratio of the node's current opinion in each workplace
    wp_edges = H.edges[workplace_layer_start:]
    sum_by_edge = np.array([
        np.sum(opinion[edge] == opinion[node]) / len(edge)
        for edge in wp_edges
    ])

    if move == 'majority':
        candidates = workplace_indices[sum_by_edge > 0.5]
        if candidates.size == 0 and fallback_to_all:
            candidates = workplace_indices.copy()
    else:
        candidates = workplace_indices.copy()

    if exclude_current:
        candidates = candidates[candidates != current_wp_idx]

    return candidates


def is_absorbing_state(H, opinion, workplace_layer_start, beta, q, r1, r2, lambda_,
                       move='majority', fallback_to_all=False, return_witness=False):
    """
    Check whether the current state is absorbing for the actual simulated dynamics.

    A state is absorbing if for every node:
      - no opinion change has positive probability, and
      - no workplace change has positive probability.

    Parameters
    ----------
    fallback_to_all : bool
        If True, workplace move logic matches the paper:
        if no majority workplace exists, choose uniformly from all workplaces.
        If False, workplace move logic matches the current code:
        if no majority workplace exists, no move occurs.

    return_witness : bool
        If True, return a witness describing the first non-absorbing node.
    """
    for node in H.nodes:
        h, w, a = node_hwa(H, opinion, node, lambda_)

        # Step 1: opinion change possible?
        if beta > 0 and a < r1:
            if return_witness:
                return False, {
                    'node': node,
                    'type': 'opinion',
                    'h': h,
                    'w': w,
                    'a': a
                }
            return False

        # Step 2: workplace change possible?
        if q > 0 and w < r2:
            candidates = get_workplace_candidates(
                H, opinion, node,
                workplace_layer_start=workplace_layer_start,
                move=move,
                fallback_to_all=fallback_to_all,
                exclude_current=True
            )
            if candidates.size > 0:
                if return_witness:
                    return False, {
                        'node': node,
                        'type': 'workplace',
                        'h': h,
                        'w': w,
                        'a': a,
                        'num_candidates': int(candidates.size)
                    }
                return False

    if return_witness:
        return True, None
    return True


def collect_snapshot(H, opinion, graph_args, workplace_layer_start, N):
    """
    Collect one snapshot of the current state for logging.
    """
    out = {}

    out['sum_A'] = int(np.sum(opinion == -1))
    out['sum_B'] = int(np.sum(opinion == 1))

    # Household opinion histogram
    opinion_sums_home = [np.sum(opinion[edge] == -1) for edge in H.edges[:workplace_layer_start]]
    z_home = np.histogram(
        opinion_sums_home,
        bins=np.arange(graph_args['args']['household size'] + 2)
    )
    out['binned_opinions_home'] = z_home[0]

    # Workplace opinion-rate histogram
    opinion_rates_wp = [
        np.sum(opinion[edge] == -1) / len(edge)
        for edge in H.edges[workplace_layer_start:]
    ]
    z_wp = np.histogram(opinion_rates_wp, bins=np.arange(11) / 10)
    out['binned_opinions_wp'] = z_wp[0]

    # Population-sized workplace statistic
    popsized = [0, 0, 0, 0]
    popsized_num = [1, 1, 1, 1]  # avoid divide-by-zero
    for act_node in H.nodes:
        id_opinions = np.array([
            np.sum(opinion[edge] == opinion[act_node]) for edge in H.edgelists[act_node]
        ])
        for j in range(1, 5):
            if id_opinions[0] == j:
                popsized[j - 1] += id_opinions[1]
                popsized_num[j - 1] += 1
    out['binned_opinions_wp_pop_sized'] = np.array(popsized) / np.array(popsized_num)

    # Workplace size histogram
    edge_sizes_wp = np.array([len(edge) for edge in H.edges[workplace_layer_start:]])
    z_sizes = np.histogram(edge_sizes_wp, bins=np.arange(15))
    out['binned_edge_sizes_wp'] = z_sizes[0]
    out['max_workplace_size'] = int(np.max(edge_sizes_wp)) if len(edge_sizes_wp) > 0 else 0

    # Homophily
    h_hh = float(
        graph_args['args']['household size']
        * (out['binned_opinions_home'][0] + out['binned_opinions_home'][-1]) / N
    )

    opinion_homo_wp = np.array([
        (np.sum(opinion[edge] == -1) == 0 or np.sum(opinion[edge] == -1) == len(edge))
        for edge in H.edges[workplace_layer_start:]
    ])
    wp_hh = float(np.sum(edge_sizes_wp[opinion_homo_wp]) / N)

    out['homophily'] = [h_hh, wp_hh]

    return out

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


def run(graph_args, beta=0.2, q=0.4, r1=0.5, r2=0.5, lambda_=0.5,
        endtime=10000, move='majority', log_freq=1000, seed=0,
        x=0.5, log_type=0, fallback_to_all=False,
        stop_if_absorbing=True, zero_streak_limit=100):
    """
    Run one simulation of the opinion/workplace dynamics.

    Parameters
    ----------
    fallback_to_all : bool
        If True, workplace move rule matches the paper:
        if no majority workplace exists, choose uniformly among all workplaces.
        If False, workplace move rule matches the current implementation:
        if no majority workplace exists, no move occurs.

    stop_if_absorbing : bool
        If True, explicitly check at log times whether the state is absorbing and stop.

    zero_streak_limit : int
        Stop early if there were no changes in this many consecutive logging windows.

    Returns
    -------
    sum_A_log, comp_sizes, homophily, max_edge
    """
    np.random.seed(seed)

    H = generate_hypergraph(graph_args['name'], graph_args['args'])
    number_of_edges = len(H.edges)
    workplace_layer_start = int(graph_args['args']['n'] / graph_args['args']['household size'])

    # Initial opinion distribution
    if x < 0:
        x = np.random.normal(loc=0.5, scale=0.75)
        while x < 0 or x > 1:
            x = np.random.normal(loc=0.5, scale=0.75)

    N = len(H.nodes)
    opinion = -1 * np.ones(N, dtype=int)
    init_B = np.random.choice(H.nodes, int(N * x), replace=False)
    opinion[init_B] = 1

    # Logs
    sum_A_log = [int(np.sum(opinion == -1))]
    sum_B_log = [int(np.sum(opinion == 1))]

    binned_opinions_home = []
    binned_opinions_wp = []
    binned_opinions_wp_pop_sized = []
    binned_edge_sizes_wp = []
    
    

    move_count = []
    op_change_count = []
    homophily = []
    max_edge = []
    move_prob_sum=[]
    change_prob_sum=[]


    move_count_act = 0
    op_change_count_act = 0
    zero_count = 0

    # Optional initial absorption check
    if stop_if_absorbing:
        absorbing_now = is_absorbing_state(
            H, opinion,
            workplace_layer_start=workplace_layer_start,
            beta=beta, q=q,
            r1=r1, r2=r2, lambda_=lambda_,
            move=move,
            fallback_to_all=fallback_to_all,
            return_witness=False
        )
        if absorbing_now:
            snap = collect_snapshot(H, opinion, graph_args, workplace_layer_start, N)
            binned_opinions_home.append(snap['binned_opinions_home'])
            binned_opinions_wp.append(snap['binned_opinions_wp'])
            binned_opinions_wp_pop_sized.append(snap['binned_opinions_wp_pop_sized'])
            binned_edge_sizes_wp.append(snap['binned_edge_sizes_wp'])
            homophily.append(snap['homophily'])
            max_edge = [snap['max_workplace_size']]

            comp_sizes, comp_opinion, comp_homo = count_components(H, opinion)
            return sum_A_log, comp_sizes, homophily, max_edge

    for i in range(endtime):
        act_node = np.random.choice(H.nodes, 1)[0]

        # Compute h, w, a for the chosen node
        h, w, a = node_hwa(H, opinion, act_node, lambda_)

        changed_this_step = False

        # Step 1: opinion change
        if a < r1 and np.random.random() < beta * (r1 - a):
            opinion[act_node] = -1 * opinion[act_node]
            op_change_count_act += 1
            changed_this_step = True

        # Step 2: workplace change
        elif w < r2 and np.random.random() < q * (r2 - w):
            candidates = get_workplace_candidates(
                H, opinion, act_node,
                workplace_layer_start=workplace_layer_start,
                move=move,
                fallback_to_all=fallback_to_all,
                exclude_current=True
            )

            if candidates.size > 0:
                edge_to_move = np.random.choice(candidates, 1)[0]

                old_wp_edge = H.edgelists[act_node][1]

                # remove from old workplace edge
                old_wp_edge.remove(act_node)

                # overwrite workplace reference safely
                H.edgelists[act_node][1] = H.edges[edge_to_move]
                H.edgelists_indices[act_node][1] = edge_to_move

                # add to new workplace edge
                H.edges[edge_to_move].append(act_node)

                move_count_act += 1
                changed_this_step = True

        # Logging
        if i % log_freq == 0:
            # Update zero_count based on activity in the last logging window
            if move_count_act == 0 and op_change_count_act == 0:
                zero_count += 1
            else:
                zero_count = 0

            # Logging variants
            if log_type == 0:
                snap = collect_snapshot(H, opinion, graph_args, workplace_layer_start, N)
                sum_A_log.append(snap['sum_A'])
                sum_B_log.append(snap['sum_B'])
                binned_opinions_home.append(snap['binned_opinions_home'])
                binned_opinions_wp.append(snap['binned_opinions_wp'])
                binned_opinions_wp_pop_sized.append(snap['binned_opinions_wp_pop_sized'])
                binned_edge_sizes_wp.append(snap['binned_edge_sizes_wp'])
                move_count.append(move_count_act)
                op_change_count.append(op_change_count_act)

            elif log_type == -1:
                snap = collect_snapshot(H, opinion, graph_args, workplace_layer_start, N)
                sum_A_log.append(snap['sum_A'])
                sum_B_log.append(snap['sum_B'])

                # overwrite latest only, to mimic your original style
                binned_opinions_home = snap['binned_opinions_home']
                binned_opinions_wp = snap['binned_opinions_wp']
                homophily.append(snap['homophily'])
                max_edge.append(snap['max_workplace_size'])
                
                move_count.append(move_count_act)
                op_change_count.append(op_change_count_act)
                ch_sum=0
                m_sum=0
                for node in H.nodes:
                    h, w, a = node_hwa(H, opinion, node, lambda_)
                    ch_sum+=beta*max(r1-a, 0)
                    m_sum+=q*max(r2-w, 0)
                move_prob_sum.append(m_sum)
                change_prob_sum.append(ch_sum)

            elif log_type == -2:
                # Only zero streak / stop logic
                pass
            
            # Explicit absorbing-state check
            absorbing_now = False
            if move_count_act == 0 and op_change_count_act == 0:
                if stop_if_absorbing:
                    absorbing_now = is_absorbing_state(
                        H, opinion,
                        workplace_layer_start=workplace_layer_start,
                        beta=beta, q=q,
                        r1=r1, r2=r2, lambda_=lambda_,
                        move=move,
                        fallback_to_all=fallback_to_all,
                        return_witness=False
                    )

            # Reset interval counters AFTER logging/checking
            move_count_act = 0
            op_change_count_act = 0

            # Stop condition
            if absorbing_now or (zero_count > zero_streak_limit) or (i == int(endtime - 1)):
                # Final snapshot
                snap = collect_snapshot(H, opinion, graph_args, workplace_layer_start, N)
                sum_A_log.append(snap['sum_A'])
                sum_B_log.append(snap['sum_B'])

                # Keep final snapshots in list form for compatibility
                if log_type == 0:
                    binned_opinions_home.append(snap['binned_opinions_home'])
                    binned_opinions_wp.append(snap['binned_opinions_wp'])
                    binned_opinions_wp_pop_sized.append(snap['binned_opinions_wp_pop_sized'])
                    binned_edge_sizes_wp.append(snap['binned_edge_sizes_wp'])
                else:
                    binned_opinions_home = [snap['binned_opinions_home']]
                    binned_opinions_wp = [snap['binned_opinions_wp']]
                    binned_opinions_wp_pop_sized.append(snap['binned_opinions_wp_pop_sized'])
                    binned_edge_sizes_wp.append(snap['binned_edge_sizes_wp'])

                homophily.append(snap['homophily'])
                max_edge.append(snap['max_workplace_size'])
                break

    comp_sizes, comp_opinion, comp_homo = count_components(H, opinion)
    return sum_A_log, comp_sizes, homophily, max_edge, change_prob_sum, move_prob_sum

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
                                results[beta_i][maxqr].append(pool.apply_async(
                                                run,
                                                args=(graph_args, beta, q, r1, r2, lambda_, endtime, move, log_freq, it, x, log_type),
                                                kwds={'fallback_to_all': False, 'stop_if_absorbing': True, 'zero_streak_limit': 500}
                                                )
                                            )
        
    pool.close()
    pool.join()
    
    
   # row_names = ['sum_opinion_A','binned_opinions_home','binned_opinions_wp','binned_edge_sizes_wp','move_count', 'op_change_count', 'component_sizes', 'component_opinion', 'component_homo','homophily']
    row_names = ['sum_opinion_A','component_sizes','homophily', 'max_workplace_size','change_prob_sum', 'move_prob_sum']

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
                                        if row_names[i] not in ['sum_opinion_A','move_count', 'op_change_count', 'component_sizes', 'component_opinion', 'component_homo','homophily', 'max_workplace_size','change_prob_sum', 'move_prob_sum']:
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
        print(f'Saving results to {dest}.pickle')
        filename = dest + '.pickle'
        # Save the results_dict in a JSON file
        with open(filename,  'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    return data
def main():
    
    #dir_to_save = 'data/majority_move_lin_long10/'
    #for n in [10,20,30,100]:
    #    graph_args={'name': '2layer', 'args': {'n': n, 'd':2,'household size':5,'workplace size':5, 'distribution':'uniform'}}
    #    data = run_parallel(graph_args, betas=np.arange(0.1,1,0.1), qs=np.arange(0,1,0.1),r1s=[1],r2s=[1],lambdas=[0.5], endtime=5*10**6,move='majority', iterations=50, processes=150,log_freq = n, dest=f'/scratch/kolokcse/logs_hypergraph/majority_move_lin_long_final/long_2_regular_edgesize_5_lin_n_{n}',log_type=-1, save=2)
    #    data = run_parallel(graph_args, betas=np.arange(0.1,1,0.1), qs=np.arange(0,1,0.1),r1s=[0.5],r2s=[0.5],lambdas=[0.5], endtime=5*10**6,move='majority', iterations=50, processes=150,log_freq = n, dest=f'/scratch/kolokcse/logs_hypergraph/majority_move_nonlin_long_final/long_2_regular_edgesize_5_nonlin_n_{n}',log_type=-1, save=2)

    #for m in [10, 20, 30, 40, 50,60,70,80,90,100]:
    #    graph_args={'name': '2layer', 'args': {'n': 1000, 'd':2,'household size':5,'workplace size':m, 'distribution':'uniform'}}
    #    data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=np.array([0.1,0.5,0.9]),r1s=[0.5],r2s=[0.5],lambdas=[0.5], endtime=5*10**6,move='majority', iterations=100, processes=100,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/majority_move_nonlin_long_final/long_2_regular_n_1000_edgesize_{m}_nonlin',log_type=-1, save=2)
    #    data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=np.array([0.1,0.5,0.9]),r1s=[1],r2s=[1],lambdas=[0.5], endtime=5*10**6,move='majority', iterations=100, processes=100,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/majority_move_lin_long_final/long_2_regular_n_1000_edgesize_{m}_lin',log_type=-1, save=2)

   #
   for q in np.array([0.1,0.5,0.9]):
        graph_args={'name': '2layer', 'args': {'n': 1000, 'd':2,'household size':5,'workplace size':5, 'distribution':'uniform'}}
        data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=[q],r1s=[1],r2s=[1],lambdas=np.linspace(0.45,0.65,10), endtime=5*10**6,move='majority', iterations=20, processes=150,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/lambda/long_2_regular_q_{q}_lin',log_type=-1, save=2)
        data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=[q],r1s=[0.5],r2s=[0.5],lambdas=np.linspace(0.45,0.65,10), endtime=5*10**6,move='majority', iterations=20, processes=150,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/lambda/long_2_regular_q_{q}_nonlin',log_type=-1, save=2)
   # for q in np.array([0.1,0.5,0.9]):
    #   	graph_args={'name': '2layer', 'args': {'n': 1000, 'd':2,'household size':5,'workplace size':5, 'distribution':'uniform'}}
    #   data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=[q],r1s=[1],r2s=np.arange(0.1,1,0.1),lambdas=[0.5], endtime=5*10**6,move='majority', iterations=100, processes=50,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/r_change/long_2_regular_q_{q}_hhlin',log_type=-1, save=2)
    #    data = run_parallel(graph_args, betas=np.array([0.1,0.5,0.9]), qs=[q],r1s=[0.5],r2s=np.arange(0.1,1,0.1),lambdas=[0.5], endtime=5*10**6,move='majority', iterations=100, processes=50,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/r_change/long_2_regular_q_{q}_hhnonlin',log_type=-1, save=2)

    #q=0.1
    #for r1 in np.arange(0.05,1.05,0.05):
    #    graph_args={'name': '2layer', 'args': {'n': 1000, 'd':2,'household size':5,'workplace size':5, 'distribution':'uniform'}}
    #    data = run_parallel(graph_args, betas=np.array([0.9]), qs=[q],r1s=[r1],r2s=np.arange(0.05,1.05,0.05),lambdas=[0.5], endtime=5*10**6,move='majority', iterations=20, processes=50,log_freq = 1000, dest=f'/scratch/kolokcse/logs_hypergraph/r_change/long_2_regular_r1_{r1}_q_{q}_beta_{0.9}',log_type=-1, save=2)
    
    #for n in [10,100]:
        #for r1 in np.arange(0.05,1.05,0.05):
            #graph_args={'name': '2layer', 'args': {'n': n, 'd':2,'household size':5,'workplace size':5, 'distribution':'binomial','binomial_n':10,'edge size':5}}
            #data = run_parallel(graph_args, betas=np.array([0.5]), qs=[q],r1s=[r1],r2s=np.arange(0.05,1.05,0.05),lambdas=[0.5], endtime=5*10**6,move='majority', iterations=20, processes=100,log_freq = n, dest=f'/scratch/kolokcse/logs_hypergraph/r_change/long_2_regular_r1_{r1}_q_{q}_beta_{0.5}_n_{n}_binom',log_type=-1, save=2)


    
if __name__ == "__main__":
    main()
