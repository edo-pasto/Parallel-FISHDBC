#!/usr/bin/env python3

# Copyright (c) 2017-2018 Symantec Corporation. All Rights Reserved. 
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# import timeit
import numpy as np
import pandas as pd
import argparse
import sys
import collections
# from functools import partial
from scipy.spatial import distance
from numba import njit
from Levenshtein import distance as lev
import sklearn.datasets
import matplotlib.pyplot as plt
from flexible_clustering import fishdbc
from flexible_clustering import hnsw_parallel
# from line_profiler import LineProfiler 
import time
import multiprocessing 
from math import dist as mathDist
from random import random
try:
    from math import log2
except ImportError: # Python 2.x or <= 3.2
    from math import log
    def log2(x):
        return log(x, 2)
MISSING = sys.maxsize
MISSING_WEIGHT = sys.float_info.max
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Show an example of running FISHDBC."
    "This will plot points that are naturally clustered and added incrementally,"
    "and then loop through all the hierarchical clusters recognized by the algorithm."
    "Original clusters are shown in different colors while each cluster found by"
    "FISHDBC is shown in red; press a key or click the mouse button to loop through clusters.")

    parser.add_argument('--dataset', type=str, default='blob',
                        help="dataset used by the algorithm (default: blob)."
                        "try with: blob, string,")
    parser.add_argument('--distance', type=str, default='hamming',
                        help="distance metrix used by FISHDBC (default: hamming)."
                        "try with: euclidean, squeclidean, cosine, dice, minkowsky, jaccard, hamming, jensenShannon, levensthein")
    parser.add_argument('--nitems', type=int, default=200,
                        help="Number of items (default 200).")
    parser.add_argument('--niters', type=int, default=2,
                        help="Clusters are shown in NITERS stage while being "
                        "added incrementally (default 4).")
    parser.add_argument('--centers', type=int, default=5,
                        help="Number of centers for the clusters generated "
                        "(default 5).")
    parser.add_argument("--parallel", type=str, default="0",
                         help="option to specify if we want to execute the fishdbc with the HNSW creation parallel (True) or not paralle (False)")    
    args = parser.parse_args()

    def plot_cluster_result(size, ctree, x, y, labels):
        plt.figure(figsize=(9, 9))
        plt.gca().set_aspect('equal')
        #     fishdbc.update(points) #points --> [[1 2],[3 4],[5 6], ...]
        nknown = size #data is a list of np array --> [array([1, 3]), array([2,4]), ...] 
        #     _, _, _, ctree, _, _ = fishdbc.cluster()# ctree is a numpy.ndarray of (,,,)--> [(200, 50, 1.87, 1),(201, 50, 1.98,1),...]
        clusters = collections.defaultdict(set) # cluster is a dict of set --> 
        #ask explanation of this for loop
        for parent, child, lambda_val, child_size in ctree[::-1]:
            if child_size == 1:
                clusters[parent].add(child) # at the set corresponding to the key parent add the integer child
            else:
                assert len(clusters[child]) == child_size
                clusters[parent].update(clusters[child]) # at the set corresponding to the key parent add 
                                                    # all the different element of the set corresponding to the key child
        clusters = sorted(clusters.items()) #order the set in the dict based on the key and now became a list 
        xknown, yknown, labels_known = x[:nknown], y[:nknown], labels[:nknown]
        color = ['rgbcmyk'[l % 7] for l in labels_known]
        plt.scatter(xknown, yknown, c=color, linewidth=0)
        plt.show(block=False)
        for _, cluster in clusters:
            plt.waitforbuttonpress()
            plt.gca().clear()
            color = ['kr'[i in cluster] for i in range(nknown)]
            plt.scatter(xknown, yknown, c=color, linewidth=0)
            plt.draw()

    dist = args.distance.lower()
    dataset = args.dataset
    parallel = int(args.parallel)
    
    if dataset == 'blob':
        data, labels = sklearn.datasets.make_blobs(args.nitems, 
                                            centers=args.centers)
        if dist == 'euclidean': 
            @njit
            def calc_dist(x,y):     
                return np.linalg.norm(x - y)
                # return distance.euclidean(x, y)
                # return mathDist(x, y)
        elif dist == 'sqeuclidean':
            def calc_dist(x,y): 
                return distance.sqeuclidean(x,y)
        elif dist == 'cosine':
            def calc_dist(x,y):
                return distance.cosine(x,y)
        elif dist == 'dice':
            def calc_dist(x,y):
                return distance.dice(x,y)
        elif dist == 'jensen-shannon':
            def calc_dist(x,y):
                return distance.jensenshannon(x,y)
        elif dist == 'jaccard':
            def calc_dist(x,y):
                return distance.jaccard(x,y)
        elif dist == 'hamming':
            def calc_dist(x,y):
                return distance.hamming(x,y)
        elif dist == "minkowski":
            def calc_dist(x,y):
                return distance.minkowski(x, y, p=2)
        else:
            raise EnvironmentError("At the moment the specified distance is not available for the blob dataset,"
                                " try with: euclidean, sqeuclidean, cosine, dice, jensen-shannon, jaccard, hamming, minkowsky")
    elif dataset == 'text':
        realData = pd.read_csv('../data/Air_Traffic_Passenger_Statistics.csv')
        li = realData.values.tolist()
        data = np.asarray(li)
        if dist == 'jaccard':
            def calc_dist(x,y):
                return distance.jaccard(x,y)
        elif dist == 'hamming':
            def calc_dist(x,y):
                return distance.hamming(x,y)
        elif dist == "levensthein":
            def calc_dist(x,y):
                return lev(x, y)
        else:
            raise EnvironmentError("At the moment the specified distance is not available for the string dataset,"
                                    " try with: jaccard, hamming, levensthein")
    else:
        raise EnvironmentError("The specified dataset doesn't exist at the moment,"
        "try with: blob, text")   

    x, y = data[:, 0], data[:, 1]
    if parallel > 0:
        start_tot = time.time()
        m = 5 
        m0 = 2 * m
        # with fork method as starting process the child processes created starting from the main process
        # should inherit the calc_distance function and the orignal dataset
        multiprocessing.set_start_method('fork')

        members = [[]]
        # members[0] = list(range(len(data)))
        levels = [ (int(-log2(random())* (1 / log2(m))) + 1) for _ in range(len(data))]
        levels = sorted(enumerate(levels), key=lambda x: x[1])
        
        j = 1
        level_j = []
        for i in levels:
            elem, level = i
            if(level > j ):
                members.append(level_j)
                level_j=[]
                j = j+1
            level_j.append(elem)
            if(j-1 > 0):
                for i in range(j-1, 0, -1):
                    members[i].append(elem)
        members.append(level_j)    
        for i, l in zip(range(len(members)), members):
            sort = sorted(l)
            members[i] = sort
        del members[0]
      
        positions = []
        for el, l in zip(members, range(len(members))): 
            positions.append({ })
            for i, x in enumerate(el):
                positions[l][x] = i
        # print("Levels: ",levels)
        # print("Members: ",members)
        # print("Positions: ",positions, "\n")
        shm_hnsw_data = multiprocessing.shared_memory.SharedMemory(create=True, size=10000000)
        shm_ent_point = multiprocessing.shared_memory.SharedMemory(create=True, size=10)
        shm_count = multiprocessing.shared_memory.SharedMemory(create=True, size=10)
        shm_adj = []
        shm_weights = []
        for i in range (len(members)):
            npArray = np.zeros(shape=(len(members[i]), m0 if i == 0 else m) , dtype=int)
            shm1 = multiprocessing.shared_memory.SharedMemory(create=True, size=npArray.nbytes)
            np.ndarray(npArray.shape, dtype=int, buffer=shm1.buf)[:, :] = MISSING
            shm_adj.append(shm1)
            shm2 = multiprocessing.shared_memory.SharedMemory(create=True, size=npArray.nbytes)
            np.ndarray(npArray.shape, dtype=float, buffer=shm2.buf)[:, :] = MISSING_WEIGHT
            shm_weights.append( shm2)
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        # graph_lock = manager.Lock()
        locks = [manager.Lock() for _ in range(6)]
        end = time.time()
        print("Execution time of preparation for HNSW Parallel :", (end-start_tot))
        hnsw = hnsw_parallel.HNSW(calc_dist, data, members, levels, positions, shm_adj, shm_weights, shm_hnsw_data, shm_ent_point, shm_count, lock, locks, m=m, m0=m0)

        #for now add the first
        #  element not in multiprocessing
        start_time = time.time()
        hnsw.hnsw_add(0)
        with multiprocessing.Pool(parallel) as pool:
            pool.map(hnsw.hnsw_add, range(len(hnsw.data))[1:])
        pool.close()
        pool.join()
        end = time.time()
        print("The time of execution of parallel HNSW is :", (end-start_time))
        # sh_count = np.ndarray(shape=(1), dtype=int, buffer=shm_count.buf)
        # print("The nbr of call to distance is :", (sh_count))

        tot_adjs = []
        tot_weights = []
        for shm1, shm2, memb, i in zip(shm_adj, shm_weights, members, range(len(members))):
            adj = np.ndarray(shape=(len(memb), m0 if i == 0 else m),
                            dtype=int, buffer=shm1.buf)
            tot_adjs.append(adj)
            weight = np.ndarray(shape=(len(memb), m0 if i == 0 else m),
                                dtype=float, buffer=shm2.buf)
            tot_weights.append(weight)

        start_fishdbc = time.time()
        fishdbcPar = fishdbc.FISHDBC(calc_dist, m, m0, vectorized=False, balanced_add=False)  
        candidate_edges = fishdbcPar.prepare_data(tot_adjs, tot_weights, positions)
        n = len(data)
        mst = fishdbcPar.create_mst(candidate_edges, n)
        _, _, _, ctree, _, _ = fishdbcPar.cluster_prova(mst)
        # print(graphs)
        # print("cand edges: ",candidate_edges, "\n\n")
        # print("mst: ",mst, "\n\n")
        # print("final clustering: ", ctree)
        end_fishdbc = time.time()
        print("The time of execution of Only FISHDBC is: ", (end_fishdbc - start_fishdbc))
        end_tot = time.time()
        print("The Tot time of execution of HNSW + FISHDBC is: ", (end_tot - start_tot))

        x, y = data[:, 0], data[:, 1]
        # plot_cluster_result(len(data), ctree, x, y, labels)

        shm_hnsw_data.close()
        shm_hnsw_data.unlink()
        shm_ent_point.close()
        shm_ent_point.unlink()
        shm_count.unlink()
        shm_count.close()
        for i in range(len(members)):
            shm_adj[i].close()
            shm_adj[i].unlink()
            shm_weights[i].close()
            shm_weights[i].unlink()

    else:
        start_time = time.time() 
        fishdbcSingle = fishdbc.FISHDBC(calc_dist, vectorized=False, balanced_add=False)            
        fishdbcSingle.update(data)
        print("The time of execution of HNSW is :", fishdbcSingle._tot_time)
        _, _, _, ctree, _, _ = fishdbcSingle.cluster()
        # print("Result of the cluster: ",ctree)
        end = time.time()
        print("The Tot time of execution of FISHDBC is :", (end-start_time))

        # plot_cluster_result(len(data), ctree, x, y, labels)




