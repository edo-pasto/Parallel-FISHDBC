import numpy as np
import sys
import pandas as pd
from functools import partial
from scipy.spatial import distance, KDTree
from Levenshtein import distance as lev
import matplotlib.pyplot as plt
from flexible_clustering import fishdbc
from flexible_clustering import hnsw_parallel
import sklearn.datasets
import collections
from sklearn.neighbors import KDTree
import time
from itertools import pairwise
from random import random
import cProfile
import multiprocessing
try:
    from math import log2
except ImportError: # Python 2.x or <= 3.2
    from math import log
    def log2(x):
        return log(x, 2)
# importlib.reload(fishdbc)
MISSING = sys.maxsize
MISSING_WEIGHT = sys.float_info.max

def calc_dist(x,y):
    return np.linalg.norm(x - y)
# def calc_dist(x,y):
#     return lev(x, y)

# def calc_dist(x,y):
#     return distance.jaccard(x,y)
def split(a, n):
    k, m = divmod(len(a), n)
    indices = [k * i + min(i, m) for i in range(n + 1)]
    return [a[l:r] for l, r in pairwise(indices)]

if __name__ == '__main__':


    # create the input dataset, data element for creating the hnsw, Y element for testing the search over it
    data, labels = sklearn.datasets.make_blobs(1000, centers=5, random_state=10)
    # np.random.shuffle(data)
    # Y = data[100000:]
    # data = data[:100000]

    m = 5
    m0 = 2 * m
## ----------------------------------- SINGLE PROCESS HNSW----------------------------------- ##
    start_single = time.time()
    fishdbc2 = fishdbc.FISHDBC(calc_dist, vectorized=False, balanced_add=False)
    single_cand_edges = fishdbc2.update(data)
    graphs = fishdbc2.the_hnsw._graphs
    time_singleHNSW = "{:.2f}".format(fishdbc2._tot_time)
    print("The time of execution Single HNSW:", (time_singleHNSW))
## ----------------------------------- PARALLEL HNSW  ----------------------------- ##
    multiprocessing.set_start_method('fork')
    
    start = time.time()

    # levels = []
    # for graph , i in zip(reversed(graphs), reversed(range(1, len(graphs)+1)) ):
    #     for g in graph.keys():
    #        if not any(element[0] == g for element in levels):
    #             levels.append((g, i))
    # levels = sorted(levels , key=lambda x: x[1])
    levels = [ (int(-log2(random())* (1 / log2(m))) + 1) for _ in range(len(data))]
    levels = sorted(enumerate(levels), key=lambda x: x[1])
    members = [[]]
    # insert the point in the list corresponding to the right level
    j = 1
    level_j = []
    for i in levels:
        elem, level = i
        if (level > j):
            members.append(level_j)
            level_j = []
            j = j+1
        level_j.append(elem)
        if (j-1 > 0):
            for i in range(j-1, 0, -1):
                members[i].append(elem)
    members.append(level_j)
    for i, l in zip(range(len(members)), members):
        sort = sorted(l)
        members[i] = sort
    del members[0]
    # print("Members: ", members, "\n")

    # create a list of dict to associate for each point in each levels its position
    positions = []
    for el, l in zip(members, range(len(members))):
        positions.append({})
        for i, x in enumerate(el):
            positions[l][x] = i
    # print("Positions: ", positions, "\n")
    # create the buffer of shared memory for each levels
    shm_hnsw_data = multiprocessing.shared_memory.SharedMemory(
        create=True, size=1000000000)
    shm_ent_point = multiprocessing.shared_memory.SharedMemory(
        create=True, size=10)
    shm_count = multiprocessing.shared_memory.SharedMemory(
    create=True, size=10)
    shm_time_localMST = multiprocessing.shared_memory.SharedMemory(
    create=True, size=10)

    shm_adj = []
    shm_weights = []
    for i in range(len(members)):
        npArray = np.zeros(shape=(len(members[i]), m0 if i == 0 else m) , dtype=int)
        shm1 = multiprocessing.shared_memory.SharedMemory(
            create=True, size=npArray.nbytes)
        np.ndarray(npArray.shape,
                dtype=int, buffer=shm1.buf)[:, :] = MISSING
        shm_adj.append(shm1)

        shm2 = multiprocessing.shared_memory.SharedMemory(
            create=True, size=npArray.nbytes)
        np.ndarray(npArray.shape, dtype=float, buffer=shm2.buf)[:, :] = MISSING_WEIGHT
        shm_weights.append(shm2)
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    # create the hnsw parallel class object and execute with pool the add function in multiprocessing
    hnsw = hnsw_parallel.HNSW(calc_dist, data, members, levels, positions,
                              shm_adj, shm_weights, shm_hnsw_data, shm_ent_point, shm_count, shm_time_localMST, lock, m=m, m0=m0, ef=32)

    end = time.time()
    print("The time of execution of preparation for hnsw:", (end-start))

    start_time = time.time()
    #for now add the first element not in multiprocessing
    start_time_hnsw_par = time.time()

    num_processes = 16

    distances_cache =[]

    candidate_edges = []

    distances_cache.append(hnsw.hnsw_add(0))
    pool = multiprocessing.Pool(num_processes)
    for dist_cache in pool.map(hnsw.hnsw_add, range(1, len(hnsw.data))):
        distances_cache.append(dist_cache)
    # dist_cache = hnsw.hnsw_add(0)
    # pool = multiprocessing.Pool(num_processes)
    # for _, candidates in pool.map(hnsw.add_and_compute_local_mst, split(range(1, len(data)), num_processes)):
    #     candidate_edges.extend(candidates)
        # candidate_edges.extend(candidates)
    pool.close()
    pool.join()

    end_time_hnsw_par = time.time()
    time_parHNSW = "{:.2f}".format(end_time_hnsw_par-start_time_hnsw_par)
    print("The time of execution of Paralell HNSW is :", (time_parHNSW))
## ------------------- TAKE AND SAVE TIME OF HNSW PARALLEL ----------------
    
    # with open("parallelHNSWText.csv", "a") as text_file:
    #     text_file.write(str(time_parHNSW) + "\n")

    # df = pd.read_csv('./parallelHNSWText.csv')
    # average = df.loc[:,"time"].mean()
    # print("Mean of execution time: ", average)
    # print("Standard Deviation of execution time: ", np.std(np.array( list(df["time"]))) )
    # print("Min: ",np.min(np.array( list(df["time"]))), "Max: ", np.max(np.array( list(df["time"]))) )
    # sh_count = np.ndarray(shape=(1), dtype=int, buffer=shm_count.buf)
    # print("The nbr of call to distance is :", (sh_count))
## ------------------------------------------------------------------------


    # # take the shared numpy array from the shared memory buffer and print them
    # start = time.time()
    # tot_adjs = []
    # tot_weights = []
    # for shm1, shm2, memb, i in zip(shm_adj, shm_weights, members, range(len(members))):
    #     adj = np.ndarray(shape=(len(memb), m0 if i == 0 else m),
    #                     dtype=int, buffer=shm1.buf)
    #     tot_adjs.append(adj)
    #     weight = np.ndarray(shape=(len(memb), m0 if i == 0 else m),
    #                         dtype=float, buffer=shm2.buf)
    #     tot_weights.append(weight)


    #     # df = pd.read_csv('./searchGraphTimes.csv')
    #     # sum_search = df.loc[:,"searchGraphTime"].mean()
    #     # print("AVG time of search graph: ", sum_search )
    #     # df2 = pd.read_csv('./selectHeuristicTimes.csv')
    #     # sum_select = df2.loc[:,"selectHeuristicTime"].mean()
    #     # print("AVG time of select Heuristic: ", sum_select)
    #     # print("weights: ", tot_weights, "\n")
    #     # print("Adjacency: ", tot_adjs, "\n")
    #     # start = time.time()



    # ## ----------------------------------- TEST FOR QUALITY OF THE SEARCH RESULTS ----------------------------- ##

    # graphs_par = []
    # for adjs, weights, i in zip(tot_adjs, tot_weights, range(len(tot_adjs))):
    #     dic = {}
    #     for adj, weight, pos in zip(adjs, weights, range(len(adjs))):
    #         dic2 = {}
    #         for a, w in zip(adj, weight):
    #             if a == MISSING:
    #                 continue
    #             dic2[a] = w
    #         idx =  list(positions[i].keys())[list(positions[i].values()).index(pos)]
    #         dic[idx] = dic2
    #     graphs_par.append(dic)
    # # if len(Y) == 2000:
    # X = data
    # kdt = KDTree(X, leaf_size=30, metric='euclidean')
    # knn_result = kdt.query(Y, k=5, return_distance=True)
    # # x, y = data[:, 7], data[:, 8]
    # # tree = KDTree(np.c_[x.ravel(), y.ravel()], metric='euclidean')
    # # dd, ii = tree.query(Y, k=5)
    # search_res_par = [hnsw.search(graphs_par, i,5) for i in Y]
    # # search_res = [fishdbc2.the_hnsw.search(graphs,i,5, test=True) for i in Y]
    # # print(len(search_res_par))
    # # print(len(dd))
    # # # compute the quality of the search results over the two hnsw with respect to a knn on a kdTree
    # diff_el_par = 0
    # diff_dist_par = 0
    # for i, j, el_par in zip(knn_result[1], knn_result[0], search_res_par,):
    #     for n1, d1, t_par in zip(i, j, el_par):
    #         n_par, d_par = t_par
    #         if n1 != n_par: diff_el_par +=1
    #         if d1 != d_par: diff_dist_par +=1
    # print(diff_el_par)
    # # with open("qualityResult100.csv", "a") as text_file:
    # #     # text_file.write("DiffDistPar: " + str(diff_dist_par)+"\n")
    # #     text_file.write(str(diff_el_par) + "\n")
    # #     # text_file.write("Different Distances NOT Par: " + str(diff_dist))
    # #     # text_file.write("Different Elements NOT Par: "+ str(diff_el))
    # # df = pd.read_csv('./qualityResult100.csv')
    # # avg_err = df.loc[:,"DiffElemParall"].mean()
    # # print("Average of Errors of HNSW Parallel: ", avg_err )
    # # perc_err = ((avg_err * 100 ) / (len(Y) * m) )
    # # print("Percentage of Error of HNSW Parallel", perc_err, "%")
    # # print("Standard Deviation of Error: ", np.std(np.array( list(df["DiffElemParall"]))) )
    # # print("Min: ",np.min(np.array( list(df["DiffElemParall"]))), "Max: ", np.max(np.array( list(df["DiffElemParall"]))) )



    # close and unlink the shared memory objects
    shm_hnsw_data.close()
    shm_hnsw_data.unlink()
    shm_ent_point.close()
    shm_ent_point.unlink()
    shm_count.unlink()
    shm_count.close()
    shm_time_localMST.unlink()
    shm_time_localMST.close()
    for i in range(len(members)):
        shm_adj[i].close()
        shm_adj[i].unlink()
        shm_weights[i].close()
        shm_weights[i].unlink()
