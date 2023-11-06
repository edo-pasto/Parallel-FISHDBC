import numpy as np
import sys
import pandas as pd
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

# import cProfile
import multiprocessing

try:
    from math import log2
except ImportError:  # Python 2.x or <= 3.2
    from math import log

    def log2(x):
        return log(x, 2)


# importlib.reload(fishdbc)
MISSING = sys.maxsize
MISSING_WEIGHT = sys.float_info.max


def calc_dist(x, y):
    return np.linalg.norm(x - y)


# def calc_dist(x,y):
#     return lev(x[0], y[0])


def split(a, n):
    k, m = divmod(len(a), n)
    indices = [k * i + min(i, m) for i in range(n + 1)]
    return [a[l:r] for l, r in pairwise(indices)]


def plot_cluster_result(size, ctree, x, y, labels):
    plt.figure(figsize=(9, 9))
    plt.gca().set_aspect("equal")
    #     fishdbc.update(points) #points --> [[1 2],[3 4],[5 6], ...]
    nknown = size  # data is a list of np array --> [array([1, 3]), array([2,4]), ...]
    #     _, _, _, ctree, _, _ = fishdbc.cluster()# ctree is a numpy.ndarray of (,,,)--> [(200, 50, 1.87, 1),(201, 50, 1.98,1),...]
    clusters = collections.defaultdict(set)  # cluster is a dict of set -->
    # ask explanation of this for loop
    for parent, child, lambda_val, child_size in ctree[::-1]:
        if child_size == 1:
            clusters[parent].add(
                child
            )  # at the set corresponding to the key parent add the integer child
        else:
            assert len(clusters[child]) == child_size
            clusters[parent].update(
                clusters[child]
            )  # at the set corresponding to the key parent add
            # all the different element of the set corresponding to the key child
    clusters = sorted(
        clusters.items()
    )  # order the set in the dict based on the key and now became a list
    xknown, yknown, labels_known = x[:nknown], y[:nknown], labels[:nknown]
    color = ["rgbcmyk"[l % 7] for l in labels_known]
    plt.scatter(xknown, yknown, c=color, linewidth=0)
    plt.show(block=False)
    for _, cluster in clusters:
        plt.waitforbuttonpress()
        plt.gca().clear()
        color = ["kr"[i in cluster] for i in range(nknown)]
        plt.scatter(xknown, yknown, c=color, linewidth=0)
        plt.draw()


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    ## ---------------------- BLOB DATASET NUMERICAL --------------------------- ##
    # create the input dataset, data element for creating the hnsw, Y element for testing the search over it
    data, labels = sklearn.datasets.make_blobs(160000, centers=5, random_state=10)
    # np.random.shuffle(data)
    # Y = data[20000:]
    # data = data[:20000]

    ## ----------------------- SYNTH DATASET TEXTUAL ---------------------------- ##
    # realData = pd.read_csv('../data/textDataset10.csv', index_col=False)
    # labels = pd.read_csv('../data/textDatasetLabels10.csv', index_col=False)
    # # labels = list(labels["label"])
    # labels = labels.values
    # li = realData.values.tolist()
    # data = np.asarray(li)
    # shuffled_indices = np.arange(len(data))
    # np.random.shuffle(shuffled_indices)
    # # Use the shuffled indices to rearrange both elements and labels
    # data = data[shuffled_indices]
    # labels = labels[shuffled_indices]
    # labels = [item for sublist in labels for item in sublist]

    ## ----------------------- SYNTH DATASET NUMERICAL ---------------------------- ##
    # realIntData = pd.read_csv('../data/banana.csv')
    # data = realIntData.values
    # data = tot_data[-1500:]
    # Y = tot_data[:-1500]

    ## -------------------------- REAL DATASET NUMERICAL ------------------------- ##
    # bunch = sklearn.datasets.fetch_california_housing()
    # data = bunch.data
    # labels = bunch.target
    # data = np.array(data)

    m = 5
    m0 = 2 * m
    ## ----------------------------------- SINGLE PROCESS ----------------------------------- ##
    print(
        "-------------------------- TIME RESULTS - SINGLE PROCESS FISHDBC --------------------------"
    )
    start_single = time.time()
    fishdbc2 = fishdbc.FISHDBC(calc_dist, vectorized=False, balanced_add=False)
    single_cand_edges = fishdbc2.update(data)
    graphs = fishdbc2.the_hnsw._graphs
    time_singleHNSW = "{:.2f}".format(fishdbc2._tot_time)
    print("The time of execution Single HNSW:", (time_singleHNSW))
    time_singleMST = "{:.2f}".format(fishdbc2._tot_MST_time)
    print("The time of execution Single MST:", (time_singleMST))
    labels_cluster, _, _, ctree, _, _ = fishdbc2.cluster(parallel=False)

    # with open("../dataResults/singleMSTText.csv", "a") as text_file:
    #     text_file.write(str(time_singleMST) + "\n")

    # df = pd.read_csv('../dataResults/singleMSTText.csv')
    # average = df.loc[:,"time"].mean()
    # print("------ SINGLE MST TIME -----")
    # print("Mean of execution time: ", average)
    # print("Standard Deviation of execution time: ", np.std(np.array( list(df["time"]))) )
    # print("Min: ",np.min(np.array( list(df["time"]))), "Max: ", np.max(np.array( list(df["time"]))) )

    print("Final Clustering NOT Parallel: ",ctree)
    # print("labels result from cluster: ", list(labels_cluster))
    end_single = time.time()
    time_singleFISHDBC = end_single - start_single
    print(
        "The time of execution Single FISHDBC:",
        "{:.3f}".format(time_singleFISHDBC),
    )

    # with open("../dataResults/singleFISHDBCText.csv", "a") as text_file:
    #     text_file.write(str(time_singleFISHDBC) + "\n")

    # df = pd.read_csv('../dataResults/singleFISHDBCText.csv')
    # average = df.loc[:,"time"].mean()
    # print("------ SINGLE FISHDBC TIME -----")
    # print("Mean of execution time: ", average)
    # print("Standard Deviation of execution time: ", np.std(np.array( list(df["time"]))) )
    # print("Min: ",np.min(np.array( list(df["time"]))), "Max: ", np.max(np.array( list(df["time"]))) )

    print(
        "___________________________________________________________________________________________\n"
    )
    ## ----------------------------------- PARALLEL HNSW E FISHDBC ----------------------------- ##
    print(
        "-------------------------- TIME RESULTS - MULTI-PROCESS FISHDBC --------------------------"
    )
    start = time.time()
    # levels = []
    # for graph , i in zip(reversed(graphs), reversed(range(1, len(graphs)+1)) ):
    #     for g in graph.keys():
    #        if not any(element[0] == g for element in levels):
    #             levels.append((g, i))
    # levels = sorted(levels , key=lambda x: x[1])
    levels = [(int(-log2(random()) * (1 / log2(m))) + 1) for _ in range(len(data))]
    levels = sorted(enumerate(levels), key=lambda x: x[1])
    members = [[]]
    # insert the point in the list corresponding to the right level
    j = 1
    level_j = []
    for i in levels:
        elem, level = i
        if level > j:
            members.append(level_j)
            level_j = []
            j = j + 1
        level_j.append(elem)
        if j - 1 > 0:
            for i in range(j - 1, 0, -1):
                members[i].append(elem)
    members.append(level_j)
    for i, l in zip(range(len(members)), members):
        sort = sorted(l)
        members[i] = sort
    del members[0]

    # create a list of dict to associate for each point in each levels its position
    positions = []
    for el, l in zip(members, range(len(members))):
        positions.append({})
        for i, x in enumerate(el):
            positions[l][x] = i

    # create the buffer of shared memory for each levels
    shm_hnsw_data = multiprocessing.shared_memory.SharedMemory(
        create=True, size=1000000000
    )
    shm_ent_point = multiprocessing.shared_memory.SharedMemory(create=True, size=10)
    shm_count = multiprocessing.shared_memory.SharedMemory(create=True, size=10)

    shm_adj = []
    shm_weights = []
    for i in range(len(members)):
        npArray = np.zeros(shape=(len(members[i]), m0 if i == 0 else m), dtype=int)
        shm1 = multiprocessing.shared_memory.SharedMemory(
            create=True, size=npArray.nbytes
        )
        np.ndarray(npArray.shape, dtype=int, buffer=shm1.buf)[:, :] = MISSING
        shm_adj.append(shm1)

        shm2 = multiprocessing.shared_memory.SharedMemory(
            create=True, size=npArray.nbytes
        )
        np.ndarray(npArray.shape, dtype=float, buffer=shm2.buf)[:, :] = MISSING_WEIGHT
        shm_weights.append(shm2)

    num_processes = 16
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    # create the hnsw parallel class object and execute with pool the add function in multiprocessing
    hnsw = hnsw_parallel.HNSW(
        calc_dist,
        data,
        members,
        levels,
        positions,
        shm_adj,
        shm_weights,
        shm_hnsw_data,
        shm_ent_point,
        shm_count,
        lock,
        m=m,
        m0=m0,
        ef=32,
    )

    end = time.time()
    print("The time of execution of preparation for hnsw:", (end - start))
    start_time = time.time()
    start_time_hnsw_par = time.time()
    ## COMPUTE ONLY PARALLEL HNSW
        # distances_cache =[]
        # distances_cache.append(hnsw.hnsw_add(0))
        # pool = multiprocessing.Pool(num_processes)
        # for dist_cache in pool.map(hnsw.hnsw_add, range(1, len(hnsw.data))):
        #     distances_cache.append(dist_cache)
        # pool.close()
        # pool.join()
        # end_time_hnsw_par = time.time()
        # time_parHNSW = "{:.2f}".format(end_time_hnsw_par-start_time_hnsw_par)
        # print("The time of execution of Paralell HNSW is :", (time_parHNSW))
    partial_mst = []
    mst_times = []
    hnsw.hnsw_add(0)
    pool = multiprocessing.Pool(num_processes)
    for local_mst, mst_time in pool.map(
        hnsw.add_and_compute_local_mst, split(range(1, len(data)), num_processes)
    ):
        # candidate_edges.extend(partial_mst)
        mst_times.append(mst_time)
        partial_mst.extend(local_mst)
    pool.close()
    pool.join()

    end_time_hnsw_par = time.time()
    time_parHNSW = "{:.2f}".format(end_time_hnsw_par - start_time_hnsw_par)
    print("The time of execution of Paralell HNSW and local MSTs is :", (time_parHNSW))
    time_localMST = np.mean(mst_times)
    print(
        "The time of execution of Paralell local MSTs is :",
        "{:.3f}".format(time_localMST),
    )

    ## ------------------- TAKE AND SAVE TIME OF HNSW PARALLEL ----------------

    # with open("../dataResults/parallelHNSW.csv", "a") as text_file:
    #     text_file.write(str(time_parHNSW) + "\n")

    # df = pd.read_csv('../dataResults/parallelHNSW.csv')
    # average = df.loc[:,"time"].mean()
    # print("Mean of execution time: ", average)
    # print("Standard Deviation of execution time: ", np.std(np.array( list(df["time"]))) )
    # print("Min: ",np.min(np.array( list(df["time"]))), "Max: ", np.max(np.array( list(df["time"]))))
    # # sh_count = np.ndarray(shape=(1), dtype=int, buffer=shm_count.buf)
    # # print("The nbr of call to distance is :", (sh_count))
    ## ------------------------------------------------------------------------

    # take the shared numpy array (The HNSW structure) from the shared memory buffer and print them

    tot_adjs = []
    tot_weights = []
    for shm1, shm2, memb, i in zip(shm_adj, shm_weights, members, range(len(members))):
        adj = np.ndarray(
            shape=(len(memb), m0 if i == 0 else m), dtype=int, buffer=shm1.buf
        )
        tot_adjs.append(adj)
        weight = np.ndarray(
            shape=(len(memb), m0 if i == 0 else m), dtype=float, buffer=shm2.buf
        )
        tot_weights.append(weight)
    # print(tot_adjs, "\n", tot_weights, "\n")

    start = time.time()
    # perform the final fishdbc operation, the creation of the mst and the final clustering
    fishdbc1 = fishdbc.FISHDBC(calc_dist, m, m0, vectorized=False, balanced_add=False)
    # final_mst = hnsw.global_mst(distances_cache)
    final_mst = hnsw.global_mst(shm_adj, shm_weights, partial_mst, len(data))
    end = time.time()
    time_globalMST = end - start
    print("The time of execution of global MST is :", "{:.3f}".format(time_globalMST))
    time_parallelMST = time_localMST + time_globalMST
    print(
        "The total time of execution of MST is :",
        "{:.3f}".format(time_parallelMST),
    )

    # with open("../dataResults/parallelMST.csv", "a") as text_file:
    #     text_file.write(str(time_parallelMST) + "\n")

    # df = pd.read_csv('../dataResults/parallelMST.csv')
    # average = df.loc[:,"time"].mean()
    # print("------ PARALLEL MST TIME -----")
    # print("Mean of execution time: ", average)
    # print("Standard Deviation of execution time: ", np.std(np.array( list(df["time"]))) )
    # print("Min: ",np.min(np.array( list(df["time"]))), "Max: ", np.max(np.array( list(df["time"]))) )

    n = len(data)
    labels_cluster_par, _, _, ctree, _, _ = fishdbc1.cluster(final_mst, parallel=True)
    # print("Final Clustering Parallel: ", ctree, "\n")
    # print("labels result from cluster PAR: ", list(labels_cluster_par), "\n")
    end = time.time()
    time_parallelFISHDBC = "{:.3f}".format(end - start_time)
    print("The time of execution of Parallel FISHDBC is :", time_parallelFISHDBC)

    # with open("../dataResults/parallelFISHDBC.csv", "a") as text_file:
    #     text_file.write(str(time_parallelFISHDBC) + "\n")

    # df = pd.read_csv('../dataResults/parallelFISHDBC.csv')
    # average = df.loc[:,"time"].mean()
    # print("------ PARALLEL FISHDBC TIME -----")
    # print("Mean of execution time: ", "{:.3f}".format(average))
    # print("Standard Deviation of execution time: ", "{:.3f}".format(np.std(np.array( list(df["time"])))))
    # print("Min: ",np.min(np.array( list(df["time"]))), "Max: ", np.max(np.array( list(df["time"]))) )

    print(
        "___________________________________________________________________________________________\n"
    )
    ## ----------------------------------- HNSW - TEST FOR QUALITY OF THE SEARCH RESULTS ----------------------------- ##

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
    # # x, y = data[:, 0], data[:, 1]
    # # tree = KDTree(np.c_[x.ravel(), y.ravel()], metric='euclidean')
    # # dd, ii = tree.query(Y, k=5)
    # search_res_par = [hnsw.search(graphs_par, i,5, ef=32) for i in Y]
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
    # print("Number of errors: ",diff_el_par)
    # with open("../dataResults/qualityResult100.csv", "a") as text_file:
    #     # text_file.write("DiffDistPar: " + str(diff_dist_par)+"\n")
    #     text_file.write(str(diff_el_par) + "\n")
    #     # text_file.write("Different Distances NOT Par: " + str(diff_dist))
    #     # text_file.write("Different Elements NOT Par: "+ str(diff_el))
    # df = pd.read_csv('../dataResults/qualityResult100.csv')
    # avg_err = df.loc[:,"DiffElemParall"].mean()
    # print("Average of Errors of HNSW Parallel: ", avg_err )
    # perc_err = ((avg_err * 100 ) / (len(Y) * m) )
    # print("Percentage of Error of HNSW Parallel", perc_err, "%")
    # print("Standard Deviation of Error: ", np.std(np.array( list(df["DiffElemParall"]))) )
    # print("Min: ",np.min(np.array( list(df["DiffElemParall"]))), "Max: ", np.max(np.array( list(df["DiffElemParall"]))) )

    ## ----------------------------------- TEST FOR QUALITY OF CLUSTERING RESULT --------------------------------- ##
    from sklearn.metrics.cluster import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        rand_score,
        normalized_mutual_info_score,
        homogeneity_completeness_v_measure,
    )

    # AMI = adjusted_mutual_info_score(labels,labels_cluster_par)
    # NMI = normalized_mutual_info_score(labels,labels_cluster_par)
    # ARI = adjusted_rand_score(labels, labels_cluster_par)
    # RI = rand_score(labels, labels_cluster_par)
    # homogeneity, completness, v_measure = homogeneity_completeness_v_measure(
    #     labels, labels_cluster_par
    # )
    # clustEval = pd.read_csv("../dataResults/clustEval.csv", index_col=False , sep=',')
    # clustEval.loc[len(clustEval)] = [AMI,NMI,ARI,RI,homogeneity, completness,v_measure]
    # avgAMI, stdAMI = (np.mean(np.array( list(clustEval["AMI"]))) , np.std(np.array( list(clustEval["AMI"]))) )
    # avgNMI, stdNMI = (np.mean(np.array( list(clustEval["NMI"]))) , np.std(np.array( list(clustEval["AMI"])))  )
    # avgARI, stdARI = (np.mean(np.array( list(clustEval["ARI"]))), np.std(np.array( list(clustEval["AMI"])))  )
    # avgRI, stdRI = (np.mean(np.array( list(clustEval["RI"]))), np.std(np.array( list(clustEval["AMI"])))  )
    # avgH, stdH = (np.mean(np.array( list(clustEval["H"]))), np.std(np.array( list(clustEval["AMI"])))  )
    # avgC, stdC = (np.mean(np.array( list(clustEval["C"]))), np.std(np.array( list(clustEval["AMI"])))  )
    # avgV, stdV = (np.mean(np.array( list(clustEval["V"]))), np.std(np.array( list(clustEval["AMI"])))  )

    # print("Mean AMI: ", "{:.2f}".format(avgAMI),", Mean NMI: ", "{:.2f}".format(avgNMI), ", Mean ARI: ",
    #       "{:.2f}".format(avgARI), ", Mean RI: ", "{:.2f}".format(avgRI),", Mean Homogeneity: ", "{:.2f}".format(avgH),
    #       ", Mean Completness: ", "{:.2f}".format(avgC), ", Mean V-measure: ", "{:.2f}".format(avgV))
    # print("Std. Dev. AMI: ", "{:.2f}".format(stdAMI),", Std. Dev. NMI: ", "{:.2f}".format(stdNMI), ", Std. Dev. ARI: ",
    #       "{:.2f}".format(stdARI), ", Std. Dev. RI: ", "{:.2f}".format(stdRI),", Std. Dev. Homogeneity: ", "{:.2f}".format(stdH),
    #       ", Std. Dev. Completness: ", "{:.2f}".format(stdC), ", Std. Dev. V-measure: ", "{:.2f}".format(stdV))
    # clustEval.to_csv('../dataResults/clustEval.csv', index=False)
    # print(
    #     "Adjsuted Mutual Info Score: ",
    #     "{:.2f}".format(adjusted_mutual_info_score(labels, labels_cluster_par)),
    # )
    # print(
    #     "Normalized Mutual Info Score: ",
    #     "{:.2f}".format(normalized_mutual_info_score(labels, labels_cluster_par)),
    # )
    # print(
    #     "Adjusted Rand Score: ",
    #     "{:.2f}".format(adjusted_rand_score(labels, labels_cluster_par)),
    # )
    # print("Rand Score: ", "{:.2f}".format(rand_score(labels, labels_cluster_par)))
    # print(
    #     "Homogeneity, Completness, V-Measure: ", (homogeneity, completness, v_measure)
    # )

    # close and unlink the shared memory objects
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
