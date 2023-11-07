import numpy as np
import time
import sklearn.datasets
import pandas as pd
import collections
import argparse
from scipy.spatial import distance, KDTree
from sklearn.neighbors import KDTree
from Levenshtein import distance as lev
import matplotlib.pyplot as plt
from flexible_clustering import fishdbc
from line_profiler import LineProfiler

import cProfile
try:
    from math import log2
except ImportError:  # Python 2.x or <= 3.2
    from math import log

def log2(x):
    return log(x, 2)


parser = argparse.ArgumentParser(description="Scacca")
parser.add_argument('--dataset', type=str, default='blob',
                        help="dataset used by the algorithm (default: blob)."
                        "try with: blob, string,")
parser.add_argument('--distance', type=str, default='hamming',
                    help="distance metrix used by FISHDBC (default: hamming)."
                    "try with: euclidean, squeclidean, cosine, dice, minkowsky, jaccard, hamming, jensenShannon, levensthein")
parser.add_argument('--nitems', type=int, default=200,
                    help="Number of items (default 200).")
parser.add_argument('--centers', type=int, default=5,
                    help="Number of centers for the clusters generated "
                    "(default 5).")
args = parser.parse_args()
dist = args.distance.lower()
dataset = args.dataset.lower()

if dataset == 'blob':
    data = []
    # data, labels = sklearn.datasets.make_blobs(args.nitems, 
    #                                     centers=args.centers,)
    bunch = sklearn.datasets.fetch_california_housing()
    data = bunch.data
    labels = bunch.target
    data = np.array(data)

    if dist == 'euclidean': 
        def calc_dist(x,y):     
            # return np.linalg.norm(x - y)
            return distance.euclidean(x, y)
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
    realData = pd.read_csv('../data/textDataset160.csv', index_col=False)
    labels = pd.read_csv('../data/textDatasetLabels160.csv', index_col=False)
    # labels = list(labels["label"])
    labels = labels.values
    li = realData.values.tolist()
    data = np.asarray(li)
    shuffled_indices = np.arange(len(data))
    np.random.shuffle(shuffled_indices)
   # Use the shuffled indices to rearrange both elements and labels
    data = data[shuffled_indices]
    labels = labels[shuffled_indices]
    labels = [item for sublist in labels for item in sublist]
    if dist == 'jaccard':
        def calc_dist(x,y):
            return distance.jaccard(x,y)
    elif dist == 'hamming':
        def calc_dist(x,y):
            return distance.hamming(x,y)
    elif dist == "levensthein":
        def calc_dist(x,y):
            return lev(x[0], y[0])
    else:
        raise EnvironmentError("At the moment the specified distance is not available for the string dataset,"
                                " try with: jaccard, hamming, levensthein")
else:
    raise EnvironmentError("The specified dataset doesn't exist at the moment,"
    "try with: blob, text")   
#create the input dataset, data element for creating the hnsw, Y element for testing the search over it
# data, labels = sklearn.datasets.make_blobs(10000 , centers=10)
# x, y = data[:, 0], data[:, 1]
# np.random.shuffle(data)
# Y = data[100000:]
# data = data[:100000]
# realData = pd.read_csv('../data/Air_Traffic_Passenger_Statistics.csv')
# li = realData.values.tolist()
# data = np.asarray(li)
m = 5
m0 = 2 * m
## ----------------------------------- NORMAL HNSW E FISHDBC ----------------------------- ##
start_time = time.time() 
fishdbc2 = fishdbc.FISHDBC(calc_dist, vectorized=False, balanced_add=False)   
# cProfile.run("fishdbc2.update(data)", "normal_hnsw_prof.prof")

fishdbc2.update(data)
time_singleHNSW = "{:.2f}".format(fishdbc2._tot_time)
print("The time of execution Single HNSW:", (time_singleHNSW))

# ------------------- SAVE HNSW TIME --------------------
with open("../dataResults/singleHNSW.csv", "a") as text_file:
    text_file.write(str(time_singleHNSW) + "\n")

df = pd.read_csv('../dataResults/singleHNSW.csv')
average = df.loc[:,"time"].mean()
print("Mean of execution time: ", average)
print("Standard Deviation of execution time: ", np.std(np.array( list(df["time"]))) )
print("Min: ",np.min(np.array( list(df["time"]))), "Max: ", np.max(np.array( list(df["time"]))) )

# # graphs = fishdbc2.the_hnsw._graphs
# # print("HNSW Graphs NOT Parallel: ",graphs, "\n")
# # start = time.time()
# _, _, _, ctree, _, _ = fishdbc2.cluster()
# # # print("Final Clustering NOT Parallel: ",ctree)
# end = time.time()
# time_singleFISHDBC = "{:.2f}".format(end-start_time)
# print("The time of execution of normal FISHDBC is :", time_singleFISHDBC)

# ------------------- SAVE FISHDBC TIME --------------------
# with open("../dataResults/singleFISHDBCText.csv", "a") as text_file:
#     text_file.write(str(time_singleFISHDBC) + "\n")

# df = pd.read_csv('../dataResults/singleFISHDBCText.csv')
# average = df.loc[:,"time"].mean()
# print("Mean of execution time: ", "{:.3f}".format(average))
# print("Standard Deviation of execution time: ", "{:.3f}".format(np.std(np.array( list(df["time"])))))
# print("Min: ",np.min(np.array( list(df["time"]))), "Max: ", np.max(np.array( list(df["time"]))) )

# x, y = data[:, 0], data[:, 1]
# tree = KDTree(np.c_[x.ravel(), y.ravel()], metric='euclidean')
# dd, ii = tree.query(Y, k=5)
# search_res = [fishdbc2.the_hnsw.search(graphs,i,5, test=True) for i in Y]
# # print(len(search_res_par))
# # print(len(dd))
# # # compute the quality of the search results over the two hnsw with respect to a knn on a kdTree
# diff_el = 0
# for i, j, el in zip(ii, dd,search_res):
#     for n1, d1, t in zip(i, j, el):
#         n, d = t
#         if n1 != n: diff_el +=1
# print(diff_el)
# with open("../dataResults/qualityResult100Single.csv", "a") as text_file:
#     # text_file.write("DiffDistPar: " + str(diff_dist_par)+"\n")
#     text_file.write(str(diff_el) + "\n")
#     # text_file.write("Different Distances NOT Par: " + str(diff_dist))
#     # text_file.write("Different Elements NOT Par: "+ str(diff_el))
# df = pd.read_csv('../dataResults/qualityResult100Single.csv')
# avg_err = df.loc[:,"DiffElemParall"].mean()
# print("Average of Errors of HNSW Parallel: ", avg_err )
# perc_err = ((avg_err * 100 ) / (len(Y) * m) )
# print("Percentage of Error of HNSW Parallel", perc_err, "%")
# print("Standard Deviation of Error: ", np.std(np.array( list(df["DiffElemParall"]))) )
# print("Min: ",np.min(np.array( list(df["DiffElemParall"]))), "Max: ", np.max(np.array( list(df["DiffElemParall"]))) )
# lp = LineProfiler()
# lp.add_function(fishdbc2.update)
# # lp.add_function(fishdbc2.add)
# # lp.add_function(fishdbc2.update_mst)
# # lp.add_function(fishdbc2.cluster)
# lp.enable_by_count()