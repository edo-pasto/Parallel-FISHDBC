
#%%
import hnswlib
import numpy as np
import pickle
import time
import sklearn.datasets
import pandas as pd
from scipy.spatial import distance, KDTree
from sklearn.neighbors import KDTree

"""
Example of search
"""
start = time.time()
dim = 2
num_elements = 20000

# Generating sample data
# data = np.float32(np.random.random((num_elements, dim)))

data, labels = sklearn.datasets.make_blobs(22000, centers=10)
np.random.shuffle(data)
Y = data[20000:]
data = data[:20000]
ids = np.arange(num_elements)
# Declaring index

p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
# p.set_num_threads(1)
# Initializing index - the maximum number of elements should be known beforehand
p.init_index(max_elements=num_elements, ef_construction=32, M=5)

# Element insertion (can be called several times):
p.add_items(data, ids)

# Controlling the recall by setting ef:
p.set_ef(32)  # ef should always be > k
end = time.time()
time_HNSWlib = "{:.2f}".format(end-start)
print("Total time of HNSWlib's HNSW creation: ", (time_HNSWlib))
# Query dataset, k - number of the closest elements (returns 2 numpy arrays)
labels_hnswlib, distances = p.knn_query(Y, k=5)

# Index objects support pickling
# WARNING: serialization via pickle.dumps(p) or p.__getstate__() is NOT thread-safe with p.add_items method!
# Note: ef parameter is included in serialization; random number generator is initialized with random_seed on Index load
# p_copy = pickle.loads(pickle.dumps(p))  # creates a copy of index p using pickle round-trip

### Index parameters are exposed as class properties:
# print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}")
# print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
# print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
# print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")

m = 5
m0 = 2 * m
# ## ----------------------------------- TEST FOR QUALITY OF THE SEARCH RESULTS ----------------------------- ##
X = data
kdt = KDTree(X, leaf_size=30, metric='euclidean')
knn_result = kdt.query(Y, k=5, return_distance=True)

diff_el_par = 0
for i, j, el_par in zip(knn_result[1], knn_result[0], labels_hnswlib):
    for n1, d1, n_par in zip(i, j, el_par):     
        if n1 != n_par: diff_el_par +=1

print(diff_el_par)
with open("qualityResultLib.csv", "a") as text_file:
    # text_file.write("DiffDistPar: " + str(diff_dist_par)+"\n")
    text_file.write(str(diff_el_par) + "\n")
    # text_file.write("Different Distances NOT Par: " + str(diff_dist))
    # text_file.write("Different Elements NOT Par: "+ str(diff_el))
df = pd.read_csv('./qualityResultLib.csv')
avg_err = df.loc[:,"DiffElemParall"].mean()
print("Average of Errors of HNSW Lib: ", avg_err )
perc_err = ((avg_err * 100 ) / (len(Y) * m) )
print("Percentage of Error of HNSW Lib", perc_err, "%")
print("Standard Deviation of Error: ", np.std(np.array( list(df["DiffElemParall"]))) )
print("Min: ",np.min(np.array( list(df["DiffElemParall"]))), "Max: ", np.max(np.array( list(df["DiffElemParall"]))) )


