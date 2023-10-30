import pandas as pd
import plotly.express as px
import numpy as np

# df1 = pd.DataFrame(dict(
#     DataSetSize = [5000, 10000, 20000, 40000, 80000],
#     Euclidean = [6.514, 13.469, 30.495, 67.469, 156.235],
#     SQEuclidean = [7.209, 14.779, 31.594, 70.127, 165.525],
#     Minkowski = [6.690, 14.515, 32.291, 74.312, 162.917],
#     Cosine = [10.900,21.225, 44.473, 94.223, 200.375 ],
# ))

# # print(df)
# fig1 = px.line(df1, x="DataSetSize", y=df1.columns[1:], title="Execution Time with various blob dataset's sizes using different distances functions ", markers=True) 
# fig1.update_layout(
#     xaxis_title="Data set Size",
#     yaxis_title="Execution Time",
#     legend_title="Distances"
# )

# fig1.show()
# ## ------------------------------------------------------- ##
# df8 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000, 160000],
#     ExecTime_single = [6.28, 13.30, 27.97, 59.03, 122.58],
#     ExecTime_parallel = [1.57, 2.82, 5.47, 10.62, 21.43 ],
   
# ))

# # print(df)
# fig8 = px.line(df8, x="DataSetSize", y=df8.columns[1:], title="Differences of execution time (mean over 10 runs) between Single and Parallel HNSW creation, Blob data set (Remote Linux machine, 16 physical core)", markers=True) 
# fig8.update_layout(
#     xaxis_title="Data set Size",
#     yaxis_title="Execution Time (mean)",
#     legend_title="Approaches"
# )

# fig8.show()

# ## ------------------------------------------------------------- ##
# df2 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000],
#     ExecTime_single = [10.81, 23.03, 47.75, 102.52],
#     ExecTime_parallel = [9.25, 21.40, 37.07, 77.32],
   
# ))

# fig2 = px.line(df2, x="DataSetSize", y=df2.columns[1:], title="Differences of execution time (mean over 10 runs) between Single and Parallel HNSW creation, Blob data set (MacBook, 4 physical core)", markers=True) 
# fig2.update_layout(
#     xaxis_title="Data set Size",
#     yaxis_title="Execution Time (mean)",
#     legend_title="Approaches"
# )

# fig2.show()
# ## ------------------------------------------------------------- ##
# df = pd.read_csv("../dataResults/qualityResult20.csv")
# data = list( (df["DiffElemParall"] / 10000) * 100 )
# sorted_data = np.sort(data)
# # Calculate the CDF values
# cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
# fig = px.line(x=sorted_data, y=cdf, title=" CDF of Result's search quality for 50 runs, 2000 input items over 20000 elements of Parallel HNSW, NO lock")

# # Customize the plot (optional)
# fig.update_xaxes(title='"%" of Errors', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig.update_yaxes(title='CDF', titlefont=dict(size=22),tickfont=dict(size=22)) 

# fig.update_yaxes(tickfont=dict(size=22)) 

# fig.show()

# ## ------------------------------------------------------------- ##
# df = pd.read_csv("../dataResults/qualityResult100.csv")
# data = list( (df["DiffElemParall"] / 50000) * 100 )
# sorted_data = np.sort(data)

# cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
# fig = px.line(x=sorted_data, y=cdf, title=" CDF of Result's search quality for 50 runs, 10000 input items over 100000 elements of Parallel HNSW, NO lock")

# fig.update_layout(
#     legend_title="Lines",
#     titlefont = dict(size=20)
# )
# fig.update_xaxes(title='"%" of Errors', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig.update_yaxes(title='CDF', titlefont=dict(size=22),tickfont=dict(size=22)) 

# fig.update_yaxes(tickfont=dict(size=22)) 

# fig.show()

# ## ------------------------------------------------------------- ##
# df = pd.read_csv("../dataResults/qualityResult20Single.csv")
# data = list( (df["DiffElemParall"] / 10000) * 100 )
# sorted_data = np.sort(data)
# cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
# fig = px.line(x=sorted_data, y=cdf, title=" CDF of Result's search quality for 50 runs, 2000 input items over 20000 elements of Single Process HNSW, NO lock")

# fig.update_layout(
#     legend_title="Lines",
#     titlefont = dict(size=20)
# )
# fig.update_xaxes(title='"%" of Errors', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig.update_yaxes(title='CDF', titlefont=dict(size=22),tickfont=dict(size=22)) 

# fig.update_yaxes(tickfont=dict(size=22)) 
# fig.show()


# ## ------------------------------------------------------------- ##
# df = pd.read_csv("../dataResults/qualityResult20Lock.csv")
# data = list( (df["DiffElemParall"] / 10000) * 100 )
# sorted_data = np.sort(data)
# cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
# fig = px.line(x=sorted_data, y=cdf, title=" CDF of Result's search quality for 50 runs, 2000 input items over 20000 elements of Parallel HNSW, WITH lock")

# fig.update_layout(
#     legend_title="Lines",
#     titlefont = dict(size=20)
# )
# fig.update_xaxes(title='"%" of Errors', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig.update_yaxes(title='CDF', titlefont=dict(size=22),tickfont=dict(size=22)) 

# fig.show()

# ## ------------------------------------------------------- ##
df = pd.read_csv("../dataResults/qualityResultLib.csv")
data = list( (df["DiffElemParall"] / 10000) * 100 )
sorted_data = np.sort(data)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
fig = px.line(x=sorted_data, y=cdf, title=" CDF of Result's search quality for 50 runs, 2000 input items over 20000 elements of Parallel Hnswlib")

fig.update_layout(
    legend_title="Lines",
    titlefont = dict(size=20)
)
fig.update_xaxes(title='"%" of Errors', titlefont=dict(size=22),tickfont=dict(size=22)) 
fig.update_yaxes(title='CDF', titlefont=dict(size=22),tickfont=dict(size=22)) 

fig.show()

# ## ------------------------------------------------------- ##
# df9 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000],
#     ExecTime_lock = [18.09,  36.74, 74.13, 150.69],
#     ExecTime_parallel = [1.57, 2.82, 5.47, 10.62 ],
   
# ))

# # print(df)
# fig9 = px.line(df9, x="DataSetSize", y=df9.columns[1:], title="Differences of execution time between Parallel HNSW creation with and without lock, Blob data set (Remote Linux machine, 16 physical core)", markers=True) 
# fig9.update_layout(
#     xaxis_title="Data set Size",
#     yaxis_title="Execution Time (mean)",
#     legend_title="Approaches"
# )

# fig9.show()
# ## ------------------------------------------------------- ##

# df9 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000, 160000],
#     ExecTime_Single = [6.28,  13.30, 27.97, 59.03, 122.58],
#     ExecTime_HNSWLib = [0.973, 1.502, 2.578, 4.871, 9.718 ],
   
# ))

# fig9 = px.line(df9, x="DataSetSize", y=df9.columns[1:], title="Differences of execution time between Single Process HNSW and Single Thread hnswlib, Blob data set (Remote Linux machine, 16 physical core)", markers=True) 

# fig9.update_layout(
#     legend_title="Approaches",
#     titlefont = dict(size=20),
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
# )
# fig9.update_xaxes(title='Data set Size', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig9.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig9.show()

## ------------------------------------------------------- ## 

# df9 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000, 160000],
#     ExecTime_Single = [1.57,  2.82, 5.47, 10.62, 21.43],
#     ExecTime_HNSWLib = [0.518, 0.565, 0.647, 0.807, 1.138 ],
   
# ))

# fig9 = px.line(df9, x="DataSetSize", y=df9.columns[1:], title="Differences of execution time between Parallel HNSW and Parallel hnswlib, Blob data set (Remote Linux machine, 16 physical core)", markers=True) 
# fig9.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig9.update_xaxes(title='Data set Size', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig9.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig9.show()
# ---------------------------------------------
# df9 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000, 160000],
#     ExecTime_Single = [6.503, 14.345, 28.238, 60.506, 129.952],
#     ExecTime_Parallel = [2.878, 6.143, 12.843, 26.772, 54.782],
   
# ))

# fig9 = px.line(df9, x="DataSetSize", y=df9.columns[1:], title="Differences of execution time between Parallel HNSW and Single HNSW, text data set (Remote Linux machine, 16 physical core)", markers=True) 
# fig9.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig9.update_xaxes(title='Data set Size', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig9.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig9.show()

# ---------------------------------------------
# 20000 parallel
# Mean of execution time:  18.408
# Standard Deviation of execution time:  0.494
# Min:  17.671 Max:  19.208

# 10000 parallel
# Mean of execution time:  8.137
# Standard Deviation of execution time:  0.233
# Min:  7.68 Max:  8.543

# parallel 40000
# Mean of execution time:  41.521
# Standard Deviation of execution time:  0.698
# Min:  40.587 Max:  42.743

# parallel 80000
# Mean of execution time:  96.544
# Standard Deviation of execution time:  2.145
# Min:  93.246 Max:  99.365

# parallel 160000
# Mean of execution time:  226.989
# Standard Deviation of execution time:  7.140
# Min:  218.351 Max:  237.817
# ---------------------------------------------
# single 10000
# Mean of execution time:  14.092
# Standard Deviation of execution time:  0.354
# Min:  13.52 Max:  14.73

# single 20000
# Mean of execution time:  30.511
# Standard Deviation of execution time:  1.247
# Min:  29.06 Max:  32.8

# single 40000
# Mean of execution time:  72.127
# Standard Deviation of execution time:  1.692
# Min:  69.89 Max:  74.43

# single 80000
# Mean of execution time:  161.852
# Standard Deviation of execution time:  3.983
# Min:  156.77 Max:  168.83

# single 160000
# Mean of execution time:  350.506
# Standard Deviation of execution time:  7.625
# Min:  337.97 Max:  359.19

# ------------------ ACCURACY FISHDBC -----------------
# blob
# Mean AMI:  0.83 , Mean NMI:  0.83 , Mean ARI:  0.83 , 
# Mean RI:  0.91 , Mean Homogeneity:  0.84 , Mean Completness: 
# 0.85 , Mean V-measure:  0.83

# synth
# Mean AMI:  0.90 , Mean NMI:  0.90 , Mean ARI:  0.93 , Mean RI:  0.97 , 
# Mean Homogeneity:  0.92 , Mean Completness:  0.89 , Mean V-measure:  0.90

# real numerical
# Mean AMI:  0.86 , Mean NMI:  0.86 , Mean ARI:  0.95 , Mean RI:  0.99 , 
# Mean Homogeneity:  0.87 , Mean Completness:  0.85 , Mean V-measure:  0.86

# text 
# Mean AMI:  0.62 , Mean NMI:  0.65 , Mean ARI:  0.40 , Mean RI:  0.89 ,
# Mean Homogeneity:  0.63 , Mean Completness:  0.66 , Mean V-measure:  0.65