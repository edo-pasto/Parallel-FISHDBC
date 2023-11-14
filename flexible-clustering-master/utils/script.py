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

# fig1 = px.line(df1, x="DataSetSize", y=df1.columns[1:], title="Execution Time with various blob dataset's sizes using different distances functions ", markers=True) 
# fig1.update_layout(
#     legend_title="Distances",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig1.update_xaxes(title='Data set Size', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig1.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig1.show()
# ## ------------------------------------------------------- ##
# df8 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000, 160000],
#     ExecTime_single = [6.28, 13.30, 27.97, 59.03, 122.58],
#     ExecTime_parallel = [1.57, 2.82, 5.47, 10.62, 21.43 ],
   
# ))

# # print(df)
# fig2 = px.line(df8, x="DataSetSize", y=df8.columns[1:], title="Differences of execution time (mean over 10 runs) between Single and Parallel HNSW creation, Blob data set (Remote Linux machine, 16 physical core)", markers=True) 

# fig2.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig2.update_xaxes(title='Data set Size', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.show()

# ## ------------------------------------------------------------- ##
# df2 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000],
#     ExecTime_single = [10.81, 23.03, 47.75, 102.52],
#     ExecTime_parallel = [9.25, 21.40, 37.07, 77.32],
   
# ))

# fig2 = px.line(df2, x="DataSetSize", y=df2.columns[1:], title="Differences of execution time (mean over 10 runs) between Single and Parallel HNSW creation, Blob data set (MacBook, 4 physical core)", markers=True) 

# fig2.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig2.update_xaxes(title='Data set Size', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
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
# df = pd.read_csv("../dataResults/qualityResultLib.csv")
# data = list( (df["DiffElemParall"] / 10000) * 100 )
# sorted_data = np.sort(data)
# cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
# fig = px.line(x=sorted_data, y=cdf, title=" CDF of Result's search quality for 50 runs, 2000 input items over 20000 elements of Parallel Hnswlib")

# fig.update_layout(
#     legend_title="Lines",
#     titlefont = dict(size=20)
# )
# fig.update_xaxes(title='"%" of Errors', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig.update_yaxes(title='CDF', titlefont=dict(size=22),tickfont=dict(size=22)) 

# fig.show()

# ## ------------------------------------------------------- ##
# df9 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000, 160000],
#     ExecTime_lock = [18.09,  36.74, 74.13, 150.69, 304.58],
#     ExecTime_NO_lock = [1.57, 2.82, 5.47, 10.62, 27.92 ],
   
# ))

# fig9 = px.line(df9, x="DataSetSize", y=df9.columns[1:], title="Differences of execution time between Parallel HNSW creation with and without lock, Blob data set (Remote Linux machine, 16 physical core)", markers=True) 

# fig9.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig9.update_xaxes(title='Number of processes', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig9.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
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

# df2 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000, 160000],
#     ExecTime_single = [  6.316, 15.658, 38.314, 93.003, 215.971],
#     ExecTime_parallel = [0.880, 2.136, 5.376, 13.281, 33.848],
   
# ))

# fig2 = px.line(df2, x="DataSetSize", y=df2.columns[1:], title="Differences of execution time (mean over 10 runs) between Single and Parallel MST creation, Blob data set (Remote machine, 16 cores)", markers=True) 

# fig2.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig2.update_xaxes(title='Data set Size', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.show()


# ---------------------------------------------

# df2 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000, 160000],
#     ExecTime_single = [ 12.799, 29.001, 66.417, 151.879, 339.532],
#     ExecTime_parallel = [ 3.323, 6.894, 14.804, 33.659, 76.783],
   
# ))

# fig2 = px.line(df2, x="DataSetSize", y=df2.columns[1:], title="Differences of execution time (mean over 10 runs) between Single and Parallel FISHDBC Algorithm, Blob data set (Remote machine, 16 cores)", markers=True) 

# fig2.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig2.update_xaxes(title='Data set Size', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.show()
# ---------------------------------------------
# df2 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000, 160000],
#     ExecTime_single = [ 9.707, 27.956, 63.832, 158.977, 365.48],
#     ExecTime_parallel = [ 1.001, 2.942, 6.203, 15.090, 37.976],
   
# ))

# fig2 = px.line(df2, x="DataSetSize", y=df2.columns[1:], title="Differences of execution time Original Single MST vs. Parallel MST creation, text data sets (remote machine, 16 physical core)", markers=True) 

# fig2.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig2.update_xaxes(title='Data set Size', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.show()

# ---------------------------------------------
# df2 = pd.DataFrame(dict(
#     DataSetSize = [10000, 20000, 40000, 80000, 160000],
#     ExecTime_single = [ 15.661, 41.757, 92.022, 218.781, 493.937],
#     ExecTime_parallel = [ 3.004, 7.134, 14.448, 31.842, 72.534],
   
# ))

# fig2 = px.line(df2, x="DataSetSize", y=df2.columns[1:], title="Differences of execution time, Single vs. Parallel FISHDBC algorithm, text data sets (remote machine, 16 physical core)", markers=True) 

# fig2.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig2.update_xaxes(title='Data set Size', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.show()
# ---------------------------------------------

# df2 = pd.DataFrame(dict(
#     NumProcs = [1, 2, 4, 8, 16],
#     ExecTime_20K = [ 17.328, 9.156, 5.080, 2.991, 2.101],
#     ExecTime_10K = [109.758, 68.322, 39.911, 25.053, 17.481],
   
# ))

# fig2 = px.line(df2, x="NumProcs", y=df2.columns[1:], title="Differences of execution time Parallel MST with different cores , blob data sets (remote machine)", markers=True) 

# fig2.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig2.update_xaxes(title='Number of processes', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.show()

# ---------------------------------------------

# df2 = pd.DataFrame(dict(
#     NumProcs = [1, 2, 4, 8, 16],
#     ExecTime_20K = [55.43,28.037,14.835,8.230,5.300],
#     ExecTime_10K = [302.671, 166.978, 89.808, 52.133, 33.122],
   
# ))

# fig2 = px.line(df2, x="NumProcs", y=df2.columns[1:], title="Differences of execution time Parallel FISHDBC with different cores , blob data sets (remote machine)", markers=True) 

# fig2.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig2.update_xaxes(title='Number of processes', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.show()

# ---------------------------------------------

# df2 = pd.DataFrame(dict(
#     NumProcs = [1, 2, 4, 8, 16],
#     ExecTime_20K = [66.289, 28.137, 14.530, 8.121, 5.592],
#     ExecTime_10K = [454.494, 285.75, 151.208, 74.494, 41.05],
   
# ))

# fig2 = px.line(df2, x="NumProcs", y=df2.columns[1:], title="Differences of execution time Parallel HNSW with different cores, blob data sets (remote machine)", markers=True) 

# fig2.update_layout(
#     legend_title="Approaches",
#     legend_title_font=dict(size=20),
#     legend_font=dict(size=16),
#     titlefont = dict(size=20)
# )
# fig2.update_xaxes(title='Number of processes', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.update_yaxes(title='Execution Time (mean)', titlefont=dict(size=22),tickfont=dict(size=22)) 
# fig2.show()




# blob 20,000
# Mean AMI:  0.96 , Mean NMI:  0.96 , Mean ARI:  0.96 , Mean RI:  0.99 , Mean Homogeneity:  0.99 , Mean Completness:  0.93 , Mean V-measure:  0.96
# Std. Dev. AMI:  0.02 , Std. Dev. NMI:  0.02 , Std. Dev. ARI:  0.02 , Std. Dev. RI:  0.02 , Std. Dev. Homogeneity:  0.02 , Std. Dev. Completness:  0.02 , Std. Dev. V-measure:  0.02

# text 20,000
# Mean AMI:  0.94 , Mean NMI:  0.94 , Mean ARI:  0.94 , Mean RI:  0.98 , Mean Homogeneity:  0.99 , Mean Completness:  0.89 , Mean V-measure:  0.94
# Std. Dev. AMI:  0.03 , Std. Dev. NMI:  0.03 , Std. Dev. ARI:  0.03 , Std. Dev. RI:  0.03 , Std. Dev. Homogeneity:  0.03 , Std. Dev. Completness:  0.03 , Std. Dev. V-measure:  0.03

# text 100,000
# Mean AMI:  0.93 , Mean NMI:  0.93 , Mean ARI:  0.93 , Mean RI:  0.98 , Mean Homogeneity:  1.00 , Mean Completness:  0.88 , Mean V-measure:  0.93
# Std. Dev. AMI:  0.03 , Std. Dev. NMI:  0.03 , Std. Dev. ARI:  0.03 , Std. Dev. RI:  0.03 , Std. Dev. Homogeneity:  0.03 , Std. Dev. Completness:  0.03 , Std. Dev. V-measure:  0.03