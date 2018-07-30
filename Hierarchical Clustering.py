#Hierarchical Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values
#Using dendrogram to find optimal no of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plot.title('Dendrogram')
plot.xlabel('Customers')
plot.ylabel('Euclidean Distance')
plot.show()
#Fit hierarchical clustering algorithm to the dataset
from sklearn.cluster import AgglomerativeClustering as ac
hc=ac(n_clusters=5,affinity='euclidean',linkage='ward')
yHc=hc.fit_predict(x)
#Visualizing the clusters
plot.scatter(x[yHc==0,0],x[yHc==0,1],s=100,c='red',label='Careful')
plot.scatter(x[yHc==1,0],x[yHc==1,1],s=100,c='blue',label='Standard')
plot.scatter(x[yHc==2,0],x[yHc==2,1],s=100,c='green',label='Target')
plot.scatter(x[yHc==3,0],x[yHc==3,1],s=100,c='cyan',label='Careless')
plot.scatter(x[yHc==4,0],x[yHc==4,1],s=100,c='magenta',label='Sensible')
plot.title('Clusters of Clients')
plot.xlabel('Annual Income in $')
plot.ylabel('Spending Score 1-100')
plot.legend()
plot.show()