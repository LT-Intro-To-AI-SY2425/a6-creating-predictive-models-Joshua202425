import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#imports the data
data = pd.read_csv("part5-unsupervised-learning/customer_data.csv")
x = data[["Annual Income", "Spending Score"]]

#standardize the data
scale = StandardScaler().fit_transform(x)

#the value of k has been defined for you
k = 5

#apply the kmeans algorithm
km=KMeans(n_clusters=k).fit(scale)

#get the centroid and label values
cen=km.cluster_centers_
lab=km.labels_
#sets the size of the graph
plt.figure(figsize=(5,4))

#use a for loop to plot the data points in each cluster
for i in range(k):
    clus = scale[lab == i]
    plt.scatter(clus[:,0], clus[:,1])

#plot the centroids
plt.scatter(cen[:, 0], cen[:, 1], marker='x', s=100,
            c='y', label='centroid')
            
#shows the graph
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()