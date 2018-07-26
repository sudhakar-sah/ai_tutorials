from sklearn import datasets, cluster
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load data 
iris = datasets.load_iris()

# extract features (petal length, width, Sepal length, width)
features = iris.data
# three different types of iris (versicolor, Sentosa, Virginica)
labels = iris.target

# k means clustering 
kmeans = cluster.KMeans(n_clusters=5)
kmeans.fit(features)
centers = kmeans.labels_


# plot the cluster in color 3D plot
fig = plt.figure(1, figsize=(8,8))
plt.clf()
ax = Axes3D(fig, rect=[0,0,1,1], elev=8, azim=200)
plt.cla()

ax.scatter(features[:,3], features[:,0], features[:,2], c= centers.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')

