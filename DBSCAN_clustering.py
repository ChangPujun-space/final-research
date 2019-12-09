import pickle
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

f = open('output/Feature_Vector_List.txt','rb')
img_vector = pickle.load(f)
f.close()
print(len(img_vector))

for i in img_vector[1:2]:
    print(i)

X = np.array(img_vector)
X.reshape((3469,25088))

import scipy
from scipy.spatial import distance
##cosine matrix
dist = scipy.spatial.distance.cdist(X, X, 'cosine')

##kmeans prediction
model = KMeans(n_clusters=10, random_state=0)
kmeans = model.fit(dist)

##find centroid
from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
clf.fit(X, kmeans.labels_)

plt.scatter(X[:, 0],X[:, 1], c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(clf.centroids_[:, 0], clf.centroids_[:, 1], c='k', s=200, alpha=0.5)
plt.savefig('output/km_clustering.png')

fileObject = open('output/clustering_centroid.txt', 'wb')
pickle.dump(list(clf.centroids_),fileObject)
fileObject.close()
fileObject = open('output/clustering_result.txt', 'wb')
pickle.dump(list(kmeans.labels_),fileObject)
fileObject.close()
