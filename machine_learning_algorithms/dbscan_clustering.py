import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets  as sd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


x,y = sd.make_blobs(n_samples = 500)


#x = np.array([[1,2],[2,3],[4,9],[6,6],[6,5],[10,9],[10,8],[9,9]])

#x = StandardScaler().fit_transform(x)


clustering = DBSCAN(eps = 3,min_samples=2).fit(x)
#print(clustering.labels_)

#print(x,y)


#print(x[0:3,1])


#df = pd.DataFrame(dict(x[:,0],x[:,1]))



#print(df)

plt.scatter(x[:,0],x[:,1])# make_bloba

#plt.scatter(x[:,0],x[:,1])
plt.show()
