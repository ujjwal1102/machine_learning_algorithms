
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv(r'D:\my world\BCA Files\clusterdata.csv')



#print(y_predicted)


scaler = MinMaxScaler()
scaler.fit(df[['Income']])
df['Income'] = scaler.transform(df[['Income']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])


km = KMeans(n_clusters = 3)
y_predicted = km.fit_predict(df[['Age','Income']])
df['Cluster'] = y_predicted



#print(df.Cluster)
#print(km.cluster_centers_)

'''
sse = []

k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(df[['Age','Income']])
    sse.append(km.inertia_)
print(sse)

plt.xlabel('K -->')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()

'''
df1 = df[df.Cluster == 0]
df2 = df[df.Cluster == 1]
df3 = df[df.Cluster == 2]
plt.scatter(df1.Age,df1['Income'],color = 'r')
plt.scatter(df2.Age,df2['Income'],color = 'g')
plt.scatter(df3.Age,df3['Income'],color = 'b')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker = '*',label = 'centroid')
plt.xlabel('Age')
plt.ylabel('Income')
#plt.legend()
plt.show()

#print(df.head())




