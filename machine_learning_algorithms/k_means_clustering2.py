import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import sklearn.datasets  as sd

data= sd.load_iris()
print(data)
df = pd.DataFrame(data.data,columns = data.feature_names)
#print(df[['sepal length (cm)','sepal width (cm)']])

s = MinMaxScaler()
s.fit(df[['sepal length (cm)']])
df['sepal length (cm)'] = s.transform(df[['sepal length (cm)']])

s.fit(df[['sepal width (cm)']])
df['sepal width (cm)'] = s.transform(df[['sepal width (cm)']])







sse = []
k_rng = range(1,80)
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(df[['sepal length (cm)','sepal width (cm)']])
    sse.append(km.inertia_)

print(sse)

plt.plot(k_rng,sse)
plt.show()

'''


km = KMeans(n_clusters = 4)
y = km.fit_predict(df[['sepal length (cm)','sepal width (cm)']])
df['Cluster'] = y

df0 = df[df['Cluster'] == 0]
df1 = df[df['Cluster'] == 1]
df2 = df[df['Cluster'] == 2]
df3 = df[df['Cluster'] == 3]






print(df)

plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color = 'r')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color = 'b')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color = 'g')
plt.scatter(df3['sepal length (cm)'],df3['sepal width (cm)'],color = 'y')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()





'''



#print( df )
