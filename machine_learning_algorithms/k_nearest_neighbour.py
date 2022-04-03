

#
#   KNN Classification
#
# 
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data,columns = iris.feature_names)



print(iris.feature_names)

print(iris.target_names)
df['target'] = iris.target
new = df[df.target == 2].head()
print(new)
# print(df.head())

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


plt.xlabel('Sepel Length')
plt.ylabel('Sepel Width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color = 'r',marker = "+")
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color = 'b',marker = "_")
# plt.show()


# train test split
X = df.drop(['target'], axis = 'columns')
Y = df.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 1)





# making knn classifier


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,Y_train)

knn.score(X_test,Y_test)

Y_pred = knn.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
print(classification_report(Y_test,Y_pred))


plt.figure(figsize = (7,5))
sns.heatmap(cm,annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()