import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.externals import joblib

from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split

import numpy as np
import pickle as p
import joblib as jb

df = pd.read_csv('D:\my world\Aman\DF.csv')
df1 = df.drop("height",axis="columns")
#%matplotlib inline

plt.xlabel('Weight')
plt.ylabel('height')
plt.scatter(df.Weight,df.height,color='b',marker='+')
print(df1)

m = linear_model.LinearRegression()

#x=Weight.reshape(-1, 1)
m.fit(df1,df.height)

print(m.predict([[44]]))
print(m.coef_)
print(m.intercept_)
#plt.show()

print(jb.dump(m,'model_joblib'))

mj = jb.load('model_joblib')
print(mj.intercept_)
print(mj.coef_)
'''
with open('model_pickle','wb') as file :
    p.dump(m,file)
with open('model_pickle','rb') as f:
    n_p = p.load(f)
    
print(n_p.intercept_)
print(n_p.coef_)


print(n_p.)

'''


'''
iris = datasets.load_iris()
x= iris.data
y= iris.target
#print(x.shape)
#print(y.shape)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)
'''
