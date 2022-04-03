import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("D:\my world\Aman\LogisticRegression.csv")


#print(df.head())
model = LogisticRegression()

xtrain,xtest,ytrain,ytest=train_test_split(df[['age']],df.bought_insurance,test_size =0.9)
plt.scatter(df.age,df.bought_insurance,marker ="+")
model.fit(xtrain,ytrain)
model.predict(xtest)
#plt.show()
print(xtest)
