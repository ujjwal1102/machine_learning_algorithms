import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()

# print(len(data.data))
# print(data.feature_names)
# print(data.target)
# print(data.target_names)


df = pd.DataFrame(np.c_[data.data,data.target], columns = [list(data.feature_names)+ ['target']])
# print(df.shape)

X = df.iloc[:, 0:-1]
Y = df.iloc[:,-1 ]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 2020)
# print(X,Y)
# print(f"Shape of X_train : {X_train.shape}")
# print(f"Shape of Y_train : {Y_train.shape}")
# print(f"Shape of X_test : {X_test.shape}")
# print(f"Shape of Y_test : {Y_test.shape}")


classifier1 = GaussianNB()
classifier2 = MultinomialNB()
classifier3 = BernoulliNB()


classifier1.fit(X_train,Y_train)
classifier2.fit(X_train,Y_train)
classifier3.fit(X_train,Y_train)


# print(classifier1.score(X_test,Y_test))

# print(classifier2.score(X_test,Y_test))

# print(classifier3.score(X_test,Y_test))



patients =[24.99,69,34.45,754.37,84,43.43,0.234,0.875,0.7563,0.9876,0.74,0.23456,0.8765,0.98765,0.25368,34,654.6,0.867,0.7463,0.12345,72,43,85,23,54,96,12,11,45,123,]
patients = np.array([patients])
print(patients)



print(classifier1.predict(patients))




'''


df = pd.read_csv('D:\\python files\\Downloads\\Datasets\\adult.csv')
df.drop(['Unknown1','Unknown2','Unknown3','Unknown4','Unknown5','Unknown6','Unknown7','Unknown8'],axis = 'columns', inplace = True )
target = df.Age
# print(df.head())
dummies = pd.get_dummies( df['Gender'] )
df = pd.concat([df,dummies],axis = 'columns')
df.drop(['Gender'],axis = 'columns')

# print(df.head())

X_train,X_test,Y_train,Y_test = train_test_split(df,target,test_size = 0.3)
print(f"X_train: {len(X_train)} X_test : {len(X_test)} Y_train : {len(Y_train)} Y_test : {len(Y_test)} ")


model = GaussianNB()

model.fit(X_train,Y_train)


'''