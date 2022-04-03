from re import X
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


data = load_breast_cancer()


df = pd.DataFrame(np.c_[data.data, data.target], columns= [list(data.feature_names)+['target']])
# print(df.tail())
# print(data.data)
# print(data.target_names)

X = df.iloc[:,0:-1]
Y = df.iloc[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2,random_state = 2020)

# print(f"Shape of X_train : {X_train.shape}")
# print(f"Shape of Y_train : {Y_train.shape}")
# print(f"Shape of X_test : {X_test.shape}")
# print(f"Shape of Y_test : {Y_test.shape}")

classifier = DecisionTreeClassifier(criterion = 'gini')

classifier.fit(X_train,Y_train)

print(classifier.score(X_test,Y_test))

classifier = DecisionTreeClassifier(criterion = 'entropy')

classifier.fit(X_train,Y_train)

# print(classifier.score(X_test,Y_test))

# feature scaling

sc = StandardScaler()

sc.fit(X_train)

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

classifier_sc = DecisionTreeClassifier(criterion = 'gini')
classifier_sc.fit(X_train_sc,Y_train)


print(classifier_sc.score(X_test_sc,Y_test))

patients =[24.99,69,34.45,754.37,84,43.43,0.234,0.875,0.7563,0.9876,0.74,0.23456,0.8765,0.98765,0.25368,34,654.6,0.867,0.7463,0.12345,72,43,85,23,54,96,12,11,45,123,]
patients = np.array([patients])
print(patients)



print(classifier.predict(patients))
