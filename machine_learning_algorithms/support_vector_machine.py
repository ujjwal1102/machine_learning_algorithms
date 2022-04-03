import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


# data loading
dig = datasets.load_digits()

# print(data['target'])

# 
clf= svm.SVC(gamma = 0.001,C = 100)

X,Y = dig.data[:-10],dig.target[:-10]

# generating model


# train the model

clf.fit(X,Y)




print(len(clf.predict(dig.data)))
plt.imshow(dig.images[6],interpolation= 'nearest')
plt.show()
 
