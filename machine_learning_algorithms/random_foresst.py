import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn .model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sb
data = load_digits()

# print(dir(data))
plt.gray()
# for i in range(4):
#     plt.matshow(data.images[i])
    
# plt.show()

df = pd.DataFrame(data.data)
# print(df.head())
df['target'] = data.target

# df.head()

X_train,X_test,Y_train,Y_test = train_test_split(df.drop(['target'],axis = 'columns'),data.target,test_size = 0.2)

model = RandomForestClassifier()
model.fit(X_train,Y_train)
score = model.score(X_test,Y_test)
# print(score)


Y_predicted = model.predict(X_test)

cm = confusion_matrix(Y_test,Y_predicted)

# print(cm)


#
#   for better visualization
#

plt.figure(figsize=(10,7))
sb.heatmap(cm, annot= True)

plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()




