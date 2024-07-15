import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("C:/Users/139334/Downloads/Datasets/train.csv")



clf = DecisionTreeClassifier()



xtrain = dataset.iloc[0:21000,1:].values
train_label = dataset.iloc[0:21000,0].values

clf.fit(xtrain, train_label)


xtest = dataset.iloc[21000:,1:].values
actual_label = dataset.iloc[21000:,0].values


d = xtest[498] 
d.shape = (28,28)
plt.imshow(255-d,cmap = "grey") 
plt.show()
print(clf.predict([xtest[498]]))
