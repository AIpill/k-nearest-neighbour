import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn_fromScratch import KNN

knn = KNeighborsClassifier(n_neighbors = 6)
iris = datasets.load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn.fit(x, y)
pred = knn.predict(x_test)
print(pred)

def accu(y_test, pred):
    acc = np.sum(y_test == pred) / len(y_test)
    return acc

print('Acc: ', accu(y_test, pred))


# KNN class from scratch
K = 6

cls_ = KNN(k = K)
cls_.fit(x_train, y_train)
predc = cls_.pred(x_test)

print('Cust Acc: ', accu(y_test, predc))



colormap = ListedColormap(['b', 'r', 'g'])
plt.figure()
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = colormap)
plt.show()

