import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

colormap = ListedColormap(['b', 'r', 'g'])
plt.figure()
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = colormap)
plt.show()
