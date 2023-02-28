import numpy as np
from collections import Counter

def ED(x1, x2):
    return np.sqrt(np.sum(x1 - x2)**2)

class KNN():
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    def pred(self, x):
        y_pred = [self.predict_most_common(x) for x in x]
        return np.array(y_pred)

    def predict_most_common(self, x):
        distances = [ED(x, x_tr) for x_tr in self.x_train]
        k_idx = np.argsort(distances)[:self.k]
        k_neighbors_lbls = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_neighbors_lbls).most_common(1)
        
        return most_common[0][0]
