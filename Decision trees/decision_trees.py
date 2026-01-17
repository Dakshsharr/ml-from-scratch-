import numpy as np

def Entropy_loss(y):
    n = len(y)
    if (len(y) ==0):
        return 0
    counts = np.bincount(y)
    p = counts / n

    L = 0
    for pi in p:
        if(pi>0):
            L = L - pi*np.log2(pi)  

    return L

def Missclass_loss(y):
    n = len(y)
    if (len(y) ==0):
        return 0
    counts = np.bincount(y)
    p = counts / n

    L = 1 - np.max(p)
    return L

def Gini_loss(y):
    n = len(y)
    if (len(y) ==0):
        return 0
    counts = np.bincount(y)
    p = counts / n

    L = 0
    for pi in p:
          L = L + pi * (1 - pi)
    return L   

def split_dataset(X, y, feature_index, threshold):
    mask = X[:, feature_index] <= threshold

    X_left = X[mask]
    y_left = y[mask]

    X_right = X[~mask]
    y_right = y[~mask]

    return X_left, y_left, X_right, y_right

def split_loss(X, y, feature_index, threshold, impurity_fn):
    X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)

    n = len(y)
    loss = (len(y_left)/n) * impurity_fn(y_left) + \
           (len(y_right)/n) * impurity_fn(y_right)

    return loss

def find_best_split(X,y,impurity_fn):
    n_samples, n_features = X.shape
    best_loss = float("inf")
    best_feature,best_threshold = None,None

    for j in range(n_features):
        values = np.unique(X[:,j])
        values.sort()
        thresholds = (values[:-1] + values[1:]) / 2

        for t in thresholds:
            loss = split_loss(X, y, j, t, impurity_fn)

            if loss < best_loss:
                best_loss = loss
                best_feature = j
                best_threshold = t
    return best_feature, best_threshold, best_loss      

def Majority_class(y):
    counts = np.bincount(y)
    value = np.argmax(counts)
    return value

class Node:
    def __init__(self, best_feature = None,best_threshold = None, left = None, right = None, value = None):
        self.best_feature = best_feature
        self.best_threshold = best_threshold
        self.left = left
        self.right = right
        self.value = value  

class DecisionTree:
    def __init__(self, impurity_fn, max_depth, min_samples_leaf = 1):
        self.impurity_fn = impurity_fn
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def build_tree(self, X, y, depth):
        if (depth >= self.max_depth or len(y) < self.min_samples_leaf or len(np.unique(y)) == 1):
            return Node(value = Majority_class(y))
    
        feature,threshold,loss = find_best_split(X, y, self.impurity_fn)
    
        Xl, yl, Xr, yr = split_dataset(X, y, feature, threshold)
    
        left_child  = self.build_tree(Xl, yl, depth + 1)
        right_child = self.build_tree(Xr, yr, depth + 1)
    
        return Node(
            best_feature = feature,
            best_threshold = threshold,
            left = left_child,
            right = right_child
        )

    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            node = self.root        
            while node.value is None:
                if x[node.best_feature] <= node.best_threshold:
                    node = node.left
                else:
                    node = node.right        
            predictions.append(node.value)
    
        return np.array(predictions)        