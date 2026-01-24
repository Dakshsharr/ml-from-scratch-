import numpy as np

class LinearRegression:
    def __init__(self,learning_rate = 0.01,iterations = 30):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None

    def prediction(self,X_aug):
        return np.dot(X_aug,self.theta)
        
    def loss(self,y, ans):
        error = 0
        for i in range(len(y)):
            diff = ans[i][0] - y[i][0]
            error = error + diff * diff
        return error/(2*len(y)) 

    def gradient(self,X_aug, y):
        m = X_aug.shape[0]
        return (X_aug.T @ ((X_aug @ self.theta) - y)) / m
        
    def fit(self,X,y):
        if X.ndim ==1:
            X = X.reshape(-1,1)
        y = y.reshape(-1,1)
        m,n = X.shape
        ones = np.ones((m, 1)) 
        X_aug = np.hstack((ones, X))        
        self.theta = np.zeros((n+1,1))
        for i in range(self.iterations):
            ans = self.prediction(X_aug)
            error = self.loss(y,ans)
            grad = self.gradient(X_aug, y)
            self.theta = self.theta - self.learning_rate * grad
            if(self.iterations - i <50):
                print(f"Epoch {i}, Loss = {error}")         
    def predict(self,X):
        if X.ndim ==1:
            X = X.reshape(-1,1)
        m,n = X.shape
        ones = np.ones((m, 1)) 
        X_aug = np.hstack((ones, X)) 
        preds = self.prediction(X_aug)
        return preds    