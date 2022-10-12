import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class dataset:

    def __init__(self,n) -> None:
        self.n = n
    
    def get(self,add_noise = False) -> pd.DataFrame:
        data = np.zeros((self.n,3))
        d = {0: (0, 0, 1), 1: (0, 3, 1)}
        
        for i in range(self.n):
            l = np.random.randint(0,2)
            h,k,r = d[l]

            x = np.random.rand()
            y = math.sqrt(r*r - x*x)

            q = np.random.randint(0,4)

            if q == 0:
                x,y = x,y
            elif q == 1:
                x,y = -x,y
            elif q == 2:
                x,y = x,-y
            else:
                x,y = -x,-y

            x = x + h
            y = y + k
            data[i] = [x,y,l]
        
        if add_noise:
            data[:,0:2] += np.random.normal(0,0.1,(self.n,2))
        df = pd.DataFrame(data, columns = ['x', 'y', 'label'])
        df.label = df.label.astype(int)
        return df


class Perceptron:
    def __init__(self,n,a = 0.000001) -> None:
        # initialize weights
        self.n = n
        self.a = a
        self.w = np.zeros(n+1)


    def step(self,z):
        # applies step function on z
        return np.heaviside(z,1)

    def fit(self, X,y,epochs=100, bias=True):
        self.w = np.zeros(self.n+1)
        if bias:
            X = np.hstack((np.ones((len(X),1)),X))
        else:
            X = np.hstack((np.zeros((len(X),1)),X))

        wrong_arr = []

        for e in range(epochs):
            wrong = 0
            for i in range(len(X)):
                z = np.dot(self.w,X[i])
                y_pred = self.step(z)
                if y_pred != y[i]:
                    wrong += 1
                    self.w += self.a*(y[i] - y_pred)*X[i]
            wrong_arr.append(wrong)
            if(wrong == 0):
                break

        return self.w, wrong_arr
    
    def predict(self,X):
        X = np.hstack((np.ones((len(X),1)),X))
        z = np.dot(self.w,X.T)
        y_pred = self.step(z)
        return y_pred

    def plot_decision_boundary(self, X,y, lim = False):

        df = pd.DataFrame(X, columns = ['x', 'y'])
        df['label'] = y
        try:
            if self.w[2]!=0:
                if lim:
                    x1 = np.array([-3,3])
                else:
                    x1 = np.array([min(X[:, 0]), max(X[:, 0])])
                x2 = -(self.w[0] + self.w[1]*x1)/self.w[2]
            elif self.w[1]!=0:
                if lim:
                    x2 = np.array([-3, 3])
                else:
                    x2 = np.array([min(X[:, 1]), max(X[:, 1])])
                x1 = -(self.w[0] + self.w[2]*x2)/self.w[1]
            else:
                print("NO boundary exists")
                return
        except:
            print("NO boundary exists")
            return
        plt.plot(x1,x2,'r-',label = 'Decision Boundary')
        sns.scatterplot(data = df, x = 'x', y='y',hue = 'label')
        plt.axis('equal')
        
        
if __name__ == "__main__":
    df = dataset(1000).get(add_noise=True)
    sns.scatterplot(data=df, x="x", y="y", hue="label")
    plt.savefig('./a.png')
    p = Perceptron()
    p.perceptron(df.drop('label', axis=1).to_numpy(),df.label.to_numpy(),epochs=100, bias=True, plot = True)

