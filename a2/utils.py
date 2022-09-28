import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class dataset:

    def __init__(self,n) -> None:
        self.n = n
        self.d = {}
        self.d[0] = (0,0,1)
        self.d[1] = (0,3,1)
    
    def get(self,add_noise = False) -> pd.DataFrame:
        data = np.zeros((self.n,3))
        
        for i in range(self.n):
            l = np.random.randint(0,2)
            h,k,r = self.d[l]

            x = np.random.rand()
            y = math.sqrt(r*r - x*x)

            q = np.random.randint(0,4)

            if q == 0:
                x = x
                y = y
            elif q == 1:
                x = -x
                y = y
            elif q == 2:
                x = x
                y = -y
            else:
                x = -x
                y = -y

            x = x + h
            y = y + k

            if add_noise:
                x = x + np.random.normal(0,1)
                y = y + np.random.normal(0,1)

            data[i][0] = x
            data[i][1] = y
            data[i][2] = l

        df = pd.DataFrame(data, columns = ['x', 'y', 'label'])
        df.label = df.label.astype(int)
        return df


class PTA:
    def __init__(self) -> None:
        pass

    def step(z):
        if z >= 0:
            return 1
        else:
            return 0

    def perceptron(self, df, a=0.0001, epochs=100, bias=True, plot = False):
        X = df[['x','y']].to_numpy()
        y = df['label'].to_numpy()
        if bias:
            X = np.hstack((np.ones((len(X),1)),X))
        else:
            X = np.hstack((np.zeros((len(X),1)),X))
        m,n = X.shape
        # theta
        theta = np.zeros(n)

        err_arr = []


        for e in range(epochs):
            err = 0
            for i in range(m):
                z = np.dot(theta,X[i])
                y_pred = PTA.step(z)
                err += abs(y[i] - y_pred)
                theta = theta + a*(y[i] - y_pred)*X[i]
            err_arr.append(err)
        
        if plot:
            PTA.plot_decision_boundary(df, theta)

        return theta, err_arr

    def plot_decision_boundary(df, theta):
        X = df[['x','y']].to_numpy()

        x1 = np.array([min(X[:,0]), max(X[:,0])])
        x2 = -(theta[0] + theta[1]*x1)/theta[2]
        plt.plot(x1,x2,'r-',label = 'Decision Boundary')
        sns.scatterplot(data = df, x = 'x', y='y',hue = 'label')
        plt.show()
        


if __name__ == "__main__":
    df = dataset(1000).get(add_noise=True)
    sns.scatterplot(data=df, x="x", y="y", hue="label")
    plt.savefig('./a.png')
    p = PTA()
    p.perceptron(df[['x','y']].to_numpy(),df['label'].to_numpy(),0.1,0.1)

