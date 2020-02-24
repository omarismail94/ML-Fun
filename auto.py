from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np


def loadClean():
    df = pd.read_csv('Auto.csv')
    df = df[df.horsepower.apply(lambda x: x.isnumeric())]
    X = np.reshape(np.array(df['horsepower']),(-1,1))
    Y = np.reshape(np.array(df['mpg']),(-1,1))

    return X,Y

def skLinReg(X,Y):
    clf = LinearRegression()
    clf.fit(X,Y)
    y_pred = clf.predict(X)
    print(r2_score(Y,y_pred))

if __name__ == "__main__":
    
    X,Y = loadClean()
    skLinReg(X,Y)




