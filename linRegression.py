import numpy as np
import matplotlib.pyplot as plt

# Here is some test comments to see if git can see it


def dataGen(sampleSize=100):
    np.random.seed(10)
    dim = 2
    meanSize = 10
    meanAndConv = np.eye(dim)

    sampleMeans = np.zeros((meanSize,dim,dim))

    X = np.zeros((sampleSize,dim,dim))
    Y = np.concatenate((np.zeros(sampleSize), np.ones(sampleSize)))

    for i in range(dim):
        sampleMeans[:,:,i] = np.random.multivariate_normal(meanAndConv[i,:] ,meanAndConv,meanSize)

    for j in range(dim):
        for i in range(sampleSize):
            X[i,:,j] = np.random.multivariate_normal(sampleMeans[np.random.randint(0, 10),:,j],meanAndConv/5)

    X =  np.concatenate((X[:,:,0], X[:,:,1]), axis=0)
    oneCol = np.ones((X.shape[0],1))
    X = np.concatenate((oneCol,X),axis=1)

    return X, Y


def solveLinReg(X,Y):  
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),Y)


if __name__ == "__main__":

    X, Y = dataGen()
    W = solveLinReg(X,Y)

    plt.figure(1)
    plt.scatter(X[:,1],X[:,2],c=Y)
    plt.show()

    plt.figure(2)
    yHat = np.where(np.matmul(X,W) > 0.5, 1, 0)

    
    plt.scatter(X[:,1],X[:,2],c=yHat)
    plt.show()

    print(np.average(yHat == Y))
