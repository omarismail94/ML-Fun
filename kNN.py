# Based off https://github.com/dgkim5360/the-elements-of-statistical-learning-notebooks/blob/master/chapter02-overview-of-supervised-learning/section3-least-squares-and-nearest-neighbors.ipynb

import numpy as np
import matplotlib.pyplot as plt

def dataGen(sampleSize=100):
    np.random.seed(456)
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
    return X, Y


def knn(k, point, data_x, data_y) :
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    distances = [(sum((x - point)**2), y) for x, y in zip(data_x, data_y)]
    distances.sort()
    return sum(y for _, y in distances[:k])/k


if __name__ == "__main__":
    X,Y = dataGen(100)


    knnGrid = np.array([(i, j)  for i in np.arange(np.amin(X,axis=0)[0], np.amax(X,axis=0)[0], .1)   for j in np.arange(np.amin(X,axis=0)[1], np.amax(X,axis=0)[1], .1)])
    knnResult = np.array([(i, j, knn(1, (i, j), X, Y)) for i, j in knnGrid])

    knnBlue = np.array([(i, j) for i, j, threshhold in knnResult if threshhold < .5 ])
    knnOrange = np.array([(i, j) for i, j, threshhold in knnResult if threshhold >= .5])

    plt.figure(1)
    plt.plot(knnBlue[:, 0], knnBlue[:, 1], 'o', alpha=.2)
    plt.plot(knnOrange[:, 0], knnOrange[:, 1], 'o', color='orange', alpha=.2)
    plt.scatter(X[:,0],X[:,1], c=Y, marker='^',cmap='jet')
    plt.show()
