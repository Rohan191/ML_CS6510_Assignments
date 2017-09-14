#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:30:39 2017

@author: rohantondulkar
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time

MIN_K = 1
MAX_K = 21

class KNearestNeighbour( object ):
    
    def __init__( self, X_train, y_train, k ):
        self.X_train   = X_train
        self.y_train   = y_train
        self.num_train = X_train.shape[0]
        self.k         = k
        
    def fit( self, X_test ):
        """Calculates the distance matrix for the test set"""
        print('Calculating entire distance matrix')
        count = 0
        self.dist_matrix = pd.DataFrame( np.zeros((X_test.shape[0], self.num_train)), dtype = object )
        self.dist_matrix.columns = self.X_train.index
        for x1 in X_test.values:
            #print('Distance vector for count:{0}'.format(count))
            self.dist_matrix.loc[count] = [ getEuclideanDistance( x1, x2) \
                                            for x2 in self.X_train.values ]
            count +=1
            
        print("Adding classes to distance matrix")
        for col in self.dist_matrix.columns:
            self.dist_matrix[col] = list(zip( self.dist_matrix[col], \
                                        np.full((self.dist_matrix.shape[0]), self.y_train.loc[col], dtype=int)))
        self.dist_matrix = self.dist_matrix.apply( np.sort, axis = 1 )
    
    def predictForK( self ):
        """Predicts the class for K"""
        print('Predicting classes for k:{0}'.format(self.k))
        prediction = pd.Series( np.zeros(self.dist_matrix.shape[0]), dtype = int)
        count = 0
        #classes    = self.y_train.unique()
        for row in self.dist_matrix.values:
            _, topK = zip(*row[:self.k])
            topK = pd.Series(topK)
            prediction.loc[count] = topK.value_counts().idxmax()
            count+=1
        return prediction
            
def createDataSet():
    """To create dataset in expected format and store in csv"""
    dataset                 = pd.DataFrame( np.zeros((1000,8), int) )
    dataset.columns         = ['ID','A', 'B', 'C', 'D', 'E', 'F', 'Class']
    dataset['ID']           = range(0, 1000)
    dataset.loc[:, 'A':'F'] = np.random.randint(1,1000, (1000,6))
    dataset.loc[:, 'Class'] = np.random.randint(0,2, (1000,1))
    print('Writing entire dataset to csv')
    dataset.to_csv('dataset.csv', index=False, encoding='utf-8')
    return dataset

def splitAndStoreDataSet( dataset ):
    """To split the dataset in 80:20 and store train and test csv"""
    x = np.arange(0,1000)
    np.random.shuffle(x)
    train = dataset.loc[x[:800]]
    test  = dataset.loc[x[800:]]
    print('Writing split (train and test) dataset to csv')
    train.to_csv('train.csv', index=False, encoding='utf-8')
    test.to_csv('test.csv', index=False, encoding='utf-8')
    return train, test
    
def getEuclideanDistance( x1, x2):
    """Expects two iterable numpy arrays or panda Series and returns Euclidean distance between them"""
    dif = x1-x2
    return np.sqrt(np.dot(dif, dif))

def calculateAccuracy( prediction, y ):
    """Return accuracy percentage between predicted and expected class"""
    return len( prediction[ prediction == y ])/len(prediction)

def plotAccuracyAndRuntime( accuracy, runtime):
    """Plot graphs with proper description"""
    plt.figure()
    plt.xlabel('Values of k')
    plt.ylabel('Accuracy for k-NN')
    plt.title('Accuracy v/s K comparison')
    plt.legend(['Personal', 'Scikitlearn'])
    plt.plot(range(1,22),accuracy[:,0], '-o', color='red', label='My kNN' )
    plt.plot(range(1,22),accuracy[:,1], '-*', color='blue', label='Sklearn kNN')
    plt.legend()
    
    plt.figure()
    plt.xlabel('Values of k')
    plt.ylabel('Runtime for k-NN')
    plt.title('Runtime v/s K comparison')
    plt.legend(['Personal', 'Scikitlearn'])
    plt.plot(range(1,22),runtime[:,0], '-o', color='red', label='My kNN' )
    plt.plot(range(1,22),runtime[:,1], '-*', color='blue', label='Sklearn kNN')
    plt.legend()
    
def runKNN():
    """"""
    dataset     = createDataSet()
    train, test = splitAndStoreDataSet( dataset )
    X_train     = train.loc[:,'A':'F']
    y_train     = train.loc[:,'Class']
    X_test      = test.loc[:,'A':'F']
    y_test      = test.loc[:,'Class']
    accuracy    = np.zeros((MAX_K-MIN_K+1,2))
    runtime     = np.zeros((MAX_K-MIN_K+1,2))
    for k in range(MIN_K, MAX_K+1):
        print('------------------------------------------------------------------')
        print('Running my implementation of kNN for k:{0}'.format(k))
        start = time.time()
        kNN = KNearestNeighbour( X_train, y_train, k )
        kNN.fit( X_test )
        prediction     = kNN.predictForK( )
        accuracy[k-1][0] = calculateAccuracy( prediction.values, y_test)
        end = time.time()
        runtime[k-1][0] = end - start
        
        print('Running sklearn implementation of kNN for k:{0}'.format(k))
        start = time.time()
        sklearn_knn    = KNeighborsClassifier(n_neighbors = k)
        sklearn_knn.fit( X_train, y_train )
        accuracy[k-1][1] = sklearn_knn.score( X_test, y_test)
        end = time.time()
        runtime[k-1][1] = end - start
        
    print(runtime)
    plotAccuracyAndRuntime( accuracy, runtime )
    


    
    