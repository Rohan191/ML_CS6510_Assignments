#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:30:39 2017

@author: rohantondulkar
"""

"""
#==============================================================================
# Call the function runKNN() form this files to run the program as shown below 
#
# import kNearestNeighbour as knn
# knn.runKNN()
#==============================================================================
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
        num_test = X_test.shape[0]
        start    = time.time()
        
        self.dist_matrix = np.zeros((num_test, self.num_train), dtype = object )
        self.dist_matrix = np.linalg.norm( X_test[:, np.newaxis] - self.X_train, axis = 2)
        print('Distance matrix took {0} secs'.format(time.time() - start))
        #print(self.dist_matrix.shape)
#        for x in X_test:
#            print('Distance vector for count:{0}'.format(count))
#            diff = self.X_train - x
#            self.dist_matrix[count, :] = np.sqrt(np.sum(diff*1*2,axis=-1))
#            count +=
            
#        for x in self.X_train:
#            print('Distance vector for count:{0}'.format(count))
#            diff = X_test - x
#            self.dist_matrix[ :, count ] = list(zip( np.sqrt(np.sum(diff**2,axis=-1)), \
#                                            np.full( X_test.shape[0], self.y_train[count]) ))
#            count +=1
            
#        for col in range(self.dist_matrix.shape[1]):
#            print('Adding classes for count:{0}'.format(col))
#            self.dist_matrix[:,col] = list(zip( self.dist_matrix[:,col], \
#                                        np.full((num_test), self.y_train[col], dtype=int)))

        #self.dist_matrix = np.sort(self.dist_matrix, axis = 1 )
        print('Sorting the distance matrix using argsort')
        self.class_matrix = np.zeros((num_test, self.num_train), dtype = int )
        start = time.time()
        count = 0
        for row in self.dist_matrix:
            self.class_matrix[count] = np.array(self.y_train[ np.argsort(row)], dtype = int)
            count +=1
        print('Sorting took {0} secs'.format(time.time() - start))
    
    def predictForK( self ):
        """Predicts the class for K"""
#        print('Predicting classes for k:{0}'.format(self.k))
#        sorted_dist = np.partition( self.dist_matrix, self.k )
#        prediction = np.zeros(self.dist_matrix.shape[0], dtype = int)
        count = 0
#        #classes    = self.y_train.unique()
#        for row in sorted_dist:
#            #print(row)
#            _, topK = zip(*row[:self.k])
#            prediction[count] = np.bincount(topK).argmax()
#            count+=1

        print('Predicting classes for k:{0}'.format(self.k))
        #sorted_dist = np.partition( self.dist_matrix, k )
        prediction = np.zeros(self.dist_matrix.shape[0], dtype = int)
        #count = 0
        #classes    = self.y_train.unique()
        #print(self.class_matrix[:,:self.k])
        #count_matrix = np.zeros( (self.class_matrix.shape[0], 2) ,dtype = int)
        #count_matrix = np.apply_along_axis(np.bincount, 1, self.class_matrix[:, :self.k])
        #prediction = count_matrix.argmax( axis = 1)
        for row in self.class_matrix:
            #print('Predicting for count: {0}'.format(count))
            prediction[count] = int(np.bincount(row[:self.k]).argmax())
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
    plt.legend(['Personal'])
    plt.plot(range(1,22),runtime[:,0], '-o', color='red', label='My kNN' )
    plt.legend()
    
    plt.figure()
    plt.xlabel('Values of k')
    plt.ylabel('Runtime for k-NN')
    plt.title('Runtime v/s K comparison')
    plt.legend(['Scikitlearn'])
    plt.plot(range(1,22),runtime[:,1], '-*', color='blue', label='Sklearn kNN')
    plt.legend()
    
def plotTrainAndTestDataset( X_train, y_train, X_test, y_test ):
    """Plot train and test dataset in one plot"""
    plt.figure()
    
    plt.subplot( 3,3,1)
    plt.title('Column A')
    plt.hist( X_train['A'], color ='red', alpha = 0.5 )
    plt.hist( X_test['A'], color ='blue', alpha = 0.5 )
    plt.legend()
    
    plt.subplot( 3,3,2)
    plt.title('Column B')
    plt.hist( X_train['B'], color ='red', alpha = 0.5 )
    plt.hist( X_test['B'], color ='blue', alpha = 0.5 )
    
    plt.subplot(3,3,3)
    plt.title('Column C')
    plt.hist( X_train['C'], color ='red', alpha = 0.5 )
    plt.hist( X_test['C'], color ='blue', alpha = 0.5 )
    
    plt.subplot( 3,3,4)
    plt.title('Column D')
    plt.hist( X_train['D'], color ='red', alpha = 0.5 )
    plt.hist( X_test['D'], color ='blue', alpha = 0.5 )
    
    plt.subplot( 3,3,5)
    plt.title('Column E')
    plt.hist( X_train['E'], color ='red', alpha = 0.5 )
    plt.hist( X_test['E'], color ='blue', alpha = 0.5 )
    
    plt.subplot( 3,3,6)
    plt.title('Column F')
    plt.hist( X_train['F'], color ='red', alpha = 0.5 )
    plt.hist( X_test['F'], color ='blue', alpha = 0.5 )
    
    plt.subplot( 3,3,7)
    plt.title('Class')
    plt.hist( y_train, color ='red', alpha = 0.5, label ='Train' )
    plt.hist( y_test, color ='blue', alpha = 0.5, label = 'Test' )
    
    plt.legend( )
    
    
def runKNN():
    """Main file to run kNN on randomly generated dataset"""
    dataset     = createDataSet()
    #scatter_matrix(dataset, alpha=0.2, figsize=(6, 6), diagonal='kde')
    #dataset.hist()
    train, test = splitAndStoreDataSet( dataset )
    X_train     = train.loc[:,'A':'F']
    y_train     = train.loc[:,'Class']
    X_test      = test.loc[:,'A':'F']
    y_test      = test.loc[:,'Class']
    plotTrainAndTestDataset( X_train, y_train, X_test, y_test )
    accuracy    = np.zeros((MAX_K-MIN_K+1,2))
    runtime     = np.zeros((MAX_K-MIN_K+1,2))
    for k in range(MIN_K, MAX_K+1):
        print('------------------------------------------------------------------')
        print('Running my implementation of kNN for k:{0}'.format(k))
        start = time.time()
        kNN   = KNearestNeighbour( X_train.values, y_train.values, k )
        kNN.fit( X_test.values )
        prediction       = kNN.predictForK( )
        accuracy[k-1][0] = calculateAccuracy( prediction, y_test.values)
        runtime[k-1][0]  = time.time() - start
        
        print('Running sklearn implementation of kNN for k:{0}'.format(k))
        start       = time.time()
        sklearn_knn = KNeighborsClassifier(n_neighbors = k)
        sklearn_knn.fit( X_train, y_train )
        accuracy[k-1][1] = sklearn_knn.score( X_test, y_test)
        runtime[k-1][1]  = time.time() - start
        
    plotAccuracyAndRuntime( accuracy, runtime )
    


    
    