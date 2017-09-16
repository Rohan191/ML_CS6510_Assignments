#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:49:16 2017

@author: rohantondulkar
"""

#SVM

"""
#==============================================================================
# Call the runSVM() method in this file to run svm on a kernel.
# Specify the kernel to be used for gram matrix calculation on line no 111, 115
# 
# import svm as s
# s.runSVM()
#==============================================================================
"""

import pandas as pd
import numpy as np
import time
from sklearn import svm

MISSING_VALUE = ' ?'
TRAIN_FILE    = 'data/train.csv'

def loadAndPreprocessDataset():
    """"""
    start = time.time()
    dataset = pd.read_csv( TRAIN_FILE, header=None)
    str_cols = [1,3,5,6,7,8,9,13]
    replace_vals = {}
    for col in str_cols:
        max_val  = dataset[col].value_counts().idxmax()
        dataset[col].replace( MISSING_VALUE, max_val, inplace = True )
        replace_vals[col] = {}
        col_vals = dataset[col].unique()
        count = 1
        for i in col_vals:
            replace_vals[col][i] = count
            count +=1
    dataset.replace( replace_vals, inplace = True )
    print('Data loading + preprocessing took {0} secs'.format(time.time() - start))
    return dataset
            
def kFoldCrossValidation( dataset, TotalK, currentK ):
    """"""
    nrows   = dataset.shape[0]
    nsplit  = int(nrows/TotalK)
    cvRange = np.arange( nsplit*(currentK-1), nsplit*currentK)
    cv      = dataset.loc[ cvRange ]
    tRange  = np.delete( np.arange( nrows ), cvRange )
    train   = dataset.loc[ tRange ]
    X_train = train.loc[:,:13].values
    y_train = train.loc[:, 14].values
    X_cv    = cv.loc[:,0:13].values
    y_cv    = cv.loc[:, 14].values
    return X_train, y_train, X_cv, y_cv
    
def calculateAccuracy( prediction, y ):
    """Return accuracy percentage between predicted and expected class"""
    return len( prediction[ prediction == y ])/len(prediction)

def linearKernel( x1, x2 ):
    """Linear kernel using formula: KLIN(xi,xj) = <xi,xj>"""
    return np.inner( x1, x2 )
    #return np.dot(x1, x2)

def polynomialKernel( x1, x2, q = 4 ):
    """Polynomial kernel using formula: KPOL(xi,xj) = (<xi,xj>+1)^q """
    return np.power(np.add(np.inner( x1, x2 ), 1), q )

def gaussianKernel( x1, x2, s = 50 ):
    """Calculates the gaussian kernel gram matrix for two vectors"""
    return np.exp( -np.divide( np.sum( np.square(np.subtract( x1[:, np.newaxis], x2)), 2), 2 * s**2 ))
    
class MultiKernelfixedrules( object ):
    
    def __init__(self, kernels ):
        self.kernels = kernels
        self.lenKernels = len( kernels )
        
    def getConvexSum( self, x1, x2, weights = [] ):
        """Get the convex sum of all kernels based on the given weights"""
        if sum( weights ) != 1:
            print(weights)
            raise Exception('Sum of weights not equal to 1 for convex sum')
        if len(weights) != self.lenKernels:
            raise Exception('Weights not assigned for all kernels')
            
        convexSum = 0
        for i in range( self.lenKernels ):
            convexSum += weights[i]*self.kernels[i]( x1, x2 )
        return convexSum
            

def runSVM():
    """Main method for this module. Calculate gram matrix using required kernel to run it"""
    dataset = loadAndPreprocessDataset()
    k = 5
    for i in range(1,k+1):
        print('----------------------------------------------------------------')
        print( 'Running SVM using {0}-fold validation for k:{1}'.format( k, i ))
        X_train, y_train, X_cv, y_cv = kFoldCrossValidation( dataset, k, i)
        start = time.time()
        multiKernel = MultiKernelfixedrules( [linearKernel, polynomialKernel, gaussianKernel] ) 
        
        #Weights for Multikernel based on performance of individual kernels
        weights = [0.5,0.25,0.25]
        
        #Calculate gram matrix for train and test using individual kernels or multikernel
        #train_gram = multiKernel.getConvexSum( X_train, X_train, weights )  #Change the kernel here
        train_gram = gaussianKernel( X_train, X_train, 10)
        train_time = time.time()
        print('Training gram matrix ready in {0} secs'.format( train_time - start ))
        #cv_gram    = multiKernel.getConvexSum( X_cv, X_train, weights )    #Change the kernel here
        cv_gram = gaussianKernel( X_cv, X_train, 10 )
        print('Cross validation gram matrix ready in {0} secs'.format( time.time() - train_time ))
        
        #Cache size set to 2GB
        clf = svm.SVC( cache_size = 2000 ) 
        clf.fit( train_gram, y_train )
        print('SVM Fit took {0} secs'.format( time.time() - start ))
        y_pred = clf.predict( cv_gram )
        print('Total time taken for SVM: {0}'.format(time.time() - start ))
        print('Accuracy is :{0}'.format( calculateAccuracy( y_pred, y_cv )))
            