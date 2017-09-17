#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:35:26 2017

@author: rohantondulkar
"""

#Naive Bayes

"""
#==============================================================================
# Call the function runNaiveBayes() form this files to run the program as shown below
# Change the value of variable DIR as per your directory of EmailsData
#
# import naiveBayes as nb
# nb.runNaiveBayes()
#==============================================================================
"""


import sklearn.feature_extraction.text as t
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import os
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import time

#Add your local path for EmailsData directory
DIR = '/Users/rohantondulkar/Projects/ML_CS6510_Assignments/naiveBayes/EmailsData'

NONSPAM_TRAIN   = 'nonspam-train'
SPAM_TRAIN      = 'spam-train'
NONSPAM_TEST    = 'nonspam-test'
SPAM_TEST       = 'spam-test'

def getTFIDFVector():
    """Send the combined list of files names (train + test) to get tfidf vector"""
    os.chdir(DIR)
    nonspam_train_files = [ '{0}/{1}'.format(NONSPAM_TRAIN, filename) for filename in os.listdir( NONSPAM_TRAIN )]
    spam_train_files    = [ '{0}/{1}'.format(SPAM_TRAIN, filename) for filename in os.listdir( SPAM_TRAIN )]
    train_files         = np.append( nonspam_train_files, spam_train_files )
    nonspam_test_files  = [ '{0}/{1}'.format(NONSPAM_TEST, filename) for filename in os.listdir( NONSPAM_TEST )]
    spam_test_files     = [ '{0}/{1}'.format(SPAM_TEST, filename) for filename in os.listdir( SPAM_TEST )]
    test_files          = np.append( nonspam_test_files, spam_test_files )
    tfidf               = t.TfidfVectorizer( 'filename', stop_words='english')
    fit = tfidf.fit_transform( np.append( train_files, test_files ) )
    return fit.toarray()
    
def getDataset():
    """Get the dataset after vectorization and dimensionality reduction"""
    dataset = getTFIDFVector()
    print('Dataset dimensions (train and test combined):{0}'.format( dataset.shape))
    y1      = np.full(350, 0, int)
    y2      = np.full(350, 1, int)
    y_train = np.append(y1, y2)
    
    y3      = np.full(130, 0, int)
    y4      = np.full(130, 1, int)
    y_test  = np.append(y3, y4)
    
    y = np.append(y_train, y_test)
    
    print('Reducing the dataset to 50 dimensions may take few minutes')
    start = time.time()
    kbest = SelectKBest( mutual_info_classif, k = 50)
    newDs = kbest.fit_transform( dataset, y)
    print('Reducing took {0} secs'.format(time.time() - start))
    print('Reduced dimensions: {0}'.format(newDs.shape))
    #np.savetxt('reducedDataset.csv', newDs, delimiter=",")
    X_train = newDs[:700,:]
    X_test  = newDs[700:, :]
    return X_train, y_train, X_test, y_test
    
def calculateGaussianLikelihood( x, mean, var, std ):
    """To calculate the likelihood of x, given mean and variance using Gaussian Distribution"""
    exp_part = np.exp(-0.5*((x-mean)/std)**2)
    return exp_part/np.sqrt( 2*np.pi* var)

def calculateAccuracy( prediction, y ):
    """Return accuracy percentage between predicted and expected class"""
    return len( prediction[ prediction == y ])/len(prediction)
    
class NaiveBayes( object ):
    
    def fit( self, X_train, y_train):
        """Save the values of mean, variance etc for training data"""
        self.class_labels   = np.unique( y_train )
        self.means          = {}
        self.variance       = {}
        self.std            = {}
        self.y_prob         = {}
        for c in self.class_labels:
            self.means[c]       = []
            self.variance[c]    = []
            self.std[c]         = []
            self.y_prob[c]      = len( y_train[ y_train == c])/len(y_train)
            temp = X_train[ y_train == c ]
            for col in range( temp.shape[1] ):
                self.means[c].append( temp[:, col].mean() )
                self.std[c].append( temp[:, col].std())
                self.variance[c].append( temp[:, col].var() )
        
    def predict( self, X_test ):
        """ Predict class for test dataset """
        prediction = np.zeros( X_test.shape[0] )
        count = 0
        for row in X_test:
            pred_classes = np.zeros( self.class_labels.shape[0] )
            for c in self.class_labels:
                p_y = self.y_prob[c]
                pred_classes[c] = 0
                prod = 1
                for i in range(len(row)):
                    mean = self.means[c][i]
                    var  = self.variance[c][i]
                    std  = self.std[c][i]
                    if var == 0:
                        continue
                    gaus = calculateGaussianLikelihood( row[i], mean, var, std )
                    #print('Gaussian prob density: {0} for count:{1}'.format(gaus, count))
                    prod = prod * gaus
                pred_classes[c] = p_y * prod
            prediction[count] = self.class_labels[ pred_classes.argmax() ]
            count+=1
        return prediction
    

def runNaiveBayes():
    """The main method in this program. To get email dataset, apply naive bayes and compare metrics"""
    X_train, y_train, X_test, y_test = getDataset()
    print('----------------------------------------------------')
    print('Running my implementation of Naive Bayes algorithm')
    nb = NaiveBayes()
    nb.fit( X_train, y_train )
    y_pred_1 = nb.predict( X_test )
    print('Accuracy achieved is :{0}'.format( calculateAccuracy( y_pred_1, y_test ) ))
    print('F1 score is :{0}'.format(f1_score( y_test, y_pred_1)))
    print('Area under ROC curve:{0}'.format(roc_auc_score( y_test, y_pred_1)))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test, y_pred_1))
    
    
    print('----------------------------------------------------')
    print('Running sklearn implementation of Naive Bayes algorithm')
    gnb      = GaussianNB()
    y_pred_2 = gnb.fit( X_train, y_train ).predict( X_test )
    print('Accuracy achieved  by scikitlearn NB is :{0}'.format( calculateAccuracy( y_pred_2, y_test ) ))
    print('F1 score is :{0}'.format(f1_score( y_test, y_pred_2)))
    print('Area under ROC curve:{0}'.format(roc_auc_score( y_test, y_pred_2)))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test, y_pred_2))