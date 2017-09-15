#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:14:59 2017

@author: rohantondulkar
"""

INTERNAL = 'Internal'
LEAF     = 'Leaf'

#def splitDataSet( dataset ):
#    """"""
#    idx      = np.arange( 0,10000 )
#    np.random.shuffle(idx)
#    train   = dataset.loc[idx[:8000]]
#    cv      = dataset.loc[idx[8000:]]
#    X_train = train.loc[:,:13].values
#    y_train = train.loc[:, 14].values
#    X_cv    = cv.loc[:,0:13].values
#    y_cv    = cv.loc[:, 14].values
#    return X_train, y_train, X_cv, y_cv

def calculateAccuracy( prediction, y ):
    """Return accuracy percentage between predicted and expected class"""
    return len( prediction[ prediction == y ])/len(prediction)

class TreeNode( object ):
    
    def __init__( self, attribute = None, ntype = INTERNAL, label = None ):
        self.attribute = attribute
        self.branches = {}
        self.ntype = ntype
        if self.ntype == LEAF:
            self.label = label
        
    def setAttribute( self, attribute = None ):
        """"""
        self.attribute = attribute
        
    def setNodeType( self, ntype = INTERNAL, label = None ):
        """"""
        self.ntype = ntype
        if self.ntype == LEAF:
            self.label = label
            
    def getLabel( self ):
        return self.label
    
    def addBranch( self, branchName, branchNode ):
        #print('Adding branch {0} for attribute {1}'.format( branchName, self.attribute ))
        self.branches[ branchName ] = branchNode

class DecisionTreesClassifier( object ):
    """
    Decision tree classifier for categorical values only.
    """
    
    def __init__( self, entropy_threshold, max_nodes, missing_vals = [ ' ?' ] ):
        import numpy as np
        import csv
        self.np     = np
        self.csv    = csv
        self.entropy_threshold = entropy_threshold
        self.maxNodes = max_nodes
        self.missing_vals = missing_vals
        #self.dept = 0
        
    def processDataSet( self, dataset ):
        """Handle the columns with continuous data by putting them in bins based on histogram analysis"""
                
        # For column 0 min = 17, max = 90. Split in bins of  size 10
        bins_0        = self.np.arange(0, 100, 10)
        dataset[:,0]  = self.np.digitize( dataset[:,0], bins_0 )
        
        # For column 2 min = 19214 max = 1184622. Split in bins of size 200000
        bins_2        = self.np.arange(0, 2000000, 200000 )
        dataset[:,2]  = self.np.digitize( dataset[:,2], bins_2 )
        
        # For column 4 min = 1 max = 16. Split in bins of size 2
        bins_4        = self.np.arange(0, 16, 2)
        dataset[:,4]  = self.np.digitize( dataset[:,4], bins_4 )
        
        # For column 10 min = 0, max = 99999. Split in bins of size 10000
        bins_10       = self.np.arange(0, 100000, 10000)
        dataset[:,10] = self.np.digitize( dataset[:,10], bins_10 )
        
        # For column 11 min = 0, max = 4356. Split in bins of size 200
        bins_11       = self.np.arange(0, 5000, 200)
        dataset[:,11] = self.np.digitize( dataset[:,11], bins_11 )
        
        # For column 12 min = 1, max = 99. Split in bins of size 10
        bins_12       = self.np.arange(0, 100, 10)
        dataset[:,12] = self.np.digitize( dataset[:,12], bins_12 )
        
        #Handling missind data ' ?' in dataset
        str_cols = [1,3,5,6,7,8,9,13]
        for col in str_cols:
            max_val = dataset[col].value_counts().idxmax()
            for j in range(dataset.shape[0]):
                if dataset.loc[j, col] in self.missing_vals:
                    dataset.loc[j,col] = max_val
            
        return dataset
    
    def fit( self, train_filename ):
        """Build a decision tree based on the given training dataset"""
        dataset = self.np.genfromtxt( train_filename, delimiter = ',', \
                                dtype = {'names':('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'), \
                                         'formats': (int, '<U20', int, '<U20', int, '<U20', '<U20', \
                                                     '<U20', '<U20', '<U20', int, int, int, '<U20', int )})
    
        
        self.dataset  = self.processDataSet( dataset )
        self.X_train  = self.dataset[:,:14]
        self.y_train  = [ int(val) for val in dataset[:, 14] ]
        self.numTrain = self.X_train.shape[0]
        self.features = self.np.arange( self.X_train.shape[1], dtype = int )
        self.root     = TreeNode()
        self.maxclass = self.np.bincount( self.y_train ).argmax()
        self.numNodes = 0
        self.generateTree( self.root, self.X_train, self.y_train, self.features )
    
    def predict( self, X_test_filename ):
        """Predict the value for given test dataset"""
        prediction = self.np.zeros( X_test.shape[0], dtype = int)
        count = 0
        for row in X_test:
            prediction[count] = self.predictRow( self.root, row )
            count +=1
        return prediction
            
    def predictRow( self, node, test_row ):
        """"""
        if node.ntype == LEAF:
            #print('Predicting class:{0}'.format(node.label))
            return node.label
        attrValue = test_row[ node.attribute ]
        #print('Going to sub-tree of attr:{0} for branch: {1}'.format(node.attribute, attrValue))
        if attrValue in node.branches:
            return self.predictRow( node.branches[ attrValue ], test_row )
        else:
            return self.maxclass
    
    def calculateEntropy( self, ytrain ):
        """Calculates the entropy using the formula : """
        """ φ(p, 1 − p) = −p log2 p − (1 − p) log2(1 − p) for two classes"""
        n = len( ytrain )
        entropy = 0
        for c in self.np.unique( ytrain ):
            p = len( ytrain[ ytrain == c ] )/n
            entropy += p * self.np.log2(p)
        return -entropy
    
    def calculateEntropyAfterSplit( self, xtrain, ytrain, feature ):
        """Calculate the entropy after split for a particular feature"""
        n = xtrain.shape[0]
        fEnt = 0
        for val in self.np.unique( xtrain[:,feature] ):
            split  = xtrain[:,feature] == val
            xsplit = xtrain[ split ] 
            ysplit = ytrain[ split ]
            nval   = xsplit.shape[0]
            valEnt = self.calculateEntropy( ysplit ) * nval/n
            fEnt  += valEnt
        #print('Entropy for feature: {0} is {1}'.format( feature, fEnt ))
        return fEnt
    
    def getSplittingAttribute( self, xtrain, ytrain, features ):
        """Returns the splitting attribute based on the least splitting entropy"""
        minEnt = 1
        splitAttr = features[0]
        for f in features:
            fEnt = self.calculateEntropyAfterSplit( xtrain, ytrain, f )
            if fEnt < minEnt:
                minEnt    = fEnt
                splitAttr = f
        return splitAttr
    
    def getUniqueValuesInAttribute( self, attr ):
        """Returns the list of unique values for that feature"""
        return self.np.unique( self.X_train[:, attr] )
    
    def generateTree( self, node, xtrain, ytrain, features ):
        """Generate the sub-tree from this node"""
        self.numNodes +=1
        if len(ytrain) == 0 or len(features) == 0 or self.numNodes > self.maxNodes:
            node.ntype = LEAF
            node.label = self.maxclass
            #print('Creating leaf node with label:{0}'.format(self.maxclass))
            return
        entropy = self.calculateEntropy( ytrain )
        if  entropy < self.entropy_threshold:
            label = self.np.bincount(ytrain).argmax() #calculates highest occuring class label
            node.ntype = LEAF
            node.label = label
            #print('Creating leaf node with label:{0} for entropy: {1}'.format(label, entropy))
            return
        attr   = self.getSplittingAttribute( xtrain, ytrain, features )
        #print('Selected attribute {0}'.format(attr))
        values = self.getUniqueValuesInAttribute( attr )
        node.setAttribute( attr )
        #print( 'Root attr:{0}'.format(self.root.attribute) )
        #print( 'Root branches:{0}'.format(self.root.branches) )
        for branchName in values:
            branchNode = TreeNode()
            node.addBranch( branchName, branchNode )
            boolXtrain = xtrain[:, attr] == branchName
            newXtrain  = xtrain[ boolXtrain ]
            newYtrain  = ytrain[ boolXtrain ]
            idx        = self.np.where( features == attr )[0][0]
            self.generateTree( branchNode, newXtrain, newYtrain, self.np.delete( features, idx ) )
            
    def displayTree( self ):
        """Print the nodes of the decision tree in human-readable format"""
        self.printNode( self.root )
        print('Number of nodes in tree: {0}'.format(self.numNodes))
    
    def printNode( self, node ):
        """"Prints the node and calls recursively for branches"""
        if node.ntype == LEAF:
            print('Leaf Node label:{0}'.format( node.label ))
            return
        print( 'Node attribute: {0}, branches: {1}'.format( node.attribute, node.branches.keys() ))
        for b in node.branches.values():
            self.printNode( b )
        
def runDecisionTrees():
    """"""
    #dataset = pd.read_csv( 'data/train.csv', header=None)
    #dataset = processDataSet( dataset )
    #X_train, y_train, X_cv, y_cv = splitDataSet( dataset )
    dTreeClf = DecisionTreesClassifier( 0.05, 20000 )
    #dTreeClf.fit( X_train, y_train )
    dTreeClf.fit( 'data/train.csv' )
    #dTreeClf.displayTree()
    y_pred = dTreeClf.predict( X_cv )
    print('Accuracy is {0}'.format(calculateAccuracy( y_pred, y_cv)))
    totalActualP = len(y_cv[y_cv==1])
    totalActualN = len(y_cv[y_cv==0])
    tp = len(y_pred[np.logical_and(y_pred == 1, y_pred == y_cv)])
    tn = len(y_pred[np.logical_and(y_pred == 0, y_pred == y_cv)])
    print('True positive: {0}, Actual positive: {1}'.format(tp, totalActualP))
    print('True negative: {0}, Actual negative: {1}'.format(tn, totalActualN))
    predP = len(y_pred[y_pred==1])
    predN = len(y_pred[y_pred==0])
    print('Precision for class 1: {0}'.format(tp/predP))
    print('Recall for class 1: {0}'.format( tp/totalActualP ))
    print('Precision for class 0: {0}'.format(tn/predN))
    print('Recall for class 0: {0}'.format( tn/totalActualN ))
    #dTreeClf.displayTree()