
#---------------Instructions------------------#
# Please read the function documentation before
# proceeding with code writing. 

# For randomizing, you will need to use following functions
# please refer to their documentation for further help.
# 1. np.random.randint
# 2. np.random.random
# 3. np.random.shuffle
# 4. np.random.normal 


# Other Helpful functions: np.atleast_2d, np.squeeze()
# scipy.stats.mode, np.newaxis

#-----------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#
import tree as tree
import numpy as np
import scipy.stats as stats
from numpy import inf
import random as rd


class RandomForest:
    ''' Implements the Random Forest For Classification... '''
    def __init__(self, ntrees=10,treedepth=5,usebagging=False,baggingfraction=0.6,
        weaklearner="Conic",
        nsplits=10,        
        nfeattest=None, posteriorprob=False,scalefeat=True ):        
        self.ntrees=ntrees
        self.treedepth=treedepth
        self.usebagging=usebagging
        self.baggingfraction=baggingfraction

        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.nfeattest=nfeattest
        
        self.posteriorprob=posteriorprob
        
        self.scalefeat=scalefeat
        
        pass

    def findScalingParameters(self,X):
        self.mean=np.mean(X,axis=0)
        self.std=np.std(X,axis=0)

    def applyScaling(self,X):
        X= X - self.mean
        X= X /self.std
        return X

    def train(self,X,Y,vX=None,vY=None):
        nexamples, nfeatures= X.shape

        self.findScalingParameters(X)
        if self.scalefeat:
            X=self.applyScaling(X)

        self.trees=[]

        for i in range(self.ntrees):
            ShufflingIndexes=list(range(len(X)))
            rd.shuffle(ShufflingIndexes)
            x,y=X[ShufflingIndexes],Y[ShufflingIndexes]
            Indices=int(len(X)*(1-rd.uniform(0,0.4)))
            Sample_X,Sample_Y=x[:Indices,:],y[:Indices]
            Tree=tree.DecisionTree(0.95,5,self.treedepth,self.weaklearner)
            Tree.train(Sample_X,Sample_Y)
            self.trees.append(Tree)

        
    def predict(self, X):
        
        """
        Test the trained RF on the given set of examples X
        
                   
            Input:
            ------
                X: [m x d] a d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        z = []
        
        if self.scalefeat:
            X=self.applyScaling(X)

        for i in range(self.ntrees):
            z.append([])
            z[i]=self.trees[i].test(X)
        z=np.transpose(z)
        Labels=[]
        for i in range(len(X)):
            UniqueLabels,Count=np.unique(z[i],return_counts=True)
            Labels.append(UniqueLabels[np.where(np.max(Count)==Count)])
        return Labels
    