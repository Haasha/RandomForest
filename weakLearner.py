

#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes.
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats
from numpy import inf
import random

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):

        self.F_Index,self.Split=-1,-1
        
    def TotalEntropy(self,Y):
        UniqueLabels,Count=np.unique(Y,return_counts=True)
        TotalCount=np.sum(Count)
        TotalEntropy=0
        for i in range(len(UniqueLabels)):
            TotalEntropy+=(-Count[i]/TotalCount)*np.log2(Count[i]/TotalCount)
        return TotalEntropy

    def calculateEntropy(self, Y, mship):
        TotalLabelsLesser,TotalLabelsGreater=Y[mship],Y[mship==0]
        UniqueLabelsGreater,CountGreater=np.unique(TotalLabelsGreater,return_counts=True)
        UniqueLabelsLesser,CountLesser=np.unique(TotalLabelsLesser,return_counts=True)
        
        TotalGreaterCount,TotalLesserCount=np.sum(CountGreater),np.sum(CountLesser)
        TotalCount=TotalGreaterCount+TotalLesserCount
        result_1=0
        result_2=0
        for k in range(len(UniqueLabelsGreater)):
            if CountGreater[k]!=0 and TotalGreaterCount!=0:
                result_1+= ((-CountGreater[k]/TotalGreaterCount)*np.log2(CountGreater[k]/TotalGreaterCount))
        for k in range(len(UniqueLabelsLesser)):
            if CountLesser[k]!=0 and TotalLesserCount!=0:
                result_2+= ((-CountLesser[k]/TotalLesserCount)*np.log2(CountLesser[k]/TotalLesserCount))

        return result_1*(TotalGreaterCount/TotalCount) + result_2*(TotalLesserCount/TotalCount)
    
    def evaluate_numerical_attribute(self,feat, Y):
        TotalEntropy=self.TotalEntropy(Y)
        UniqueLabels,Count=np.unique(Y,return_counts=True)
        Index=0
        TargetedEntropy=float('Inf')
        
        for i in range(len(feat)):
            Point=feat[i]
            mship=feat<=Point
            Temp=self.calculateEntropy(Y,mship)
            if Temp<TargetedEntropy:
                TargetedEntropy=Temp
                Index=i
        score=TotalEntropy-TargetedEntropy
        split=feat[Index]
        RightChildInd=feat>split
        LeftChildInd=feat<=split
        return split, score,LeftChildInd,RightChildInd
    
    def train(self, X, Y):
        
        nexamples,nfeatures=X.shape
        split,score,LeftChildInd,RightChildInd=0,-float('inf'),0,0
        FeatureIndex=-1
        for i in range(nfeatures):
            split_temp,score_temp,LeftChildInd_temp,RightChildInd_temp=self.evaluate_numerical_attribute(X[:,i],Y)
            if i!=0:
                if score_temp>score:
                    score=score_temp
                    split=split_temp
                    LeftChildInd=LeftChildInd_temp
                    RightChildInd=RightChildInd_temp
                    FeatureIndex=i
            else:
                score=score_temp
                split=split_temp
                LeftChildInd=LeftChildInd_temp
                RightChildInd=RightChildInd_temp
                FeatureIndex=i
        self.F_Index=FeatureIndex
        self.Split=split
        return split,score,LeftChildInd,RightChildInd

    def evaluate(self,X):
        if X[self.F_Index]<=self.Split:
                return True
        return False

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....


    def __init__(self, nsplits=+np.inf, nrandfeat=None):

        WeakLearner.__init__(self) # calling base class constructor...
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
        self.fidx=-1
        self.split=-1
        #pass

    def train(self, X, Y):

        #print "Inside the train of Random"

        nexamples,nfeatures=X.shape

        #print "Train has X of length ", X.shape


        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        Index=-1
        split,score,LeftChildInd,RightChildInd=0,-float('inf'),0,0
        Features=np.random.randint(0,nfeatures,self.nrandfeat)
        for i in range(len(Features)):
            split_temp,score_temp,LeftChildInd_temp,RightChildInd_temp=self.findBestRandomSplit(X[:,Features[i]],Y)
            if score_temp>score:
                score=score_temp
                split=split_temp
                LeftChildInd=LeftChildInd_temp
                RightChildInd=RightChildInd_temp
                Index=Features[i]
        self.F_Index=Index
        self.Split=split
        return split,score,LeftChildInd,RightChildInd
        #---------End of Your Code-------------------------#
    

    def findBestRandomSplit(self,feat,Y):

        frange=np.max(feat)-np.min(feat)


        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        TotalEntropy=self.TotalEntropy(Y)
        
        splitvalue,TargetEntropy,Index=0,float('inf'),-1
        for i in range(self.nsplits):
            RandomIndex=random.randrange(len(feat))
            mship=feat<=feat[RandomIndex]
            score_temp=self.calculateEntropy(Y,mship)
            
            if score_temp<=TargetEntropy:
                TargetEntropy=score_temp
                splitvalue=feat[RandomIndex]
                Index=RandomIndex
        #---------End of Your Code-------------------------#
        score=TotalEntropy-TargetEntropy
        LeftChildInd=feat<=feat[Index]
        RightChildInd=feat>feat[Index]
        return splitvalue, score, LeftChildInd, RightChildInd

    

    def evaluate(self, X):

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        #print X.shape, self.fidx, "xshape"
        if X[self.F_Index]<=self.Split:
                return True
        return False
        #---------End of Your Code-------------------------#



# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....

    def __init__(self, nsplits=10):
        self.a=0
        self.b=0
        self.c=0
        self.F1=0
        self.F2=0

        RandomWeakLearner.__init__(self,nsplits)

        #pass

    def train(self,X, Y):

        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        MinScore,Index=float('Inf'),-1
        self.F1,self.F2=random.sample(range(0, nfeatures), 2)
        Left=[]
        Right=[]
        for i in range(self.nsplits):
            Temp_a,Temp_b,Temp_c=np.random.uniform(-4,4,3)
            Results=[]
            for j in range(len(X)):
                if (Temp_a*X[j,self.F1]+Temp_b*X[j,self.F2]+Temp_c)<=0:
                    Results.append(1)
                else:
                    Results.append(0)
            
            if np.sum(Results)>0 and np.sum(Results)<len(X):
                Score=super().calculateEntropy(Y,list(Results))
                if MinScore>=Score:
                    MinScore=Score
                    self.a=Temp_a
                    self.b=Temp_b
                    self.c=Temp_c
                    Index=i
                    Left=Results
                    Right=Results
        for i in range(len(Right)):
            if Right[i]==1:
                Right[i]=0
            else:
                Right[i]=1
        return 0, MinScore, Left, Right


    def evaluate(self,X):
        if self.a*X[self.F1]+self.b*X[self.F2]+self.c<=0:
            return True
        return False


#build a classifier a*x^2+b*y^2+c*x*y+ d*x+e*y+f
class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....

    def __init__(self, nsplits=10):
        self.a=0
        self.b=0
        self.c=0
        self.d=0
        self.e=0
        self.f=0
        self.F1=0
        self.F2=0

        RandomWeakLearner.__init__(self,nsplits)
        
        pass

    
    def train(self,X, Y):
        
        nexamples,nfeatures=X.shape
        MinScore,Index=float('Inf'),-1
        self.F1,self.F2=random.sample(range(0, nfeatures), 2)
        Left=[]
        Right=[]
        for i in range(self.nsplits):
            Temp_a,Temp_b,Temp_c,Temp_d,Temp_e,Temp_f=np.random.uniform(-4,4,6)
            Results=[]
            for j in range(len(X)):
                if (Temp_a*(X[j,self.F1]**2)+Temp_b*(X[j,self.F2]**2)+Temp_c*X[j,self.F1]*X[j,self.F2] + Temp_d*X[j,self.F1]+ Temp_e*X[j,self.F2] +Temp_f<=0)<=0:
                    Results.append(1)
                else:
                    Results.append(0)
            if np.sum(Results)>0 and np.sum(Results)<len(X):
                Score=super().calculateEntropy(Y,list(Results))
                if MinScore>=Score:
                    MinScore=Score
                    self.a=Temp_a
                    self.b=Temp_b
                    self.c=Temp_c
                    self.d=Temp_d
                    self.e=Temp_e
                    self.f=Temp_f
                    Index=i
                    Left=Results
        for i in range(len(Left)):
            if Left[i]==1:
                Right.append(0)
            else:
                Right.append(1)
        return 0, MinScore, Left, Right

    def evaluate(self,X):
        if self.a*(X[self.F1]**2)+self.b*(X[self.F2]**2)+self.c*X[self.F1]*X[self.F2] + self.d*X[self.F1]+ self.e*X[self.F2] +self.f<=0:
            return True
        return False
