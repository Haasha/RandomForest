
# A good heuristic is to choose sqrt(nfeatures) to consider for each node...
import weakLearner as wl
import numpy as np
import scipy.stats as stats
from numpy import inf

#---------------Instructions------------------#

# Here you will have to reproduce the code you have already written in
# your previous assignment.

# However one major difference is that now each node non-terminal node of the
# tree  object will have  an instance of weaklearner...

# Look for the missing code sections and fill them.
#-------------------------------------------#



class Node:
    def __init__(self,purity,klasslabel='',pdistribution=[],score=0,wlearner=None):
        """
               Input:
               --------------------------
               klasslabel: to use for leaf node
               pdistribution: posteriorprob class probability at the node
               score: split score
               weaklearner: which weaklearner to use this node, an object of WeakLearner class or its childs...

        """

        self.lchild=None
        self.rchild=None
        self.klasslabel=klasslabel
        self.pdistribution=pdistribution
        self.score=score
        self.wlearner=wlearner
        self.purity = purity

    def set_childs(self,lchild,rchild):
        
        self.lchild=lchild
        self.rchild=rchild

        
    def isleaf(self):
        if self.lchild or self.rchild:
            return False
        return True

    def isless_than_eq(self, X):
        return self.wlearner.evaluate(X)


    def get_str(self):
        """
            returns a string representing the node information...
        """
        if self.isleaf():
            return 'C(posterior={},class={},Purity={})'.format(self.pdistribution, self.klasslabel,self.purity)
        else:
            return 'I(Fidx={},Score={},Split={})'.format(self.fidx,self.score,self.split)


class DecisionTree:
    ''' Implements the Decision Tree For Classification With Information Gain
        as Splitting Criterion....
    '''
    def __init__(self, purity, exthreshold=5, maxdepth=10,
     weaklearner="Conic", pdist=False, nsplits=10, nfeattest=None):
        '''
        Input:
        -----------------
            exthreshold: Number of examples to stop splitting, i.e. stop if number examples at a given node are less than exthreshold
            maxdepth: maximum depth of tree upto which we should grow the tree. Remember a tree with depth=10
            has 2^10=1K child nodes.
            weaklearner: weaklearner to use at each internal node.
            pdist: return posterior class distribution or not...
            nsplits: number of splits to use for weaklearner
        '''
        self.purity = purity
        self.maxdepth=maxdepth
        self.exthreshold=exthreshold
        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.pdist=pdist
        self.nfeattest=nfeattest
        assert (weaklearner in ["Conic", "Linear","Axis-Aligned","Axis-Aligned-Random"])
        #pass
    def FindImpurity(self,Y):
        UniqueLabels,Count=np.unique(Y,return_counts=True)
        if len(Count)!=1:    
            Max=np.max(Count)
            Sum=np.sum(Count)
            Impurity=Max/Sum
            Label=UniqueLabels[np.where(Count==Max)][0]
        else:
            Impurity=1
            Label=UniqueLabels[0]
        return Label,Impurity
    
    def getWeakLearner(self):
        if self.weaklearner == "Conic":
            return wl.ConicWeakLearner(self.nsplits)
        elif self.weaklearner== "Linear":
            return wl.LinearWeakLearner(self.nsplits)
        elif self.weaklearner == "Axis-Aligned":
            return wl.WeakLearner()
        else:
            return wl.RandomWeakLearner(self.nsplits,self.nfeattest)
        
        #pass

    def train(self, X, Y):
        nexamples,nfeatures=X.shape
        self.tree=self.build_tree(X,Y,self.maxdepth)
        
        
        
    def build_tree(self, X, Y, depth):
        """ 
            Function is used to recursively build the decision Tree 
          
            Input
            -----
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns
            -------
            root node of the built tree...
        """
        
        nexamples, nfeatures=X.shape
        # YOUR CODE HERE
        Split=0
        InfoGain=-float('Inf')
        RightChildInd=0
        LeftChildInd=0
        FeatureIndex=-1
        Label,Impurity=self.FindImpurity(Y)
        Learner=self.getWeakLearner()
        if depth==0 or len(X)<=self.exthreshold or Impurity>=self.purity:
            return Node(Impurity,Label,0,0,Learner)
        
        Split,InfoGain,LeftChildInd,RightChildInd=Learner.train(X,Y)
        Temp_X,Temp_Y=X[LeftChildInd],X[RightChildInd]
        
        if len(Temp_X)==0 or len(Temp_Y)==0:
            return Node(Impurity,Label,0,0,Learner)
            
                
        node=Node(Impurity,Label,0,InfoGain,Learner)
        RightNode=self.build_tree(X[RightChildInd],Y[RightChildInd],depth-1)
        LeftNode=self.build_tree(X[LeftChildInd],Y[LeftChildInd],depth-1)
        node.set_childs(LeftNode,RightNode)
        return node
        
    def test(self, X):
        nexamples, nfeatures=X.shape
        pclasses=self.predict(X)
        return np.array(pclasses)

    def predict(self, X):
        pclass=[]
        for i in range(len(X)):
            Point=X[i]
            pclass.append(self._predict(self.tree,Point))
        return pclass

    def _predict(self,node, X):
        # YOUR CODE HERE
        if node.isleaf():
            return node.klasslabel
        else:
            if node.isless_than_eq(X):
                return self._predict(node.lchild,X)
            else:
                return self._predict(node.rchild,X)
    
    def __str__(self):
        """
            overloaded function used by print function for printing the current tree in a
            string format
        """
        str = '---------------------------------------------------'
        str += '\n A Decision Tree With Depth={}'.format(self.find_depth())
        str += self.__print(self.tree)
        str += '\n---------------------------------------------------'
        return str  # self.__print(self.tree)


    def _print(self, node):
        """
                Recursive function traverse each node and extract each node information
                in a string and finally returns a single string for complete tree for printing purposes
        """
        if not node:
            return
        if node.isleaf():
            return node.get_str()

        string = node.get_str() + self._print(node.lchild)
        return string + node.get_str() + self._print(node.rchild)

    def find_depth(self):
        """
            returns the depth of the tree...
        """
        return self._find_depth(self.tree)

    def _find_depth(self, node):
        """
            recursively traverse the tree to the depth of the tree and return the depth...
        """
        if not node:
            return
        if node.isleaf():
            return 1
        else:
            return max(self._find_depth(node.lchild), self._find_depth(node.rchild)) + 1

    def __print(self, node, depth=0):
        """

        """
        ret = ""

        # Print right branch
        if node.rchild:
            ret += self.__print(node.rchild, depth + 1)

        # Print own value

        ret += "\n" + ("    "*depth) + node.get_str()

        # Print left branch
        if node.lchild:
            ret += self.__print(node.lchild, depth + 1)

        return ret
