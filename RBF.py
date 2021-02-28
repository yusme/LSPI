from scipy import *
from scipy.linalg import norm, pinv

import numpy as np


class BasisFunction:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [np.random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        print "Centers",self.centers
        self.beta = 8
        self.W = np.random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        norm_1 = (c-d)/((c**2)+(d**2))**(1/2)
        print (c-d),(norm_1)
        return np.exp(-self.beta *( (c-d)[0]** 2))

    # Berechnen de basis FUnction for each sample and gaussion
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]

        #print "center", self.centers
        # calculate activations of RBFs
        G = self._calcAct(X)
        #print G


        # calculate output weights (pseudoinverse)
        Maximun_likelihood= pinv(G) #pseudoinverse
        self.W = np.dot(Maximun_likelihood, Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y

