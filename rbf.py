import numpy as np


class Basis_Function:
    def __init__(self, input_dim, num_means, num_actions, gamma ):
        self.input_dim = input_dim
        self.num_means = num_means
        self.gamma = 5
        self.num_actions = num_actions


        self.means = [np.random.uniform(-1, 1, input_dim) for i in xrange(self.num_means )]

        self.beta = 8
        #self.W = np.random.random((self.num_basis_func, self.num_actions ))





    def _num_basis(self):


        return (len(self.means)+1 ) * self.num_actions



    def __calc_basis_component(self,state, mean,gamma):

        mean_diff = (state - mean)**2
        return np.exp(-gamma * np.sum(mean_diff) )




    def evaluate(self, state, action):

        if state.shape != self.means[0].shape:
            print state.shape, self.means[0].shape
            raise ValueError('Dimensions of state no match dimensions of means')


        phi = np.zeros((self._num_basis(),))
        offset = (len(self.means[0])+1 ) * action

        rbf = [self.__calc_basis_component(state, mean, self.gamma )
               for mean in self.means]

        #print np.sum(rbf,axis=0),1/(np.sum(rbf,axis=0))

        phi[offset] = 1.
        phi[offset + 1 : offset +1 + len(rbf)] = rbf


        return phi

    #@staticmethod



