from __future__ import division

import numpy as np
from rbf import Basis_Function


class Policy:

    def __init__(self,basis, num_theta, theta=None ):
        self.basis_function=basis
        self.actions = [0, 1, 2]

        self.num_theta=num_theta

        # uniform distribution of the actions


        self.theta_behavior= theta

        if theta is None:
            self.weights = np.random.uniform(-1.0, 1.0, size=(num_theta,))

        else:
            self.weights=theta

    def set_theta(self, theta):
        self.weights = (self.weights+theta)*0*5


    def behavior(self,state,action):
        prob=0.0
        if self.theta_behavior is None:
            self.theta_behavior = np.random.uniform(-1.0, 1.0, size=(self.num_theta,))


        vector_basis = self.basis_function.evaluate(state, action)
        return np.dot(vector_basis, self.theta_behavior)




    def q_value_function(self, state, action ):
        vector_basis = self.basis_function.evaluate(state,action)
        return np.dot(vector_basis,self.weights)

    def get_actions(self, state):


        q_state_action=[self.q_value_function(state,self.actions[i]) for i in range(len(self.actions))]
        q_state_action = np.reshape(q_state_action,[len(q_state_action),1])# convert to column vector

        index = np.argmax(q_state_action)
        q_max = q_state_action[index]


        best_actions = [self.actions[index]]
        ind =[index]

        for i in range(len(q_state_action)):
            if q_state_action[i]==q_max and index!=i:
                best_actions.append(self.actions[i])
                ind.append(i)



        return best_actions







