
import numpy as np
from rbf import Basis_Function
from policy import Policy

"""
important property of LSPI is that it does not require an approximate policy representation,
At each iteration, a different policy is evaluated
and certain sets of basis functions may be more appropriate than others for representing
the state-action value function for each of these policies.

since LSPI approximates state-action value
functions, it can use samples from any policy to estimate the state-action value function of
another policy. This focuses attention more clearly on the issue of exploration since any
policy can be followed while collecting samples.
"""


class LSPI:

    def __init__(self, num_actions=3, num_means=2 ,gamma=0.99 ):

        print num_actions, num_means

        self.basis_function = Basis_Function(num_means, num_means, num_actions, gamma)
        num_basis = self.basis_function._num_basis()

        self.policy = Policy(self.basis_function, num_basis)
        self.lstdq  = LSTDQ(self.basis_function, gamma, self.policy)

        self.stop_criterium= 10**-5
        self.gamma = gamma



    #def agent (self,sample,total_iterationen):


    def _act(self,state):
        index =  self.policy.get_actions(state)  # TODO: validation for random actions
        action = self.policy.actions[index[0]]
        return action



    def train( self,  sample,  total_iterationen, w_important_Sampling=False  ):

        error = float('inf')
        num_interation=0
        epsilon = 0.001

        #print "policy weights", self.policy.weights

        while  (epsilon * (1 - self.gamma) / self.gamma) < error and num_interation< total_iterationen :

            if w_important_Sampling:
                new_weights = self.lstdq.train_weight_parameter ( sample,
                                                                  self.policy,
                                                                  self.basis_function )
            else:
                new_weights = self.lstdq.train_parameter(sample,
                                                         self.policy,
                                                         self.basis_function)


            error = np.linalg.norm((new_weights - self.policy.weights))#difference between current policy and target policy
            self.policy.theta_behavior  = self.policy.weights
            self.policy.weights = new_weights
            #print "new weights", self.policy.weights


            num_interation += 1


        return self.policy


    def td_error(self, sample):

        states = sample[0]
        actions = sample[1]
        rewards = sample[2]
        next_states = sample[3]
        sample_size = len(states)
        td_e = 0.0

        for i in range(sample_size):

            index = self.policy.get_actions(next_states[i])  # TODO: validation in case more actions
            action = self.policy.actions[index[0]]

            index = self.policy.get_actions(states[i])  # TODO: validation in case more actions
            act = self.policy.actions[index[0]]

            Vst = self.policy.q_value_function(next_states[i], action)
            Vs = self.policy.q_value_function(states[i], act)

            td_e += ((rewards[i] + self.gamma * Vst) - Vs) ** 2
            # td_e = (rewards[i]- Vs)**2

        print"td_error=", (td_e / float(sample_size))


        # return (td_e/sample_size)


class LSTDQ:
    def __init__(self,basis_function, gamma, init_policy):
        self.basis_function = basis_function
        self.gamma = gamma
        self.policy = init_policy
        self.gready =[]





    def train_parameter (self, sample, policy , basis_function ):
        r""" Computer Q value function of current policy
            to obtain the gready policy
        """
        k = basis_function._num_basis()

        A=np.zeros([k,k])
        b=np.zeros([k,1])
        np.fill_diagonal(A, .1)

        states      = sample[0]
        actions     = sample[1]
        rewards     = sample[2]
        next_states = sample[3]

        for i in range(len(states)):

            # take action from the gready target policy

            index = policy.get_actions(next_states[i]) # TODO: validation in case more actions
            action= policy.actions[index[0]]


            phi =      self.basis_function.evaluate(states[i], actions[i])
            phi_next = self.basis_function.evaluate(next_states[i], action)

            loss = (phi - self.gamma * phi_next)
            phi  = np.resize(phi, [k, 1])

            phi = np.resize(phi, [k, 1])
            loss = np.resize(phi, [1, len(loss)])

            A = A + np.dot(phi, loss)
            b = b + (phi * rewards[i])

            #A = A +np.dot(loss.transpose(),loss)
            #b = b + (loss.transpose() * rewards[i])

        inv_A = np.linalg.inv(A)

        theta= np.dot(inv_A,b)


        return theta

    def train_weight_parameter(self, sample, policy, basis_function):
        r""" Computer Q value function of current policy
            to obtain the gready policy
        """

        k = basis_function._num_basis()
        A = np.zeros([k, k])
        b = np.zeros([k, 1])
        np.fill_diagonal(A, .1)

        states  = sample[0]
        actions = sample[1]
        rewards = sample[2]
        next_states = sample[3]
        sample_size= len(states)

        self.gready =np.zeros_like(actions)
        self.gready = np.reshape(self.gready,[1,len(actions)])
        self.gready=self.gready[0]


        sum_W = 0.0
        W = 1.0
        for i in range(sample_size):

            act = policy.get_actions(states[i])
            prob_target =   policy.q_value_function(states[i], act[0])
            prob_behavior = policy.behavior(states[i], actions[i])


            #exp = (i - sample_size)
            if prob_behavior==0.0:
                W=0
            else:

                W = (prob_target / prob_behavior)
                sum_W = sum_W + W


        for i in range(sample_size):
            # take action from the gready target policy

            index = policy.get_actions(next_states[i])
            action = policy.actions[index[0]]

            phi =      self.basis_function.evaluate(states[i], actions[i])
            phi_next = self.basis_function.evaluate(next_states[i], action)

            act=policy.get_actions(states[i])

            prob_target =  policy.q_value_function( states[i], act[0] )
            prob_behavior= policy.behavior(states[i], actions[i] )

            self.gready[i] = act[0]


            #print"prob Target", prob_target, "beharvior ",prob_behavior

            exp=(i-sample_size)

            norm_W = (prob_target/prob_behavior  )/sum_W


            #important weigthing on the whole transition

            loss = norm_W *(phi - self.gamma *  phi_next)
            #print "norm W", norm_W


            phi = np.resize(phi, [k, 1])
            loss= np.resize(phi, [1,len(loss)])

            A = A + np.dot(phi, loss)
            b = b + (phi * rewards[i] )
            #print "b=",(phi * rewards[i] )
            #print "b_norm=",norm_W*(phi * rewards[i] )


        inv_A = np.linalg.inv(A)

        theta = np.dot(inv_A, b)
        policy.theta_behavior=policy.weights

        #print "actions=", np.reshape(actions,[1,len(actions)])
        #print "gready=", self.gready,"\n"


        return theta
    """
        lstd has problem mit singular matrix a

        one qay to avoid this problem is to initialize A matrix
        mulitple identify matrix phi*indenty_matrix
        for a robuste inversion use singular value desomposition

        learn the value function Q
    """