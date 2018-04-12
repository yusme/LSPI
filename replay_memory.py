from random import sample as random
import collections as memory
import numpy as np
import random

alpha=0.7
beta=0.5


from collections import defaultdict

class Memory:
    # This is unsere MDP model
    #TODO: contruct the MDP chain of state

     def __init__(self,MemorySize, batch_size, act_dim,obs_dim):
         self.Memorysize = MemorySize
         #self.batch_size = batch_size
         self.container= memory.deque()
         self.containerSize = 0
         self.priority=1
         self.act_dim=act_dim
         self.obs_dim=obs_dim



     def get_size(self):
         return self.batch_size

     def size(self):
         return self.containerSize

     def select_batch(self, batchSize):
         return random.sample(self.container, batchSize)

     def add(self, experience):

         experience.append(self.priority)

         if self.containerSize < self.Memorysize:
            self.container.append(experience)
            self.containerSize = self.containerSize+1


         else:
             self.container.popleft()
             self.container.append(experience)


     def transform_sample(self,sample,batch_size):




         obs_dim=self.obs_dim
         act_dim = self.act_dim

         current_state=  [x[0] for x in sample]
         actions =     np.asarray([x[1] for x in sample])
         rewards =     [x[2] for x in sample]
         next_state=   [x[3] for x in sample]
         done =        [x[4] for x in sample]


         current_state = np.resize(current_state,[batch_size,obs_dim])
         actions       = np.resize(actions, [batch_size, act_dim])
         rewards       = np.resize(rewards, [batch_size, act_dim])
         next_state    = np.resize(next_state, [batch_size, obs_dim])
         done          = np.resize(done, [batch_size, act_dim])


         return [current_state,actions,rewards,next_state,done]

     def select_sample(self,batch_size):
         print "container size",self.containerSize
         sample = random.sample(self.container, batch_size)
         return self.transform_sample(sample,batch_size)

     def clear_memory(self):
         self.container = memory.deque()
         self.containerSize=0
         self.num_experiences = 0


     def important_sampling(self, batch_size,policy):
         current_state, actions, rewards, next_state, done\
             = self.select_sample(batch_size)
         discount_factor=0.8

         G = 0.0
         W = 1.0
         C = np.zeros(3)

         for i in range(batch_size):
            G =+ discount_factor*rewards[i]
            C+=W
            q_state_action=policy.q_value_function(current_state[i],actions[i])
            new_q = (W/C)*(G - q_state_action)
            W = W * 1. / behavior_policy(state)[action]











