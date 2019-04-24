from matplotlib import pyplot as pl
from replay_memory import Memory
from policy import Policy
from rbf import Basis_Function
from lspi import LSPI, LSTDQ
import gym
import scipy 
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#Settings
TRANSITION=15000
EPISODE = 1000
BATCH_SIZE=400
MEMORY_SIZE=TRANSITION+1000
important_sampling = None
lspi_interation = 20
num_actions=3
num_means=4
gamma=0.99




mean_reward1=[]
mean_reward2=[]

def test_policy(policy,env, state,agent):   
    print ("Test")
    Best_policy= 0   
    total_reward = 0.0
    state = env.reset()
    for j in range(1):
        state = env.reset()
        for i in range(5000):
            env.render()
            index = policy.get_actions(state)  
            #action = policy.actions[index[0]]  
            action=agent._act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += gamma * reward            
            if done:
                Best_policy=agent.policy
                print("done")
                break
    return total_reward, Best_policy



def _initial_sample2(env, memory,agent ):
    state = env.reset()
    ##action = env.action_space.sample()
    total_reward = -4000
    best_reward=-4000
    Best_agent=None
    found=False
    best_theta=False
    for j in range(EPISODE):
        state = env.reset()
        best_theta = False
        for i in range(TRANSITION):
           # env.render()
            #action = agent._act(state)
            if best_reward >= total_reward and found==False:
                action = env.action_space.sample()
            else:
                agent=Best_agent
                best_theta = True
                action = agent._act(state)
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                print ("done interation=",i)
                break
        if j>0:
            if done:
                sample = memory.select_sample(j)
            else:
                sample = memory.select_sample(TRANSITION)
            policy = agent.train(sample, lspi_interation, important_sampling)
            total_reward, policy_test =test_policy(policy, env, state,agent)
            if best_reward < total_reward:
                Best_agent=agent
                best_reward=total_reward
                total_reward=-4950.0
                #found=True
            print "TEST---",j
            print "total_reward",total_reward
            if best_theta:
                memory.clear_memory()
    memory.clear_memory()
    return mean_reward1

def _reuse_sample2(env, memory, agent ):
    state = env.reset()
    total_reward = 0.0
    important_sampling = False
    for j in range(EPISODE):
        print ("episode", j, "/", EPISODE)
        state = env.reset()
        total_reward = 0.0
        for i in range(TRANSITION):
            env.render()
            if j<50:
                action = env.action_space.sample()
            else:
                action = agent._act(state)
                sample = memory.select_sample(BATCH_SIZE)  # [current_state, actions, rewards, next_state, done]
                agent.train(sample, lspi_interation, important_sampling)
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                print ("done")
                break

        sample = memory.select_sample(BATCH_SIZE)  # [current_state, actions, rewards, next_state, done]
        agent.train(sample, lspi_interation, important_sampling)
        #memory.clear_memory()
        print (memory.containerSize)
    return mean_reward2


def experiment_1():
    import gym
    #env = gym.make('InvertedPendulum-v1')
    env = gym.make('Acrobot-v0');
    state = env.reset()

    num_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    action_dim = 1

    memory = Memory(MEMORY_SIZE, BATCH_SIZE,
                    action_dim,  obs_dim)
    print (num_actions,obs_dim)

    agent =  LSPI(num_actions,obs_dim )
    return agent, env, memory


def experiment_2():
    env = gym.make('MountainCar-v0');
    state = env.reset()

    action_dim = 1
    num_actions = env.action_space.n
    obs_dim     = env.observation_space.shape[0]
    memory      = Memory(MEMORY_SIZE, BATCH_SIZE,
                    action_dim,  obs_dim)

    agent =  LSPI(num_actions,obs_dim)

    return agent, env , memory




def main():
    #agent, env, memory = experiment_1()
    agent, env, memory = experiment_2()

    print ("memory size", memory.containerSize)

    y2 =_reuse_sample2(env, memory, agent)
    y1 = _initial_sample2(env, memory, agent)

    x  = np.arange(0, len(mean_reward1))

    np.reshape(mean_reward1,x.shape)
    print (x.shape, mean_reward1, mean_reward2, x)

    import plot as pl
    pj = pl.Plot()
    pj.plot_rewad(x, y1,y2)




def _initial_sample(env, memory, policy=None):
    impotant_samplig = False
    lspi_interation = 20
    lspi = LSPI()
    state = env.reset()
    action = env.action_space.sample()
    for j in range(EPISODE):
        print ("episode-", j, "/", EPISODE, j / float(EPISODE))
        for i in range(TRANSITION):
            env.render()
            if policy is None:
                action = env.action_space.sample()
            else:
                sample = memory.select_sample(BATCH_SIZE)  # [current_state, actions, rewards, next_state, done]
                policy = lspi.train(sample, lspi_interation, impotant_samplig)
                index  = policy.get_actions(state)  # TODO: for random actions
                action = policy.actions[index[0]]
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                print ("done")
                test_policy(policy, env, state)
                break
        sample = memory.select_sample(BATCH_SIZE)
        policy = lspi.train(sample, lspi_interation, impotant_samplig)
    return policy


def _reuse_sample(env, memory):
    total_reward = 0.0
    lspi = LSPI()
    policy = lspi.policy
    state = env.reset()
    for j in range(EPISODE):
        print "episode-",j,"/",EPISODE ,j/float(EPISODE)
        for i in range(TRANSITION):
            env.render()
            if memory.containerSize < BATCH_SIZE:
                index = policy.get_actions(state)  # TODO: validation for random actions
                action = policy.actions[index[0]]

            else:
                sample = memory.select_sample(BATCH_SIZE)  # [current_state, actions, rewards, next_state, done]
                policy = lspi.train(sample, lspi_interation, important_sampling)
                index = policy.get_actions(state)
                action = policy.actions[index[0]]
                memory.clear_memory()
            next_state, reward, done, info = env.step(action)
            memory.add([state, action, reward, next_state, done])
            state = next_state
            if done:
                print ("done")
                state = env.reset()
                break
            mean_reward1.append(total_reward)
            total_reward = 0.0
            # R=test_policy(policy, env,state)
            # mean_reward.append([np.mean(R),np.max(R),np.min(R)])


def exmaple_RBF():
    
    rbf= BasisFunction2(1, 13, 1)
    #rbf =BasisFunction()
    n = 500
    x = np.mgrid[-1:1:complex(0, n)].reshape(n, 1)
    #print "vector" ,np.mgrid[-1:1:complex(0, n)]
    # set y and add random noise
    y = np.sin(3 * (x + 0.5) ** 3 - 50)
    # y += random.normal(0, 0.1, y.shape)
    # rbf regression
    rbf.train(x, y)
    z = rbf.test(x)
    plot_function(x, y, z,rbf)


if __name__ == '__main__':
    main()






