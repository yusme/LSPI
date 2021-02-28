# LSPI  Least-Squares Policy Iteration

(LSPI) reinforcement learning algorithm is a model-free, off-policy method

https://www2.cs.duke.edu/research/AI/LSPI/nips01.pdf

The goal of these algorithms is to perform

LSPI is model-free and uses the results of LSQ to form an approximate policy iteration algorithm. 
This algorithm combines the policy search efficiency of policy iteration with the data efficiency of LSTD

Since LSPI uses LSQ to compute approximate Q functions, it can use any data source for samples.
A single set of samples may be used for the entire optimization, or additional samples may be acquired, 
either through trajectories or some other scheme, for each iteration of policy iteration
 
LSQ: Learning the State-Action Value Function
LSPI uses LSQ to compute approximate Q function



- Solving Acrobot env with LSPI 
-
![](Acrobot.gif)


- Solving mountainCar-v0 env with LSPI

![](MountainCar.gif)


## TODO
- Weighted importance sampling for off-policy
