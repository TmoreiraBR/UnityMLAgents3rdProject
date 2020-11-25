<<<<<<< HEAD
[//]: # (Image References)

[image2]: https://github.com/TmoreiraBR/UnityMLAgents3rdProject/blob/main/TrainedResults.jpg  "Training Agents"

# Report for Project 2: Continuous Control

### Introduction

For solving this project a DDPG Algorithim, with 4 neural networks (target and local networks for Actor and Critic, respectivelly), was utilized.

The Algorithim, based on [[1]](#1), can be interpreted as an approximate DQN for continuous action spaces [[2]](#2).

Similarly to DQN, the Critic part of DDPG utilizes Experience Replay to train a parametrized action value function <img src="https://render.githubusercontent.com/render/math?math=\hat{q}_{\pi}(s,a,\theta)">, in an off-policy manner (<img src="https://render.githubusercontent.com/render/math?math=\theta"> are the neural network weights).

Target and local networks, with weights <img src="https://render.githubusercontent.com/render/math?math=\theta_{frozen}"> and <img src="https://render.githubusercontent.com/render/math?math=\theta"> respectively, are also utilized by the Critic to avoid unstable learning ([[3]](#3), [[4]](#4)) when minimizing the loss function [[2]](#2):

<img src="https://render.githubusercontent.com/render/math?math=L(\theta) = \hat{E}_{(s,a,r',s')}[sum(r',  \gamma \hat{q}(s',a^*',\theta_{frozen})) - \hat{q}(s,a,\theta)]^2">,

where <img src="https://render.githubusercontent.com/render/math?math=\gamma"> is the discount factor, <img src="https://render.githubusercontent.com/render/math?math=a^*'"> the optimium action to take at state <img src="https://render.githubusercontent.com/render/math?math=s',"><img src="https://render.githubusercontent.com/render/math?math=\hat{E}"> is a sample-based estimate for the expectation, where batches of experience are sampled from the replay buffer and ' denotes a forward time-step.

Now, differently from DQN, DDPG utilizes a parameterized deterministic policy network to approximate the optimum continuous action <img src="https://render.githubusercontent.com/render/math?math=a^*"> for any given state:

<img src="https://render.githubusercontent.com/render/math?math=a^*' = \mu(s', \phi)">,

where <img src="https://render.githubusercontent.com/render/math?math=\phi"> are the network weights for the policy network.

Substitution of the deterministic policy into the loss function gives us the objective function to minimize (with respect to <img src="https://render.githubusercontent.com/render/math?math=\theta">) for the Critic part of DDPG:

<img src="https://render.githubusercontent.com/render/math?math=L(\theta) = \hat{E}_{(s,a,r',s')}[sum(r',  \gamma \hat{q}(s',\mu(s', \phi_{frozen}),\theta_{frozen})) - \hat{q}(s,a,\theta)]^2">,

where <img src="https://render.githubusercontent.com/render/math?math=\phi_{frozen}"> are the target weights for the parameterized deterministic policy network.

In order to train the Actor portion of DDPG we utilize the output of the deterministic policy network <img src="https://render.githubusercontent.com/render/math?math=\mu(s, \phi)"> as an input to our parametrized action value function <img src="https://render.githubusercontent.com/render/math?math=\hat{q}_{\pi}(s, \mu(s, \phi), \theta)">, so that our objective function is to maximize the expected value of the action value function with respect to <img src="https://render.githubusercontent.com/render/math?math=\phi">:

<img src="https://render.githubusercontent.com/render/math?math=J(\phi) = \hat{E}_{(s)}[\hat{q}_{\pi}(s, \mu(s, \phi), \theta)]">.

Maximization of the equation above with respect to <img src="https://render.githubusercontent.com/render/math?math=\phi"> is obtained algorithmically through the gradient of the loss function:

<img src="https://render.githubusercontent.com/render/math?math=\nabla_{\phi} J(\phi) = \hat{E}_{(s)}[\nabla_{\mu(s, \phi)}\hat{q}_{\pi}(s, \mu(s, \phi), \theta) \nabla_{\phi} \mu(s, \phi)]">.

In order to deal with the exploration-exploitation dillema for deterministic policies, Gaussian noise is introduced to the actions selected by the policy before performing the gradient function above:

<img src="https://render.githubusercontent.com/render/math?math=a_t = sum(\mu(s, \phi), G_t)">.

Finally, differently from DQN, DDPG applies a soft update to the target network weights every time-step for both the Actor and Critic networks, grealy increasing the stability of learning:

<img src="https://render.githubusercontent.com/render/math?math=\theta_{frozen} \leftarrow \tau \theta + (1-\tau) \theta">, and <img src="https://render.githubusercontent.com/render/math?math=\phi_{frozen} \leftarrow \tau \phi + (1-\tau) \phi">,

where <img src="https://render.githubusercontent.com/render/math?math=\tau"> is a hyperparameter << 1 that controls the target networks update speed.

The solution for this work is based on the environment with 20 independent arms. The experience for each arm is stored in the replay buffer during the interaction with the environment. The usage of multiple agents greatly increases the generalization capabilities of the algorithim.

## Algorithm

Detailed Algorithim pseudocode, edited from [[1]](#1)

**Algorithm 1: DDPG algorithm**
* Randomly initialize critic network <img src="https://render.githubusercontent.com/render/math?math=\hat{q}(s,a,\theta)"> and actor <img src="https://render.githubusercontent.com/render/math?math=\mu(s, \phi)"> with weights <img src="https://render.githubusercontent.com/render/math?math=\theta"> and <img src="https://render.githubusercontent.com/render/math?math=\phi">.
* Initialize target networks with weights <img src="https://render.githubusercontent.com/render/math?math=\theta_{frozen} \leftarrow \theta">, <img src="https://render.githubusercontent.com/render/math?math=\phi_{frozen} \leftarrow \phi"> 
* Initialize replay buffer **R**
* **For** episode = 1,M **do**
  * Initialize a random process <img src="https://render.githubusercontent.com/render/math?math=G"> (Gaussian Noise) for action exploration
  * Receive initial observation state <img src="https://render.githubusercontent.com/render/math?math=s_1">
  * **For** t = 1,T **do**
    * Select action <img src="https://render.githubusercontent.com/render/math?math=a_t = sum(\mu(s, \phi), G_t)">  according to the current policy and exploration noise
    * Execute action <img src="https://render.githubusercontent.com/render/math?math=a_t"> and observe reward <img src="https://render.githubusercontent.com/render/math?math=r'"> and new state <img src="https://render.githubusercontent.com/render/math?math=s'"> (' = t + 1)
    * Store transition <img src="https://render.githubusercontent.com/render/math?math=(s_t,a_t,r',s')"> in **R**
    * Sample a random minibatch of **T** transitions <img src="https://render.githubusercontent.com/render/math?math=(s_i,a_i,r',s')"> from **R**
    * Set <img src="https://render.githubusercontent.com/render/math?math=y_i=sum(r', \gamma q(s',\mu(s', \phi_{frozen}),\theta_{frozen})))">
    * Update critic weights by minimizing the loss <img src="https://render.githubusercontent.com/render/math?math=L(\theta) = \frac{1}{N}\sum_i [y_i - q(s,a,\theta)]^2">
    * Update the actor policy weights using the sampled policy gradient:
    * <img src="https://render.githubusercontent.com/render/math?math=\nabla_{\phi} J(\phi) = \frac{1}{N}\sum_i[\nabla_{\mu(s, \phi)}q(s, \mu(s, \phi), \theta) \nabla_{\phi} \mu(s, \phi)]">
    * Update the target networks:
    * <img src="https://render.githubusercontent.com/render/math?math=\theta_{frozen} \leftarrow \tau \theta + (1-\tau) \theta">
    * <img src="https://render.githubusercontent.com/render/math?math=\phi_{frozen} \leftarrow \tau \phi + (1-\tau) \phi">
    
## Hyperparameters and Neural Network Architecture

After a couple of attempts hyperparameter values that could reach the minimum of 30+ cumulative rewards in 100 episodes were obtained. These are:

Hyperparameter value  | Description
------------- | -------------
n_episodes=500  | maximum number of training episodes
max_t=1000  | maximum number of timesteps per episode
BUFFER_SIZE = int(1e6)   | replay buffer size
BATCH_SIZE = 256 | minibatch size
GAMMA = 0.99   | discount factor
TAU = 1e-3  | Value between 0 and 1 -> The closer to 1 the greater the target weights update will be (if TAU = 1, then <img src="https://render.githubusercontent.com/render/math?math=\theta_{frozen} = \theta">)
LR_ACTOR = 1e-4  | learning rate for updating Actor policy network weights
LR_CRITIC = 1e-4  | learning rate for updating Critic policy network weights

Neural Network Layers (local and target networks)  | Number of nodes 
------------- | -------------
Input Layer  | 33 Input States
1st Hidden Layer  | 128 (followed by ReLu Activation function)
2nd Hidden Layer  | 128 (followed by ReLu Activation function)
3rd Hidden Layer  | 64 (followed by ReLu Activation function)
Output Layer  | 4 Continuous Actions (torques applied to joints)


## Results

A plot for the mean return every 100 episodes is shown below. We see that our trained Agent is capable of surparsing the requirements of 30+ rewards after approximately 300 Episodes. The weights for the Actor and Critic policy networks are saved in checkpoint_actor.pth and checkpoint_critic.pth, respectively.

![Training Agents][image2]


## Future Ideas

Implement and compare current results with other deep reinforcement learning algorithims for continuous control, such as PPO, A2C and A3C.

## References
<a id="1">[1]</a> 
Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D. and Wierstra, D., 2015. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.

<a id="2">[2]</a> 
Miguel Morales, Grokkiing, Deep Reinforcement Learning

<a id="3">[3]</a> 
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. nature, 518(7540), pp.529-533.

<a id="4">[4]</a> 
https://github.com/TmoreiraBR/UnityMLAgents1stProject/blob/main/Report.md
