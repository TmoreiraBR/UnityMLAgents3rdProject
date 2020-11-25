<<<<<<< HEAD
[//]: # (Image References)

[image1]: https://github.com/TmoreiraBR/UnityMLAgents3rdProject/blob/main/TrainedResults.jpg  "Training Agents"
[image2]: https://github.com/TmoreiraBR/UnityMLAgents3rdProject/blob/main/MADDPG.PNG  "ImageArticle"


# Report for Project 3: Collaboration and Competition

### Introduction

For solving this project a multi-agent deep deterministic policy gradient (MADDPG) Algorithim, was utilized.

The Algorithim, based on [[1]](#1), works in a very similar fashion to DDPG [[2]](#2), with the main differences being:

* Each individual Agent (in a competitive or cooperative setting) contains its own Actor (policy) and Critic (state-action value function estimate) Networks.

* The Critic of each Agent is augmented in order to have a state (or state-action) representation of the whole environment (including states and actions from other Agents).

## DDPG Recap (see https://github.com/TmoreiraBR/UnityMLAgents2ndProject-MultiAgent/blob/main/Report.md):

DDPG can be interpreted as an approximate DQN for continuous action spaces [[3]](#3).

Similarly to DQN, the Critic part of DDPG utilizes Experience Replay to train a parametrized action value function <img src="https://render.githubusercontent.com/render/math?math=\hat{q}_{\pi}(s,a,\theta)">, in an off-policy manner (<img src="https://render.githubusercontent.com/render/math?math=\theta"> are the neural network weights).

Target and local networks, with weights <img src="https://render.githubusercontent.com/render/math?math=\theta_{frozen}"> and <img src="https://render.githubusercontent.com/render/math?math=\theta"> respectively, are also utilized by the Critic to avoid unstable learning ([[4]](#4), [[5]](#5)) when minimizing the loss function [[3]](#3):

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

## MADDPG

In a multi-agent setting, where each Agent has its own Actor and Critic networks (as in DDPG), if the state representation for each Agent is only local (i.e. each Agent is not aware of states and actions of other agents) the environment can appear Non-Stationary, which violates the Markov Assumption for convergence [[1]](#1). 

In other to solve this issue, MADDPG proposes an augmented Critic for each Agent "i", named centralized action-value function, that receives observations and actions for all agents in the environment:

<img src="https://render.githubusercontent.com/render/math?math=q_i^{\pi} (\vec x, \vec a)">,

where <img src="https://render.githubusercontent.com/render/math?math=\vec x"> is a vector with all observation from all N Agents <img src="https://render.githubusercontent.com/render/math?math=\vec x = [o_1, o_2, ..., o_N]"> and <img src="https://render.githubusercontent.com/render/math?math=\vec a"> is a vector with all actions performed by all N Agents <img src="https://render.githubusercontent.com/render/math?math=\vec a = [a_1, a_2, ..., a_N]">.

The Actor for each Agent remains unaltered, where each Agent's action only depends on the Agent's local observations. Visually, the MADDPG framework can be seen as:

![ImageArticle][image2]
Image taken from [[1]](#1).

Modification of the Critic´s action-value function estimate, updates the loss function from DDPG (to be minimized for each Agent "i") into:

<img src="https://render.githubusercontent.com/render/math?math=L(\theta) = \hat{E}_{(\vec x,\vec a,r',\vec x')}[sum(r',  \gamma \hat{q_i}(\vec x', \vec{\mu'}(\vec{\phi}_{frozen}),\theta_{frozen})) - \hat{q}(\vec x, \vec{\mu}(\vec{\phi}),\theta)]^2">, where

<img src="https://render.githubusercontent.com/render/math?math=\vec{\mu}(\vec{\phi}) = [\mu_1(o_1, \phi_1), \mu_2(o_2, \phi_2), ...,  \mu_N(o_N, \phi_N),]"> is the vector of deterministic policies from Agents 1 to N.

Also as a consequence of modifying the action-value function, the gradient of the loss function for the Actor of each Agent "i" changes into:
 
<img src="https://render.githubusercontent.com/render/math?math=\nabla_{\phi_i} J(\mu_i) = \hat{E}_{(\vec x, \vec{\mu})}[\nabla_{\mu_i}\hat{q_i}(\vec x, \vec{\mu}(\vec{\phi}), \theta) \nabla_{\phi_i} \mu_i(o_i, \phi_i)]">.

## Algorithm

Detailed Algorithim pseudocode, edited from [[1]](#1)

**Algorithm 1: MADDPG algorithm**

**For** episode = 1,M **do**
* Initialize a random process <img src="https://render.githubusercontent.com/render/math?math=R"> for action exploration
* Receive initial state <img src="https://render.githubusercontent.com/render/math?math=\vec x">
* **For** t = 1 to max-episode-length **do**
  * for each agent i, select action <img src="https://render.githubusercontent.com/render/math?math=a_i = sum(\mu_i(o_i, \phi_i), R_t)"> w.r.t. the current policy and exploration
  * Execute actions <img src="https://render.githubusercontent.com/render/math?math=\vec a = [a_1, a_2, ..., a_N]"> and observe reward <img src="https://render.githubusercontent.com/render/math?math=r"> and new state <img src="https://render.githubusercontent.com/render/math?math=\vec x'">
  * Store <img src="https://render.githubusercontent.com/render/math?math=(\vec x, \vec a, \vec r, \vec x')"> in a replay buffer D
  * <img src="https://render.githubusercontent.com/render/math?math=\vec x \leftarrow \vec x'">
  * **For** Agent i=1 to N **do**
    * Sample a random minibatch of **S** samples <img src="https://render.githubusercontent.com/render/math?math=(\vec x^j, \vec a^j, \vec r^j, \vec x'^j)"> from D
    * Set <img src="https://render.githubusercontent.com/render/math?math=y^j = sum(r_i^j, \gamma \hat{q_i}(\vec x', \vec a',\theta_{frozen}))">, where <img src="https://render.githubusercontent.com/render/math?math=\vec a' = \vec{\mu'}(\vec{\phi}_{frozen})">
    * Update Critic by minimizing the loss <img src="https://render.githubusercontent.com/render/math?math=L(\theta_i) = \frac{1}{S} \sum_j [y^j - \hat{q}(\vec x^j, \vec{\mu^j}(\vec{\phi}),\theta)]^2">
    * Update Actor using the sampled policy gradient: <img src="https://render.githubusercontent.com/render/math?math=\nabla_{\phi_i} J(\mu_i) = \frac{1}{S} \sum_j \nabla_{\mu_i}\hat{q_i}(\vec x^j, \vec{\mu^j}(\vec{\phi}), \theta) \nabla_{\phi_i} \mu_i(o_i^j, \phi_i)">
  * Update target network parameters for each Agent i: 
  * <img src="https://render.githubusercontent.com/render/math?math=\theta' \leftarrow sum(\tau \theta_i, (1 - \tau) \theta_i')">
  * <img src="https://render.githubusercontent.com/render/math?math=\phi' \leftarrow sum(\tau \phi_i, (1 - \tau) \phi_i')">
    
## Hyperparameters and Neural Network Architecture

After a couple of attempts, hyperparameter values that could reach the minimum of average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents) were obtained. These are:

Hyperparameter value  | Description
------------- | -------------
n_episodes=4000  | maximum number of training episodes
BUFFER_SIZE = int(1e6)   | replay buffer size
BATCH_SIZE = 256 | minibatch size
GAMMA = 0.99   | discount factor
TAU = 1e-3  | Value between 0 and 1 -> The closer to 1 the greater the target weights update will be (if TAU = 1, then <img src="https://render.githubusercontent.com/render/math?math=\theta_{frozen} = \theta">)
LR_ACTOR = 1e-4  | learning rate for updating Actor policy network weights
LR_CRITIC = 1e-4  | learning rate for updating Critic policy network weights
theta = .15  | Ornstein-Uhlenbeck
sigma = 0.4  | Ornstein-Uhlenbeck

Neural Network Layers for Actor (local and target networks)  | Number of nodes 
------------- | -------------
Input Layer  | 24 Input States
1st Hidden Layer  | 256 (followed by ReLu Activation function)
2nd Hidden Layer  | 128 (followed by ReLu Activation function)
3rd Hidden Layer  | 64 (followed by ReLu Activation function)
Output Layer  | 2 Continuous Actions (x-y racket movement)

Neural Network Layers for Critic (local and target networks)  | Number of nodes 
------------- | -------------
Input Layer  | 48 Input States (All states for both Agents)
1st Hidden Layer  | 256 (followed by ReLu Activation function)
2nd Hidden Layer  | 128 (followed by ReLu Activation function)
3rd Hidden Layer  | 64 (followed by ReLu Activation function)
Output Layer  | 1 (action-value estimate)


## Results

A plot for the mean return every 100 episodes is shown below. We see that our trained Agent is capable of surparsing the requirements of .5 score after approximately 2700 Episodes. The weights for each Agent's Actor and Critic policy networks are saved in agent1_checkpoint_actor.pth, agent1_checkpoint_critic.pth and agent2_checkpoint_actor.pth, agent2_checkpoint_critic.pth, respectivelly.

![Training Agents][image1]


## Future Ideas

Great improvement of results were obtained when fine-tunning the noise function. Investigation of different Noise functions, with noise decay for instance, could prove very beneficial for speeding up training. This could also be an indicative that the exploration-exploitation dillema still has room from improvement in MADDPG.

## References
<a id="1">[1]</a> 
Lowe, R., Wu, Y.I., Tamar, A., Harb, J., Pieter Abbeel, O. and Mordatch, I., 2017. Multi-agent actor-critic for mixed cooperative-competitive environments. Advances in neural information processing systems, 30, pp.6379-6390.

<a id="2">[2]</a> 
Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D. and Wierstra, D., 2015. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.

<a id="3">[3]</a> 
Miguel Morales, Grokkiing, Deep Reinforcement Learning

<a id="4">[4]</a> 
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. nature, 518(7540), pp.529-533.

<a id="5">[5]</a> 
https://github.com/TmoreiraBR/UnityMLAgents1stProject/blob/main/Report.md
