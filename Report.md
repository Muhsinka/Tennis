## Introduction

The project aims to provide a deeper understanding of Deep Deterministic policy Gradient(DDPG) that consist of actor-critic methods. The goal of this project to teach two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. The enviroment ends when the avarage rewards of both agents reach 0.50.

The DDPG pseudocode used to solve the enviroment is shown below.

![ddpg_algorithm](https://user-images.githubusercontent.com/43606874/52708863-43c68c80-2f9c-11e9-9001-20c619bd057d.png)

## Neural network

The DDPG Algorithm has 2 neural networks actor and critic network. The critic network calculates the state action pairs and the actor network calculates the policy state of the agent. Both of the actor-crtitic networks consists of the same neural network that can be found in the Udacity Deep RL repository DDPG Algorithm.

The unimproved neural network code consist of same layers can be found in the Udacity Deep RL repository 
includes the following:

- Fully connected layer - input: 33 (state size) | output: 128
- ReLU layer - activation function
- Batch normalization
- Fully connected layer - input: 128 |  output 128
- ReLU layer - activation function
- Fully connected layer - input: 128 | output: (action size)
- Output activation layer - tanh function

Hyperparameters used in the DDPG algorithm:

- BUFFER_SIZE = int(1e6)
- BATCH_SIZE = 128 
- GAMMA = 0.99
- TAU = 1e-3
- LR_ACTOR = 5e-5
- LR_CRITIC = 6e-5 
- WEIGHT_DECAY = 0.0001


## Improvements

First attemps with unimporeved code shown no signs of learning, the agent always got around 0-0.3 points in average.

Hyperparameters

- BUFFER_SIZE = int(1e5)
- BATCH_SIZE = 128 
- GAMMA = 0.99
- TAU = 1e-3
- LR_ACTOR = 1e-4 
- LR_CRITIC = 1e-3 
- WEIGHT_DECAY = 0

The Network 

Additional 2 batch normalization layers implemented to both actor-critic networks.
```
    def __init__(self, state_size, action_size, seed=0, fc1_units=128, fc2_units=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()
```

```
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
```

## Results

After the implementations the agent was able to solve the task in 197 episodes
```
Episode 50	Average Score: 0.00	Score: -0.005
Episode 100	Average Score: 0.00	Score: -0.005
Episode 150	Average Score: 0.00	Score: -0.005
Episode 200	Average Score: 0.00	Score: -0.005
Episode 250	Average Score: 0.00	Score: -0.005
Episode 300	Average Score: 0.00	Score: -0.005
Episode 350	Average Score: 0.00	Score: -0.005
Episode 400	Average Score: 0.00	Score: -0.005
Episode 450	Average Score: 0.00	Score: -0.005
Episode 500	Average Score: 0.00	Score: -0.005
Episode 550	Average Score: 0.00	Score: -0.005
Episode 600	Average Score: 0.00	Score: -0.005
Episode 650	Average Score: 0.02	Score: -0.005
Episode 700	Average Score: 0.02	Score: -0.005
Episode 750	Average Score: 0.02	Score: 0.0455
Episode 800	Average Score: 0.03	Score: 0.0455
Episode 850	Average Score: 0.04	Score: -0.005
Episode 900	Average Score: 0.04	Score: -0.005
Episode 950	Average Score: 0.06	Score: 0.0455
Episode 1000	Average Score: 0.07	Score: 0.045
Episode 1050	Average Score: 0.09	Score: -0.005
Episode 1100	Average Score: 0.11	Score: 0.0455
Episode 1150	Average Score: 0.12	Score: 0.1955
Episode 1200	Average Score: 0.13	Score: 0.0455
Episode 1250	Average Score: 0.11	Score: 0.0955
Episode 1300	Average Score: 0.13	Score: 0.195
Episode 1350	Average Score: 0.14	Score: 0.0455
Episode 1400	Average Score: 0.18	Score: 0.0455
Episode 1450	Average Score: 0.25	Score: -0.005
Episode 1500	Average Score: 0.39	Score: 0.6955
Episode 1522	Average Score: 0.50	Score: 0.8455

Environment solved in 1522 episodes!
```

![download](https://user-images.githubusercontent.com/43606874/53333153-2089ce80-3906-11e9-8bd1-19fc2b64b1ed.png)

## Future Improvements

The batch normalization showed significant improvemtns to the agent needs further learning.

Trying different learning rates and batch size.

Adding dropout layer can improve the model

Adding prioritized experience replay

D4PG algorithm in the Google Deep Mind's paper can be implemented.

TRPO algorithm in the Trust Region Policy Optimization paper can be inplemented.

Try a Competition instead of Collaboration.

## Reference
[DDPG paper](https://arxiv.org/pdf/1509.02971.pdf).

[Google DeepMind´s paper](https://openreview.net/pdf?id=SyZipzbCb).

[Trust Region Policy Optimization paper](https://arxiv.org/pdf/1502.05477.pdf)..
