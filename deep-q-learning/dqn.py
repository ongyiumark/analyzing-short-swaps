# IMPORT LIBRARIES
import random
import itertools
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from shortswaps import ShortSwap

# HYPERPARAMETERS
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
LEARNING_RATE = 0.00001
PERM_N = 4
NUM_HIDDEN_NODES = 100

# NEURAL NETWORK
class Network(nn.Module):
  def __init__(self, env):
    super().__init__()

    in_features = int(np.prod(env.observation_space_shape))

    self.net = nn.Sequential(
      nn.Linear(in_features, NUM_HIDDEN_NODES),
      nn.Tanh(),
      nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
      nn.Tanh(),
      nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
      nn.Tanh(),
      nn.Linear(NUM_HIDDEN_NODES, env.action_space_size)
    )
  
  def forward(self, x):
    return self.net(x)
  
  def act(self, obs):
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    q_values = self(obs_t.unsqueeze(0))

    max_q_index = torch.argmax(q_values, dim=1)[0]
    action = max_q_index.detach().item()
    
    return action

# DEEP Q-LEARNING
env = ShortSwap(PERM_N)

replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)
target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), LEARNING_RATE)

# Initialize Replay Buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
  action = env.action_space_sample()
  
  new_obs, reward, done = env.step(action)
  transition = (obs, action, reward, done, new_obs)
  replay_buffer.append(transition)
  obs = new_obs

  if done:
    obs = env.reset()

# Main Training Loop
obs = env.reset()
for step in itertools.count():
  epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

  rnd_sample = random.random()
  if rnd_sample <= epsilon:
    action = env.action_space_sample()
  else:
    action = online_net.act(obs)

  new_obs, reward, done = env.step(action)
  transition = (obs, action, reward, done, new_obs)
  replay_buffer.append(transition)
  obs = new_obs
  episode_reward += reward

  if done:
    obs = env.reset()
    reward_buffer.append(episode_reward)
    episode_reward = 0.0

  # Watch Play After Solve
  if len(reward_buffer) >= 100 and np.mean(reward_buffer) >= 98:
    while True:
      action = online_net.act(obs)
      obs, _, done = env.step(action)
      env.render()
      if done:
        env.reset()

  # Gradient Step
  transitions = random.sample(replay_buffer, BATCH_SIZE)
  obses, actions, rewards, dones, new_obses = map(np.asarray, zip(*transitions))

  obses_t = torch.as_tensor(obses, dtype=torch.float32)
  actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
  rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
  dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
  new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

  # Compute Targets
  target_q_values = target_net(new_obses_t)
  max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
  
  targets = rewards_t + GAMMA * (1-dones_t) * max_target_q_values

  # Compute Loss
  q_values = online_net(obses_t)
  action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

  loss = F.smooth_l1_loss(action_q_values, targets)

  # Gradient Descent
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # Update Target Network
  if step % TARGET_UPDATE_FREQ == 0:
    target_net.load_state_dict(online_net.state_dict())

  # Logging
  if step % 1000 == 0:
    print()
    print(f"Step: {step}, avg Reward: {np.mean(reward_buffer)}")
