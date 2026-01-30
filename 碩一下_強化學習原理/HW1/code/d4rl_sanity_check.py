#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 535514 Reinforcement Learning, HW1

The sanity check suggested by D4RL official repo
Source: https://github.com/Farama-Foundation/D4RL
"""

import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import numpy as np

# Create the environment
#env = gym.make('maze2d-umaze-v1')
env = gym.make('halfcheetah-medium-v2')
# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos

dataset = env.get_dataset()
print(f"Observations shape: {dataset['observations'].shape}")
print(dataset['observations']) # An N x dim_observation Numpy array of observations
print(f"Min: {np.min(dataset['observations'], axis=0)}")
print(f"Max: {np.max(dataset['observations'], axis=0)}")
print(f"Mean: {np.mean(dataset['observations'], axis=0)}")

print(f"Actions shape: {dataset['actions'].shape}")
print(dataset['actions'])
print(f"Min: {np.min(dataset['actions'])}")
print(f"Max: {np.max(dataset['actions'])}")
print(f"Mean: {np.mean(dataset['actions'])}")

print(f"Rewards shape: {dataset['rewards'].shape}")
print(dataset['rewards'])
print(f"Min: {np.min(dataset['rewards'])}")
print(f"Max: {np.max(dataset['rewards'])}")
print(f"Mean: {np.mean(dataset['rewards'])}")

print(f"Terminals shape: {dataset['terminals'].shape}")
print(dataset['terminals'])






# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)