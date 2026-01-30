# Spring 2025, 535514 Reinforcement Learning
# HW2: DDPG

import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as Scheduler
import optuna

# Define a tensorboard writer
#writer = SummaryWriter("./tb_record_3/HalfCheetah_CDQ")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network
        self.layer1 = nn.Linear(num_inputs, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_outputs)

        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
       ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        x = self.layer1(inputs)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.tanh(x)  # the feasible action is [-1,1]
        return x
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network
        self.layer1 = nn.Linear(num_inputs+num_outputs, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)

        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network
        inputs = torch.cat((inputs, actions), dim = -1)
        x = self.layer1(inputs)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x       
        
        ########## END OF YOUR CODE ##########        
        

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma, tau,lr_a, lr_c,step_size, lr_a_decay, lr_c_decay, hidden_size=128):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)
        self.actor_scheduler = Scheduler.StepLR(self.actor_optim, step_size=step_size, gamma=lr_a_decay)

        self.critic1 = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic2 = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target1 = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target2 = Critic(hidden_size, self.num_inputs, self.action_space)

        self.critic_optim1 = Adam(self.critic1.parameters(), lr=lr_c)
        self.critic_optim2 = Adam(self.critic2.parameters(), lr=lr_c)
        self.critic_scheduler1 = Scheduler.StepLR(self.critic_optim1, step_size=step_size, gamma=lr_c_decay)
        self.critic_scheduler2 = Scheduler.StepLR(self.critic_optim2, step_size=step_size, gamma=lr_c_decay)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target1, self.critic1)
        hard_update(self.critic_target2, self.critic2)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed 
        noise = [0.0] if action_noise is None else action_noise.noise()
        noise = torch.FloatTensor(noise)
        action = mu + noise
        action = torch.clip(action, min=-1, max=1)  # feasible action space
        return action
        ########## END OF YOUR CODE ##########


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat([t.state for t in batch]))
        action_batch = Variable(torch.cat([t.action for t in batch]))
        reward_batch = Variable(torch.cat([t.reward for t in batch]))
        mask_batch = Variable(torch.cat([t.mask for t in batch]))
        next_state_batch = Variable(torch.cat([t.next_state for t in batch]))
        
        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic          
        with torch.no_grad():
            next_action_batch = self.actor_target(next_state_batch)
            # compute the two Q_wi'(s', pi(s')), use the min as the Q_w'(s', pi(s'))
            next_Q_batch1 = self.critic_target1(next_state_batch, next_action_batch)
            next_Q_batch2 = self.critic_target2(next_state_batch, next_action_batch)
            y_batch = reward_batch.view(-1,1) + self.gamma * torch.min(next_Q_batch1,next_Q_batch2)  * (1-mask_batch.view(-1,1))
            
        Q_batch1 = self.critic1(state_batch, action_batch)
        Q_batch2 = self.critic2(state_batch, action_batch)
        
        value_loss1 = F.mse_loss(y_batch, Q_batch1)
        value_loss2 = F.mse_loss(y_batch, Q_batch2)
        
        self.critic_optim1.zero_grad()
        value_loss1.backward(retain_graph=True)
        self.critic_optim1.step()

        self.critic_optim2.zero_grad()
        value_loss2.backward()
        self.critic_optim2.step()
        
        action_batch_ = self.actor(state_batch)
        Q_batch_ = self.critic1(state_batch, action_batch_)
        policy_loss = -Q_batch_.mean()
        
        self.actor_optim.zero_grad()   
        policy_loss.backward()
        self.actor_optim.step() 

        ########## END OF YOUR CODE ########## 

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target1, self.critic1, self.tau)
        soft_update(self.critic_target2, self.critic2, self.tau)

        return value_loss1.item(), value_loss2.item(), policy_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path1=None, critic_path2=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path1 is None:
            critic_path1 = "preTrained/ddpg_critic1_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path2 is None:
            critic_path2 = "preTrained/ddpg_critic2_{}_{}_{}".format(env_name, timestamp, suffix) 

        print('Saving models to {} , {} and {}'.format(actor_path, critic_path1, critic_path2))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic_path1)
        torch.save(self.critic2.state_dict(), critic_path2)

    def load_model(self, actor_path, critic_path1, critic_path2):
        print('Loading models from {} , {} and {}'.format(actor_path, critic_path1, critic_path2))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path1 is not None: 
            self.critic1.load_state_dict(torch.load(critic_path1))
        if critic_path2 is not None: 
            self.critic2.load_state_dict(torch.load(critic_path2))

def train(env_name, writer, gamma, lr_a, lr_c, noise_scale, step_size, lr_a_decay, lr_c_decay):    
    num_episodes = 500
    #gamma = 0.99
    tau = 0.005
    #lr_a=1e-3
    #lr_c=5e-3
    hidden_size = 128
    #noise_scale = 0.5
    replay_size = 100000
    batch_size = 512
    updates_per_step = 1
    print_freq = 1
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau,lr_a, lr_c,step_size, lr_a_decay, lr_c_decay, hidden_size)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)
    
    for i_episode in range(num_episodes):
        
        ounoise.scale = noise_scale
        ounoise.reset()
        
        state = torch.Tensor([env.reset()])

        episode_reward = 0
        value_losses1 = []
        value_losses2 = []
        policy_losses = []
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic
            total_numsteps+=1
            action = agent.select_action(state, ounoise)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            next_state = torch.Tensor([next_state])
            done = torch.Tensor([done])
            reward = torch.Tensor([reward])
            memory.push(state, action, done, next_state,reward)            
    
            
            if len(memory)>= batch_size and total_numsteps% updates_per_step ==0:                     
                batch = memory.sample(batch_size)
                value_loss1, value_loss2, policy_loss = agent.update_parameters(batch)
                value_losses1.append(value_loss1)
                value_losses2.append(value_loss2)
                policy_losses.append(policy_loss)
                
            episode_reward+=reward
            state = next_state
            if done:
                break    
            
            ########## END OF YOUR CODE ########## 
            # For wandb logging
            # wandb.log({"actor_loss": actor_loss, "critic_loss": critic_loss})

        rewards.append(episode_reward)
        value_losses1 = np.mean(value_losses1)
        value_losses2 = np.mean(value_losses2)
        policy_losses = np.mean(policy_losses)        
        
        t = 0
        if i_episode % print_freq == 0:
            state = torch.Tensor([env.reset()])
            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.numpy()[0])
                
                #env.render()
                
                episode_reward += reward

                next_state = torch.Tensor([next_state])

                state = next_state
                
                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)           
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))
            
            writer.add_scalar('Reward/ Episodic', episode_reward, i_episode)
            writer.add_scalar('Reward/ EWMA', ewma_reward, i_episode)
            writer.add_scalar('Loss/Policy Loss', policy_losses, i_episode)
            writer.add_scalar('Loss/Value Loss1', value_losses1, i_episode)
            writer.add_scalar('Loss/Value Loss2', value_losses2, i_episode)
        
        agent.actor_scheduler.step()
        agent.critic_scheduler1.step()
        agent.critic_scheduler2.step()
        
        if (i_episode==100) and (ewma_reward<0):
            break 
        
        if (i_episode==200) and (ewma_reward<1800):
            break     
              
            
    if ewma_reward > 5000:
        agent.save_model(f"{env_name}_{ewma_reward}", '.pth')
    return ewma_reward
       
 
 
def objective(trial):
    # Suggest hyperparameters
    lr_a = trial.suggest_loguniform('lr_a', 5e-5, 1e-2)
    lr_c = trial.suggest_loguniform('lr_c', 5e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.90, 0.999)
    noise_scale = trial.suggest_uniform('noise_scale', 0.1, 0.5)
    step_size = trial.suggest_int('step_size', 100, 300)
    lr_a_decay = trial.suggest_uniform('lr_a_decay', 0.5, 0.99)
    lr_c_decay = trial.suggest_uniform('lr_c_decay', 0.5, 0.99)

    # TensorBoard log path
    logdir = f"Tunning2/lr_a{lr_a:.1e}_lr_c{lr_c:.1e}_gamma{gamma:.3f}_noise{noise_scale:.2f}_stepsize{step_size}_lr_a_decay{lr_a_decay:.2f}_lr_c_decay{lr_c_decay:.2f}"
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)

    # Modify your train function to accept these arguments
    final_ewma = train(
        env_name="HalfCheetah_CDQ",
        writer=writer,
        gamma=gamma,
        lr_a=lr_a,
        lr_c=lr_c,
        noise_scale=noise_scale,
        step_size=step_size,
        lr_a_decay=lr_a_decay,
        lr_c_decay=lr_c_decay
    )

    writer.close()

    # Objective: maximize final EWMA reward
    return final_ewma



if __name__ == '__main__':
    # For reproducibility
    random_seed = 10
    torch.manual_seed(random_seed)
    env = gym.make('HalfCheetah-v2')
    env.seed(random_seed)

    # Create study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=50)

