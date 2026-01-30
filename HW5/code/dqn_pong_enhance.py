# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import optuna
from torch.optim.lr_scheduler import StepLR

# Set seed
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                    nn.ReLU(True),
                                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                    nn.ReLU(True),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                    nn.ReLU(True)
                                    )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_actions)
                                        )
        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.epsilon = 1e-9

    def add(self, transition, error=1):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if len(self.buffer)< self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = (abs(error) + self.epsilon)** self.alpha
        self.pos = (self.pos+1)% self.capacity                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.pos] 
            
        probs = priorities/priorities.sum()
        indices = np.random.choice(len(probs), batch_size, p = probs)
        samples = [self.buffer[idx] for idx in indices]
            
        weights = (len(self.buffer)* probs[indices])**(-self.beta)
        weights/= weights.max()
        return indices, samples, weights
        ########## END OF YOUR CODE (for Task 3) ##########
        
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for idx, err in zip(indices, errors):
            self.priorities[idx] = (abs(err) + 1e-6) ** self.alpha   
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    
class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(num_actions=self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        #self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=args.lr, alpha=0.95, eps=1e-2)
        
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay_steps = args.epsilon_decay_steps

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.memory = PrioritizedReplayBuffer(capacity=args.memory_size)

        self.n_steps = args.n_steps
        self.nstep_buffer = deque(maxlen=self.n_steps) 
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes= 1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0
            
            if step_count == 0:
                self.nstep_buffer.clear()

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocessor.step(next_obs)
                reward = np.clip(reward, -1, 1) # stable                

                self.nstep_buffer.append( (state, action, reward, next_state, done))
                
                if len(self.nstep_buffer) == self.n_steps:
                    R = 0.0
                    s_n = next_state
                    done_n = done
                    for k, (_, _, r, s_next, d) in enumerate(reversed(self.nstep_buffer)):
                        R = r + (self.gamma * R * (1 - d))
                        if d: 
                            s_n = s_next
                            done_n = True
                            break
                    s_0, a_0, _, _, _ = self.nstep_buffer[0]
                    
                    if len(self.memory.buffer):
                        priority = self.memory.priorities[: len(self.memory.buffer)].max()
                        self.memory.add((s_0, a_0, R, s_n, done_n), error=priority)
                    else:
                        self.memory.add((s_0, a_0, R, s_n, done_n), error = 1)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                
                if self.env_count % 200000 ==0:
                    model_path = os.path.join(self.save_dir, f"env_count_model{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved env_count {self.env_count} model to {model_path}")

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward >= self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })
                
            if self.train_count >=1e6:
                break
                

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):
        if len(self.memory.buffer) < self.replay_start_size:
            return 
        
        #Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.train_count += 1

       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer 
        indices, batch, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)   
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states), dim = 1) 
            q_nexts = self.target_net(next_states.squeeze(-1)).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_targets = rewards+ (self.gamma ** self.n_steps) * q_nexts*(1-dones)
        
        
        BM_errors = (q_values - q_targets.squeeze()).detach().cpu().numpy()
        self.memory.update_priorities(indices, BM_errors)
                
 
        loss = (weights * BM_errors.pow(2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        wandb.log({
            "Train Loss": loss.item(),
            "Q Mean": q_values.mean().item(),
            "Q Std": q_values.std().item()
        })        
      
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
           print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="pong-run")
    parser.add_argument("--wandb-project-name", type=str, default="DLP-Lab5-DQN-Pong") 
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99999)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=250000)
    parser.add_argument("--target-update-frequency", type=int, default=5000)
    parser.add_argument("--replay-start-size", type=int, default=30000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=3)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=4)
    args = parser.parse_args()
    
    
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run(args.num_epochs)
    
