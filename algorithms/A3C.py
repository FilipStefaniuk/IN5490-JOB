import numpy as np 
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
import time

from sys import stdout
import copy
import random
from tqdm import tqdm

import threading


ACTION_DISCRETE = 0
ACTION_CONTINUOUS = 1



class Actor(torch.nn.Module):
    def __init__(self, num_features, num_actions, action_type, distribution):
        super(Actor, self).__init__()
        self.action_type = action_type
        self.distribution = distribution
        self.linear = torch.nn.Linear(num_features, 64, bias=True)
        self.linear2 = torch.nn.Linear(64, 64, bias=True)

        if action_type == ACTION_DISCRETE:
            self.linear3 = torch.nn.Linear(64, num_actions, bias=True)
        else:
            self.linear3 = torch.nn.Linear(64, 64, bias=True)
            self.linear_param1 = torch.nn.Linear(64, num_actions, bias=True)
            self.linear_param2 = torch.nn.Linear(64, num_actions, bias=True)
            
        self.num_actions = num_actions


    def forward(self, x):
        x = self.linear( x )
        x = F.tanh(x)
        
        if self.action_type == ACTION_DISCRETE:
            x = self.linear2(x)
            x = F.tanh(x)
            x = self.linear3(x)
            x = x - torch.max( x )
            x = F.softmax( x )
            return x
        else:
            x1 = self.linear2(x)
            x1 = F.tanh(x1)
            x2 = self.linear3(x)
            x2 = F.tanh(x2)

            x1 = self.linear_param1( x1 )
            x2 = self.linear_param2( x2 )
            if self.distribution == "normal":
                return torch.sigmoid(x1), torch.sigmoid( x2 ) / 10.0
            
            elif self.distribution == "beta":
                return F.softplus( x1 ), F.softplus( x2 )
            else:
                assert(False)


    def get_distribution(self, params1, params2):
        if self.distribution == "normal":
            dist = Normal( params1, params2)
        elif self.distribution == "beta":
            dist = Beta(params1, params2)

        return dist

    def select_action(self, phi):

        x = torch.from_numpy(phi).float()

        # print(softmax)
        if self.action_type == ACTION_CONTINUOUS:
            param1, param2 = self.forward( x )
            param1 = param1.view(1,-1)
            param2 = param2.view(1,-1)

            dist = self.get_distribution(param1, param2)
            action = dist.sample()[0]
            action = action.detach().numpy()
        elif self.action_type == ACTION_DISCRETE:
            action = self.forward( x )

            action = action.detach().numpy()
            action = np.random.choice(range(action.shape[0]), p=action)
        else:
            assert(False)
            
        return action



class Critic(torch.nn.Module):
    def __init__(self, num_features):
        super(Critic, self).__init__()
        self.linear = torch.nn.Linear(num_features, 64, bias=True)
        self.linear2 = torch.nn.Linear(64, 64, bias=True)
        self.linear3 = torch.nn.Linear(64, 1, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)
        x = self.linear3(x)

        return x

def get_phi(env, state):
        st = state.reshape( (state.shape[0],) )
        return st
        st = (st - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) 
        return st

        return phi


class AdvAgent():

    GPU = False
    
    def __init__(self, env, actor, critic, gamma, t_length, action_type, distribution):
        self.env = env
        print(env.action_space.__dict__)
        print(env.observation_space.__dict__)

        self.s = env.reset()
        phi = get_phi(self.env, self.s.reshape(-1,1))
        self.actor = actor
        self.critic = critic
        self.action_type = action_type
        self.distribution = distribution
        self.gamma = gamma

        self.episode_rewards = []
        self.episode_steps = []
        self.buffer = []

        self.clear_history()
        self.buffer_size = 16
        self.t_length = t_length

        self.pause_collecting = False
        self.terminate_collecting = False

        self.done = True
        self.episodes = 0


    def clear_history(self):
        self.hist_s, self.hist_sn, self.hist_a, self.hist_r, self.hist_term = None, None, None, None, None


    def reset(self):
        self.s = self.env.reset()
        self.clear_history()
    
    def calculate_losses(self):
        s, sn, r, a, term = self.buffer[-1]
        P = self.actor.forward( s )
        V = self.critic.forward( s )

        R = 0.0 if term[-1] == 0.0 else V[-1]
        actor_loss, critic_loss = 0.0, 0.0
        for t in range(s.shape[0]-2, -1, -1):
            R = r[t] + self.gamma * R

            if self.action_type == ACTION_DISCRETE:
                
                actor_loss += torch.log( P[t,a[t]] ) * (R - V[t])            
                critic_loss += (R - V[t])**2

            elif self.action_type == ACTION_CONTINUOUS:
                p1, p2 = P

                dist = self.actor.get_distribution(p1, p2)

                actor_loss += torch.sum(dist.log_prob( a ) * (R - V[t]) )              
                critic_loss += (R - V[t])**2
        
        return actor_loss, critic_loss


    def add_hist_to_buffer(self):
        x = (self.hist_s.copy(), self.hist_sn.copy(), self.hist_r.copy(), self.hist_a.copy(), self.hist_term.copy())        
        x = (torch.from_numpy(x[0]).float(), torch.from_numpy(x[1]).float(), torch.from_numpy(x[2]).float(),
            torch.from_numpy(x[3]), torch.from_numpy(x[4]).float())

        if self.GPU:
            x = (x[0].cuda(), x[1].cuda(), x[2].cuda(), x[3].cuda(), x[4].cuda())

        self.buffer.append( x )


    def add_to_history(self, tup):
        s, sn, r, a, term = tup
        if self.hist_s is None:
            self.hist_s = s.reshape(1,-1)
            self.hist_sn = sn.reshape(1,-1)
            self.hist_a = a
            self.hist_r = r
            self.hist_term = term
        else:
            self.hist_s = np.vstack( (self.hist_s, s))
            self.hist_sn = np.vstack( (self.hist_sn, sn))
            self.hist_a = np.vstack( (self.hist_a, a))
            self.hist_r = np.vstack( (self.hist_r, r))
            self.hist_term = np.vstack( (self.hist_term, term))


    def _step(self, action):
        if self.action_type == ACTION_CONTINUOUS:
            if self.distribution == 'beta':
                a = (action * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low)
            elif self.distribution == 'normal':
                # a = action
                a = np.clip(action, np.zeros(action.shape), np.ones(action.shape))
                a = (a * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low)
        else:
            a = action
        
        state_next, reward, done, _ = self.env.step( a )

        return state_next, reward, done, _


    def collect_samples(self, episodes, max_steps):
        self.buffer = []
        self.total_reward = 0.0
        self.total_step = 0
        while self.terminate_collecting == False:
            if self.done == True:
                self.s = self.env.reset()
                done = False
                self.episodes += 1
                self.episode_rewards.append( self.total_reward )
                self.episode_steps.append( self.total_step )
                self.total_reward = 0.0
                self.total_step = 0


            s = get_phi(self.env, self.s.reshape(-1,1)).T
            
            action = self.actor.select_action(s)

            state_next, reward, done, _ = self._step(action)

            self.total_reward += reward
            self.total_step += 1
            
            sn = get_phi(self.env, state_next.reshape(-1,1)).T
            r = np.array([[reward]])
            a = np.array([[action]]) if self.action_type == ACTION_DISCRETE else np.array([action])
            nonterm = np.array([[0]]) if done else np.array([[1]])

            self.add_to_history( (s, sn, r, a, nonterm) )

            self.done = done

            if self.hist_s.shape[0] == self.t_length or done:
                self.add_hist_to_buffer()
                self.actor_loss, self.critic_loss = self.calculate_losses()
                self.pause_collecting = True 

            while self.pause_collecting:
                time.sleep(0.1)

            self.s = state_next


class A3C():

    GPU = False
    
    
    def __init__(self, envs, alpha, beta, gamma, epochs, t_length, action_type=ACTION_DISCRETE, distribution="normal", verbose=True):
        # super(ActorCriticPytorch, self).__init__()
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma

        self.action_type = action_type
        self.num_actions = envs[0].action_space.n if action_type == ACTION_DISCRETE else envs[0].action_space.low.shape[0]
        self.distribution = distribution

        if action_type == ACTION_CONTINUOUS:
            self.torch_high = torch.from_numpy(envs[0].action_space.high)
            self.torch_low = torch.from_numpy(envs[0].action_space.low)
        
        self.num_features = get_phi(envs[0], envs[0].reset()).shape[0]
        self.epochs = epochs

        self.critic = Critic(self.num_features).cuda() if self.GPU == True else Critic(self.num_features)
            
        self.actor = Actor(self.num_features, self.num_actions, action_type, distribution)
        

        # for param in self.actor_update.parameters():
        #    param.data /= 10.0
        
        # for param in self.actor_old.parameters():
        #    param.data /= 10.0

        # for param in self.critic.parameters():
        #    param.data /= 10.0


        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.beta)

        self.agents = [AdvAgent(env, self.actor, self.critic, self.gamma, t_length, action_type, distribution) for env in envs[:-1]]
        self.test_agent = AdvAgent(envs[-1], self.actor, self.critic, self.gamma, t_length, action_type, distribution)


    def set_envs(self, envs):
        for i, agent in enumerate(self.agents):
            agent.env = envs[i]
        self.test_agent.env = envs[-1]


    def train(self, max_steps, episodes, render=False, verbose=False, save_path=None):
        self.train_threads = []
        for agent in self.agents:
            thread = threading.Thread(target=agent.collect_samples, args=(episodes, max_steps))
            self.train_threads.append( thread )
        
        
        for thread in self.train_threads:
            thread.start()

        done = False
        updates = 0
        current_episodes = 0
        agents_on_hold = []
        while not done:
            for k, agent in enumerate(self.agents):
                if agent.pause_collecting and agent not in agents_on_hold:
                    agents_on_hold.append( agent )


            if len(agents_on_hold) == len(self.agents):
                agents_on_hold = []

                start = time.time()
                for _ in range(self.epochs):
                    self.update( verbose )
                end = time.time()

                for agent in self.agents:
                    agent.clear_history()
                    agent.buffer = []
                    agent.actor_loss, agent.critic_loss = None, None
                    agent.pause_collecting = False

                updates += 1

                if current_episodes != self.agents[0].episodes:
                    current_episodes = self.agents[0].episodes
                    rewards = [a.episode_rewards[-1] for a in self.agents]
                    steps = [a.episode_steps[-1] for a in self.agents]

                    print("Reward: " + str(round(np.mean(rewards), 2)) + " Steps: " 
                            + str(round(np.mean(steps),2)) + " Update: " + str(updates) 
                            + " Time: " + str(round(end-start,2)) + " Episodes: " + str(current_episodes))

                if save_path is not None:
                    self.save_model(save_path, current_episodes)

                if current_episodes % 50 == 0:
                    self.run(max_steps=1000, render=True)
                
                done = (current_episodes >= episodes)
                
        rewards, steps = [], []
        for k in range(len(self.agents[0].episode_rewards)):
            r, s = [], []
            for x in self.agents:
                if k < len(x.episode_rewards):
                    r.append( x.episode_rewards[k] )
                    s.append( x.episode_steps[k] )
            rewards.append( np.mean(r) )
            steps.append( np.mean(s) )

        for agent in self.agents:
            agent.pause_collecting = False
            agent.terminate_collecting = True     
        
        for thread in self.train_threads:
            thread.join()
        
        return rewards, steps         


    def run(self, max_steps, render):
        total_reward = 0.0
        self.test_agent.reset()
        for step in range(max_steps):

            action = self.test_agent.actor.select_action(self.test_agent.s)
            
            state_next, reward, done, _ = self.test_agent._step( action )

            if render == True:
                self.test_agent.env.render( )

            total_reward += reward

            self.test_agent.s = state_next
            if done:
                break
        return total_reward, step

    
    def update(self, verbose):
        
        actor_losses, critic_losses = 0.0, 0.0
        for agent in self.agents:
            # al, cl = agent.calculate_losses()
            actor_losses += agent.actor_loss
            critic_losses += agent.critic_loss

        actor_losses = -torch.mean(actor_losses)
        critic_losses = torch.mean(critic_losses)
        
        self.optimizer_actor.zero_grad()
        actor_losses.backward(retain_graph=True)
        
        self.optimizer_critic.zero_grad()
        critic_losses.backward(retain_graph=True)
        self.optimizer_actor.step() 
        self.optimizer_critic.step()

            

    def save_model(self, path, i):
        torch.save(self.actor, path+"actor_%f_%f_%f_%d.pt" %(self.alpha, self.beta, self.gamma, i))
        critic = self.critic.cpu() if self.GPU == True else self.critic
        
        torch.save(critic, path+"critic_%f_%f_%f_%d.pt" %(self.alpha, self.beta, self.gamma, i))

    
    def load_model(self, actor, critic):
        self.actor.load_state_dict(torch.load(actor))
        self.critic.load_state_dict(torch.load(critic))



if __name__ == "__main__":
    import gym
    # import roboschool
    import matplotlib.pyplot as plt

    num_agents = 6

    # env = [gym.make("CartPole-v0") for _ in range(num_agents+1)]
    # env = [gym.make("RoboschoolInvertedPendulum-v1") for _ in range(num_agents+1)]
    # env = [gym.make("LunarLander-v2") for _ in range(num_agents+1)]
    env = [gym.make("MountainCarContinuous-v0") for _ in range(num_agents+1)]
    agent = A3C(env, alpha=1e-5, beta=1e-4, gamma=0.99, epochs=4, t_length=8,
                action_type=ACTION_CONTINUOUS, distribution="beta", verbose=False)

    rewards, steps = agent.train(max_steps=500, episodes=1000, render=False, verbose=False)
    
    agent.run(100000, render=True)

    rewards = [ np.mean(rewards[i:i+10]) for i in range(len(rewards))]

    plt.plot(range(len(rewards)), rewards)
    plt.show()