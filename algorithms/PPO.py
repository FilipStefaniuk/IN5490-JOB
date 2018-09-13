
import numpy as np 
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
import time

from sys import stdout
# from Features import fourier_basis
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
        self.linear = torch.nn.Linear(num_features, 32, bias=True)
        self.linear2 = torch.nn.Linear(32, 32, bias=True)

        if action_type == ACTION_DISCRETE:
            self.linear3 = torch.nn.Linear(32, num_actions, bias=True)
        else:
            self.linear3 = torch.nn.Linear(32, 32, bias=True)
            self.linear_param1 = torch.nn.Linear(32, num_actions, bias=True)
            self.linear_param2 = torch.nn.Linear(32, num_actions, bias=True)
            
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
                return torch.sigmoid(x1), torch.sigmoid( x2 ) / 2.0
            
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

    def select_action(self, phi, print_action=False):

        x = torch.from_numpy(phi).float()

        # print(softmax)
        if self.action_type == ACTION_CONTINUOUS:
            param1, param2 = self.forward( x )
            param1 = param1.view(1,-1)
            param2 = param2.view(1,-1)
            if print_action:
                print(param1)
                print(param2)

            dist = self.get_distribution(param1, param2)
            action = dist.sample()[0]
            action = action.detach().numpy()
        elif self.action_type == ACTION_DISCRETE:
            action = self.forward( x )
            if print_action:
                print(action)

            action = action.detach().numpy()
            action = np.random.choice(range(action.shape[0]), p=action)
        else:
            assert(False)
            
        return action



class Critic(torch.nn.Module):
    def __init__(self, num_features):
        super(Critic, self).__init__()
        self.linear = torch.nn.Linear(num_features, 32, bias=True)
        self.linear2 = torch.nn.Linear(32, 32, bias=True)
        self.linear3 = torch.nn.Linear(32, 1, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)
        x = self.linear3(x)

        return x

def get_phi(env, state):
        st = state.reshape( (state.shape[0],) )
        return st / 100.0
        st = (st - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) 
        return st

        return phi


class PPOAgent():

    GPU = False
    
    def __init__(self, env, actor, t_length, action_type, distribution):
        self.env = env
        print(env.action_space.__dict__)
        print(env.observation_space.__dict__)

        self.s = env.reset()
        phi = get_phi(self.env, self.s.reshape(-1,1))
        self.actor = actor
        self.action_type = action_type
        self.distribution = distribution

        self.episode_rewards = []
        self.episode_steps = []
        self.buffer = []

        self.clear_history()
        self.buffer_size = 16
        self.t_length = t_length

        self.done = True
        self.pause_collecting = False
        self.terminate_collecting = False
        self.episodes = 0


    def clear_history(self):
        self.hist_s, self.hist_sn, self.hist_a, self.hist_r, self.hist_term = None, None, None, None, None


    def reset(self):
        self.s = self.env.reset()
    
    def add_hist_to_buffer(self):
        x = (self.hist_s.copy(), self.hist_sn.copy(), self.hist_r.copy(), self.hist_a.copy(), self.hist_term.copy())        
        x = (torch.from_numpy(x[0]).float(), torch.from_numpy(x[1]).float(), torch.from_numpy(x[2]).float(),
            torch.from_numpy(x[3]), torch.from_numpy(x[4]).float())

        if self.GPU:
            x = (x[0].cuda(), x[1].cuda(), x[2].cuda(), x[3].cuda(), x[4].cuda())

        self.buffer.append( x )
        if len(self.buffer) == self.buffer_size:
            self.pause_collecting = True


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

            self.done = done or (self.total_step == max_steps)

            if self.hist_s.shape[0] == self.t_length or done:
                self.add_hist_to_buffer()
                self.pause_collecting = True 

            while self.pause_collecting:
                time.sleep(0.1)

            self.s = state_next
        
        
class PPO():
    
    GPU = False
    
    
    def __init__(self, envs, alpha, beta, gamma, 
                lambda_=0.95, batch_size=32, t_length=32, epochs=4,
                action_type=ACTION_DISCRETE, distribution="normal", verbose=True):
        # super(ActorCriticPytorch, self).__init__()
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.lamba_ = lambda_

        self.t_length = t_length

        self.action_type = action_type
        self.num_actions = envs[0].action_space.n if action_type == ACTION_DISCRETE else envs[0].action_space.low.shape[0]
        self.distribution = distribution

        if action_type == ACTION_CONTINUOUS:
            self.torch_high = torch.from_numpy(envs[0].action_space.high)
            self.torch_low = torch.from_numpy(envs[0].action_space.low)
        
        self.num_features = get_phi(envs[0], envs[0].reset()).shape[0]
        self.epochs = epochs

        self.max_grad_norm = 2.0
        

        self.buffer_size = 1000
        self.batch_size = batch_size
        self.buffer = []


        self.critic = Critic(self.num_features).cuda() if self.GPU == True else Critic(self.num_features)
            
        self.actor_act = Actor(self.num_features, self.num_actions, action_type, distribution)
        self.actor_update = self.actor_act
        if self.GPU:
            self.actor_update = Actor(self.num_features, self.num_actions, action_type, distribution)
            self.actor_update.load_state_dict(self.actor_act.state_dict())
            self.actor_update = self.actor_update.cuda()

        self.actor_old = Actor(self.num_features, self.num_actions, action_type, distribution).cuda() if self.GPU == True else Actor(self.num_features, self.num_actions, action_type, distribution)


        self.optimizer_actor = torch.optim.Adam(self.actor_update.parameters(), lr=self.alpha)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.beta)

        self.agents = [PPOAgent(env, self.actor_act, t_length, action_type, distribution) for env in envs[:-1]]
        self.test_agent = PPOAgent(envs[-1], self.actor_act, t_length, action_type, distribution)


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
                    if len(self.buffer) >= self.buffer_size:
                        for _ in range(len(agent.buffer)):
                            self.buffer.pop( 0 )

                    agents_on_hold.append( agent )

                    self.buffer.extend( agent.buffer )
                    agent.clear_history()
                    agent.buffer = []

            if len(agents_on_hold) == len(self.agents):
                agents_on_hold = []

                for agent in self.agents:
                    agent.pause_collecting = False

                start = time.time()
                self.update( verbose )
                end = time.time()
                # print("Time: " + str(end-start))
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

                    if current_episodes % 20 == 0:
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

            action = self.test_agent.actor.select_action(self.test_agent.s, print_action=False)
            
            state_next, reward, done, _ = self.test_agent._step( action )

            if render == True:
                self.test_agent.env.render( )

            total_reward += reward

            self.test_agent.s = state_next
            if done:
                break
        return total_reward, step



    def compute_loss_terms(self, samples):

        Ls = torch.zeros( (self.t_length, len(samples))) if self.GPU == False else torch.cuda.FloatTensor(self.t_length, len(samples)).fill_(0)
        vfs = torch.zeros( (self.t_length, len(samples))) if self.GPU == False else torch.cuda.FloatTensor(self.t_length, len(samples)).fill_(0)
        ents = torch.zeros( (self.t_length, len(samples))) if self.GPU == False else torch.cuda.FloatTensor(self.t_length, len(samples)).fill_(0)

        for s_index, hist in enumerate(samples):
            s, sn, r, a, nonterm = hist

            vn = self.critic.forward( sn )
            v = self.critic.forward( s )


            deltas = (r + (nonterm * self.gamma * vn)) - v

            A = torch.zeros((s.shape[0], 1)) if self.GPU == False else torch.cuda.FloatTensor(s.shape[0],1).fill_(0)

            for i in range(s.shape[0]):
                A[i] = sum([ (self.gamma*self.lamba_)**(k-i) * deltas[k] for k in range(i, s.shape[0])])
                 
            if self.action_type == ACTION_DISCRETE: 
                p = self.actor_update.forward( s )
                p_o = self.actor_old.forward( s )

                R = torch.exp( torch.log(p.gather(1, a)) - torch.log(p_o.gather(1, a)) )
                H = -torch.sum( p * torch.log(p), dim=1).view(-1,1)
            else:
                p1, p2 = self.actor_update.forward( s )
                p_o1, p_o2 = self.actor_old.forward( s )

                dist = self.actor_update.get_distribution(p1, p2)
                dist_old = self.actor_update.get_distribution(p_o1, p_o2)
                

                R = torch.exp(dist.log_prob(a) - dist_old.log_prob(a))
                H = torch.sum( dist.entropy(), dim=1 ).view(-1,1)

            e = 0.2
            l1 = R*A 
            l2 = torch.clamp(R, 1-e, 1+e) * A

            L = torch.mean( torch.min(l1, l2), dim=1 )
            
            for t_index in range(L.shape[0]):
                Ls[t_index, s_index] = L[t_index]

            for t_index in range(deltas.shape[0]):
                vfs[t_index, s_index] = deltas[t_index]**2

            for t_index in range(H.shape[0]):
                ents[t_index, s_index] = torch.mean(H[t_index])


        L = torch.sum( torch.mean(Ls, dim=1) )
        vf = torch.sum( torch.mean(vfs, dim=1) )
        ent = 0.0 * torch.sum( torch.mean(ents, dim=1) )        
        

        return L, vf, ent


    
    def update(self, verbose):
        k = self.batch_size if self.batch_size < len(self.buffer) else len(self.buffer)

        d = self.actor_update.state_dict().copy()
        for _ in range(self.epochs):
            samples = random.sample( self.buffer, k=k-1)
            samples.append( self.buffer[-1] )
            L, vf, ent = self.compute_loss_terms(samples)

            
            if verbose:
                print("Total loss: %f - L: %f - delta: %f - entropy: %f"
                    %(L-vf+ent, L, vf, ent))
            


            self.optimizer_actor.zero_grad()
            actor_loss = -(L + ent)
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(self.actor_update.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            vf.backward()
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()

        self.actor_old.load_state_dict(d)


        
        if self.GPU == True:
            self.actor_act.load_state_dict(self.actor_update.state_dict())



    def save_model(self, path, i):
        torch.save(self.actor_update, path+"actor_%f_%f_%f_%d.pt" %(self.alpha, self.beta, self.gamma, i))
        critic = self.critic.cpu() if self.GPU == True else self.critic
        
        torch.save(critic, path+"critic_%f_%f_%f_%d.pt" %(self.alpha, self.beta, self.gamma, i))

    
    def load_model(self, actor, critic):
        self.actor_update.load_state_dict(torch.load(actor))
        self.critic.load_state_dict(torch.load(critic))





if __name__ == "__main__":
    import gym
    # import roboschool
    import matplotlib.pyplot as plt

    num_agents = 6

    # env = [gym.make("Acrobot-v1") for _ in range(num_agents+1)]
    # env = [gym.make("CartPole-v1") for _ in range(num_agents+1)]
    env = [gym.make("LunarLanderContinuous-v2") for _ in range(num_agents+1)]
    # env = [gym.make("RoboschoolInvertedPendulum-v1") for _ in range(num_agents+1)]
    agent = PPO(env, alpha=1e-3, beta=1e-2, gamma=0.99, 
            lambda_=0.95, batch_size=64, t_length=16, epochs=8,
            action_type=ACTION_CONTINUOUS, distribution="normal", verbose=False)

    rewards, steps = agent.train(max_steps=500, episodes=1000, render=False, verbose=False)
    
    agent.run(100000, render=True)

    rewards = [ np.mean(rewards[i:i+10]) for i in range(len(rewards))]

    plt.plot(range(len(rewards)), rewards)
    plt.show()
