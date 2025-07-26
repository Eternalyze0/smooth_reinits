import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, experimental=False):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.experimental = experimental
        self.scaling_factor = 1.0

    def forward(self, x):
        if self.experimental:
            with torch.no_grad():
                # self.fc1.weight.data.add_(torch.randn_like(self.fc1.weight) * self.fc1.weight * self.scaling_factor)
                self.fc2.weight.data.add_(torch.randn_like(self.fc2.weight) * self.fc2.weight * self.scaling_factor)
                # self.fc3.weight.data.add_(torch.randn_like(self.fc3.weight) * self.fc3.weight * self.scaling_factor)
                # self.fc1.bias.data.add_(torch.randn_like(self.fc1.bias) * self.fc1.bias * self.scaling_factor)
                self.fc2.bias.data.add_(torch.randn_like(self.fc2.bias) * self.fc2.bias * self.scaling_factor)
                # self.fc3.bias.data.add_(torch.randn_like(self.fc3.bias) * self.fc3.bias * self.scaling_factor)
                self.scaling_factor *= 1.0
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) + x
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main(experimental, seed):
    torch.manual_seed(seed)
    env = gym.make('CartPole-v1')
    q = Qnet(experimental=experimental)
    q_target = Qnet(experimental=experimental)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        # epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        epsilon = 0.0
        s, _ = env.reset()
        done = False

        # while not done:
        for ep_dur in range(100000):
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if ep_dur + 1 >= 100000 or n_epi + 1 >= 10000:
            return n_epi
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("experimental:", experimental, "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    speed_run = []
    speed_run_exp = []
    for seed in range(10):
        speed_run.append(main(experimental=False, seed=seed))
        speed_run_exp.append(main(experimental=True, seed=seed))
    plt.plot(speed_run, label='baseline')
    plt.plot(speed_run_exp, label='experimental')
    plt.xlabel("Seed")
    plt.ylabel("Time It Takes to Win at Cartpole with Epsilon=0.0")
    plt.legend()
    plt.savefig("dqn.png")
