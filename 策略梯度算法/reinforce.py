import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ========== 策略网络 ==========
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


# ========== REINFORCE 算法 ==========
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device

# “R 的期望值（平均值）”
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        rewards = transition_dict['rewards']
        states = transition_dict['states']
        actions = transition_dict['actions']
#如果某个动作带来了高回报，就提高它的概率；
# 如果带来低回报，就降低它的概率。”
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            state = torch.tensor(states[i], dtype=torch.float32).to(self.device)
            action = torch.tensor([actions[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action) + 1e-8)
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()


# ========== 主程序 ==========
def main():
    lr = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.action_space.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, lr, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                transition_dict = {'states': [], 'actions': [], 'rewards': []}
                state, _ = env.reset(seed=0)
                done = False
                episode_return = 0
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['rewards'].append(reward)
                    state = next_state
                    episode_return += reward
                agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    # 绘图
    plt.plot(return_list)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'REINFORCE on {env_name}')
    plt.show()

    # 平滑处理
    mv_return = np.convolve(return_list, np.ones(9)/9, mode='valid')
    plt.plot(mv_return)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Return')
    plt.title(f'Moving Average Return ({env_name})')
    plt.show()


if __name__ == "__main__":
    main()
