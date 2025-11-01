import gym
import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

# ==================== 兼容性修复 ====================
if not hasattr(np, "bool8"):   # 兼容 NumPy 2.0 删除的 bool8
    np.bool8 = np.bool_


# ==================== 策略网络 ====================
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

# ==================== 价值网络 ====================
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ==================== Actor-Critic 智能体 ====================
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        # ✅ 高效写法
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(state)  # 作用：把状态送进 actor（策略网络），得到 每个动作的概率分布
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        # ========= 时序差分目标 =========
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        # ✅ TD误差标准化（关键稳定技巧）
        td_delta = (td_delta - td_delta.mean()) / (td_delta.std() + 1e-8)

        # ========= 策略损失 =========
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # ========= 价值损失 =========
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())

        # ========= 参数更新 =========
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

# ==================== 训练函数 ====================
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(num_episodes):
        state, _ = env.reset(seed=0)
        done, episode_return = False, 0
        transition_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}

        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ✅ 稍微缩放奖励，减轻数值震荡
            reward /= 10.0

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)

            state = next_state
            episode_return += reward * 10  # 还原显示真实奖励

        agent.update(transition_dict)
        return_list.append(episode_return)

        if (i + 1) % 50 == 0:
            avg_r = np.mean(return_list[-50:])
            print(f"Episode {i+1}/{num_episodes}, Average Return (last 50): {avg_r:.1f}")

    return return_list


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # ✅ 固定随机种子，保证可复现
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    env = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ✅ 调参区
    hidden_dim = 256
    actor_lr = 1e-4
    critic_lr = 5e-3
    gamma = 0.99
    num_episodes = 1000

    agent = ActorCritic(state_dim, hidden_dim, action_dim,
                        actor_lr, critic_lr, gamma, device)

    returns = train_on_policy_agent(env, agent, num_episodes)

    # ========= 绘制曲线 =========
    plt.figure(figsize=(8, 5))
    plt.plot(returns, label="Return", alpha=0.4)
    mv_return = np.convolve(returns, np.ones(9) / 9, mode='valid')
    plt.plot(mv_return, label="Moving Average", linewidth=2, color="orange")
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Actor-Critic (Stable Version) on CartPole-v1')
    plt.legend()
    plt.show()
