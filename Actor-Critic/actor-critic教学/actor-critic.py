import gym
import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

# ==================== 兼容性修复 ====================
# ✅ NumPy 2.0 删除了 bool8 类型，而 gym 旧版本还会用到它
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# ===================================================
#                   策略网络（Actor）
# ===================================================
class PolicyNet(torch.nn.Module):
    """输入状态 → 输出每个动作的概率分布"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))                  # 隐藏层 ReLU 激活
        return F.softmax(self.fc2(x), dim=1)     # 输出动作概率（归一化）

# ===================================================
#                   价值网络（Critic）
# ===================================================
class ValueNet(torch.nn.Module):
    """输入状态 → 输出状态价值 V(s)"""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)                       # 输出标量（V(s)）

# ===================================================
#                   Actor-Critic 智能体
# ===================================================
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, gamma, device):
        # ✅ 初始化两个网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        # ✅ 为两个网络分别创建优化器（设置学习率）
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma      # 折扣因子 γ：越接近1越看重未来奖励
        self.device = device

    # ---------------------------------------------------
    # 按当前策略 π(a|s) 采样动作
    # ---------------------------------------------------
    def take_action(self, state):
        # 环境返回的是一维状态向量，先加 batch 维度
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 前向传播得到动作概率分布
        probs = self.actor(state)

        # 用 Categorical 分布按概率采样动作（不是取最大！）
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()

        return action.item()  # 返回 int 型动作

    # ---------------------------------------------------
    # 使用一条轨迹 (s,a,r,s') 更新网络
    # ---------------------------------------------------
    def update(self, transition_dict):
        # 将收集到的轨迹字典转成 tensor 批数据
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        # ========= 1️⃣ 计算时序差分目标 (TD target) =========
        # y = r + γ * V(s') * (1 - done)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        # ========= 2️⃣ 计算时序差分误差 δ =========
        td_delta = td_target - self.critic(states)

        # ✅ 小技巧：标准化 δ，让训练更稳定（避免梯度爆炸/消失）
        td_delta = (td_delta - td_delta.mean()) / (td_delta.std() + 1e-8)

        # ========= 3️⃣ 策略损失 (Actor Loss) =========
        # log π(a|s)：取出当前动作的对数概率
        log_probs = torch.log(self.actor(states).gather(1, actions))
        # 策略梯度公式：max E[log π(a|s) * advantage]
        # 因为 PyTorch 默认最小化 loss，加负号
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # ========= 4️⃣ 价值损失 (Critic Loss) =========
        # 让 V(s) 拟合 TD 目标（普通 MSE 回归）
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())

        # ========= 5️⃣ 参数更新 =========
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()   # ← 使用 actor_lr 更新策略参数
        self.critic_optimizer.step()  # ← 使用 critic_lr 更新价值参数

# ===================================================
#              训练主循环（on-policy）
# ===================================================
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []

    for i in range(num_episodes):
        # 初始化环境（返回 state 和 info）
        state, _ = env.reset(seed=0)
        done, episode_return = False, 0

        # 存储本回合轨迹
        transition_dict = {
            'states': [], 'actions': [], 'rewards': [],
            'next_states': [], 'dones': []
        }

        # ========== 一回合交互 ==========
        while not done:
            action = agent.take_action(state)                 # 采样动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated                    # 判定 episode 是否结束

            reward /= 10.0  # ✅ 缩放奖励：让梯度更平稳

            # 保存轨迹
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)

            state = next_state
            episode_return += reward * 10  # 乘回去只是为了打印原奖励

        # ========== 更新网络 ==========
        agent.update(transition_dict)
        return_list.append(episode_return)

        # 每 50 回合打印一次平均收益
        if (i + 1) % 50 == 0:
            avg_r = np.mean(return_list[-50:])
            print(f"Episode {i+1}/{num_episodes}, "
                  f"Average Return (last 50): {avg_r:.1f}")

    return return_list

# ===================================================
#                   主程序入口
# ===================================================
if __name__ == "__main__":
    # ✅ 固定随机种子，保证可复现
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # 创建环境
    env = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ✅ 调参区
    hidden_dim = 256           # 网络隐藏层宽度
    actor_lr = 1e-4            # 策略网络学习率（小 → 稳定）
    critic_lr = 5e-3           # 价值网络学习率（大 → 学得快）
    gamma = 0.99               # 折扣因子（越大越远见）
    num_episodes = 1000        # 训练回合数

    # 创建智能体
    agent = ActorCritic(state_dim, hidden_dim, action_dim,
                        actor_lr, critic_lr, gamma, device)

    # 开始训练
    returns = train_on_policy_agent(env, agent, num_episodes)

    # ========= 绘制学习曲线 =========
    plt.figure(figsize=(8, 5))
    plt.plot(returns, label="Return", alpha=0.4)
    # 平滑处理（9点滑动平均）
    mv_return = np.convolve(returns, np.ones(9) / 9, mode='valid')
    plt.plot(mv_return, label="Moving Average", linewidth=2, color="orange")
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Actor-Critic (Stable Version) on CartPole-v1')
    plt.legend()
    plt.show()
