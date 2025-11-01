import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym


# ========== Replay Buffer ==========

class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (np.stack(state, axis=0),
                np.array(action, dtype=np.int64),
                np.array(reward, dtype=np.float32),
                np.stack(next_state, axis=0),
                np.array(done, dtype=np.float32))

    def size(self):
        return len(self.buffer)  # 返回大小


# ========== 平滑函数 ==========

def moving_average(x, window_size):
    if len(x) == 0:
        return []
    window = min(window_size, len(x))
    cumsum = np.cumsum(np.insert(np.array(x, dtype=np.float32), 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    head = [ma[0]] * (len(x) - len(ma))
    return head + ma.tolist()


# ========== Q 网络 ==========

class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ========== Double DQN ==========

class DoubleDQN:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # 目标网络一开始拷贝一份
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):
        # ε-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        # 确保是 np.array -> torch
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state)

        return q_values.argmax(dim=1).item()

    def max_q_value(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32, device=self.device).view(-1, 1)

        # 当前 Q(s,a) —— 用在线网
        q_values = self.q_net(states).gather(1, actions)

        # ===== Double DQN 核心两步 =====
        # 1) 在线网在 s' 上选动作 a* = argmax_a Q_online(s', a)
        next_q_online = self.q_net(next_states)
        max_actions = next_q_online.argmax(dim=1, keepdim=True)   # (batch, 1)

        # 2) 目标网用刚刚挑出来的动作算 Q_target(s', a*)
        next_q_target = self.target_q_net(next_states)
        max_next_q_values = next_q_target.gather(1, max_actions)  # (batch, 1)

        # 目标 y = r + γ * Q_target(s', a*) * (1 - done)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每次更新都 +1，然后按频率同步目标网络
        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())


# ========== 训练 ==========

def train_dqn(agent, env, num_episodes, replay_buffer, minimal_size, batch_size, action_dim):
    return_list = []
    max_q_value_list = []
    max_q_value = 0.0

    # 把 [-2, 2] 均分成 action_dim 份
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]

    def dis_to_con(discrete_action: int):
        return action_low + (discrete_action / (action_dim - 1)) * (action_high - action_low)

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                state, _ = env.reset()
                done = False
                episode_return = 0.0

                while not done:
                    # 1. 离散动作
                    action = agent.take_action(state)
                    # 2. 转回连续动作
                    action_continuous = dis_to_con(action)

                    # gymnasium: 5 个返回值
                    next_state, reward, terminated, truncated, _ = env.step([action_continuous])
                    done = terminated or truncated

                    # 存经验
                    replay_buffer.add(state, action, reward, next_state, done)

                    # 记录平滑 Q 值
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
                    max_q_value_list.append(max_q_value)

                    state = next_state
                    episode_return += reward

                    # 开始训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'rewards': b_r,
                            'next_states': b_ns,
                            'dones': b_d
                        }
                        agent.update(transition_dict)

                return_list.append(episode_return)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': f'{num_episodes / 10 * i + i_episode + 1}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)

    return return_list, max_q_value_list


# ========== 主函数 ==========

def main():
    # 超参
    lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 环境
    env_name = "Pendulum-v1"
    env = gym.make(env_name)

    # 随机种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = 11  # 把连续动作离散成 11 个

    replay_buffer = ReplayBuffer(buffer_size)
    agent = DoubleDQN(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon,
        target_update=target_update,
        device=device,
    )

    return_list, max_q_value_list = train_dqn(
        agent, env, num_episodes,
        replay_buffer, minimal_size,
        batch_size, action_dim
    )

    # 画回报
    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'Double DQN on {env_name}')
    plt.show()

    # 画 Q 值
    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, ls='--', c='orange')
    plt.axhline(10, ls='--', c='red')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title(f'Double DQN on {env_name}')
    plt.show()


if __name__ == "__main__":
    main()
