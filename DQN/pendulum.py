import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym


# ================= 工具（Gymnasium 适配） =================
def make_env(env_id="Pendulum-v1", seed=0):
    env = gym.make(env_id)
    try: env.action_space.seed(seed)
    except: pass
    try: env.observation_space.seed(seed)
    except: pass
    return env


def env_reset(env, seed=None):
    obs, info = env.reset(seed=seed)
    return obs


def env_step(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, (terminated or truncated), info


def moving_average(x, window_size):
    if len(x) == 0: return []
    w = min(window_size, len(x))
    cs = np.cumsum(np.insert(np.array(x, dtype=np.float32), 0, 0.0))
    ma = (cs[w:] - cs[:-w]) / w
    head = [ma[0]] * (len(x) - len(ma))
    return head + ma.tolist()


# =============== 离散动作包装（把 Box 动作离散化） ===============
class DiscreteActionWrapper:
    """
    将连续动作 Box([-a, a], shape=(1,)) 离散为 N 个力矩档位：
    idx ∈ [0, N-1]  ->  torque ∈ linspace(-a, a, N)
    """
    def __init__(self, env, num_actions=11):
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (1,)
        self.env = env
        self.num_actions = num_actions
        self.max_torque = float(env.action_space.high[0])
        self.action_table = np.linspace(-self.max_torque, self.max_torque, num_actions, dtype=np.float32)

        # 伪装成离散动作空间，便于上层算法使用
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return env_reset(self.env, **kwargs)

    def step(self, discrete_idx):
        torque = np.array([self.action_table[int(discrete_idx)]], dtype=np.float32)
        return env_step(self.env, torque)


# ================= Replay Buffer =================
class ReplayBuffer:
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
        s, a, r, ns, d = zip(*transitions)
        return (np.stack(s, axis=0),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.stack(ns, axis=0),
                np.array(d, dtype=np.float32))

    def size(self):
        return len(self.buffer)


# ================= 网络定义 =================
class Qnet(torch.nn.Module):
    """基础 Q 网络：两层 MLP"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VAnet(torch.nn.Module):
    """Dueling：共享干道 + A/V 两头"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1  = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = F.relu(self.fc1(x))           # 只前向一次
        A = self.fc_A(shared)                  # (B, A)
        V = self.fc_V(shared)                  # (B, 1)
        A = A - A.mean(dim=1, keepdim=True)    # 去均值，避免不可辨识性
        return V + A                           # (B, A)



# ================= DQN / Double / Dueling =================
class DQN:
    """
    dqn_type: 'VanillaDQN' / 'DoubleDQN' / 'DuelingDQN'
    - Double 的目标：argmax 用 online，估值用 target
    - Dueling 使用 V/A 结构；目标计算默认按 Double（常用组合：Dueling + Double）
    """
    def __init__(self, state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon,
                 target_update, device,
                 dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.dqn_type = dqn_type

        if dqn_type == 'DuelingDQN':
            self.q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_net(s).argmax(dim=1).item()

    def max_q_value(self, state):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_net(s).max().item()

    def update(self, transition_dict):
        states      = torch.as_tensor(transition_dict['states'], dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(transition_dict['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        rewards     = torch.as_tensor(transition_dict['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.as_tensor(transition_dict['next_states'], dtype=torch.float32, device=self.device)
        dones       = torch.as_tensor(transition_dict['dones'], dtype=torch.float32, device=self.device).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            if self.dqn_type in ('DoubleDQN', 'DuelingDQN'):
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                max_next_q = self.target_q_net(next_states).gather(1, next_actions)
            else:
                max_next_q = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]
            q_targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

# ================= 训练循环 =================
def train(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    returns, q_curve = [], []
    ema_q = 0.0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc=f'Iteration {i}') as pbar:
            for j in range(int(num_episodes/10)):
                state = env.reset()
                if isinstance(state, tuple): state = state[0]  # 兼容性
                done, ep_ret = False, 0.0
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)

                    q_now = agent.max_q_value(state)
                    ema_q = 0.995 * ema_q + 0.005 * q_now
                    q_curve.append(ema_q)

                    state = next_state
                    ep_ret += reward

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        agent.update({
                            'states': b_s,
                            'actions': b_a,
                            'rewards': b_r,
                            'next_states': b_ns,
                            'dones': b_d
                        })
                returns.append(ep_ret)
                if (j+1) % 5 == 0:
                    pbar.set_postfix({'ret': f'{np.mean(returns[-5:]):.1f}'})
                pbar.update(1)
    return returns, q_curve

# ================= 主函数 =================
def main():
    # -------- 超参 --------
    lr = 1e-3
    num_episodes = 400
    hidden_dim = 256
    gamma = 0.99
    epsilon = 0.1
    target_update = 200
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 128
    ACTION_SPACE = 11
    dqn_type = 'DuelingDQN'   # 'VanillaDQN' / 'DoubleDQN' / 'DuelingDQN'

    seed = 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_env = make_env('Pendulum-v1', seed=seed)
    env = DiscreteActionWrapper(base_env, num_actions=ACTION_SPACE)

    state_dim = env.observation_space.shape[0]   # Pendulum: 3 (cosθ, sinθ, θ_dot)
    action_dim = env.action_space.n              # 离散动作数

    buffer = ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device, dqn_type=dqn_type)

    # 训练
    returns, q_curve = train(agent, env, num_episodes, buffer, minimal_size, batch_size)

    # 曲线
    plt.figure()
    plt.plot(range(len(returns)), moving_average(returns, 9))
    plt.xlabel('Episodes'); plt.ylabel('Returns'); plt.title(f'{dqn_type} (Discrete) on Pendulum-v1')
    plt.show()

    plt.figure()
    plt.plot(range(len(q_curve)), q_curve)
    plt.axhline(0, ls='--', c='orange')
    plt.xlabel('Frames'); plt.ylabel('Q value (EMA)'); plt.title(f'{dqn_type} (Discrete) on Pendulum-v1')
    plt.show()

if __name__ == "__main__":
    # 为了兼容 tqdm 的 notebook/终端显示，这里不做额外花活
    main()
