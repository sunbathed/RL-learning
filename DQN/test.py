import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym


# ========== 环境工具 ==========CartPole 环境

def make_env(env_id="CartPole-v1", seed=0):
    """创建并设定随机种子（推荐方式：对 action_space / observation_space 设种子）"""
    env = gym.make(env_id)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    try:
        env.observation_space.seed(seed)
    except Exception:
        pass
    return env


def env_reset(env, seed=None):
    obs, info = env.reset(seed=seed)
    return obs


def env_step(env, action):
    """gymnasium.step() 返回 (obs, reward, terminated, truncated, info)"""
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return obs, reward, done, info


# ========== 实用小函数 ==========输入一串数，输出一串更平滑的数
def moving_average(x, window_size):
    if len(x) == 0:
        return []
    window = min(window_size, len(x))
    cumsum = np.cumsum(np.insert(np.array(x, dtype=np.float32), 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    # 与原始长度对齐：前面用首值填充
    head = [ma[0]] * (len(x) - len(ma))
    return head + ma.tolist()


# ========== Replay Buffer ==========
class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity):
        # maxlen=capacity 表示队列满了以后，自动丢掉最老的数据（先进先出）
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # 存 numpy float32，减少后续转换开销
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),   # 把输入 x 转成 NumPy 数组 ,不重复复制数据
            bool(done),
        ))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # 从经验回放池里面取出batch_size个不同的数据（五元组）
        state, action, reward, next_state, done = zip(*transitions)
        return (np.stack(state, axis=0),
                np.array(action, dtype=np.int64),
                np.array(reward, dtype=np.float32),
                np.stack(next_state, axis=0),
                np.array(done, dtype=np.float32))

    def size(self):
        return len(self.buffer)


# ========== Q 网络 ==========
# for just 计算Q值
#  继承nn.Module 是定义所有神经网络模块的起点
# state_dim：输入数据的维度，即状态空间的维度。
# hidden_dim：隐藏层神经元的数量，决定了模型的容量。
# action_dim：输出数据的维度，即动作空间的维度
class Qnet(torch.nn.Module):
    """两层 MLP（1层隐藏层）"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()  # 调用父类nn.Module的初始化方法[6,7](@ref)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 定义第一个全连接层（输入层到隐藏层）[9,10](@ref)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 定义第二个全连接层（隐藏层到输出层）[9,10](@ref)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 数据通过第一层后应用ReLU激活函数[1,2](@ref)
        return self.fc2(x)  # 数据通过第二层（输出层），得到最终结果[1](@ref)


# ========== DQN 算法 ==========
class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim

        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)  # 预测网络
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)   # 目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 把当前网络的参数复制给目标网络。

        #使用 Adam 优化器去更新 q_net 的参数。self.q_net.parameters() 是模型所有可学习的权重和偏置。
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        # ：.to(device)方法可以自由地将向量和模型在CPU和GPU等设备中迁移
        self.device = device

        #当你将模型实例像函数一样调用时，__call__方法会自动帮你调用你定义在类中的forward方法
    def take_action(self, state):
        # state 可能是 list/np.ndarray；确保为 (1, state_dim) 的 FloatTensor
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():  # 表示只前向计算，不需要梯度（节省显存和时间）。
            state_t = torch.tensor(
                np.asarray(state),
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)  # 神经网络要求输入是 [batch_size, features]，即便只有一条数据也要包装成“批次”
            # .argmax(dim=1)：找到 Q 值最大的动作。
            # .item() 把结果从张量转成普通 Python 数字。
            return self.q_net(state_t).argmax(dim=1).item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32, device=self.device)
        # .view(-1,1) 把一维数组变成二维列向量，
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32, device=self.device).view(-1, 1)
        # 精准地挑选出智能体实际执行的那个动作所对应的Q值。
        # 当前 Q(s,a)
        #当 dim=0时，对于输出张量中每个位置 (i, j)，其值的计算公式为：output[i][j] = input[ index[i][j] ][j]
        q_values = self.q_net(states).gather(1, actions)  # just 按照行取列
        # 目标 Q
        with torch.no_grad():
            max_next_q = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]
            #  如果 done=1（episode 结束），则不考虑未来奖励；
            # 如果 done=0，则考虑折扣后的未来奖励
            q_targets = rewards + self.gamma * max_next_q * (1.0 - dones)
        #  使用均方误差（MSE）来衡量当前Q值估计（q_values）与目标Q值（q_targets）之间的差异。
        #  最小化这个损失就是让Q网络的预测逐渐逼近更优的TD目标
        loss = F.mse_loss(q_values, q_targets)
        #  经典三部曲
        #  将模型所有可学习参数（如Q网络权重）的 .grad属性（即梯度）重置为零
        self.optimizer.zero_grad()
        #
        loss.backward()
        self.optimizer.step()

        # 软频率更新：每 step 判断一次
        # 每执行 target_update 步，就把 q_net 参数复制到 target_q_net。
        # 这样目标网络是「延迟更新」的，从而让训练更平稳。
        # self.count 是用来计步的。
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


# ========== 训练入口 ==========
def main():
    # 超参
    lr = 2e-3  # 学习率。更新神经网络参数时的步长
    num_episodes = 500 # 总训练回合数
    hidden_dim = 128
    gamma = 0.98
    # epsilon(探索率): 在ε-贪婪策略中，控制随机探索的概率。
    # target_update(目标网络更新频率): 经过多少回合训练后，将在线网络的参数同步到目标网络。
    # buffer_size(经验回放池容量): 最多能存储的经验条数。
    # minimal_size(最小经验数量): 当回放池中的经验数超过此值后，才开始采样训练。
    epsilon = 0.01              # ε-贪心：小概率探索
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64

    # 随机种子 & 设备
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 环境（gymnasium）
    env_name = 'CartPole-v1'
    env = make_env(env_name, seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Agent & Buffer 初始化智能体和经验池
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    replay_buffer = ReplayBuffer(buffer_size)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                # gymnasium 要记得 reset 返回的是 (obs, info)
                state = env_reset(env, seed=seed if (i == 0 and i_episode == 0) else None)
                episode_return = 0.0
                done = False

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env_step(env, action)

                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

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
                        'episode': f'{int(num_episodes / 10 * i + i_episode + 1)}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)

    # 曲线
    episodes_list = list(range(len(return_list)))
    plt.figure()
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'DQN on {env_name} (raw)')
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.figure()
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns (MA, w=9)')
    plt.title(f'DQN on {env_name} (moving average)')
    plt.show()


if __name__ == "__main__":
    main()
