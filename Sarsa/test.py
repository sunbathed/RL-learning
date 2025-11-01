import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        # col 为列 ，row为行
        self.nrow = nrow
        self.ncol = ncol
        # 初始坐标的定义
        self.x = 0 # 横坐标
        self.y = self.nrow - 1 # 纵坐标

    def step(self, action):
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # 上下左右
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False

        # 判断是否到达悬崖或终点
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100  # 掉下悬崖
        return next_state, reward, done

    # 重新开始
    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


class Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q表格，Q_table[s, a]表示在状态 s下采取动作 a的长期期望回报
        self.n_action = n_action  # 动作数量（4个：上、下、左、右）
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索概率（ε-greedy策略）

    def take_action(self, state):
        #NumPy库中用于生成一个[0.0, 1.0)范围内
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_action)
        # 用于找到数组或列表中最大值的索引，这个索引就是Q值最大的动作编号
        return np.argmax(self.Q_table[state])

    #take_action方法使用ε - greedy策略（包含随机探索）不同，
    # best_action完全基于当前学习到的Q值表，贪婪地选择已知的最佳动作
    def best_action(self, state):
        # 先找到Q（动作价值函数）的最大值
        Q_max = np.max(self.Q_table[state])
        # 使用列表推导式标记最优动作
        return [1 if self.Q_table[state, i] == Q_max else 0 for i in range(self.n_action)]

    # Q值更新公式
    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


def train_sarsa(num_episodes=500, ncol=12, nrow=4, epsilon=0.1, alpha=0.1, gamma=0.9):
    env = CliffWalkingEnv(ncol, nrow) # 创建悬崖漫步的环境
    agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)  # 创建Sarsa智能体
    np.random.seed(0) # 随机数种子

    return_list = [] # 记录每一个回合的总回报
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0  #当前回合的奖励
                state = env.reset()  # 重置环境 ，当前的state为默认值
                action = agent.take_action(state)
                done = False  # 回合是否结束标志

                while not done:
                    # 执行动作 ， 获得环境反馈
                    next_state, reward, done = env.step(action)
                    # 选择下一个动作
                    next_action = agent.take_action(next_state)
                    # 更新Q值
                    agent.update(state, action, reward, next_state, next_action)
                    # 累计奖励 ，更新状态与Q值
                    episode_return += reward
                    state, action = next_state, next_action

                return_list.append(episode_return)
                # 每10个回合更新一次进度条显示 不是很懂
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': f'{int(num_episodes / 10 * i + i_episode + 1)}',
                        'avg_return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)

    return agent, env, return_list


def plot_results(return_list):
    episodes = np.arange(len(return_list))
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on Cliff Walking')
    plt.grid(True)
    plt.show()


def print_agent(agent, env, action_meaning, disaster=None, end=None):
    if disaster is None:
        disaster = []
    if end is None:
        end = []

    print("\n策略图示 (**** 为悬崖, EEEE 为终点):\n")
    for i in range(env.nrow):
        for j in range(env.ncol):
            state = i * env.ncol + j
            if state in disaster:
                print('****', end=' ')
            elif state in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(state)
                pi_str = ''.join([action_meaning[k] if a[k] > 0 else 'o' for k in range(len(action_meaning))])
                print(pi_str, end=' ')
        print()


if __name__ == '__main__':
    agent, env, returns = train_sarsa(num_episodes=500)
    plot_results(returns)
    action_meaning = ['^', 'v', '<', '>']
    print('Sarsa算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, disaster=list(range(37, 47)), end=[47])
