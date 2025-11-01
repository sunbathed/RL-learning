import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ------------------------
# 环境定义（与教材一致）
# ------------------------
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0
        self.y = self.nrow - 1  # 起点在左下角

    def step(self, action):
        # action: 0=上, 1=下, 2=左, 3=右
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))

        reward = -1
        done = False

        # 悬崖区
        if self.y == self.nrow - 1 and 1 <= self.x <= self.ncol - 2:
            reward = -100
            done = True  # 掉崖直接结束

        # 终点
        elif self.y == self.nrow - 1 and self.x == self.ncol - 1:
            done = True

        next_state = self.y * self.ncol + self.x
        return next_state, reward, done

    def reset(self):
        self.x, self.y = 0, self.nrow - 1
        return self.y * self.ncol + self.x


# ------------------------
# SARSA算法（教材1-step）
# ------------------------
class Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_action)
        return np.argmax(self.Q[state])

    def best_action(self, state):
        Q_max = np.max(self.Q[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_target = r + self.gamma * self.Q[s1, a1]
        td_error = td_target - self.Q[s0, a0]
        self.Q[s0, a0] += self.alpha * td_error


# ------------------------
# 训练主程序
# ------------------------
def train_sarsa(num_episodes=500, ncol=12, nrow=4, epsilon=0.1, alpha=0.1, gamma=0.9):
    env = CliffWalkingEnv(ncol, nrow)
    agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    np.random.seed(0)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                state = env.reset()
                action = agent.take_action(state)
                done = False
                episode_return = 0
                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)
                    agent.update(state, action, reward, next_state, next_action)
                    state, action = next_state, next_action
                    episode_return += reward
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'avg_return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return agent, env, return_list


# ------------------------
# 策略打印函数
# ------------------------
def print_agent(agent, env, action_meaning, disaster=None, end=None):
    if disaster is None:
        disaster = []
    if end is None:
        end = []
    print("\n策略图示（****为悬崖, EEEE为终点）:\n")
    for i in range(env.nrow):
        for j in range(env.ncol):
            s = i * env.ncol + j
            if s in disaster:
                print('****', end=' ')
            elif s in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(s)
                pi_str = ''.join(action_meaning[k] if a[k] > 0 else 'o' for k in range(len(action_meaning)))
                print(pi_str, end=' ')
        print()


# ------------------------
# 主运行入口
# ------------------------
if __name__ == '__main__':
    agent, env, returns = train_sarsa(num_episodes=500)
    # 平滑曲线
    episodes = np.arange(len(returns))
    plt.plot(episodes, returns)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SARSA on Cliff Walking (Textbook Version)')
    plt.grid(True)
    plt.show()

    # 打印策略
    action_meaning = ['^', 'v', '<', '>']
    agent.epsilon = 0  # 打印确定性策略
    print('SARSA算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, disaster=list(range(37, 47)), end=[47])
