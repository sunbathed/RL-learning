import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalkingEnv:
    def __init__(self,ncol , nrow):
        self.nrow = nrow
        self.ncol = ncol
        #  初始坐标定义
        self.x = 0
        self.y = self.nrow - 1  # 纵坐标

    def step(self,action):
        change = [[0,-1] ,[0,1] ,[-1 , 0],[1,0]]
        self.x = min(self.ncol - 1 ,max(0 ,self.x + change[action][0]))
        self.y = min(self.nrow - 1 ,max(0 ,self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False

        # 判断是否到达悬崖或者终点
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol -1:
                reward = -100 #   掉下悬崖
            return next_state , reward , done

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

class Sarsa:
    def __init__(self , ncol ,nrow ,epsilon , alpha , gamma ,n_action=4):
        self.Q_table = np.zeros([nrow * ncol , n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self , state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_action)
        return np.argmax(self.Q_table[state])

    def best_acton(self ,state):
        Q_max = np.max(self.Q_table[state])
        return [1 if self.Q_table[state , i] == Q_max else 0 for i in range(self.n_action)]

    def update(self ,s0 ,a0 ,r ,s1 ,a1):
        td_error = r + self.gamma * self.Q_table[s1,a1] - self.Q_table[s0,a0]
        self.Q_table[s0,a0] += self.alpha * td_error


def train_sarsa(num_episodes=500 ,ncol=12,nrow=4 ,epsilon =0.1 ,alpha=0.1,gamma = 0.9):
    env = CliffWalkingEnv(ncol , nrow)
    agent = Sarsa(ncol , nrow , epsilon , alpha , gamma)
    np.random.seed(0)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/ 10) , desc = f'Iteration {i + 1}') as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state ,reward ,done = env.step(action)
                    next_action = agent.take_action(next_state)
                    agent.update(state ,action ,next_state ,next_action)
                    episode_return += reward
                    state , action = next_state ,next_action
                return_list.append(episode_return)
                # 每10个回合更新一次进度条显示 不是很懂
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': f'{int(num_episodes / 10 * i + i_episode + 1)}',
                        'avg_return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)
    return agent ,env , return_list

def plot_results(return_list):
    episodes = np.arange(len(return_list))
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on Cliff Walking')
    plt.grid(True)
    plt.show()









