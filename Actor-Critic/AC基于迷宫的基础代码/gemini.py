# -*- coding: utf-8 -*-
"""
Actor-Critic 迷宫导航 (可视化+稳定版)
-----------------------------------------
特性：
 1. 可视化：每隔 N 回合演示一次移动过程 (由 SHOW_EVERY 控制)
 2. 稳定性：使用 One-Hot 状态编码 + 单步 TD(0) 更新
 3. 效率：后台训练飞快，演示时直观
"""

import numpy as np
import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import matplotlib.pyplot as plt

# ==========================================
#  ⚙️ 配置区域 (修改这里来控制动画)
# ==========================================
UNIT = 40  # 格子像素大小
MAZE_H = 4  # 迷宫高度
MAZE_W = 4  # 迷宫宽度
SHOW_EVERY = 1  # 每隔多少回合显示一次动画 (设为 1 则每次都显示)
ANIMATION_SPEED = 0.05  # 动画延迟时间 (秒)，越小越快


# ==========================================


# ============================================================
# 环境定义：Maze 类 (负责画图和逻辑)
# ============================================================
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'r', 'l']
        self.n_actions = len(self.action_space)
        self.title('Actor-Critic Maze')
        self.geometry(f'{MAZE_W * UNIT}x{MAZE_H * UNIT}')
        self._build_maze()
        self._init_rc_refs()
        self._build_bfs_distance()

    def _build_maze(self):
        """构建迷宫界面"""
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # 绘制网格
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT)
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r)

        origin = np.array([20, 20])

        # === 障碍物设置 (黑色) ===
        # 障碍 1
        pos1 = origin + np.array([UNIT * 2, UNIT * 1])
        self.hell1 = self.canvas.create_rectangle(
            pos1[0] - 15, pos1[1] - 15, pos1[0] + 15, pos1[1] + 15, fill='black')
        # 障碍 2
        pos2 = origin + np.array([UNIT * 1, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            pos2[0] - 15, pos2[1] - 15, pos2[0] + 15, pos2[1] + 15, fill='black')

        # === 终点 (黄色圆) ===
        oval_pos = origin + np.array([UNIT * 2, UNIT * 2])
        self.oval = self.canvas.create_oval(
            oval_pos[0] - 15, oval_pos[1] - 15, oval_pos[0] + 15, oval_pos[1] + 15, fill='yellow')

        # === 智能体 (红色方块) ===
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15, origin[0] + 15, origin[1] + 15, fill='red')

        self.canvas.pack()

    def _coords_to_rc(self, coords):
        x1, y1, x2, y2 = coords
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        c = int(round((cx - UNIT / 2) / UNIT))
        r = int(round((cy - UNIT / 2) / UNIT))
        return r, c

    def _init_rc_refs(self):
        self.start_rc = self._coords_to_rc(self.canvas.coords(self.rect))
        self.goal_rc = self._coords_to_rc(self.canvas.coords(self.oval))
        self.walls_rc = {
            self._coords_to_rc(self.canvas.coords(self.hell1)),
            self._coords_to_rc(self.canvas.coords(self.hell2)),
        }

    def _build_bfs_distance(self):
        """计算每个格子到终点的最短路，用于引导奖励"""
        H, W = MAZE_H, MAZE_W
        dist = [[float('inf')] * W for _ in range(H)]
        from collections import deque
        q = deque([self.goal_rc])
        dist[self.goal_rc[0]][self.goal_rc[1]] = 0
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in self.walls_rc:
                    if dist[nr][nc] > dist[r][c] + 1:
                        dist[nr][nc] = dist[r][c] + 1
                        q.append((nr, nc))
        self.dist_map = dist

    def reset(self):
        self.update()
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15, origin[0] + 15, origin[1] + 15, fill='red')
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        r, c = self._coords_to_rc(s)

        base_action = np.array([0, 0])
        if action == 0:
            base_action[1] -= UNIT  # 上
        elif action == 1:
            base_action[1] += UNIT  # 下
        elif action == 2:
            base_action[0] += UNIT  # 右
        elif action == 3:
            base_action[0] -= UNIT  # 左

        # 移动前预测位置
        next_coords = [s[0] + base_action[0], s[1] + base_action[1],
                       s[2] + base_action[0], s[3] + base_action[1]]
        cx, cy = (next_coords[0] + next_coords[2]) / 2, (next_coords[1] + next_coords[3]) / 2
        nr = int(round((cy - UNIT / 2) / UNIT))
        nc = int(round((cx - UNIT / 2) / UNIT))

        # 边界检测：如果在界内，移动；否则不动
        if 0 <= nr < MAZE_H and 0 <= nc < MAZE_W:
            self.canvas.move(self.rect, base_action[0], base_action[1])
            r_, c_ = nr, nc
        else:
            r_, c_ = r, c  # 撞墙保持不动

        s_ = self.canvas.coords(self.rect)

        # === 奖励计算 ===
        reward = -0.02  # 基础步数消耗
        done = False

        if (r_, c_) == self.goal_rc:
            reward = 5.0
            done = True
            s_ = 'terminal'
        elif (r_, c_) in self.walls_rc:
            reward = -1.0
            done = True
            s_ = 'terminal'
        else:
            # 距离引导奖励 (Shaping)
            d_prev = self.dist_map[r][c]
            d_now = self.dist_map[r_][c_]
            if d_now < float('inf'):
                reward += 0.3 * (d_prev - d_now)
            if (r, c) == (r_, c_):  # 撞墙/出界惩罚
                reward -= 0.1

        return s_, reward, done

    def render(self):
        self.update()


# ============================================================
# 神经网络模型
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================
# AC 智能体
# ============================================================
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.device = device
        self.entropy_coef = 0.05

    def take_action(self, state_vec):
        """根据概率选择动作"""
        state = torch.tensor(state_vec, dtype=torch.float32).to(self.device).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        """单步更新网络参数"""
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.int64).to(self.device).view(-1, 1)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device).view(-1, 1)
        done_val = torch.tensor([float(done)], dtype=torch.float32).to(self.device).view(-1, 1)

        # TD Target
        v_next = self.critic(next_state)
        td_target = reward + self.gamma * v_next * (1 - done_val)
        v_current = self.critic(state)
        td_delta = td_target - v_current

        # 1. Critic Loss
        critic_loss = F.mse_loss(v_current, td_target.detach())

        # 2. Actor Loss
        probs = self.actor(state)
        log_probs = torch.log(probs.gather(1, action) + 1e-9)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()
        actor_loss = -torch.mean(log_probs * td_delta.detach()) - self.entropy_coef * entropy

        # Backprop
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()


# ============================================================
# 训练流程 (含可视化逻辑)
# ============================================================
class Trainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.episode = 0
        self.max_episodes = 600
        self.rewards = []

    def _obs_to_one_hot(self, obs):
        """One-Hot 编码: 将位置转为向量"""
        vec = np.zeros(MAZE_H * MAZE_W, dtype=np.float32)
        if obs == 'terminal': return vec
        r, c = self.env._coords_to_rc(obs)
        index = r * MAZE_W + c
        if 0 <= index < len(vec):
            vec[index] = 1.0
        return vec

    def train(self):
        if self.episode >= self.max_episodes:
            print("训练结束！")
            self._plot()
            return  # 停止

        # 判断本回合是否需要显示动画
        is_rendering = (self.episode % SHOW_EVERY == 0) or (self.episode > self.max_episodes - 5)

        obs = self.env.reset()
        state = self._obs_to_one_hot(obs)
        ep_reward = 0

        # 动态调整熵系数 (后期减少探索)
        self.agent.entropy_coef = 0.05 if self.episode < 300 else 0.01

        # 单回合循环
        for step in range(100):
            # === 可视化核心逻辑 ===
            if is_rendering:
                self.env.render()
                time.sleep(ANIMATION_SPEED)  # 暂停一下让人眼看清
            # ===================

            action = self.agent.take_action(state)
            obs_, reward, done = self.env.step(action)
            next_state = self._obs_to_one_hot(obs_)

            # 实时更新网络
            self.agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            if done:
                break

        self.rewards.append(ep_reward)

        # 打印进度

        avg = np.mean(self.rewards[-10:])
        print(f"Episode {self.episode} | Reward: {ep_reward:.2f} | Avg(10): {avg:.2f}")

        self.episode += 1
        # 自动进行下一回合
        self.env.after(10, self.train)

    def _plot(self):
        """绘制结果曲线"""
        plt.figure(figsize=(8, 5))
        plt.plot(self.rewards, alpha=0.3, label='Raw')
        # 平滑处理
        kernel = np.ones(20) / 20
        smooth = np.convolve(self.rewards, kernel, mode='valid')
        plt.plot(smooth, color='red', label='Smoothed')
        plt.title('Actor-Critic Training')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()


# ============================================================
# 🚀 运行入口
# ============================================================
if __name__ == "__main__":
    # 随机种子
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # 初始化
    env = Maze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化智能体 (针对 4x4 One-Hot 优化的参数)
    agent = ActorCritic(
        state_dim=MAZE_H * MAZE_W,
        hidden_dim=64,
        action_dim=4,
        actor_lr=0.002,
        critic_lr=0.01,
        gamma=0.95,
        device=device
    )

    print("🚀 开始训练...")
    print(f"👀 每 {SHOW_EVERY} 回合演示一次移动过程")

    trainer = Trainer(env, agent)
    env.after(100, trainer.train)  # 100ms 后启动 train
    env.mainloop()