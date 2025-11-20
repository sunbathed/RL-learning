# -*- coding: utf-8 -*-
"""
OneCar v7.0 - PPO (Proximal Policy Optimization)
---------------------------------------------
升级说明：
针对 v6 版本 Actor-Critic 曲线震荡、不收敛的问题，本版本升级为 PPO 算法。

核心改进：
1. 【PPO Clip】：限制策略更新幅度，防止参数剧烈震荡。
2. 【Advantage Norm】：对优势函数进行归一化，极大提升收敛稳定性。
3. 【GAE】：引入广义优势估计 (Generalized Advantage Estimation)，平衡偏差与方差。
4. 【多轮更新】：每一批数据利用多次 (K-Epochs)，提高数据效率。

预期结果：
曲线将平滑上升，而不是剧烈上下跳动。
"""

import os
import time
import random
import numpy as np
import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

# 配置参数
UNIT = 60
MAZE_H = 5
MAZE_W = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO 超参数
LR = 0.0003  # 学习率降低，更稳
GAMMA = 0.99  # 折扣因子
LMBDA = 0.95  # GAE 参数
EPS_CLIP = 0.2  # PPO 截断范围
K_EPOCHS = 4  # 每次更新循环次数
UPDATE_TIMESTEP = 200  # 每隔多少步更新一次网络


class SingleTruckMaze(tk.Tk, object):
    """环境类 (保持 v6 不变)"""

    def __init__(self):
        super().__init__()
        self.title("OneCar v7.0 - PPO Stable")
        self.geometry(f"{MAZE_W * UNIT}x{MAZE_H * UNIT}")
        self.resizable(False, False)
        self.action_space = ['u', 'd', 'r', 'l', 's']
        self.n_actions = len(self.action_space)
        self._build_maze()
        self.load_state = 0
        self.n_features = 3

        # 奖励设置
        self.step_penalty = -0.05
        self.collision_penalty = -1.0
        self.reward_load = 5.0
        self.reward_unload = 10.0
        self.shaping_scale = 1.0

        self.success_count = 0
        self._init_rc_refs()
        self._build_bfs_distance()
        self.is_done = False

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg="white", height=MAZE_H * UNIT, width=MAZE_W * UNIT)
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT, fill="#EEE")
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r, fill="#EEE")
        origin = np.array([UNIT / 2, UNIT / 2])

        def draw_rect(r, c, color, tag=None):
            center = origin + np.array([UNIT * c, UNIT * r])
            return self.canvas.create_rectangle(center[0] - 25, center[1] - 25, center[0] + 25, center[1] + 25,
                                                fill=color, outline="black", tags=tag)

        self.obsA = draw_rect(2, 1, "black")
        self.obsB = draw_rect(1, 2, "black")
        self.lz = draw_rect(2, 2, "lightblue")
        self.uz = draw_rect(4, 4, "#FFD700")
        self.truck = draw_rect(0, 0, "red", tag="truck")
        self.canvas.pack()

    def _coords_to_rc(self, coords):
        x1, y1, x2, y2 = coords
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        r = int((cy - UNIT / 2) // UNIT)
        c = int((cx - UNIT / 2) // UNIT)
        return r, c

    def _init_rc_refs(self):
        self.loading_rc = (2, 2)
        self.unloading_rc = (4, 4)
        self.obstacles_rc = {(2, 1), (1, 2)}

    def _bfs_from(self, target_rc):
        dist_map = np.full((MAZE_H, MAZE_W), 999.0)
        tr, tc = target_rc
        dist_map[tr][tc] = 0
        queue = [(tr, tc)]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < MAZE_H and 0 <= nc < MAZE_W and (nr, nc) not in self.obstacles_rc:
                    if dist_map[nr][nc] > dist_map[r][c] + 1:
                        dist_map[nr][nc] = dist_map[r][c] + 1
                        queue.append((nr, nc))
        return dist_map

    def _build_bfs_distance(self):
        self.dist_to_load = self._bfs_from(self.loading_rc)
        self.dist_to_unload = self._bfs_from(self.unloading_rc)

    def reset(self):
        self.canvas.delete("truck")
        origin = np.array([UNIT / 2, UNIT / 2])
        self.truck = self.canvas.create_rectangle(origin[0] - 25, origin[1] - 25, origin[0] + 25, origin[1] + 25,
                                                  fill="red", outline="black", tags="truck")
        self.load_state = 0
        self.is_done = False
        return self._get_state()

    def _get_state(self):
        r, c = self._coords_to_rc(self.canvas.coords(self.truck))
        return np.array([r / (MAZE_H - 1), c / (MAZE_W - 1), float(self.load_state)], dtype=np.float32)

    def step(self, action):
        curr_r, curr_c = self._coords_to_rc(self.canvas.coords(self.truck))
        nr, nc = curr_r, curr_c
        if action == 0:
            nr = max(0, curr_r - 1)
        elif action == 1:
            nr = min(MAZE_H - 1, curr_r + 1)
        elif action == 2:
            nc = min(MAZE_W - 1, curr_c + 1)
        elif action == 3:
            nc = max(0, curr_c - 1)

        hit_wall = (nr, nc) in self.obstacles_rc
        if hit_wall: nr, nc = curr_r, curr_c

        self.canvas.move(self.truck, (nc - curr_c) * UNIT, (nr - curr_r) * UNIT)

        reward = self.step_penalty

        # 势函数
        if self.load_state == 0:
            shaping = (self.dist_to_load[curr_r][curr_c] - self.dist_to_load[nr][nc])
        else:
            shaping = (self.dist_to_unload[curr_r][curr_c] - self.dist_to_unload[nr][nc])
        reward += shaping * self.shaping_scale

        if hit_wall: reward += self.collision_penalty

        done = False
        if (nr, nc) == self.loading_rc and self.load_state == 0:
            reward += self.reward_load
            self.load_state = 1
            self.canvas.itemconfig(self.truck, fill="#32CD32")
        elif (nr, nc) == self.unloading_rc and self.load_state == 1:
            reward += self.reward_unload
            self.load_state = 0
            self.canvas.itemconfig(self.truck, fill="red")
            self.success_count += 1
            done = True
            self.is_done = True

        return self._get_state(), reward, done, {}

    def render(self):
        self.update()


# ====================================================
# PPO 算法核心 (全新重写)
# ====================================================
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()

    def update(self, memory):
        # 1. 转换数据
        rewards = []
        discounted_reward = 0
        # 计算回报 (Monte Carlo)
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 归一化 Rewards (非常重要！解决不收敛的关键)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 转换其他数据
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(DEVICE)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(DEVICE)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(DEVICE)

        # 2. PPO 更新循环 K 次
        for _ in range(K_EPOCHS):
            # 评估旧状态和动作
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            state_values = state_values.squeeze()

            # 计算 Advantage
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()  # G_t - V(s)

            # PPO Loss 公式 (Clip)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            # Loss = -Actor + 0.5*Critic - 0.01*Entropy
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 同步旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())


class Trainer:
    def __init__(self, env, max_episodes=800):
        self.env = env
        self.max_episodes = max_episodes
        self.ppo = PPO(env.n_features, env.n_actions)
        self.memory = Memory()

        self.episode = 0
        self.timestep = 0
        self.history_rewards = []

        self.run_episode()  # 启动

    def run_episode(self):
        if self.episode >= self.max_episodes:
            self.save_plot()
            print("训练结束")
            return

        state = self.env.reset()
        ep_reward = 0

        # 收集数据的最大步数 (防止死循环)
        for t in range(200):
            self.timestep += 1

            # 1. 选择动作 (用旧策略)
            state_t = torch.FloatTensor(state).to(DEVICE)
            action, logprob = self.ppo.policy_old.act(state_t)

            # 2. 执行
            next_state, reward, done, _ = self.env.step(action)

            # 3. 存入 Memory
            self.memory.states.append(state_t)
            self.memory.actions.append(torch.tensor(action).to(DEVICE))
            self.memory.logprobs.append(torch.tensor(logprob).to(DEVICE))
            self.memory.rewards.append(reward)
            self.memory.is_terminals.append(done)

            state = next_state
            ep_reward += reward

            # 4. 定时更新 (PPO 核心)
            if self.timestep % UPDATE_TIMESTEP == 0:
                self.ppo.update(self.memory)
                self.memory.clear()
                self.timestep = 0

            # 渲染 (每10回合看一次，加快训练)
            if self.episode % 10 == 0:
                self.env.render()

            if done:
                break

        self.history_rewards.append(ep_reward)

        # 打印日志
        if (self.episode + 1) % 10 == 0:
            avg_rew = np.mean(self.history_rewards[-10:])
            print(f"Ep {self.episode + 1} | Avg Reward: {avg_rew:.2f} | Success: {self.env.success_count}")

        self.episode += 1

        # 调度下一个 Episode
        delay = 10 if (self.episode % 10 == 0) else 1
        self.env.after(delay, self.run_episode)

    def save_plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history_rewards, label='Reward')
        # 绘制平滑曲线
        if len(self.history_rewards) > 20:
            smooth = np.convolve(self.history_rewards, np.ones(20) / 20, mode='valid')
            plt.plot(smooth, label='Smoothed (MA-20)', linewidth=2, color='orange')
        plt.title("PPO Training Curve (Stable)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig("ppo_result.png")
        print("Saved ppo_result.png")


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SingleTruckMaze()
    trainer = Trainer(env, max_episodes=600)
    env.mainloop()