# -*- coding: utf-8 -*-
"""
Actor-Critic Maze v2.0（稳定增强版）
-----------------------------------------
增强内容：
 - Advantage Normalize（优势归一化）
 - TD-Target clipping（防爆）
 - 动态 gamma & 熵衰减
 - 改进 BFS 势函数奖励（更平滑）
 - 加入移动奖励、重复访问惩罚
 - 修复 log_prob 维度
"""

import numpy as np
import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import matplotlib.pyplot as plt
import os

UNIT = 40
MAZE_H = 4
MAZE_W = 4


# ============================================================
# 环境定义
# ============================================================
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'r', 'l']
        self.n_actions = len(self.action_space)
        self.title('Actor-Critic Maze v2.0 Stable')
        self.geometry(f'{MAZE_W * UNIT}x{MAZE_H * UNIT}')
        self._build_maze()

        self.step_penalty = -0.01
        self.move_bonus = +0.02
        self.is_done = False
        self.prev_rc = None
        self.visit_count = {}

        self._init_rc_refs()
        self._build_bfs_distance()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 网格
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT)
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r)

        origin = np.array([20, 20])

        # 障碍
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black'
        )
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black'
        )

        # 终点
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow'
        )

        # 起点
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )
        self.canvas.pack()

    def _coords_to_rc(self, coords):
        x1, y1, x2, y2 = coords
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return int(round((cy - UNIT/2) / UNIT)), int(round((cx - UNIT/2) / UNIT))

    def _init_rc_refs(self):
        self.start_rc = self._coords_to_rc(self.canvas.coords(self.rect))
        self.goal_rc = self._coords_to_rc(self.canvas.coords(self.oval))
        self.walls_rc = {
            self._coords_to_rc(self.canvas.coords(self.hell1)),
            self._coords_to_rc(self.canvas.coords(self.hell2)),
        }

    def _build_bfs_distance(self):
        H, W = MAZE_H, MAZE_W
        dist = [[999] * W for _ in range(H)]
        from collections import deque
        q = deque([self.goal_rc])
        dist[self.goal_rc[0]][self.goal_rc[1]] = 0

        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in self.walls_rc:
                    if dist[nr][nc] > dist[r][c] + 1:
                        dist[nr][nc] = dist[r][c] + 1
                        q.append((nr,nc))

        self.dist_map = dist

    def reset(self):
        origin = np.array([20, 20])
        self.canvas.coords(self.rect,
                           origin[0]-15, origin[1]-15,
                           origin[0]+15, origin[1]+15)

        self.is_done = False
        self.prev_rc = None
        self.visit_count = {}
        self.update()
        return self.canvas.coords(self.rect)

    def step(self, action):
        if self.is_done:
            return 'terminal', 0.0, True

        s = self.canvas.coords(self.rect)
        rc = self._coords_to_rc(s)

        move = [0, 0]
        if action == 0 and rc[0] > 0: move[1] -= UNIT
        elif action == 1 and rc[0] < MAZE_H - 1: move[1] += UNIT
        elif action == 2 and rc[1] < MAZE_W - 1: move[0] += UNIT
        elif action == 3 and rc[1] > 0: move[0] -= UNIT

        self.canvas.move(self.rect, move[0], move[1])
        s_ = self.canvas.coords(self.rect)
        rc_ = self._coords_to_rc(s_)

        # 终点
        if rc_ == self.goal_rc:
            self.is_done = True
            return 'terminal', 5.0, True
        if rc_ in self.walls_rc:
            self.is_done = True
            return 'terminal', -1.0, True

        # 奖励
        reward = self.step_penalty

        if move != [0,0]:
            reward += self.move_bonus

        # BFS 势函数（平滑）
        d_prev = self.dist_map[rc[0]][rc[1]]
        d_now = self.dist_map[rc_[0]][rc_[1]]

        diff = d_prev - d_now
        reward += 0.3 * np.tanh(diff)

        # 来回惩罚
        if self.prev_rc == rc_:
            reward -= 0.05
        self.prev_rc = rc

        # 重复访问
        self.visit_count[rc_] = self.visit_count.get(rc_, 0) + 1
        if self.visit_count[rc_] > 3:
            reward -= 0.05

        return s_, reward, False

    def render(self):
        time.sleep(0.01)
        self.update()


# ============================================================
# Actor / Critic
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================
# Actor-Critic（增强版）
# ============================================================
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, gamma, device, entropy_coef=0.05):

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device

    def take_action(self, state_vec):
        state = torch.tensor(state_vec, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def update(self, traj):
        states = torch.tensor(np.array(traj['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(traj['actions'], dtype=torch.int64).view(-1,1).to(self.device)
        rewards = torch.tensor(traj['rewards'], dtype=torch.float32).view(-1,1).to(self.device)
        next_states = torch.tensor(np.array(traj['next_states']), dtype=torch.float32).to(self.device)
        dones = torch.tensor(traj['dones'], dtype=torch.float32).view(-1,1).to(self.device)

        # TD Target（稳定版：clip）
        with torch.no_grad():
            next_v = self.critic(next_states)
            td_target = rewards + self.gamma * next_v * (1 - dones)
            td_target = torch.clamp(td_target, -10.0, 10.0)

        v = self.critic(states)
        advantage = td_target - v

        # ---- Advantage Normalization ----
        adv = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        # ---- Actor loss ----
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions.squeeze()).view(-1,1)

        entropy = dist.entropy().mean()
        actor_loss = torch.mean(-log_probs * adv.detach()) \
                     - self.entropy_coef * entropy

        # ---- Critic loss ----
        critic_loss = F.mse_loss(v, td_target.detach())

        # --- Update ---
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()


# ============================================================
# Trainer
# ============================================================
class Trainer:
    def __init__(self, env, agent, max_episodes=600, max_steps=80):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.episode = 0
        self.return_list = []

    def _obs_to_state_vec(self, obs):
        if obs == 'terminal':
            return np.zeros(2, dtype=np.float32)
        r, c = self.env._coords_to_rc(obs)
        return np.array([r/(MAZE_H-1), c/(MAZE_W-1)], dtype=np.float32)

    def train_one_episode(self):
        obs = self.env.reset()
        state = self._obs_to_state_vec(obs)

        traj = {'states': [], 'actions': [], 'rewards': [],
                'next_states': [], 'dones': []}

        ep_ret = 0

        for _ in range(self.max_steps):
            self.env.render()

            action = self.agent.take_action(state)
            obs_, reward, done = self.env.step(action)
            next_state = self._obs_to_state_vec(obs_)

            traj['states'].append(state.copy())
            traj['actions'].append(action)
            traj['rewards'].append(reward)
            traj['next_states'].append(next_state.copy())
            traj['dones'].append(done)

            state = next_state
            ep_ret += reward

            if done:
                break

        if traj['states']:
            self.agent.update(traj)

        return ep_ret

    def run(self):
        if self.episode >= self.max_episodes:
            print("训练结束 ✔")
            self._plot()
            return

        # 动态熵衰减
        if self.episode <= 200:
            self.agent.entropy_coef = 0.05
        elif self.episode <= 350:
            self.agent.entropy_coef *= 0.995
            self.agent.entropy_coef = max(0.02, self.agent.entropy_coef)
        else:
            self.agent.entropy_coef = 0.01

        ep_ret = self.train_one_episode()
        self.return_list.append(ep_ret)

        if (self.episode + 1) % 20 == 0:
            avg20 = np.mean(self.return_list[-20:])
            print(f"Ep {self.episode+1}/{self.max_episodes} | "
                  f"Ret={ep_ret:.2f} | Avg20={avg20:.2f} | "
                  f"entropy={self.agent.entropy_coef:.3f}")

        self.episode += 1
        self.env.after(40, self.run)

    def _plot(self):
        plt.figure(figsize=(8,5))
        def ma(x,w=15):
            x = np.array(x)
            if len(x)<w: return x
            c = np.cumsum(np.insert(x,0,0))
            s = (c[w:]-c[:-w])/w
            head = [s[0]] * (len(x)-len(s))
            return head + s.tolist()

        plt.plot(ma(self.return_list), lw=2)
        plt.grid(alpha=.3)
        plt.title("AC Maze v2.0 Stable Return Curve")
        plt.savefig("maze_v2_return.png", dpi=300)
        plt.show()


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    env = Maze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = ActorCritic(
        state_dim=2,
        hidden_dim=128,
        action_dim=4,
        actor_lr=1e-3,
        critic_lr=2e-3,
        gamma=0.9,
        device=device,
        entropy_coef=0.05,
    )

    trainer = Trainer(env, agent, max_episodes=600, max_steps=80)
    env.after(100, trainer.run)
    env.mainloop()
