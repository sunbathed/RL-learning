# -*- coding: utf-8 -*-
"""
OneCar v6.1 — 最终稳定版（含 has_load + 修复 Loss + 极速训练模式）
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

UNIT = 60
MAZE_H = 5
MAZE_W = 5


# ====================================================
# 环境
# ====================================================
class SingleTruckMaze(tk.Tk, object):
    """单车矿区环境"""

    def __init__(self):
        super().__init__()
        self.title("OneCar v6.1 - Stable")
        self.geometry(f"{MAZE_W * UNIT}x{MAZE_H * UNIT}")

        # 上下左右停
        self.action_space = ['u', 'd', 'r', 'l', 's']
        self.n_actions = len(self.action_space)

        self._build_maze()

        # 载重状态
        self.load_state = "empty"  # empty / full

        # 奖励
        self.step_penalty = -0.02
        self.move_bonus = +0.05
        self.collision_penalty = -1.0
        self.reward_load = +8.0
        self.reward_unload = +12.0

        # BFS-shaping 系数
        self.shaping_coef = 0.8

        # 热力图
        self.visit_map = np.zeros((MAZE_H, MAZE_W), dtype=np.int32)

        # 成功统计
        self.success_count = 0
        self.total_count = 0

        # 预计算 BFS
        self._init_rc_refs()
        self._build_bfs_distance()

        self.is_done = False

    # --------------------------------------------------------
    # 地图
    # --------------------------------------------------------
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg="white",
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 网格线
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT)
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r)

        origin = np.array([UNIT / 2, UNIT / 2])

        # 障碍
        obsA = origin + np.array([UNIT * 1, UNIT * 2])
        self.obsA = self.canvas.create_rectangle(
            obsA[0] - 25, obsA[1] - 25,
            obsA[0] + 25, obsA[1] + 25,
            fill="black"
        )

        obsB = origin + np.array([UNIT * 2, UNIT * 1])
        self.obsB = self.canvas.create_rectangle(
            obsB[0] - 25, obsB[1] - 25,
            obsB[0] + 25, obsB[1] + 25,
            fill="black"
        )

        # 装载区
        lz = origin + np.array([UNIT * 2, UNIT * 2])
        self.lz = self.canvas.create_rectangle(
            lz[0] - 25, lz[1] - 25,
            lz[0] + 25, lz[1] + 25,
            fill="lightblue"
        )

        # 卸载区
        uz = origin + np.array([UNIT * 4, UNIT * 4])
        self.uz = self.canvas.create_rectangle(
            uz[0] - 25, uz[1] - 25,
            uz[0] + 25, uz[1] + 25,
            fill="orange"
        )

        # 起点
        st = origin + np.array([0, 0])
        self.truck_item = self.canvas.create_rectangle(
            st[0] - 25, st[1] - 25,
            st[0] + 25, st[1] + 25,
            fill="red"
        )

        self.canvas.pack()

    def _coords_to_rc(self, coords):
        x1, y1, x2, y2 = coords
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return int(round((cy - UNIT/2) / UNIT)), int(round((cx - UNIT/2) / UNIT))

    def _init_rc_refs(self):
        self.loading_rc = self._coords_to_rc(self.canvas.coords(self.lz))
        self.unloading_rc = self._coords_to_rc(self.canvas.coords(self.uz))
        self.obstacles_rc = {
            self._coords_to_rc(self.canvas.coords(self.obsA)),
            self._coords_to_rc(self.canvas.coords(self.obsB))
        }
        self.start_rc = (0, 0)

    # --------------------------------------------------------
    # BFS
    # --------------------------------------------------------
    def _bfs_from(self, target):
        H, W = MAZE_H, MAZE_W
        dist = [[999] * W for _ in range(H)]

        from collections import deque
        q = deque([target])
        dist[target[0]][target[1]] = 0

        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) in self.obstacles_rc:
                        continue
                    if dist[nr][nc] > dist[r][c] + 1:
                        dist[nr][nc] = dist[r][c] + 1
                        q.append((nr, nc))
        return dist

    def _build_bfs_distance(self):
        self.dist_to_loading = self._bfs_from(self.loading_rc)
        self.dist_to_unloading = self._bfs_from(self.unloading_rc)

    # --------------------------------------------------------
    # reset
    # --------------------------------------------------------
    def reset(self):
        origin = np.array([UNIT/2, UNIT/2])
        self.canvas.coords(self.truck_item,
                           origin[0]-25, origin[1]-25,
                           origin[0]+25, origin[1]+25)

        self.load_state = "empty"
        self.is_done = False

        self.total_count += 1
        self.visit_map[:] = 0

        self.update()
        return self._get_obs()

    def _get_obs(self):
        """返回 (r, c)"""
        return np.array(self._coords_to_rc(self.canvas.coords(self.truck_item)),
                        dtype=np.float32)

    # --------------------------------------------------------
    # step
    # --------------------------------------------------------
    def step(self, action):

        r, c = self._coords_to_rc(self.canvas.coords(self.truck_item))
        nr, nc = r, c
        moved = False
        collision = False

        # movement
        if action != 4:
            if action == 0 and r > 0: nr -= 1; moved = True
            elif action == 1 and r < MAZE_H-1: nr += 1; moved = True
            elif action == 2 and c <  MAZE_W-1: nc += 1; moved = True
            elif action == 3 and c > 0: nc -= 1; moved = True

        # hit obstacle
        if (nr, nc) in self.obstacles_rc:
            nr, nc = r, c
            collision = True
            moved = False

        # Reward
        reward = self.step_penalty
        if moved: reward += self.move_bonus
        if collision: reward += self.collision_penalty

        # BFS shaping
        if self.load_state == "empty":
            d_prev = self.dist_to_loading[r][c]
            d_now  = self.dist_to_loading[nr][nc]
        else:
            d_prev = self.dist_to_unloading[r][c]
            d_now  = self.dist_to_unloading[nr][nc]

        diff = d_prev - d_now

        # 反向势函数 (靠近 +0.8, 远离 -1.2)
        if diff > 0:
            reward += 0.8 * diff
        else:
            reward += 1.2 * diff  # negative

        # 方向一致性奖励
        if d_now < d_prev:
            reward += 0.1

        # 路径惩罚：重复访问
        self.visit_map[nr][nc] += 1
        if self.visit_map[nr][nc] > 2:
            reward -= 0.15

        # move
        self.canvas.move(self.truck_item, (nc-c)*UNIT, (nr-r)*UNIT)

        # 装载
        if (nr, nc) == self.loading_rc and self.load_state == "empty":
            reward += self.reward_load
            self.load_state = "full"

        # 卸载
        done = False
        if (nr, nc) == self.unloading_rc and self.load_state == "full":
            reward += self.reward_unload
            self.load_state = "empty"
            self.is_done = True
            done = True
            self.success_count += 1

        self.update()
        return self._get_obs(), float(reward), done, {}

    def render(self, delay=0.01):
        time.sleep(delay)
        self.update()


# ====================================================
# Actor / Critic
# ====================================================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
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


# ====================================================
# Actor-Critic
# ====================================================
class SingleAgentActorCritic:
    def __init__(self, state_dim, hidden_dim,
                 n_actions, actor_lr, critic_lr,
                 gamma, device, entropy_coef=0.05):

        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.actor = PolicyNet(state_dim, hidden_dim, n_actions).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

    # ------ sample 动作（无 argmax）------
    def take_action(self, state_vec, episode=None):
        state = torch.tensor(state_vec, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    # ------ update ------
    def update(self, traj):

        states = torch.tensor(np.array(traj["states"]),
                              dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(traj["actions"]),
                               dtype=torch.int64, device=self.device)
        rewards = torch.tensor(np.array(traj["rewards"]),
                               dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(np.array(traj["next_states"]),
                                   dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(traj["dones"]),
                             dtype=torch.float32, device=self.device).view(-1, 1)

        # Critic
        with torch.no_grad():
            next_v = self.critic(next_states)
            td_target = rewards + self.gamma * next_v * (1 - dones)
            td_target = torch.clamp(td_target, -30, 30)

        v = self.critic(states)
        td_delta = td_target - v  # [T,1]

        # 修复维度：让 log_prob 与 advantage 对齐
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions).view(-1, 1)  # [T,1]
        advantage = td_delta.detach()

        actor_loss = torch.mean(-log_probs * advantage) \
                     - self.entropy_coef * dist.entropy().mean()

        critic_loss = F.mse_loss(v, td_target.detach())

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()


# ====================================================
# Trainer
# ====================================================
class Trainer:
    def __init__(self, env, agent, max_episodes=650, max_steps=80,
                 fast_train=True, render_every=10):

        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps

        self.fast_train = fast_train
        self.render_every = render_every

        self.episode = 0
        self.return_list = []
        self.success_list = []
        self.heat_accum = np.zeros((MAZE_H, MAZE_W), dtype=np.int32)

    # ------ 状态向量：加入 has_load ------
    def _obs_to_state_vec(self, obs):
        r, c = obs.astype(np.float32)
        r_norm = r / (MAZE_H - 1)
        c_norm = c / (MAZE_W - 1)
        has_load = 1.0 if self.env.load_state == "full" else 0.0
        return np.array([r_norm, c_norm, has_load], dtype=np.float32)

    # --------------------------------------------------------
    def train_one_episode(self):
        prev_success = self.env.success_count

        obs = self.env.reset()
        state = self._obs_to_state_vec(obs)

        traj = {
            "states": [], "actions": [], "rewards": [],
            "next_states": [], "dones": []
        }

        ep_ret = 0

        for t in range(self.max_steps):

            # ----------- 极速训练渲染模式 ------------
            if (not self.fast_train) or (self.episode % self.render_every == 0):
                self.env.render(delay=0.01)
            else:
                self.env.update_idletasks()
                self.env.update()

            # 动作
            action = self.agent.take_action(state, episode=self.episode)

            obs_next, reward, done, info = self.env.step(action)
            next_state = self._obs_to_state_vec(obs_next)

            traj["states"].append(state.copy())
            traj["actions"].append(action)
            traj["rewards"].append(reward)
            traj["next_states"].append(next_state.copy())
            traj["dones"].append(done)

            state = next_state
            ep_ret += reward

            if done:
                break

        if traj["states"]:
            self.agent.update(traj)

        success = self.env.success_count - prev_success
        self.success_list.append(success)

        self.heat_accum += self.env.visit_map

        return ep_ret

    # --------------------------------------------------------
    def run(self):
        if self.episode >= self.max_episodes:
            print("训练结束 ✔")
            self._plot_and_save()
            return

        # 熵衰减
        if self.episode <= 200:
            self.agent.entropy_coef = 0.05
        elif 200 < self.episode <= 350:
            self.agent.entropy_coef *= 0.995
            self.agent.entropy_coef = max(0.02, self.agent.entropy_coef)
        else:
            self.agent.entropy_coef = 0.02

        ep_ret = self.train_one_episode()
        self.return_list.append(ep_ret)

        if (self.episode + 1) % 20 == 0:
            avg20 = np.mean(self.return_list[-20:])
            sr20 = np.mean(self.success_list[-20:])
            print(f"Ep {self.episode+1}/{self.max_episodes} | "
                  f"Ret={ep_ret:.2f} | Avg20={avg20:.2f} | "
                  f"Success20={sr20:.2f} | "
                  f"entropy={self.agent.entropy_coef:.3f}")

        self.episode += 1
        self.env.after(1, self.run)

    # --------------------------------------------------------
    def _plot_and_save(self):
        os.makedirs("", exist_ok=True)

        # 回报
        plt.figure(figsize=(8,4))
        def ma(x, w=20):
            x = np.array(x)
            if len(x) < w: return x
            c = np.cumsum(np.insert(x, 0, 0))
            s = (c[w:] - c[:-w]) / w
            head = [s[0]] * (len(x) - len(s))
            return head + s.tolist()
        plt.plot(ma(self.return_list), lw=2)
        plt.grid(alpha=.3)
        plt.title("OneCar v6.1 Return Curve")
        plt.savefig("success-results_v6p1/return.png", dpi=300)
        plt.close()

        # 成功率
        window=20
        sr=[]
        for i in range(len(self.success_list)):
            s=np.mean(self.success_list[max(0,i-window+1):i+1])
            sr.append(s)
        plt.figure(figsize=(8,4))
        plt.plot(sr,lw=2)
        plt.ylim(0,1)
        plt.grid(alpha=.3)
        plt.title("OneCar v6.1 Success Rate")
        plt.savefig("success-results_v6p1/success.png", dpi=300)
        plt.close()

        # 热力图
        plt.figure(figsize=(5,5))
        plt.imshow(self.heat_accum, cmap="hot", origin="upper")
        plt.colorbar()
        plt.xticks(range(MAZE_W))
        plt.yticks(range(MAZE_H))
        plt.title("OneCar v6.1 Heatmap")
        plt.savefig("success-results_v6p1/heatmap.png", dpi=300)
        plt.close()


# ====================================================
# main
# ====================================================
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    env = SingleTruckMaze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 状态维度 = 3（加入 has_load）
    state_dim = 3

    agent = SingleAgentActorCritic(
        state_dim=state_dim,
        hidden_dim=64,
        n_actions=env.n_actions,
        actor_lr=1e-3,
        critic_lr=5e-4,
        gamma=0.95,
        device=device,
        entropy_coef=0.05
    )

    trainer = Trainer(env, agent,
                      max_episodes=650,
                      max_steps=80,
                      fast_train=True,     # 极速训练开关
                      render_every=10)

    env.after(10, trainer.run)
    env.mainloop()
