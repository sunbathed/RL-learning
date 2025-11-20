# -*- coding: utf-8 -*-
"""

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
    def __init__(self):
        super().__init__()
        self.title("SingleTruckMaze - test4 curve style v4")
        self.geometry(f"{MAZE_W * UNIT}x{MAZE_H * UNIT}")

        # 动作：0上 1下 2右 3左 4停
        self.action_space = ['u', 'd', 'r', 'l', 's']
        self.n_actions = len(self.action_space)

        self._build_maze()

        self.load_state = "empty"
        self.prev_rc = None

        # ===== reward（完全保持你原来的）=====
        self.step_penalty = -0.02
        self.move_bonus = +0.05
        self.collision_penalty = -1.0
        self.reward_load = +8.0
        self.reward_unload = +12.0

        # 轨迹热图（虽然 v4 不画，可不删）
        self.visit_map = np.zeros((MAZE_H, MAZE_W), dtype=np.int32)

        # BFS 势函数初始化
        self._init_rc_refs()
        self._build_bfs_distance()
        self.is_done = False

    # ---------------------------------------------------
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg="white",
                                height=MAZE_H*UNIT,
                                width=MAZE_W*UNIT)
        for c in range(0, MAZE_W*UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H*UNIT)
        for r in range(0, MAZE_H*UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W*UNIT, r)

        origin = np.array([UNIT/2, UNIT/2])

        # 障碍 1
        obs1 = origin + np.array([UNIT*1, UNIT*2])
        self.obsA = self.canvas.create_rectangle(
            obs1[0]-25, obs1[1]-25, obs1[0]+25, obs1[1]+25,
            fill="black"
        )

        # 障碍 2
        obs2 = origin + np.array([UNIT*2, UNIT*1])
        self.obsB = self.canvas.create_rectangle(
            obs2[0]-25, obs2[1]-25, obs2[0]+25, obs2[1]+25,
            fill="black"
        )

        # 装载区
        lz = origin + np.array([UNIT*2, UNIT*2])
        self.lz = self.canvas.create_rectangle(
            lz[0]-25, lz[1]-25, lz[0]+25, lz[1]+25,
            fill="lightblue"
        )

        # 卸载区
        uz = origin + np.array([UNIT*4, UNIT*4])
        self.uz = self.canvas.create_rectangle(
            uz[0]-25, uz[1]-25, uz[0]+25, uz[1]+25,
            fill="orange"
        )

        # 车辆
        pos = origin
        self.truck_item = self.canvas.create_rectangle(
            pos[0]-25, pos[1]-25, pos[0]+25, pos[1]+25,
            fill="red"
        )

        self.canvas.pack()

    # ---------------------------------------------------
    def _coords_to_rc(self, coords):
        x1, y1, x2, y2 = coords
        cx, cy = (x1+x2)/2, (y1+y2)/2
        return int(round((cy - UNIT/2)/UNIT)), int(round((cx - UNIT/2)/UNIT))

    # ---------------------------------------------------
    def _init_rc_refs(self):
        self.loading_rc = self._coords_to_rc(self.canvas.coords(self.lz))
        self.unloading_rc = self._coords_to_rc(self.canvas.coords(self.uz))
        self.obstacles_rc = {
            self._coords_to_rc(self.canvas.coords(self.obsA)),
            self._coords_to_rc(self.canvas.coords(self.obsB)),
        }
        self.start_rc = (0, 0)

    # ---------------------------------------------------
    def _bfs_from(self, target):
        H, W = MAZE_H, MAZE_W
        dist = [[999]*W for _ in range(H)]
        from collections import deque
        q = deque([target])
        dist[target[0]][target[1]] = 0
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W:
                    if (nr,nc) in self.obstacles_rc: continue
                    if dist[nr][nc] > dist[r][c] + 1:
                        dist[nr][nc] = dist[r][c] + 1
                        q.append((nr,nc))
        return dist

    def _build_bfs_distance(self):
        self.dist_to_loading = self._bfs_from(self.loading_rc)
        self.dist_to_unloading = self._bfs_from(self.unloading_rc)

    # ---------------------------------------------------
    def reset(self):
        origin = np.array([UNIT/2, UNIT/2])
        self.canvas.coords(self.truck_item,
                           origin[0]-25, origin[1]-25,
                           origin[0]+25, origin[1]+25)
        self.is_done = False
        self.load_state = "empty"
        self.prev_rc = None
        self.visit_map[:] = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array(self._coords_to_rc(self.canvas.coords(self.truck_item)),
                        dtype=np.float32)

    # ---------------------------------------------------
    def step(self, action):

        r, c = self._coords_to_rc(self.canvas.coords(self.truck_item))
        nr, nc = r, c
        moved = False
        collision = False

        if action != 4:
            if action == 0 and r>0: nr -=1; moved=True
            elif action == 1 and r<MAZE_H-1: nr +=1; moved=True
            elif action == 2 and c<MAZE_W-1: nc +=1; moved=True
            elif action == 3 and c>0: nc -=1; moved=True

        if (nr,nc) in self.obstacles_rc:
            nr, nc = r, c
            collision = True
            moved = False

        # ============= reward =============
        reward = self.step_penalty
        if moved: reward += self.move_bonus

        # --- BFS tanh 势函数
        if self.load_state == "empty":
            diff = self.dist_to_loading[r][c] - self.dist_to_loading[nr][nc]
        else:
            diff = self.dist_to_unloading[r][c] - self.dist_to_unloading[nr][nc]

        reward += 0.3 * np.tanh(diff)

        # ---- Anti-Oscillation：来回跳惩罚 ----
        # 如果当前点 == 上一回合的点 → 来回横跳
        if self.prev_rc == (nr, nc):
            reward -= 0.1          # ⭐ 加重惩罚，强制阻止 oscillation

        self.prev_rc = (r, c)      # 更新“上一个点”

        if collision:
            reward += self.collision_penalty

        self.canvas.move(self.truck_item, (nc-c)*UNIT, (nr-r)*UNIT)
        self.visit_map[nr][nc] += 1

        # 装载
        if (nr,nc)==self.loading_rc and self.load_state=="empty":
            reward += self.reward_load
            self.load_state = "full"

        # 卸载
        done = False
        if (nr,nc)==self.unloading_rc and self.load_state=="full":
            reward += self.reward_unload
            self.load_state = "empty"
            self.is_done = True
            done = True

        return self._get_obs(), float(reward), done, {}

    def render(self, delay=0.01):
        time.sleep(delay)
        self.update()


# ====================================================
# Policy / Value
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
    def __init__(self, state_dim, hidden_dim, n_actions,
                 actor_lr, critic_lr, gamma, device,
                 entropy_coef=0.05):

        self.actor = PolicyNet(state_dim, hidden_dim, n_actions).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.device = device
        self.entropy_coef = entropy_coef

    # --------- test4 + anti-oscillation 探索策略 -------
    def take_action(self, state_vec, episode):

        state = torch.tensor(state_vec, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)

        # ---- Phase 1：完全探索（0~250）----
        if episode <= 250:
            return dist.sample().item()

        # ---- Phase 2：逐步减小探索（250~400）----
        elif episode <= 400:
            explore_prob = max(0.1, 1.0 - (episode - 250) / 150)
            if random.random() < explore_prob:
                return dist.sample().item()
            return torch.argmax(probs).item()

        # ---- Phase 3：稳定阶段（>400）：5% 探索 → 防止横跳 ----
        else:
            explore_prob = 0.05     # ⭐ 从 1% → 5%
            if random.random() < explore_prob:
                return dist.sample().item()
            return torch.argmax(probs).item()

    # ---------------------------------------------------
    def update(self, traj):
        states = torch.tensor(np.array(traj["states"]),
                              dtype=torch.float32, device=self.device)
        actions = torch.tensor(traj["actions"], dtype=torch.int64,
                               device=self.device).view(-1, 1)
        rewards = torch.tensor(traj["rewards"], dtype=torch.float32,
                               device=self.device).view(-1, 1)
        next_states = torch.tensor(np.array(traj["next_states"]),
                                   dtype=torch.float32, device=self.device)
        dones = torch.tensor(traj["dones"], dtype=torch.float32,
                             device=self.device).view(-1, 1)

        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

            # ⭐ test4 风格 clamp（更稳定）
            td_target = torch.clamp(td_target, -5.0, 5.0)

        v = self.critic(states)
        td_delta = td_target - v

        probs = self.actor(states)
        log_probs = torch.log(probs.gather(1, actions) + 1e-8)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True).mean()

        actor_loss = torch.mean(-log_probs * td_delta.detach()) \
                     - self.entropy_coef * entropy
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
    def __init__(self, env, agent, max_episodes, max_steps):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.episode = 0
        self.return_list = []

    def _obs_to_state_vec(self, obs):
        r, c = obs
        return np.array([r/(MAZE_H-1), c/(MAZE_W-1)], dtype=np.float32)

    # ---------------------------------------------------
    def train_one_episode(self):
        obs = self.env.reset()
        state = self._obs_to_state_vec(obs)

        traj = {"states": [], "actions": [], "rewards": [],
                "next_states": [], "dones": []}

        ep_ret = 0.0

        for _ in range(self.max_steps):
            self.env.render()
            action = self.agent.take_action(state, self.episode)
            obs_next, reward, done, _ = self.env.step(action)
            next_state = self._obs_to_state_vec(obs_next)

            traj["states"].append(state)
            traj["actions"].append(action)
            traj["rewards"].append(reward)
            traj["next_states"].append(next_state)
            traj["dones"].append(done)

            ep_ret += reward
            state = next_state

            if done:
                break

        self.agent.update(traj)
        return ep_ret

    # ---------------------------------------------------
    def run(self):
        if self.episode >= self.max_episodes:
            self.plot_curve()
            print("训练完成")
            return

        # --- 动态 gamma（test4 风格）---
        decay_start = int(self.max_episodes * 0.7)
        if self.episode >= decay_start:
            ratio = (self.episode - decay_start) / (self.max_episodes - decay_start)
            self.agent.gamma = max(0.85, 0.9 - 0.05 * ratio)

        # --- 熵衰减（不动 test4 结构）---
        if self.episode > 250:
            self.agent.entropy_coef *= 0.995
            self.agent.entropy_coef = max(0.01, self.agent.entropy_coef)
        if self.episode > 400:
            self.agent.entropy_coef = 0.01

        ep_ret = self.train_one_episode()
        self.return_list.append(ep_ret)

        if (self.episode+1)%20==0:
            avg_ret = np.mean(self.return_list[-20:])
            print(f"Ep {self.episode+1} | Ret={ep_ret:.2f} | Avg={avg_ret:.2f} | "
                  f"gamma={self.agent.gamma:.3f}")

        self.episode += 1
        self.env.after(30, self.run)

    # ---------------------------------------------------
    def plot_curve(self):
        os.makedirs("results", exist_ok=True)

        def smooth(x, w=20):
            if len(x)<w: return x
            c = np.cumsum([0]+x)
            sm = (c[w:] - c[:-w]) / w
            return [sm[0]]*(len(x)-len(sm)) + sm.tolist()

        plt.figure(figsize=(8,5))
        plt.plot(smooth(self.return_list, 15), lw=2)
        plt.title("OneCar v4 - Test4 Curve Style + Anti-Oscillation")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.grid(alpha=0.3)

        path = "results/onecar_v4_curve.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"曲线已保存：{path}")


# ====================================================
# main
# ====================================================
if __name__ == "__main__":

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    env = SingleTruckMaze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SingleAgentActorCritic(
        state_dim=2,
        hidden_dim=128,
        n_actions=5,
        actor_lr=1e-3,
        critic_lr=2e-3,
        gamma=0.9,
        device=device,
        entropy_coef=0.05,
    )

    trainer = Trainer(env, agent, max_episodes=600, max_steps=80)
    env.after(10, trainer.run)
    env.mainloop()
