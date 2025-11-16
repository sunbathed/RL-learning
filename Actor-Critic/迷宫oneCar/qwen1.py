# -*- coding: utf-8 -*-
"""
单车 Actor-Critic 示例（Phase-Based 最强版）
--------------------------------
目标：让智能体稳定学会
    空车 → 先到装载区 → 再到卸载区 → 结束回合

设计要点：
1. 两阶段势函数（BFS）：
   - 空车：朝装载区的 BFS 距离 dist_to_loading
   - 满载：朝卸载区的 BFS 距离 dist_to_unloading
2. 阶段奖励（Phase Completion Reward）：
   - 完成装载阶段：+8.0
   - 完成卸载阶段：+12.0
3. 基础奖励：
   - 每步 -0.02，成功移动 +0.05
   - 撞障碍 -1.0
4. 训练曲线保存到 results/onecar.png
"""

import os
import time
import random
import numpy as np
import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim import Adam

# ===================== 全局参数 =====================
UNIT = 60   # 每格像素
MAZE_H = 5  # 行
MAZE_W = 5  # 列


# ====================================================
# 环境：SingleTruckMaze
# ====================================================
class SingleTruckMaze(tk.Tk, object):
    """单车矿区环境（装载/卸载 + 障碍）"""

    def __init__(self):
        super().__init__()
        self.title("SingleTruckMaze - RL Mining Env")
        self.geometry(f"{MAZE_W * UNIT}x{MAZE_H * UNIT}")

        # 动作空间：0上/1下/2右/3左/4停
        self.action_space = ['u', 'd', 'r', 'l', 's']
        self.n_actions = len(self.action_space)

        # 画地图
        self._build_maze()

        # 载重状态：empty / full
        self.load_state = "empty"

        # ==== Reward 配置（核心）====
        # 基础步进相关
        self.step_penalty = -0.02      # 每步基础惩罚：防止无意义拖时间
        self.move_bonus = +0.05        # 成功移动奖励：鼓励前进
        self.collision_penalty = -1.0  # 撞障碍惩罚：明确告诉“不要撞”

        # Phase completion 奖励（子任务完成奖励）
        self.reward_load_phase = +8.0   # 完成装载阶段
        self.reward_unload_phase = +12.0  # 完成卸载阶段（整个回合）

        # 两阶段势函数系数
        self.shaping_coef_load = 1.0    # 空车 → 装载区
        self.shaping_coef_unload = 1.3  # 满载 → 卸载区

        # 终止标志
        self.is_done = False

        # 初始化坐标引用与 BFS 势函数
        self._init_rc_refs()
        self._build_bfs_distance()

    # ---------- 画地图 ----------
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

        # 固定障碍 A： (2,1)
        obsA_center = origin + np.array([UNIT * 1, UNIT * 2])
        self.obsA = self.canvas.create_rectangle(
            obsA_center[0] - UNIT / 2 + 5, obsA_center[1] - UNIT / 2 + 5,
            obsA_center[0] + UNIT / 2 - 5, obsA_center[1] + UNIT / 2 - 5,
            fill="black"
        )

        # 固定障碍 B： (1,2)
        obsB_center = origin + np.array([UNIT * 2, UNIT * 1])
        self.obsB = self.canvas.create_rectangle(
            obsB_center[0] - UNIT / 2 + 5, obsB_center[1] - UNIT / 2 + 5,
            obsB_center[0] + UNIT / 2 - 5, obsB_center[1] + UNIT / 2 - 5,
            fill="black"
        )

        # 装载区： (2,2)
        lz_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.lz = self.canvas.create_rectangle(
            lz_center[0] - UNIT / 2 + 5, lz_center[1] - UNIT / 2 + 5,
            lz_center[0] + UNIT / 2 - 5, lz_center[1] + UNIT / 2 - 5,
            fill="lightblue"
        )

        # 卸载区： (4,4)
        uz_center = origin + np.array([UNIT * 4, UNIT * 4])
        self.uz = self.canvas.create_rectangle(
            uz_center[0] - UNIT / 2 + 5, uz_center[1] - UNIT / 2 + 5,
            uz_center[0] + UNIT / 2 - 5, uz_center[1] + UNIT / 2 - 5,
            fill="orange"
        )

        # 单车起点： (0,0)
        start_pos = origin + np.array([UNIT * 0, UNIT * 0])
        self.truck_item = self.canvas.create_rectangle(
            start_pos[0] - UNIT / 2 + 5, start_pos[1] - UNIT / 2 + 5,
            start_pos[0] + UNIT / 2 - 5, start_pos[1] + UNIT / 2 - 5,
            fill="red"
        )

        self.canvas.pack()

    # ---------- 坐标转换 & 初始化 ----------
    def _coords_to_rc(self, coords):
        x1, y1, x2, y2 = coords
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        c = int(round((cx - UNIT / 2) / UNIT))
        r = int(round((cy - UNIT / 2) / UNIT))
        return r, c

    def _init_rc_refs(self):
        # 记录装载区、卸载区、障碍的位置（格坐标）
        self.loading_rc = self._coords_to_rc(self.canvas.coords(self.lz))
        self.unloading_rc = self._coords_to_rc(self.canvas.coords(self.uz))
        self.obstacles_rc = {
            self._coords_to_rc(self.canvas.coords(self.obsA)),
            self._coords_to_rc(self.canvas.coords(self.obsB))
        }
        self.start_rc = (0, 0)  # 起点坐标

    # ---------- BFS 势函数 ----------
    def _bfs_from(self, target_rc):
        """
        从 target_rc 出发做一次 BFS，得到每个格子到该点的最短步数。
        障碍格不可通行。
        """
        H, W = MAZE_H, MAZE_W
        dist = [[999] * W for _ in range(H)]
        from collections import deque

        q = deque([target_rc])
        r0, c0 = target_rc
        dist[r0][c0] = 0

        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) in self.obstacles_rc:
                        continue
                    if dist[nr][nc] > dist[r][c] + 1:
                        dist[nr][nc] = dist[r][c] + 1
                        q.append((nr, nc))
        return dist

    def _build_bfs_distance(self):
        """
        构建两张势函数距离图：
        - dist_to_loading   ：空车使用，鼓励靠近装载区
        - dist_to_unloading ：满载使用，鼓励靠近卸载区
        """
        self.dist_to_loading = self._bfs_from(self.loading_rc)
        self.dist_to_unloading = self._bfs_from(self.unloading_rc)

    # ---------- reset & 观测 ----------
    def reset(self):
        origin = np.array([UNIT / 2, UNIT / 2])
        start_pos = origin + np.array([UNIT * 0, UNIT * 0])
        self.canvas.coords(
            self.truck_item,
            start_pos[0] - UNIT / 2 + 5, start_pos[1] - UNIT / 2 + 5,
            start_pos[0] + UNIT / 2 - 5, start_pos[1] + UNIT / 2 - 5
        )

        self.is_done = False
        self.load_state = "empty"

        self.update()
        return self._get_obs()

    def _get_obs(self):
        """返回单车的 (r,c) 位置"""
        return np.array(self._coords_to_rc(self.canvas.coords(self.truck_item)),
                        dtype=np.float32)

    # ---------- step ----------
    def step(self, action):
        """
        action: int (0-4)
        返回：
            obs: 新状态
            reward: 即时奖励
            done: 是否完成卸载
            info: 详细信息
        """
        if self.is_done:
            return self._get_obs(), 0.0, True, {}

        # 获取当前位置
        r, c = self._coords_to_rc(self.canvas.coords(self.truck_item))
        nr, nc = r, c
        moved = False
        collision = False
        done = False

        # 暂时不做“可行驶距离限制”，只要不是“停”就能动
        if action != 4:  # 4=停
            if action == 0 and r > 0:  # 上
                nr -= 1
                moved = True
            elif action == 1 and r < MAZE_H - 1:  # 下
                nr += 1
                moved = True
            elif action == 2 and c < MAZE_W - 1:  # 右
                nc += 1
                moved = True
            elif action == 3 and c > 0:  # 左
                nc -= 1
                moved = True

        # 检查障碍碰撞
        if (nr, nc) in self.obstacles_rc:
            nr, nc = r, c  # 撞障碍回退
            collision = True
            moved = False

        # ======== Phase-based Reward Shaping（最强版本）========

        reward = 0.0

        # 1. 每步基础惩罚
        reward += self.step_penalty  # -0.02

        # 2. 成功移动奖励
        if moved:
            reward += self.move_bonus  # +0.05

        # 3. Phase-based Potential Shaping（两阶段势函数）
        if self.load_state == "empty":
            # Phase 1：朝装载区
            d_prev = self.dist_to_loading[r][c]
            d_now = self.dist_to_loading[nr][nc]
            diff = d_prev - d_now
            reward += self.shaping_coef_load * diff     # 一步靠近 → +1.0
        else:
            # Phase 2：朝卸载区
            d_prev = self.dist_to_unloading[r][c]
            d_now = self.dist_to_unloading[nr][nc]
            diff = d_prev - d_now
            reward += self.shaping_coef_unload * diff   # 一步靠近 → +1.3

        # 4. 撞障碍惩罚
        if collision:
            reward += self.collision_penalty  # -1.0

        # 5. Phase Completion Reward（阶段完成奖励）

        # 完成装载区（phase 1 完成）
        if (nr, nc) == self.loading_rc and self.load_state == "empty":
            reward += self.reward_load_phase   # +8.0
            self.load_state = "full"

        # 完成卸载区（phase 2 完成 → episode 结束）
        if (nr, nc) == self.unloading_rc and self.load_state == "full":
            reward += self.reward_unload_phase  # +12.0
            done = True
            self.is_done = True

        # 画面移动
        dr = (nc - c) * UNIT  # x方向：列
        dc = (nr - r) * UNIT  # y方向：行
        self.canvas.move(self.truck_item, dr, dc)

        self.update()

        return self._get_obs(), float(reward), done, {
            "moved": moved,
            "collision": collision,
            "load_state": self.load_state,
        }

    def render(self, delay=0.01):
        time.sleep(delay)
        self.update()


# ====================================================
# 策略网络（Actor）与价值网络（Critic）
# ====================================================
class PolicyNet(nn.Module):
    """策略网络：输入单车状态，输出动作分布"""

    def __init__(self, state_dim, hidden_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        """
        输入: x  [B, state_dim]
        输出: 概率 [B, n_actions]
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        return probs


class ValueNet(nn.Module):
    """价值网络：输入单车状态，输出 V(s)"""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ====================================================
# 单车 Actor-Critic
# ====================================================
class SingleAgentActorCritic:
    def __init__(self, state_dim, hidden_dim,
                 n_actions, actor_lr, critic_lr,
                 gamma, device, entropy_coef=0.05):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.actor = PolicyNet(state_dim, hidden_dim, n_actions).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state_vec):
        """
        输入: state_vec: numpy [2] (r, c)
        输出: action: int (0-4)
        """
        state = torch.tensor(state_vec, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        probs = self.actor(state)  # [1, n_actions]
        dist = torch.distributions.Categorical(probs)

        action = dist.sample().item()
        return action

    def update(self, traj):
        """
        traj: dict
          'states':      [T, 2] numpy
          'actions':     [T]    int
          'rewards':     [T]    float
          'next_states': [T, 2] numpy
          'dones':       [T]    bool/0/1
        """
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

        # Critic 目标：TD target
        with torch.no_grad():
            next_v = self.critic(next_states)
            td_target = rewards + self.gamma * next_v * (1 - dones)
            td_target = torch.clamp(td_target, -20.0, 20.0)  # 防止过大值

        v = self.critic(states)
        td_delta = td_target - v  # [T,1]

        # Actor 损失
        probs = self.actor(states)  # [T, n_actions]
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)  # [T]

        # 熵（鼓励探索）
        entropy = dist.entropy().mean()

        actor_loss = torch.mean(-log_probs * td_delta.detach()) - \
                     self.entropy_coef * entropy
        critic_loss = F.mse_loss(v, td_target.detach())

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()


# ====================================================
# 训练器
# ====================================================
class Trainer:
    def __init__(self, env, agent, max_episodes=600, max_steps=80):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.episode = 0
        self.return_list = []

    def _obs_to_state_vec(self, obs):
        """
        obs: numpy [2] -> 归一化到 [0,1]
        """
        obs = obs.astype(np.float32).copy()
        obs[0] = obs[0] / (MAZE_H - 1)  # row
        obs[1] = obs[1] / (MAZE_W - 1)  # col
        return obs

    def train_one_episode(self):
        obs = self.env.reset()
        state = self._obs_to_state_vec(obs)
        traj = {"states": [], "actions": [], "rewards": [],
                "next_states": [], "dones": []}
        ep_ret = 0.0

        for _ in range(self.max_steps):
            self.env.render(delay=0.01)

            action = self.agent.take_action(state)
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

        return ep_ret

    def run(self):
        if self.episode >= self.max_episodes:
            print("训练结束 ✅")
            self._plot_and_save_curve()
            return

        # 熵衰减：前 200 回合多探索，后期收敛
        if self.episode > 200:
            self.agent.entropy_coef *= 0.995
            self.agent.entropy_coef = max(0.01, self.agent.entropy_coef)

        ep_ret = self.train_one_episode()
        self.return_list.append(ep_ret)

        if (self.episode + 1) % 20 == 0:
            avg_ret = np.mean(self.return_list[-20:])
            print(
                f"Episode {self.episode + 1}/{self.max_episodes} | "
                f"Return={ep_ret:.3f} | Avg={avg_ret:.3f} | "
                f"gamma={self.agent.gamma:.3f} | entropy={self.agent.entropy_coef:.4f}"
            )

        self.episode += 1
        self.env.after(30, self.run)

    def _plot_and_save_curve(self):
        """绘制并保存回报曲线（平滑）"""

        def moving_average(x, window=10):
            if len(x) < window:
                return x
            x = np.array(x)
            cumsum = np.cumsum(np.insert(x, 0, 0))
            smooth = (cumsum[window:] - cumsum[:-window]) / window
            head = [smooth[0]] * (len(x) - len(smooth))
            return head + smooth.tolist()

        os.makedirs("results", exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(
            moving_average(self.return_list, 10),
            linewidth=2,
            label="Smoothed Return (w=10)",
        )
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Single Truck Actor-Critic on Mining Maze")
        plt.grid(alpha=0.3)
        plt.legend()
        save_path = os.path.join("results", "onecar.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"✅ 回报曲线已保存到: {save_path}")


# ====================================================
# 主程序入口
# ====================================================
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    env = SingleTruckMaze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    state_dim = 2  # (r, c)
    n_actions = env.n_actions

    agent = SingleAgentActorCritic(
        state_dim=state_dim,
        hidden_dim=64,
        n_actions=n_actions,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.95,
        device=device,
        entropy_coef=0.05,
    )

    trainer = Trainer(env, agent, max_episodes=600, max_steps=80)
    env.after(10, trainer.run)
    env.mainloop()
