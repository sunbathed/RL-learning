# -*- coding: utf-8 -*-
"""
Multi-Truck Actor-Critic 示例（改良稳定版）
-------------------------------------------
改动要点（相对于你那版）：

1）势函数 BFS 分成两张图：
   - dist_to_loading   ：空车时鼓励靠近装载区 (2,2)
   - dist_to_unloading ：满载时鼓励靠近卸载区 (4,4)
   -> 完整体现 “先装载再卸载” 的作业流程。

2）Reward 重新平衡：
   - 步惩罚 step_penalty          = -0.005  （更轻一点）
   - 被挡惩罚 block_penalty       = -0.02   （大幅减轻，避免学成“全停”）
   - 主动移动奖励 move_bonus      = +0.01   （鼓励动起来）
   - 势函数系数 shaping_coef      = 0.4     （鼓励朝目标走）
   - 装载奖励 +3，卸载奖励 +5 保持不变
   -> 期望行为：车会积极移动 + 先去装载区 + 再去卸载区，且尽量避免冲突。

3）其它：
   - 仍然使用多智能体共享 Actor，多头输出，每辆车一个动作分布；
   - Critic 集中式，输入全局状态；
   - 仍保留 gamma / 熵系数动态衰减机制；
   - global_reward 依然使用“所有车 reward 的平均值”，但在新 reward 结构下不会塌缩到“全停”策略。
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
UNIT = 60        # 每格像素
MAZE_H = 5       # 行
MAZE_W = 5       # 列


# ====================================================
# 环境：MultiTruckMaze
# ====================================================
class MultiTruckMaze(tk.Tk, object):
    """多智能体矿区环境（装载/卸载 + 障碍 + 优先级 + 协作）"""
    def __init__(self, n_trucks=3):
        super().__init__()
        assert n_trucks <= 4, "5×5 环境最多摆 4 辆车"
        self.n_trucks = n_trucks

        # 动作空间：0上/1下/2右/3左/4停
        self.action_space = ['u', 'd', 'r', 'l', 's']
        self.n_actions = len(self.action_space)

        self.title("MultiTruckMaze - RL Mining Env (Improved)")
        self.geometry(f"{MAZE_W * UNIT}x{MAZE_H * UNIT}")

        # 画地图
        self._build_maze()

        # 优先级（数值越大优先级越高），这里简单固定：最后一辆优先级最高
        self.priorities = list(range(self.n_trucks))[::-1]

        # 动态可行驶距离（简单版：最多连续移动 3 步）
        self.max_travel = 3
        self.remain_travel = [self.max_travel] * self.n_trucks

        # 载重状态：empty / full
        self.load_state = ["empty"] * self.n_trucks

        # ==== Reward 配置（已重新平衡）====
        self.step_penalty = -0.005       # 每步基础惩罚
        self.block_penalty = -0.02       # 被挡惩罚（相比原来的 -0.05 小很多）
        self.move_bonus = +0.01          # 只要成功移动就给一点小奖励
        self.reward_load = +3.0          # 空车到装载区
        self.reward_unload = +5.0        # 满载到卸载区
        self.shaping_coef = 0.4          # 势函数系数（比原来更强）

        # 终止标志：任意一辆车成功完成卸载就结束本回合
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

        # 3 辆车起点： (0,0), (4,0), (0,4)
        self.truck_items = []
        start_positions = [
            origin + np.array([UNIT * 0, UNIT * 0]),  # 左上
            origin + np.array([UNIT * 4, UNIT * 0]),  # 右上
            origin + np.array([UNIT * 0, UNIT * 4]),  # 左下
        ]
        colors = ["red", "blue", "green", "purple"]

        for i in range(self.n_trucks):
            center = start_positions[i]
            item = self.canvas.create_rectangle(
                center[0] - UNIT / 2 + 5, center[1] - UNIT / 2 + 5,
                center[0] + UNIT / 2 - 5, center[1] + UNIT / 2 - 5,
                fill=colors[i]
            )
            self.truck_items.append(item)

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
        start_positions = [
            origin + np.array([UNIT * 0, UNIT * 0]),
            origin + np.array([UNIT * 4, UNIT * 0]),
            origin + np.array([UNIT * 0, UNIT * 4]),
        ]
        for i, item in enumerate(self.truck_items):
            center = start_positions[i]
            self.canvas.coords(
                item,
                center[0] - UNIT / 2 + 5, center[1] - UNIT / 2 + 5,
                center[0] + UNIT / 2 - 5, center[1] + UNIT / 2 - 5
            )

        self.is_done = False
        self.remain_travel = [self.max_travel] * self.n_trucks
        self.load_state = ["empty"] * self.n_trucks

        self.update()
        return self._get_obs()

    def _get_obs(self):
        """返回所有车的 (r,c) 拼接：长度 = 2 * n_trucks"""
        obs = []
        for item in self.truck_items:
            obs.extend(self._coords_to_rc(self.canvas.coords(item)))
        return np.array(obs, dtype=np.float32)

    # ---------- 计算候选移动 ----------
    def _compute_candidate_moves(self, actions):
        rc_list, cand_rc = [], []
        for item in self.truck_items:
            rc_list.append(self._coords_to_rc(self.canvas.coords(item)))

        for i, (r, c) in enumerate(rc_list):
            a = actions[i]
            nr, nc = r, c

            # 若可行驶距离用完，则只能停
            if self.remain_travel[i] <= 0:
                cand_rc.append((nr, nc))
                continue

            if a == 4:  # 主动选择停
                cand_rc.append((nr, nc))
                continue

            # 上下左右移动
            if a == 0 and r > 0:
                nr -= 1
            elif a == 1 and r < MAZE_H - 1:
                nr += 1
            elif a == 2 and c < MAZE_W - 1:
                nc += 1
            elif a == 3 and c > 0:
                nc -= 1

            cand_rc.append((nr, nc))
        return rc_list, cand_rc

    # ---------- 冲突解决 ----------
    def _resolve_conflicts(self, rc_list, cand_rc):
        occupied = set()
        final_rc = list(cand_rc)
        blocked = [False] * self.n_trucks

        # 按优先级顺序决定谁先占坑
        order = sorted(range(self.n_trucks),
                       key=lambda i: self.priorities[i],
                       reverse=True)

        for i in order:
            cand = cand_rc[i]

            # 撞障碍：回原地 + 标记阻塞
            if cand in self.obstacles_rc:
                final_rc[i] = rc_list[i]
                blocked[i] = True
                continue

            # 两车争同一格：优先级高的先占，后来的被挡
            if cand in occupied:
                final_rc[i] = rc_list[i]
                blocked[i] = True
                continue

            occupied.add(cand)

        return final_rc, blocked

    # ---------- step ----------
    def step(self, actions):
        """
        actions: list[int]，长度 = n_trucks
        返回：
            obs: 新状态
            global_reward: 所有车 reward 的平均
            done: 是否任意车成功卸载
            info: 详细信息
        """
        if self.is_done:
            return self._get_obs(), 0.0, True, {}

        rc_list, cand_rc = self._compute_candidate_moves(actions)
        final_rc, blocked = self._resolve_conflicts(rc_list, cand_rc)

        rewards_each = [self.step_penalty] * self.n_trucks
        done_flags = [False] * self.n_trucks

        for i, item in enumerate(self.truck_items):
            old = rc_list[i]
            new = final_rc[i]

            # 是否真的移动
            moved = (new != old)

            # 画面移动
            dr = (new[1] - old[1]) * UNIT
            dc = (new[0] - old[0]) * UNIT
            self.canvas.move(item, dr, dc)

            if moved:
                # 消耗可行驶距离
                self.remain_travel[i] -= 1
                # 鼓励主动移动
                rewards_each[i] += self.move_bonus

            # ==== 势函数奖励：分两阶段 ====
            if self.load_state[i] == "empty":
                d_prev = self.dist_to_loading[old[0]][old[1]]
                d_now = self.dist_to_loading[new[0]][new[1]]
            else:
                d_prev = self.dist_to_unloading[old[0]][old[1]]
                d_now = self.dist_to_unloading[new[0]][new[1]]

            # 距离减少 -> diff > 0 -> tanh(diff) 正；距离增加则相反
            diff = d_prev - d_now
            rewards_each[i] += self.shaping_coef * np.tanh(diff)

            # ==== 冲突惩罚：被挡住且没动 ====
            if blocked[i] and (not moved):
                rewards_each[i] += self.block_penalty

            # ==== 装载逻辑（空车到装载区）====
            if new == self.loading_rc and self.load_state[i] == "empty":
                rewards_each[i] += self.reward_load
                self.load_state[i] = "full"

            # ==== 卸载逻辑（满载到卸载区）====
            if new == self.unloading_rc and self.load_state[i] == "full":
                rewards_each[i] += self.reward_unload
                self.load_state[i] = "empty"
                done_flags[i] = True

        # 是否有车完成卸载
        done = any(done_flags)
        if done:
            self.is_done = True

        # 简单版“动态可行驶距离”重置：
        # 若已用完，则在本 step 后直接重置为 max_travel
        for i in range(self.n_trucks):
            if self.remain_travel[i] <= 0:
                self.remain_travel[i] = self.max_travel

        self.update()

        global_reward = float(np.mean(rewards_each))
        return self._get_obs(), global_reward, done, {
            "rewards_each": rewards_each,
            "blocked": blocked,
            "done_flags": done_flags
        }

    def render(self, delay=0.01):
        time.sleep(delay)
        self.update()


# ====================================================
# 策略网络（Actor）与价值网络（Critic）
# ====================================================
class PolicyNet(nn.Module):
    """策略网络：输入全局状态，输出所有车的动作分布"""
    def __init__(self, state_dim, hidden_dim, n_trucks, n_actions_per_truck):
        super().__init__()
        self.n_trucks = n_trucks
        self.n_actions_per_truck = n_actions_per_truck
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_trucks * n_actions_per_truck)

    def forward(self, x):
        """
        输入: x  [B, state_dim]
        输出: 概率 [B, n_trucks * n_actions_per_truck]
        使用时可 reshape 为 [B, n_trucks, n_actions_per_truck]
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        return probs


class ValueNet(nn.Module):
    """价值网络：输入全局状态，输出 V(s)"""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ====================================================
# 多智能体 Actor-Critic
# ====================================================
class MultiAgentActorCritic:
    def __init__(self, state_dim, hidden_dim,
                 n_trucks, n_actions_per_truck,
                 actor_lr, critic_lr, gamma, device,
                 entropy_coef=0.05):
        self.n_trucks = n_trucks
        self.n_actions_per_truck = n_actions_per_truck
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.actor = PolicyNet(state_dim, hidden_dim,
                               n_trucks, n_actions_per_truck).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state_vec):
        """
        输入: state_vec: numpy 1D [state_dim]
        输出: actions: list[int]，长度 = n_trucks
        """
        state = torch.tensor(state_vec, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        probs_flat = self.actor(state)  # [1, n_trucks*n_actions]
        probs_flat = probs_flat.view(1, self.n_trucks, self.n_actions_per_truck)
        probs = probs_flat[0]  # [n_trucks, n_actions]

        actions = []
        for i in range(self.n_trucks):
            dist = torch.distributions.Categorical(probs[i])
            a = dist.sample().item()
            actions.append(a)
        return actions

    def update(self, traj):
        """
        traj: dict
          'states':      [T, state_dim] numpy
          'actions':     [T, n_trucks]  int
          'rewards':     [T] 或 [T,1]   float（global reward）
          'next_states': [T, state_dim]
          'dones':       [T] 或 [T,1]   bool/0/1
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
            td_target = torch.clamp(td_target, -5.0, 5.0)

        v = self.critic(states)
        td_delta = td_target - v  # [T,1]

        # Actor 概率
        probs_flat = self.actor(states)  # [T, n_trucks*n_actions]
        probs_flat = probs_flat.view(-1, self.n_trucks, self.n_actions_per_truck)

        # 取出各车的被选动作概率
        actions_expanded = actions.unsqueeze(-1)  # [T, n_trucks, 1]
        chosen_probs = probs_flat.gather(dim=2, index=actions_expanded).squeeze(-1)
        log_probs = torch.log(chosen_probs + 1e-8)             # [T, n_trucks]
        log_probs_sum = log_probs.sum(dim=1, keepdim=True)     # [T,1] 联合 log_prob

        # 熵（鼓励探索）
        entropy_all = -(probs_flat * torch.log(probs_flat + 1e-8)).sum(dim=2)
        entropy = entropy_all.mean()

        actor_loss = torch.mean(-log_probs_sum * td_delta.detach()) - \
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
        obs: numpy [2 * n_trucks] -> 归一化到 [0,1]
        """
        obs = obs.astype(np.float32).copy()
        for i in range(0, len(obs), 2):
            obs[i] = obs[i] / (MAZE_H - 1)      # row
            obs[i + 1] = obs[i + 1] / (MAZE_W - 1)  # col
        return obs

    def train_one_episode(self):
        obs = self.env.reset()
        state = self._obs_to_state_vec(obs)
        traj = {"states": [], "actions": [], "rewards": [],
                "next_states": [], "dones": []}
        ep_ret = 0.0

        for _ in range(self.max_steps):
            self.env.render(delay=0.03)

            actions = self.agent.take_action(state)
            obs_next, reward, done, info = self.env.step(actions)
            next_state = self._obs_to_state_vec(obs_next)

            traj["states"].append(state)
            traj["actions"].append(actions)
            traj["rewards"].append(reward)
            traj["next_states"].append(next_state)
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

        # 动态 Gamma 衰减：后期略微降低远期权重，稳定估计
        decay_start = int(self.max_episodes * 0.7)
        if self.episode >= decay_start:
            ratio = (self.episode - decay_start) / (self.max_episodes - decay_start)
            new_gamma = 0.9 - 0.05 * ratio
            self.agent.gamma = max(0.85, new_gamma)

        # 熵系数衰减：前期探索多，后期收敛
        if self.episode > 250:
            self.agent.entropy_coef *= 0.995
            self.agent.entropy_coef = max(0.01, self.agent.entropy_coef)
        if self.episode > 400:
            self.agent.entropy_coef = 0.01

        ep_ret = self.train_one_episode()
        self.return_list.append(ep_ret)

        if (self.episode + 1) % 20 == 0:
            avg_ret = np.mean(self.return_list[-20:])
            print(
                f"Episode {self.episode + 1}/{self.max_episodes} | "
                f"Return={ep_ret:.3f} | Avg={avg_ret:.3f} | "
                f"gamma={self.agent.gamma:.3f} | entropy={self.agent.entropy_coef:.3f}"
            )

        self.episode += 1
        self.env.after(40, self.run)

    def _plot_and_save_curve(self):
        """绘制并保存回报曲线（平滑）"""

        def moving_average(x, window=15):
            if len(x) < window:
                return x
            x = np.array(x)
            cumsum = np.cumsum(np.insert(x, 0, 0))
            smooth = (cumsum[window:] - cumsum[:-window]) / window
            head = [smooth[0]] * (len(x) - len(smooth))
            return head + smooth.tolist()

        os.makedirs("../results", exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(
            moving_average(self.return_list, 15),
            linewidth=2,
            label="Smoothed Return (w=15)",
        )
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Multi-Truck Actor-Critic on Mining Maze (Improved)")
        plt.grid(alpha=0.3)
        plt.legend()
        save_path = os.path.join("../results", "multi_truck_maze_return_improved.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"✅ 回报曲线已保存到: {save_path}")


# ====================================================
# 主程序入口
# ====================================================
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    env = MultiTruckMaze(n_trucks=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    state_dim = env.n_trucks * 2
    n_trucks = env.n_trucks
    n_actions_per_truck = env.n_actions

    agent = MultiAgentActorCritic(
        state_dim=state_dim,
        hidden_dim=128,
        n_trucks=n_trucks,
        n_actions_per_truck=n_actions_per_truck,
        actor_lr=1e-3,
        critic_lr=2e-3,
        gamma=0.9,
        device=device,
        entropy_coef=0.05,
    )

    trainer = Trainer(env, agent, max_episodes=600, max_steps=80)
    env.after(100, trainer.run)
    env.mainloop()
