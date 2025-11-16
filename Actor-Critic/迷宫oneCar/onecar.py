# -*- coding: utf-8 -*-
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
    """单车矿区环境（装载/卸载 + 障碍）"""

    def __init__(self):
        super().__init__()
        self.title("SingleTruckMaze - RL Mining Env")
        self.geometry(f"{MAZE_W * UNIT}x{MAZE_H * UNIT}")

        # 动作空间：0上/1下/2右/3左/4停
        self.action_space = ['u', 'd', 'r', 'l', 's']
        self.n_actions = len(self.action_space)

        self._build_maze()

        # 载重状态
        self.load_state = "empty"

        # Reward
        self.step_penalty = -0.02
        self.move_bonus = +0.05
        self.collision_penalty = -1.0
        self.reward_load = +8.0
        self.reward_unload = +12.0
        self.shaping_coef = 0.8

        # 路径记录（用于热力图）
        self.visit_map = np.zeros((MAZE_H, MAZE_W), dtype=np.int32)

        # 成功率统计
        self.success_count = 0
        self.total_count = 0

        # BFS 势函数
        self._init_rc_refs()
        self._build_bfs_distance()

        self.is_done = False

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

            # 障碍
            obsA_center = origin + np.array([UNIT * 1, UNIT * 2])
            self.obsA = self.canvas.create_rectangle(
                obsA_center[0] - 25, obsA_center[1] - 25,
                obsA_center[0] + 25, obsA_center[1] + 25,
                fill="black"
            )

            obsB_center = origin + np.array([UNIT * 2, UNIT * 1])
            self.obsB = self.canvas.create_rectangle(
                obsB_center[0] - 25, obsB_center[1] - 25,
                obsB_center[0] + 25, obsB_center[1] + 25,
                fill="black"
            )

            # 装载区
            lz_center = origin + np.array([UNIT * 2, UNIT * 2])
            self.lz = self.canvas.create_rectangle(
                lz_center[0] - 25, lz_center[1] - 25,
                lz_center[0] + 25, lz_center[1] + 25,
                fill="lightblue"
            )

            # 卸载区
            uz_center = origin + np.array([UNIT * 4, UNIT * 4])
            self.uz = self.canvas.create_rectangle(
                uz_center[0] - 25, uz_center[1] - 25,
                uz_center[0] + 25, uz_center[1] + 25,
                fill="orange"
            )

            # 起点
            start_pos = origin + np.array([0, 0])
            self.truck_item = self.canvas.create_rectangle(
                start_pos[0] - 25, start_pos[1] - 25,
                start_pos[0] + 25, start_pos[1] + 25,
                fill="red"
            )

            self.canvas.pack()
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

        # 障碍
        obsA_center = origin + np.array([UNIT * 1, UNIT * 2])
        self.obsA = self.canvas.create_rectangle(
            obsA_center[0] - 25, obsA_center[1] - 25,
            obsA_center[0] + 25, obsA_center[1] + 25,
            fill="black"
        )

        obsB_center = origin + np.array([UNIT * 2, UNIT * 1])
        self.obsB = self.canvas.create_rectangle(
            obsB_center[0] - 25, obsB_center[1] - 25,
            obsB_center[0] + 25, obsB_center[1] + 25,
            fill="black"
        )

        # 装载区
        lz_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.lz = self.canvas.create_rectangle(
            lz_center[0] - 25, lz_center[1] - 25,
            lz_center[0] + 25, lz_center[1] + 25,
            fill="lightblue"
        )

        # 卸载区
        uz_center = origin + np.array([UNIT * 4, UNIT * 4])
        self.uz = self.canvas.create_rectangle(
            uz_center[0] - 25, uz_center[1] - 25,
            uz_center[0] + 25, uz_center[1] + 25,
            fill="orange"
        )

        # 起点
        start_pos = origin + np.array([0, 0])
        self.truck_item = self.canvas.create_rectangle(
            start_pos[0] - 25, start_pos[1] - 25,
            start_pos[0] + 25, start_pos[1] + 25,
            fill="red"
        )

        self.canvas.pack()
    def _coords_to_rc(self, coords):
        x1, y1, x2, y2 = coords
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return int(round((cy - UNIT / 2) / UNIT)), int(round((cx - UNIT / 2) / UNIT))

    def _init_rc_refs(self):
        self.loading_rc = self._coords_to_rc(self.canvas.coords(self.lz))
        self.unloading_rc = self._coords_to_rc(self.canvas.coords(self.uz))
        self.obstacles_rc = {
            self._coords_to_rc(self.canvas.coords(self.obsA)),
            self._coords_to_rc(self.canvas.coords(self.obsB))
        }
        self.start_rc = (0, 0)

    # ---------- BFS ----------
    def _bfs_from(self, target):
        H, W = MAZE_H, MAZE_W
        dist = [[999] * W for _ in range(H)]
        from collections import deque
        q = deque([target])
        dist[target[0]][target[1]] = 0

        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr,nc) in self.obstacles_rc:
                        continue
                    if dist[nr][nc] > dist[r][c] + 1:
                        dist[nr][nc] = dist[r][c] + 1
                        q.append((nr,nc))
        return dist

    def _build_bfs_distance(self):
        self.dist_to_loading = self._bfs_from(self.loading_rc)
        self.dist_to_unloading = self._bfs_from(self.unloading_rc)

    def reset(self):
        # 回到起点
        origin = np.array([UNIT/2, UNIT/2])
        start_pos = origin
        self.canvas.coords(self.truck_item,
                           start_pos[0]-25, start_pos[1]-25,
                           start_pos[0]+25, start_pos[1]+25)

        self.is_done = False
        self.load_state = "empty"

        # 统计
        self.total_count += 1

        # 清空路径热力图
        self.visit_map[:] = 0

        self.update()
        return self._get_obs()

    def _get_obs(self):
        return np.array(self._coords_to_rc(self.canvas.coords(self.truck_item)), dtype=np.float32)

    # ---------- 环境步进 ----------
    def step(self, action):

        r, c = self._coords_to_rc(self.canvas.coords(self.truck_item))
        nr, nc = r, c
        moved = False
        collision = False

        if action != 4:  # 不是停
            if action == 0 and r>0: nr -=1; moved=True
            elif action == 1 and r<MAZE_H-1: nr +=1; moved=True
            elif action == 2 and c<MAZE_W-1: nc +=1; moved=True
            elif action == 3 and c>0: nc -=1; moved=True

        # 障碍
        if (nr,nc) in self.obstacles_rc:
            nr,nc = r,c
            collision=True
            moved=False

        # ===== reward =====
        reward = self.step_penalty
        if moved: reward += self.move_bonus

        # 势函数 shaping
        if self.load_state=="empty":
            diff = self.dist_to_loading[r][c] - self.dist_to_loading[nr][nc]
        else:
            diff = self.dist_to_unloading[r][c] - self.dist_to_unloading[nr][nc]
        reward += self.shaping_coef * diff

        # 撞
        if collision: reward += self.collision_penalty

        # move
        self.canvas.move(self.truck_item, (nc-c)*UNIT, (nr-r)*UNIT)

        # 路径记录
        self.visit_map[nr][nc] += 1

        # 装载
        if (nr,nc)==self.loading_rc and self.load_state=="empty":
            reward += self.reward_load
            self.load_state="full"

        # 卸载
        done = False
        if (nr,nc)==self.unloading_rc and self.load_state=="full":
            reward += self.reward_unload
            self.load_state="empty"
            self.is_done=True
            done=True
            self.success_count += 1

        self.update()
        return self._get_obs(), float(reward), done, {}

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

    def take_action(self, state_vec, episode=None):
        """
        输入: state_vec: numpy [2] (r, c)
        输出: action: int (0-4)

        前期 episode <= 300：按概率 sample（随机策略，便于探索）
        后期 episode  > 300：使用 argmax（确定性策略，稳定收敛）
        """
        state = torch.tensor(state_vec, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        probs = self.actor(state)  # [1, n_actions]
        dist = torch.distributions.Categorical(probs)

        if (episode is not None) and (episode > 550):
            # 后期：确定性策略
            if random.random() < 0.01:  # 1% 随机
                action = dist.sample().item()
            else:
                action = torch.argmax(probs, dim=1).item()
        else:
            # 前期：随机探索
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

        self.return_list = []       # 每回合回报
        self.success_list = []      # 每回合是否成功（1/0）
        self.heat_accum = np.zeros((MAZE_H, MAZE_W), dtype=np.int32)  # 轨迹累计

    def _obs_to_state_vec(self, obs):
        """
        obs: numpy [2] -> 归一化到 [0,1]
        """
        obs = obs.astype(np.float32).copy()
        obs[0] = obs[0] / (MAZE_H - 1)  # row
        obs[1] = obs[1] / (MAZE_W - 1)  # col
        return obs

    def train_one_episode(self):
        # 记录本回合是否成功的基线
        prev_success = self.env.success_count

        obs = self.env.reset()
        state = self._obs_to_state_vec(obs)
        traj = {"states": [], "actions": [], "rewards": [],
                "next_states": [], "dones": []}
        ep_ret = 0.0

        for _ in range(self.max_steps):
            self.env.render(delay=0.01)

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

        # 是否成功（本回合装+卸成功）
        success = self.env.success_count - prev_success  # 0 or 1
        self.success_list.append(success)

        # 累计路径热力图
        self.heat_accum += self.env.visit_map

        return ep_ret

    def run(self):
        if self.episode >= self.max_episodes:
            print("训练结束 ✅")
            self._plot_and_save_curves()
            return

        # 熵衰减：200 前保持较大探索；200~300 平滑衰减；300 后完全 0
        if self.episode <= 200:
            self.agent.entropy_coef = 0.05
        elif 200 < self.episode <= 300:
            self.agent.entropy_coef *= 0.995
            self.agent.entropy_coef = max(0.01, self.agent.entropy_coef)
        else:
            self.agent.entropy_coef = 0.0

        ep_ret = self.train_one_episode()
        self.return_list.append(ep_ret)

        # 最近 20 回合平均回报 & 成功率
        window = 20
        recent_returns = self.return_list[-window:]
        avg_ret = np.mean(recent_returns) if recent_returns else ep_ret

        recent_successes = self.success_list[-window:]
        success_rate = np.mean(recent_successes) if recent_successes else 0.0

        if (self.episode + 1) % 20 == 0:
            print(
                f"Episode {self.episode + 1}/{self.max_episodes} | "
                f"Return={ep_ret:.3f} | Avg={avg_ret:.3f} | "
                f"SuccRate(last {window})={success_rate:.2f} | "
                f"gamma={self.agent.gamma:.3f} | entropy={self.agent.entropy_coef:.4f}"
            )

        self.episode += 1
        self.env.after(30, self.run)

    def _plot_and_save_curves(self):
        os.makedirs("results", exist_ok=True)

        # ---------- 1）回报曲线 ----------
        def moving_average(x, window=10):
            if len(x) < window:
                return x
            x = np.array(x)
            cumsum = np.cumsum(np.insert(x, 0, 0))
            smooth = (cumsum[window:] - cumsum[:-window]) / window
            head = [smooth[0]] * (len(x) - len(smooth))
            return head + smooth.tolist()

        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(self.return_list, 10),
                 linewidth=2, label="Smoothed Return (w=10)")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Single Truck Actor-Critic - Return")
        plt.grid(alpha=0.3)
        plt.legend()
        ret_path = os.path.join("results", "onecar.png")
        plt.savefig(ret_path, dpi=300)
        plt.close()
        print(f"✅ 回报曲线已保存到: {ret_path}")

        # ---------- 2）成功率曲线 ----------
        if self.success_list:
            window = 20
            success_rates = []
            for i in range(len(self.success_list)):
                start = max(0, i - window + 1)
                sr = np.mean(self.success_list[start:i + 1])
                success_rates.append(sr)

            plt.figure(figsize=(8, 4))
            plt.plot(success_rates, linewidth=2, label=f"Success Rate (w={window})")
            plt.ylim(0.0, 1.05)
            plt.xlabel("Episode")
            plt.ylabel("Success Rate")
            plt.title("Single Truck Actor-Critic - Success Rate")
            plt.grid(alpha=0.3)
            plt.legend()
            succ_path = os.path.join("results", "success_rate.png")
            plt.savefig(succ_path, dpi=300)
            plt.close()
            print(f"✅ 成功率曲线已保存到: {succ_path}")

        # ---------- 3）轨迹热力图 ----------
        if self.heat_accum.sum() > 0:
            plt.figure(figsize=(5, 5))
            plt.imshow(self.heat_accum, cmap="hot", origin="upper")
            plt.colorbar(label="Visit Count")
            plt.xticks(range(MAZE_W))
            plt.yticks(range(MAZE_H))
            plt.title("Visit Heatmap (Single Truck)")
            heat_path = os.path.join("results", "heatmap.png")
            plt.savefig(heat_path, dpi=300)
            plt.close()
            print(f"✅ 轨迹热力图已保存到: {heat_path}")

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
        critic_lr=5e-4,   # critic 稍微小一点更稳
        gamma=0.95,
        device=device,
        entropy_coef=0.05,
    )

    trainer = Trainer(env, agent, max_episodes=600, max_steps=80)
    env.after(10, trainer.run)
    env.mainloop()






