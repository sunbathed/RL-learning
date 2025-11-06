# -*- coding: utf-8 -*-
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


# ======================= 环境 =======================
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'r', 'l']
        self.n_actions = len(self.action_space)
        self.title('Actor-Critic Maze (平滑版)')
        self.geometry(f'{MAZE_W * UNIT}x{MAZE_H * UNIT}')
        self._build_maze()

        self.step_penalty = -0.01
        self.is_done = False
        self.prev_rc = None
        self.visit_count = {}
        self._init_rc_refs()
        self._build_bfs_distance()

    # ======================= 画出迷宫 =======================
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT)
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r)
        origin = np.array([20, 20])

        # 黑格障碍
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

        # 终点（黄）
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow'
        )

        # 起点（红）
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )
        self.canvas.pack()

    # ======================= 工具函数 =======================
    def _coords_to_rc(self, coords):
        x1, y1, x2, y2 = coords
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        c = int(round((cx - UNIT/2) / UNIT))
        r = int(round((cy - UNIT/2) / UNIT))
        return r, c

    def _init_rc_refs(self):
        self.start_rc = self._coords_to_rc(self.canvas.coords(self.rect))
        self.goal_rc = self._coords_to_rc(self.canvas.coords(self.oval))
        self.walls_rc = {
            self._coords_to_rc(self.canvas.coords(self.hell1)),
            self._coords_to_rc(self.canvas.coords(self.hell2)),
        }

    def _build_bfs_distance(self):
        """预计算 BFS 距离，作为奖励辅助（势函数）"""
        H, W = MAZE_H, MAZE_W
        dist = [[float('inf')] * W for _ in range(H)]
        from collections import deque
        q = deque([self.goal_rc])
        gr, gc = self.goal_rc
        dist[gr][gc] = 0
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in self.walls_rc:
                    if dist[nr][nc] > dist[r][c] + 1:
                        dist[nr][nc] = dist[r][c] + 1
                        q.append((nr, nc))
        for r, c in self.walls_rc:
            dist[r][c] = float('inf')
        self.dist_map = dist

    def reset(self):
        """重置环境"""
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )
        self.is_done = False
        self.prev_rc = None
        self.visit_count = {}
        self.update()
        return self.canvas.coords(self.rect)

    def step(self, action):
        """执行一步动作，返回下一状态、奖励、是否结束"""
        if self.is_done:
            return 'terminal', 0.0, True

        s = self.canvas.coords(self.rect)
        rc = self._coords_to_rc(s)
        move = np.array([0, 0])

        # 上下左右
        if action == 0 and rc[0] > 0:
            move[1] -= UNIT
        elif action == 1 and rc[0] < MAZE_H - 1:
            move[1] += UNIT
        elif action == 2 and rc[1] < MAZE_W - 1:
            move[0] += UNIT
        elif action == 3 and rc[1] > 0:
            move[0] -= UNIT

        self.canvas.move(self.rect, move[0], move[1])
        s_ = self.canvas.coords(self.rect)
        rc_ = self._coords_to_rc(s_)

        # 终点奖励 / 障碍惩罚
        if rc_ == self.goal_rc:
            self.is_done = True
            return 'terminal', 5.0, True
        if rc_ in self.walls_rc:
            self.is_done = True
            return 'terminal', -1.0, True

        # 基础步惩罚
        reward = self.step_penalty

        # 势函数奖励（靠近目标加分）
        d_prev = self.dist_map[rc[0]][rc[1]]
        d_now = self.dist_map[rc_[0]][rc_[1]]
        if np.isfinite(d_prev) and np.isfinite(d_now):
            reward += 0.25 * (d_prev - d_now)

        # 原地惩罚 / 重复访问惩罚
        if self.prev_rc == rc_:
            reward -= 0.05
        self.prev_rc = rc
        self.visit_count[rc_] = self.visit_count.get(rc_, 0) + 1
        if self.visit_count[rc_] > 4:
            reward -= 0.01
        return s_, float(reward), False

    def render(self):
        time.sleep(0.01)
        self.update()


# ======================= 网络结构 =======================
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


# ======================= Actor-Critic =======================
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, gamma, device, entropy_coef=0.01):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device
        self.entropy_coef = entropy_coef

    def take_action(self, state_vec):
        state = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def update(self, traj):
        states = torch.tensor(np.array(traj['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(traj['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(traj['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(traj['next_states']), dtype=torch.float32).to(self.device)
        dones = torch.tensor(traj['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        probs = self.actor(states)
        log_probs = torch.log(probs.gather(1, actions) + 1e-8)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True).mean()

        actor_loss = torch.mean(-log_probs * td_delta.detach()) - self.entropy_coef * entropy
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()


# ======================= 训练器 =======================
class Trainer:
    def __init__(self, env, agent, max_episodes=600, max_steps=60):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.episode = 0
        self.return_list = []

    def _obs_to_state_vec(self, obs):
        if obs == 'terminal':
            return np.zeros(2, dtype=np.float32)
        row, col = self.env._coords_to_rc(obs)
        return np.array([row / (MAZE_H - 1), col / (MAZE_W - 1)], dtype=np.float32)

    def train_one_episode(self):
        obs = self.env.reset()
        state = self._obs_to_state_vec(obs)
        traj = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        ep_ret = 0.0
        for _ in range(self.max_steps):
            self.env.render()
            action = self.agent.take_action(state)
            obs_, reward, done = self.env.step(action)
            next_state = self._obs_to_state_vec(obs_)
            traj['states'].append(state)
            traj['actions'].append(action)
            traj['rewards'].append(reward)
            traj['next_states'].append(next_state)
            traj['dones'].append(done)
            state = next_state
            ep_ret += reward
            if done:
                break
        if traj['states']:
            self.agent.update(traj)
        return ep_ret

    def run(self):
        # ========== 训练结束判定 ==========
        if self.episode >= self.max_episodes:
            print("训练结束 ✅")
            self._plot_and_save_curve()
            return

        # ========== 动态调整 gamma（折扣因子） ==========
        # 前 50% 训练保持 0.9，后期线性衰减到 0.8
        decay_start = int(self.max_episodes * 0.5)
        if self.episode >= decay_start:
            new_gamma = 0.9 - 0.1 * ((self.episode - decay_start) / (self.max_episodes - decay_start))
            new_gamma = max(0.8, new_gamma)
            self.agent.gamma = new_gamma  # 实时更新 agent.gamma

        # ========== 单回合训练 ==========
        ep_ret = self.train_one_episode()
        self.return_list.append(ep_ret)

        # ========== 打印训练进度 ==========
        if (self.episode + 1) % 20 == 0:
            avg_ret = np.mean(self.return_list[-20:])
            print(f"Episode {self.episode+1}/{self.max_episodes} | "
                  f"Return={ep_ret:.3f} | Avg={avg_ret:.3f} | gamma={self.agent.gamma:.3f}")

        # 下一轮
        self.episode += 1
        self.env.after(40, self.run)

    # === 平滑曲线 + 保存 ===
    def _plot_and_save_curve(self):
        def moving_average(x, window=20):
            if len(x) < window:
                return x
            cumsum = np.cumsum(np.insert(np.array(x), 0, 0))
            smooth = (cumsum[window:] - cumsum[:-window]) / window
            head = [smooth[0]] * (len(x) - len(smooth))
            return head + smooth.tolist()

        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(moving_average(self.return_list, 15), color='royalblue', linewidth=2, label='Smoothed (w=15)')
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Actor-Critic Maze 学习曲线（动态γ）")
        plt.legend()
        plt.grid(alpha=0.3)
        save_path = os.path.join("results", "maze_plot.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"✅ 图像已保存到：{save_path}")


# ======================= 主程序入口 =======================
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    env = Maze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Actor-critic 初始化
    agent = ActorCritic(
        state_dim=2, hidden_dim=128, action_dim=4,
        actor_lr=1e-3, critic_lr=5e-3,
        gamma=0.9, device=device, entropy_coef=0.05
    )

    trainer = Trainer(env, agent, max_episodes=600, max_steps=60)

    # 以 Tkinter 的 after() 异步启动
    env.after(100, trainer.run)
    env.mainloop()
