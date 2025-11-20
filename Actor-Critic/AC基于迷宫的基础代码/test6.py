# -*- coding: utf-8 -*-
"""
Actor-Critic 强化学习示例：迷宫导航（V6 单步更新稳定版）
---------------------------------------------------------
红方块：智能体
黑方块：障碍（奖励 -1）
黄圆：终点（奖励 +5）
白格：普通地面（每步 -0.01）

特性：
 - 使用 Actor-Critic（TD(0)）单步更新策略和价值函数
 - 奖励含有 BFS 势函数项（鼓励靠近终点）+ 防横跳 + 重复访问惩罚
 - 动态衰减 γ 和 熵系数，增强后期收敛稳定性
 - 仅采样策略（不使用 argmax），避免后期在两格之间横跳
 - 自动绘制训练回报曲线 + 成功率曲线（平滑处理）
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
# 环境定义：Maze 类
# ============================================================
class Maze(tk.Tk, object):
    """Tkinter 迷宫环境：
       - 用 Canvas 绘制网格、障碍、起点、终点
       - 提供 step() 与 reset() 接口供智能体交互
    """
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'r', 'l']
        self.n_actions = len(self.action_space)
        self.title('Actor-Critic Maze (V6)')
        self.geometry(f'{MAZE_W * UNIT}x{MAZE_H * UNIT}')
        self._build_maze()           # 绘制迷宫布局

        # ===== 奖励参数配置 =====
        self.step_penalty = -0.01    # 每步小惩罚，避免无意义游走
        self.is_done = False         # 是否到终点/撞墙
        self.prev_rc = None          # 上一步所在格子（用于防来回横跳）
        self.visit_count = {}        # 每个格子访问次数（防刷格子）

        # BFS 势函数用
        self._init_rc_refs()         # 记录起点 / 终点 / 障碍格坐标
        self._build_bfs_distance()   # 计算各格到终点的 BFS 距离

    # ------------------------------------------------------------
    def _build_maze(self):
        """绘制迷宫：网格、障碍、终点、起点"""
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)
        # 绘制网格线
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT)
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r)

        origin = np.array([20, 20])

        # 障碍 (黑色格子)：(1,2) & (2,1)
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

        # 终点 (黄色圆)：(2,2)
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow'
        )

        # 起点 (红色方块)：(0,0)
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )
        self.canvas.pack()

    # ------------------------------------------------------------
    def _coords_to_rc(self, coords):
        """将 Canvas 像素坐标转换为 (row, col) 网格索引"""
        x1, y1, x2, y2 = coords
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        c = int(round((cx - UNIT / 2) / UNIT))
        r = int(round((cy - UNIT / 2) / UNIT))
        return r, c

    # ------------------------------------------------------------
    def _init_rc_refs(self):
        """初始化关键坐标（起点、终点、障碍）"""
        self.start_rc = self._coords_to_rc(self.canvas.coords(self.rect))
        self.goal_rc = self._coords_to_rc(self.canvas.coords(self.oval))
        self.walls_rc = {
            self._coords_to_rc(self.canvas.coords(self.hell1)),
            self._coords_to_rc(self.canvas.coords(self.hell2)),
        }

    # ------------------------------------------------------------
    def _build_bfs_distance(self):
        """构建 BFS 距离图，用于势函数奖励：
           每个格子存储其到终点的最短步数
        """
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
        # 障碍格距离设为 inf
        for r, c in self.walls_rc:
            dist[r][c] = float('inf')
        self.dist_map = dist

    # ------------------------------------------------------------
    def reset(self):
        """重置环境至起点"""
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

    # ------------------------------------------------------------
    def step(self, action):
        """执行一个动作，返回 (next_state_coords / 'terminal', reward, done)"""
        if self.is_done:
            # 终止之后如果还被调用，就直接保持终止
            return 'terminal', 0.0, True

        s = self.canvas.coords(self.rect)
        rc = self._coords_to_rc(s)

        move = np.array([0, 0])

        # 动作映射：0上 1下 2右 3左
        if action == 0 and rc[0] > 0:
            move[1] -= UNIT
        elif action == 1 and rc[0] < MAZE_H - 1:
            move[1] += UNIT
        elif action == 2 and rc[1] < MAZE_W - 1:
            move[0] += UNIT
        elif action == 3 and rc[1] > 0:
            move[0] -= UNIT
        # 否则撞边界：保持不动（只吃步惩罚）

        self.canvas.move(self.rect, move[0], move[1])
        s_ = self.canvas.coords(self.rect)
        rc_ = self._coords_to_rc(s_)

        # ===== 终止条件：到终点 / 撞障碍 =====
        if rc_ == self.goal_rc:
            self.is_done = True
            return 'terminal', 5.0, True
        if rc_ in self.walls_rc:
            self.is_done = True
            return 'terminal', -1.0, True

        # ===== 中间奖励计算 =====
        reward = self.step_penalty

        # 势函数奖励：离终点更近，加分（BFS 距离差）
        d_prev = self.dist_map[rc[0]][rc[1]]
        d_now = self.dist_map[rc_[0]][rc_[1]]
        if np.isfinite(d_prev) and np.isfinite(d_now):
            diff = d_prev - d_now         # >0 表示靠近终点
            reward += 0.25 * np.tanh(diff)

        # 距离越远，每步额外惩罚越大（鼓励尽快靠近终点）
        if np.isfinite(self.dist_map[rc_[0]][rc_[1]]):
            norm_d = self.dist_map[rc_[0]][rc_[1]] / (MAZE_H + MAZE_W)
            reward += -0.005 * norm_d

        # 来回横跳惩罚：这一步回到上一格
        if self.prev_rc == rc_:
            reward -= 0.05
        self.prev_rc = rc

        # 重复访问惩罚：多次踩同一格，轻微减分
        self.visit_count[rc_] = self.visit_count.get(rc_, 0) + 1
        if self.visit_count[rc_] > 4:
            reward -= 0.01

        return s_, float(reward), False

    def render(self):
        """延迟刷新界面，便于观察动画"""
        time.sleep(0.01)
        self.update()


# ============================================================
# 策略网络（Actor）与价值网络（Critic）
# ============================================================
class PolicyNet(nn.Module):
    """策略网络：输入状态向量，输出各动作概率分布"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(nn.Module):
    """价值网络：输入状态，输出其 V(s) 估计值"""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================
# Actor-Critic 算法实现（TD(0) 单步更新版）
# ============================================================
class ActorCritic:
    """联合更新策略与价值函数的智能体（单步 TD(0)）"""
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, gamma, device, entropy_coef=0.05):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device
        self.entropy_coef = entropy_coef  # 熵系数：鼓励探索

    def take_action(self, state_vec):
        """根据当前策略分布采样动作（始终采样，不用 argmax）"""
        state = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(state)           # [1, n_actions]
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def update(self, s, a, r, s_next, done):
        """
        单步 TD(0) 更新：
        输入是一个 transition: (s, a, r, s', done)
        """
        # 转 tensor
        state = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)       # [1, state_dim]
        next_state = torch.tensor(s_next, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor([[a]], dtype=torch.int64, device=self.device)                # [1, 1]
        reward = torch.tensor([[r]], dtype=torch.float32, device=self.device)              # [1, 1]
        done_t = torch.tensor([[float(done)]], dtype=torch.float32, device=self.device)    # [1, 1]

        # V(s) 与 V(s')
        v_s = self.critic(state)                  # [1,1]
        with torch.no_grad():
            v_next = self.critic(next_state) * (1.0 - done_t)

        # TD 目标 & TD 误差
        td_target = reward + self.gamma * v_next          # [1,1]
        td_target = torch.clamp(td_target, -5.0, 5.0)     # 防止数值爆炸
        td_delta = td_target - v_s                        # [1,1]

        # 策略梯度项
        probs = self.actor(state)                         # [1, action_dim]
        log_prob = torch.log(probs.gather(1, action) + 1e-8)  # [1,1]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)  # [1,1]

        actor_loss = -log_prob * td_delta.detach() - self.entropy_coef * entropy
        critic_loss = F.mse_loss(v_s, td_target.detach())

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()


# ============================================================
# 训练器：统一控制训练流程、衰减参数、绘图等
# ============================================================
class Trainer:
    """训练调度类，控制回合循环与可视化"""
    def __init__(self, env, agent, max_episodes=600, max_steps=80):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.episode = 0
        self.return_list = []
        self.success_list = []   # 成功=1，失败=0

    def _obs_to_state_vec(self, obs):
        """将 Canvas 坐标转为归一化状态向量 (row, col)"""
        if obs == 'terminal':
            return np.zeros(2, dtype=np.float32)
        row, col = self.env._coords_to_rc(obs)
        return np.array([row / (MAZE_H - 1), col / (MAZE_W - 1)], dtype=np.float32)

    def train_one_episode(self):
        """训练单个 episode（TD(0) 单步更新）"""
        obs = self.env.reset()
        state = self._obs_to_state_vec(obs)

        ep_ret = 0.0
        success = 0

        for _ in range(self.max_steps):
            self.env.render()
            action = self.agent.take_action(state)
            obs_, reward, done = self.env.step(action)
            next_state = self._obs_to_state_vec(obs_)

            # 单步 TD(0) 更新
            self.agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_ret += reward

            if done:
                # 判断是否成功到终点
                if obs_ != 'terminal':
                    rc_ = self.env._coords_to_rc(obs_)
                    if rc_ == self.env.goal_rc:
                        success = 1
                else:
                    # obs_ == 'terminal' 时，再从 env.rect 读一次位置
                    rc_ = self.env._coords_to_rc(self.env.canvas.coords(self.env.rect))
                    if rc_ == self.env.goal_rc:
                        success = 1
                break

        return ep_ret, success

    def run(self):
        """主训练循环（含动态 γ / 熵衰减与曲线绘制）"""
        if self.episode >= self.max_episodes:
            print("训练结束 ✅")
            self._plot_and_save_curve()
            return

        # ===== 动态调整 gamma：后期略降，稳定价值估计 =====
        decay_start = int(self.max_episodes * 0.7)
        if self.episode >= decay_start:
            ratio = (self.episode - decay_start) / (self.max_episodes - decay_start)
            new_gamma = 0.90 - 0.04 * ratio      # 0.90 -> 0.86
            self.agent.gamma = max(0.86, new_gamma)

        # ===== 动态调整熵系数：前期探索多，后期逐步收紧到 0 =====
        if self.episode > 200:
            self.agent.entropy_coef *= 0.995
            self.agent.entropy_coef = max(0.0, self.agent.entropy_coef)
        if self.episode > 450:
            self.agent.entropy_coef = 0.0

        ep_ret, success = self.train_one_episode()
        self.return_list.append(ep_ret)
        self.success_list.append(success)

        if (self.episode + 1) % 20 == 0:
            avg_ret = np.mean(self.return_list[-20:])
            avg_succ = np.mean(self.success_list[-20:]) if self.success_list else 0.0
            print(f"Episode {self.episode+1}/{self.max_episodes} | "
                  f"Return={ep_ret:.3f} | AvgRet={avg_ret:.3f} | "
                  f"SuccRate(20ep)={avg_succ:.2f} | "
                  f"gamma={self.agent.gamma:.3f} | entropy={self.agent.entropy_coef:.3f}")

        self.episode += 1
        self.env.after(40, self.run)

    def _plot_and_save_curve(self):
        """绘制并保存回报曲线 + 成功率曲线"""
        def moving_average(x, window=20):
            if len(x) < window:
                return x
            cumsum = np.cumsum(np.insert(np.array(x), 0, 0.0))
            smooth = (cumsum[window:] - cumsum[:-window]) / window
            head = [smooth[0]] * (len(x) - len(smooth))
            return head + smooth.tolist()

        os.makedirs("results", exist_ok=True)

        episodes = np.arange(1, len(self.return_list) + 1)

        plt.figure(figsize=(10, 6))

        # 子图 1：回报曲线
        plt.subplot(2, 1, 1)
        plt.plot(episodes, moving_average(self.return_list, 15),
                 linewidth=2, label='Return (MA, w=15)')
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Actor-Critic Maze (V6) - Return")
        plt.legend()
        plt.grid(alpha=0.3)

        # 子图 2：成功率曲线（滑动窗口）
        if self.success_list:
            succ_ma = moving_average(self.success_list, 20)
            plt.subplot(2, 1, 2)
            plt.plot(episodes, succ_ma, linewidth=2, label='Success Rate (MA, w=20)')
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Episode")
            plt.ylabel("Success Rate")
            plt.title("Actor-Critic Maze (V6) - Success Rate")
            plt.legend()
            plt.grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join("results", "maze_ac_v6.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"✅ 图像已保存到：{save_path}")


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    # 随机种子
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    env = Maze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Actor 小学习率、Critic 略大，收敛更稳
    agent = ActorCritic(
        state_dim=2,
        hidden_dim=128,
        action_dim=4,
        actor_lr=1e-3,
        critic_lr=2e-3,
        gamma=0.90,
        device=device,
        entropy_coef=0.05,
    )

    trainer = Trainer(env, agent, max_episodes=600, max_steps=80)
    env.after(100, trainer.run)
    env.mainloop()
