import numpy as np
import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

# ========== 迷宫基础参数 ==========
UNIT = 40
MAZE_H = 4
MAZE_W = 4


# ======================= 环境 =======================
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Actor-Critic Maze (easy reward)')
        self.geometry(f'{MAZE_W * UNIT}x{MAZE_H * UNIT}')
        self._build_maze()
        self._build_bfs_distance()   # ⭐ 用 BFS 先把可达最短步数表准备好
        # 这一回合目前为止离终点的“最好距离”
        self.best_dist_in_ep = None
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 画网格
        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT)
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r)

        origin = np.array([20, 20])

        # 黑格1 (1,2)
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black'
        )

        # 黑格2 (2,1)
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black'
        )

        # 终点 (2,2)
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
        self.goal_center = np.array(oval_center, dtype=np.float32)

    def _build_bfs_distance(self):
        """对这张 4×4 图做一次 BFS，得到真正能到终点的最短步数"""
        goal = (2, 2)
        walls = {(1, 2), (2, 1)}
        dist = [[float('inf')] * MAZE_W for _ in range(MAZE_H)]
        q = deque()
        dist[goal[0]][goal[1]] = 0
        q.append(goal)

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < MAZE_H and 0 <= nc < MAZE_W:
                    if (nr, nc) in walls:
                        continue
                    if dist[nr][nc] > dist[r][c] + 1:
                        dist[nr][nc] = dist[r][c] + 1
                        q.append((nr, nc))

        # 把黑格也标成 inf
        for (r, c) in walls:
            dist[r][c] = float('inf')

        self.dist_map = dist

    def coords_to_pos(self, coords):
        x1, y1, x2, y2 = coords
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        col = int((cx - 20) / UNIT + 1e-6)
        row = int((cy - 20) / UNIT + 1e-6)
        return row, col

    def reset(self):
        # 起点还是 (0,0)，不动
        self.update()
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )
        # ⭐ 回合一开始，记录当前这格（0,0）到终点的 BFS 距离
        start_row, start_col = 0, 0
        self.best_dist_in_ep = self.dist_map[start_row][start_col]
        return self.canvas.coords(self.rect)

    def step(self, action):
        if not self.canvas.winfo_exists():
            return 'terminal', 0.0, True

        s = self.canvas.coords(self.rect)
        old_row, old_col = self.coords_to_pos(s)

        # 动作
        move = np.array([0, 0])
        if action == 0 and s[1] > UNIT:  # up
            move[1] -= UNIT
        elif action == 1 and s[1] < (MAZE_H - 1) * UNIT:  # down
            move[1] += UNIT
        elif action == 2 and s[0] < (MAZE_W - 1) * UNIT:  # right
            move[0] += UNIT
        elif action == 3 and s[0] > UNIT:  # left
            move[0] -= UNIT

        self.canvas.move(self.rect, move[0], move[1])
        s_ = self.canvas.coords(self.rect)
        new_row, new_col = self.coords_to_pos(s_)

        # ===== 终点和黑格，还是原来的 =====
        if s_ == self.canvas.coords(self.oval):
            # 真正到终点，给大糖
            return 'terminal', 3.0, True
        if s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            return 'terminal', -1.0, True

        # ===== 下面是关键：只奖励“比这一回合历史最好的还好”的那一步 =====
        old_best = self.best_dist_in_ep
        cur_d = self.dist_map[new_row][new_col]

        # 一点小的时间惩罚，防止无限游走
        base_penalty = -0.01

        if not np.isfinite(cur_d):
            # 不可达的/黑格周围的，就当你走错了
            reward = -0.08 + base_penalty
            done = False
        else:
            if cur_d < old_best:
                # ✅ 你这一步是“回合内最好的一次”
                # 更新这一回合的最好距离
                self.best_dist_in_ep = cur_d
                # 奖励可以给得稍微大一点
                reward = 0.25 + base_penalty
                done = False
            else:
                # ❌ 没有比自己更好（不管你是横跳还是往回走），都不给你蹭
                # 这里扣得稍微重一点，这样它就会想办法去打破 best
                reward = -0.05 + base_penalty
                done = False

        return s_, reward, done

    def render(self):
        # 不要在 Tk 回调里 sleep
        self.update()


# ======================= 网络 =======================
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
        action = dist.sample()
        return action.item()

    def update(self, traj):
        states = torch.tensor(np.array(traj['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(traj['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(traj['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(traj['next_states']), dtype=torch.float32).to(self.device)
        dones = torch.tensor(traj['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        probs = self.actor(states)
        chosen_probs = probs.gather(1, actions)
        log_probs = torch.log(chosen_probs + 1e-8)

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
    def __init__(self, env: Maze, agent: ActorCritic,
                 max_episodes=500, max_steps=60):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.episode = 0

    def _obs_to_state_vec(self, obs):
        if obs == 'terminal':
            return np.zeros(2, dtype=np.float32)
        row, col = self.env.coords_to_pos(obs)
        return np.array([row / (MAZE_H - 1), col / (MAZE_W - 1)], dtype=np.float32)

    def train_one_episode(self):
        obs = self.env.reset()
        state = self._obs_to_state_vec(obs)

        traj = {'states': [], 'actions': [], 'rewards': [],
                'next_states': [], 'dones': []}

        ep_ret = 0.0

        for t in range(self.max_steps):
            # 只在前 50 回合渲染，后面静默训练
            if self.episode < 50:
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
        if self.episode >= self.max_episodes:
            print("训练结束")
            return

        ep_ret = self.train_one_episode()
        self.episode += 1

        if self.episode % 20 == 0:
            print(f"Episode {self.episode}/{self.max_episodes}, return={ep_ret:.3f}")

        self.env.after(100, self.run)


# ======================= 主程序 =======================
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    env = Maze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = 2
    hidden_dim = 128
    action_dim = 4
    actor_lr = 1e-3
    critic_lr = 5e-3
    gamma = 0.9

    agent = ActorCritic(state_dim, hidden_dim, action_dim,
                        actor_lr, critic_lr, gamma, device,
                        entropy_coef=0.05)

    trainer = Trainer(env, agent,
                      max_episodes=400,
                      max_steps=60)

    env.after(100, trainer.run)
    env.mainloop()
