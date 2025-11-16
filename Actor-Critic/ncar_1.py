#

import numpy as np
import tkinter as tk
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
from ncarmenu import test  # 多车环境

UNIT = 40
MAZE_H = 4
MAZE_W = 4

# ====================== 集中式 Actor-Critic ======================
class MultiHeadPolicy(nn.Module):
    """
    输入: state (B, state_dim)
    输出: probs (B, n_trucks, n_actions)
    """
    def __init__(self, state_dim, hidden_dim, n_trucks, n_actions):
        super().__init__()
        self.n_trucks = n_trucks
        self.n_actions = n_actions
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_trucks * n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)                      # (B, n_trucks * n_actions)
        logits = logits.view(-1, self.n_trucks, self.n_actions)
        probs = F.softmax(logits, dim=-1)
        return probs


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CentralActorCritic:
    def __init__(self, state_dim, hidden_dim, n_trucks, n_actions,
                 actor_lr, critic_lr, gamma, device, entropy_coef=0.01):
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.n_trucks = n_trucks
        self.n_actions = n_actions

        self.actor = MultiHeadPolicy(state_dim, hidden_dim, n_trucks, n_actions).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state_vec):
        """
        state_vec: np.array(state_dim,)
        返回: [a0, a1, ..., a_(n_trucks-1)]
        """
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(s)          # (1, n_trucks, n_actions)
        probs = probs[0]               # (n_trucks, n_actions)
        actions = []
        for i in range(self.n_trucks):
            dist = torch.distributions.Categorical(probs[i])
            a = dist.sample().item()
            actions.append(a)
        return actions

    def update(self, traj):
        states = torch.tensor(np.array(traj['states']),
                              dtype=torch.float32, device=self.device)                 # (T, S)
        next_states = torch.tensor(np.array(traj['next_states']),
                                   dtype=torch.float32, device=self.device)           # (T, S)
        rewards = torch.tensor(traj['rewards'],
                               dtype=torch.float32, device=self.device).view(-1, 1)   # (T, 1)
        dones = torch.tensor(traj['dones'],
                             dtype=torch.float32, device=self.device).view(-1, 1)     # (T, 1)
        actions = torch.tensor(traj['actions'],
                               dtype=torch.int64, device=self.device)                 # (T, n_trucks)

        # ----- Critic -----
        with torch.no_grad():
            v_next = self.critic(next_states)
            td_target = rewards + self.gamma * v_next * (1 - dones)
            # 防爆：裁剪 TD 目标
            td_target = torch.clamp(td_target, -5.0, 5.0)

        v = self.critic(states)
        td_delta = td_target - v

        critic_loss = F.mse_loss(v, td_target)

        # ----- Actor (集中式) -----
        probs = self.actor(states)  # (T, n_trucks, n_actions)
        # 取出每辆车实际执行动作的概率
        chosen_probs = probs.gather(2, actions.unsqueeze(-1))   # (T, n_trucks, 1)
        log_probs = torch.log(chosen_probs + 1e-8).squeeze(-1)  # (T, n_trucks)
        # 联合动作 log_prob = 所有车 log_prob 之和
        joint_log_prob = log_probs.sum(dim=1, keepdim=True)     # (T, 1)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=2).mean()

        actor_loss = (-joint_log_prob * td_delta.detach()).mean() - self.entropy_coef * entropy

        # ----- 更新 -----
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()


# ====================== 训练调度 ======================
class Trainer:
    def __init__(self, env, agent, max_episodes=600, max_steps=80):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.episode = 0
        self.return_list = []

    def train_one_episode(self):
        obs = self.env.reset()
        state = obs
        traj = {'states': [], 'actions': [], 'rewards': [],
                'next_states': [], 'dones': []}
        ep_ret = 0.0

        for t in range(self.max_steps):
            self.env.render(delay=0.03)
            action_list = self.agent.take_action(state)   # [a0, a1, a2]
            next_obs, reward, done, info = self.env.step(action_list)

            traj['states'].append(state)
            traj['actions'].append(action_list)
            traj['rewards'].append(reward)
            traj['next_states'].append(next_obs)
            traj['dones'].append(done)

            state = next_obs
            ep_ret += reward
            if done:
                break

        if traj['states']:
            self.agent.update(traj)
        return ep_ret

    def run(self):
        if self.episode >= self.max_episodes:
            print("训练结束 ✅")
            self._plot_curve()
            return

        ep_ret = self.train_one_episode()
        self.return_list.append(ep_ret)

        if (self.episode + 1) % 20 == 0:
            avg_ret = np.mean(self.return_list[-20:])
            print(f"Episode {self.episode+1}/{self.max_episodes} | "
                  f"Return={ep_ret:.3f} | Avg={avg_ret:.3f}")

        self.episode += 1
        self.env.after(40, self.run)

    def _plot_curve(self):
        if not self.return_list:
            return

        def moving_average(x, window=20):
            if len(x) < window:
                return x
            x = np.array(x, dtype=np.float32)
            cumsum = np.cumsum(np.insert(x, 0, 0.0))
            ma = (cumsum[window:] - cumsum[:-window]) / window
            head = [ma[0]] * (len(x) - len(ma))
            return head + ma.tolist()

        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(moving_average(self.return_list, 15), linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Centralized Actor-Critic on Multi-Truck Maze")
        plt.grid(alpha=0.3)
        path = os.path.join("results", "central_ac_multi_truck.png")
        plt.savefig(path, dpi=300)
        plt.show()
        print("曲线已保存到:", path)


# ====================== 主函数 ======================
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    env = MultiTruckMaze(n_trucks=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    n_trucks = env.n_trucks
    n_actions = env.n_actions
    state_dim = 2 * n_trucks     # [r0,c0,r1,c1,r2,c2]

    agent = CentralActorCritic(
        state_dim=state_dim,
        hidden_dim=128,
        n_trucks=n_trucks,
        n_actions=n_actions,
        actor_lr=1e-3,
        critic_lr=2e-3,
        gamma=0.9,
        device=device,
        entropy_coef=0.02,
    )

    trainer = Trainer(env, agent, max_episodes=600, max_steps=80)
    env.after(200, trainer.run)
    env.mainloop()
