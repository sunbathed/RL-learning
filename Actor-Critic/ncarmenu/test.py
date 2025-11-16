import tkinter as tk
import numpy as np
import time

UNIT = 60     # 每格像素大小
MAZE_H = 5    # 行数
MAZE_W = 5    # 列数


class MultiTruckMaze(tk.Tk, object):
    def __init__(self, n_trucks=3):
        super().__init__()
        assert n_trucks <= 4, "这个 5×5 小迷宫最多放 4 辆车演示就够了"
        self.n_trucks = n_trucks

        self.action_space = ['u', 'd', 'r', 'l', 's']
        self.n_actions = len(self.action_space)

        self.title('Multi-Truck Maze (Mining Transport Simplified)')
        self.geometry(f'{MAZE_W * UNIT}x{MAZE_H * UNIT}')

        self._build_maze()

        self.priorities = list(range(self.n_trucks))[::-1]
        self.remain_travel = [3] * self.n_trucks

        self.step_penalty = -0.01
        self.block_penalty = -0.05

        self.is_done = False

        self._init_rc_refs()
        self._build_bfs_distance()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT)
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r)

        origin = np.array([UNIT/2, UNIT/2])

        # 障碍 两个固定
        obs_A_center = origin + np.array([UNIT * 1, UNIT * 2])
        self.obsA = self.canvas.create_rectangle(
            obs_A_center[0] - UNIT/2 + 5, obs_A_center[1] - UNIT/2 + 5,
            obs_A_center[0] + UNIT/2 - 5, obs_A_center[1] + UNIT/2 - 5,
            fill='black'
        )
        obs_B_center = origin + np.array([UNIT * 2, UNIT * 1])
        self.obsB = self.canvas.create_rectangle(
            obs_B_center[0] - UNIT/2 + 5, obs_B_center[1] - UNIT/2 + 5,
            obs_B_center[0] + UNIT/2 - 5, obs_B_center[1] + UNIT/2 - 5,
            fill='black'
        )

        # 装载区 (2,2)
        lz_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.lz = self.canvas.create_rectangle(
            lz_center[0] - UNIT/2 + 5, lz_center[1] - UNIT/2 + 5,
            lz_center[0] + UNIT/2 - 5, lz_center[1] + UNIT/2 - 5,
            fill='lightblue'
        )

        # 卸载区 (4,4)（可按你需求调整）
        uz_center = origin + np.array([UNIT * 4, UNIT * 4])
        self.uz = self.canvas.create_rectangle(
            uz_center[0] - UNIT/2 + 5, uz_center[1] - UNIT/2 + 5,
            uz_center[0] + UNIT/2 - 5, uz_center[1] + UNIT/2 - 5,
            fill='orange'
        )

        # 起始位置：为 3 辆车设定三个不同起点
        self.truck_items = []
        start_positions = [
            origin + np.array([0, 0]),           # 第1辆：0,0
            origin + np.array([UNIT * 0, UNIT * 4]),  # 第2辆：0,4
            origin + np.array([UNIT * 4, 0]),        # 第3辆：4,0
        ]
        colors = ['red', 'blue', 'green', 'purple']
        for i in range(self.n_trucks):
            center = start_positions[i]
            item = self.canvas.create_rectangle(
                center[0] - UNIT/2 + 5, center[1] - UNIT/2 + 5,
                center[0] + UNIT/2 - 5, center[1] + UNIT/2 - 5,
                fill=colors[i]
            )
            self.truck_items.append(item)

        self.canvas.pack()

    def _coords_to_rc(self, coords):
        x1, y1, x2, y2 = coords
        cx, cy = (x1 + x2)/2.0, (y1 + y2)/2.0
        c = int(round((cx - UNIT/2) / UNIT))
        r = int(round((cy - UNIT/2) / UNIT))
        return r, c

    def _init_rc_refs(self):
        self.loading_rc = self._coords_to_rc(self.canvas.coords(self.lz))
        self.unloading_rc = self._coords_to_rc(self.canvas.coords(self.uz))
        self.obstacles_rc = {
            self._coords_to_rc(self.canvas.coords(self.obsA)),
            self._coords_to_rc(self.canvas.coords(self.obsB))
        }

    def _build_bfs_distance(self):
        H, W = MAZE_H, MAZE_W
        dist = [[float('inf')] * W for _ in range(H)]
        from collections import deque
        q = deque([self.unloading_rc])
        gr, gc = self.unloading_rc
        dist[gr][gc] = 0
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in self.obstacles_rc:
                    if dist[nr][nc] > dist[r][c] + 1:
                        dist[nr][nc] = dist[r][c] + 1
                        q.append((nr, nc))
        self.dist_map = dist

    def _get_obs(self):
        obs = []
        for item in self.truck_items:
            r, c = self._coords_to_rc(self.canvas.coords(item))
            obs.extend([r, c])
        return np.array(obs, dtype=np.float32)

    def reset(self):
        origin = np.array([UNIT/2, UNIT/2])
        start_positions = [
            origin + np.array([0, 0]),
            origin + np.array([0, UNIT * 4]),
            origin + np.array([UNIT * 4, 0]),
        ]
        for i, item in enumerate(self.truck_items):
            center = start_positions[i]
            self.canvas.coords(
                item,
                center[0] - UNIT/2 +5, center[1] - UNIT/2 +5,
                center[0] + UNIT/2 -5, center[1] + UNIT/2 -5
            )
        self.is_done = False
        self.remain_travel = [3] * self.n_trucks
        self.update()
        return self._get_obs()

    def _compute_candidate_moves(self, actions):
        rc_list = []
        cand_rc = []
        for item in self.truck_items:
            rc_list.append(self._coords_to_rc(self.canvas.coords(item)))

        for i, (rc, a) in enumerate(zip(rc_list, actions)):
            r, c = rc
            nr, nc = r, c
            if self.remain_travel[i] <= 0 or a == 4:
                cand_rc.append((nr, nc))
                continue
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

    def _resolve_conflicts_with_priority(self, rc_list, cand_rc):
        occupied = set()
        final_rc = list(cand_rc)
        blocked = [False] * self.n_trucks
        order = sorted(range(self.n_trucks),
                       key=lambda i: self.priorities[i],
                       reverse=True)
        for i in order:
            target = cand_rc[i]
            if target in self.obstacles_rc:
                final_rc[i] = rc_list[i]
                blocked[i] = True
                continue
            if target in occupied:
                final_rc[i] = rc_list[i]
                blocked[i] = True
            else:
                occupied.add(target)
        return final_rc, blocked

    def step(self, actions):
        if self.is_done:
            return self._get_obs(), [0.0]*self.n_trucks, True, {}

        rc_list, cand_rc = self._compute_candidate_moves(actions)
        final_rc, blocked = self._resolve_conflicts_with_priority(rc_list, cand_rc)

        rewards_each = [self.step_penalty] * self.n_trucks
        done_flags = [False]*self.n_trucks

        for i, item in enumerate(self.truck_items):
            old_rc = rc_list[i]
            new_rc = final_rc[i]

            # — 移动画面 —
            dr = (new_rc[1] - old_rc[1]) * UNIT
            dc = (new_rc[0] - old_rc[0]) * UNIT
            self.canvas.move(item, dr, dc)

            # 如果真的移动了，消耗可行驶步数
            if new_rc != old_rc:
                self.remain_travel[i] -= 1

            # — 距离势函数奖励 —
            d_prev = self.dist_map[old_rc[0]][old_rc[1]]
            d_now  = self.dist_map[new_rc[0]][new_rc[1]]
            if np.isfinite(d_prev) and np.isfinite(d_now):
                diff = d_prev - d_now
                rewards_each[i] += 0.3 * np.tanh(diff)

            # — 卡住惩罚 —
            if blocked[i] and new_rc == old_rc:
                rewards_each[i] += self.block_penalty

            # — 装载逻辑（空车到装载区）—
            if new_rc == self.loading_rc and getattr(self, "load_state", None) is not None:
                if self.load_state[i] == "empty":
                    rewards_each[i] += 3.0      # 装载奖励
                    self.load_state[i] = "full"

            # — 卸载逻辑（满载到卸载区）—
            if new_rc == self.unloading_rc and getattr(self, "load_state", None) is not None:
                if self.load_state[i] == "full":
                    rewards_each[i] += 5.0      # 卸载任务完成奖励
                    self.load_state[i] = "empty"
                    done_flags[i] = True

            # — 空车占道／等待惩罚（示例）—
            if self.load_state[i] == "empty":
                # 假设有一个函数 is_on_critical_channel(rc) 判断是否处于关键通道
                if is_on_critical_channel(new_rc) and maybe_waiting_too_long(i):
                    rewards_each[i] += -0.1     # 空车占道惩罚

        done = any(done_flags)
        if done:
            self.is_done = True

        # — 重置可行驶距离 —
        for i in range(self.n_trucks):
            if self.remain_travel[i] <= 0:
                r, c = final_rc[i]
                if np.isfinite(self.dist_map[r][c]):
                    self.remain_travel[i] = 3

        self.update()

        global_reward = float(np.mean(rewards_each))
        return self._get_obs(), global_reward, done, {
            "rewards_each": rewards_each,
            "blocked": blocked,
            "done_flags": done_flags,
        }


    def render(self, delay=0.05):
        time.sleep(delay)
        self.update()
