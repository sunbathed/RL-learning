"""
简单的贪吃蛇小游戏（Tkinter 版本）
--------------------------------
操作方式:
 - 使用键盘方向键 ↑ ↓ ← → 控制蛇的移动
 - 吃到食物得分 +1，蛇身增长一格
 - 撞到墙或自身则游戏结束并自动重置

依赖:
 - Python 标准库 (tkinter)

运行方式:
    python snake_game.py
"""

import random
import tkinter as tk
from dataclasses import dataclass


# ==================== 配置参数 ====================
CELL_SIZE = 20       # 每个网格的像素大小
GRID_WIDTH = 30      # 网格列数
GRID_HEIGHT = 20     # 网格行数
INITIAL_SPEED = 150  # 初始刷新速度（毫秒）
SPEED_STEP = 5       # 每吃一次食物稍微加速（毫秒）
MIN_SPEED = 60       # 最快速度（毫秒）


@dataclass
class Point:
    row: int
    col: int


class SnakeGame(tk.Tk):
    """Tkinter 实现的贪吃蛇游戏。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("Snake Game (Tkinter)")
        self.resizable(False, False)

        # 画布尺寸
        width = GRID_WIDTH * CELL_SIZE
        height = GRID_HEIGHT * CELL_SIZE
        self.canvas = tk.Canvas(self, width=width, height=height, bg="#111111")
        self.canvas.pack()

        # 状态变量
        self.snake: list[Point] = []
        self.direction = Point(0, 1)  # 初始向右
        self.next_direction = self.direction
        self.food: Point | None = None
        self.score = 0
        self.high_score = 0
        self.speed = INITIAL_SPEED
        self.game_running = False

        # 顶部信息栏
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(
            self,
            textvariable=self.status_var,
            font=("Consolas", 12),
            fg="#ffffff",
            bg="#222222",
        )
        self.status_label.pack(fill="x")

        self.bind("<KeyPress>", self._on_key_press)
        self._build_grid()
        self.reset_game()
        self.after(self.speed, self.game_step)

    # -------------------- UI 构建 --------------------
    def _build_grid(self) -> None:
        """绘制背景网格线（装饰）。"""
        for r in range(0, GRID_HEIGHT * CELL_SIZE, CELL_SIZE):
            self.canvas.create_line(0, r, GRID_WIDTH * CELL_SIZE, r, fill="#1e1e1e")
        for c in range(0, GRID_WIDTH * CELL_SIZE, CELL_SIZE):
            self.canvas.create_line(c, 0, c, GRID_HEIGHT * CELL_SIZE, fill="#1e1e1e")

    def _draw_cell(self, point: Point, color: str) -> int:
        """根据网格坐标绘制一个矩形并返回其 ID。"""
        x1 = point.col * CELL_SIZE
        y1 = point.row * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE
        return self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    # -------------------- 游戏逻辑 --------------------
    def reset_game(self) -> None:
        """重置蛇、食物、分数等状态。"""
        self.canvas.delete("snake")
        self.canvas.delete("food")
        self.snake.clear()

        # 初始蛇身（长度 3）
        start_row = GRID_HEIGHT // 2
        start_col = GRID_WIDTH // 2
        self.snake = [
            Point(start_row, start_col - 1),
            Point(start_row, start_col),
            Point(start_row, start_col + 1),
        ]

        self.direction = Point(0, 1)
        self.next_direction = self.direction
        self.score = 0
        self.speed = INITIAL_SPEED
        self.game_running = True

        self._spawn_food()
        self._update_status()
        self._render_snake()

    def _spawn_food(self) -> None:
        """随机生成食物，确保不与蛇身重叠。"""
        while True:
            row = random.randint(0, GRID_HEIGHT - 1)
            col = random.randint(0, GRID_WIDTH - 1)
            point = Point(row, col)
            if point not in self.snake:
                self.food = point
                break
        self.canvas.delete("food")
        if self.food:
            self._draw_cell(self.food, "#e0a800")
            self.canvas.itemconfigure("food", tags="food")

    def _update_status(self, message: str | None = None) -> None:
        """刷新顶部状态栏信息。"""
        if message:
            self.status_var.set(message)
            return
        self.high_score = max(self.high_score, self.score)
        self.status_var.set(
            f"Score: {self.score}   High Score: {self.high_score}   Speed: {int(1000 / self.speed)} fps"
        )

    def _render_snake(self) -> None:
        """绘制蛇身。"""
        self.canvas.delete("snake")
        if not self.snake:
            return
        for index, segment in enumerate(self.snake):
            color = "#4CAF50" if index == len(self.snake) - 1 else "#2e7d32"
            rect_id = self._draw_cell(segment, color)
            self.canvas.itemconfigure(rect_id, tags="snake")

    def _move_snake(self) -> bool:
        """移动蛇的逻辑，返回是否存活。"""
        head = self.snake[-1]
        new_head = Point(head.row + self.direction.row, head.col + self.direction.col)

        # 撞墙判定
        if (
            new_head.row < 0
            or new_head.col < 0
            or new_head.row >= GRID_HEIGHT
            or new_head.col >= GRID_WIDTH
        ):
            return False

        # 撞到自己
        if new_head in self.snake:
            return False

        self.snake.append(new_head)

        # 吃到食物
        if self.food and new_head == self.food:
            self.score += 1
            self.speed = max(MIN_SPEED, self.speed - SPEED_STEP)
            self._spawn_food()
        else:
            # 移除尾巴保持长度
            self.snake.pop(0)

        return True

    def game_step(self) -> None:
        """游戏主循环。"""
        if not self.game_running:
            self.after(300, self.game_step)
            return

        self.direction = self.next_direction
        alive = self._move_snake()
        if not alive:
            self.game_running = False
            self._update_status("Game Over! Press Space to restart.")
            self.after(800, self.game_step)
            return

        self._render_snake()
        self._update_status()
        self.after(self.speed, self.game_step)

    # -------------------- 事件处理 --------------------
    def _on_key_press(self, event: tk.Event) -> None:
        """处理方向键或重启键。"""
        key = event.keysym.lower()
        if key == "space":
            self.reset_game()
            return

        direction_map = {
            "up": Point(-1, 0),
            "down": Point(1, 0),
            "left": Point(0, -1),
            "right": Point(0, 1),
        }
        if key not in direction_map:
            return

        new_dir = direction_map[key]

        # 防止直接掉头
        if len(self.snake) > 1:
            head = self.snake[-1]
            neck = self.snake[-2]
            current_dir = Point(head.row - neck.row, head.col - neck.col)
            if (new_dir.row, new_dir.col) == (-current_dir.row, -current_dir.col):
                return

        self.next_direction = new_dir


if __name__ == "__main__":
    random.seed()
    game = SnakeGame()
    game.mainloop()

