import gymnasium as gym
import numpy as np


def test_gymnasium_no_render():
    print("=== Gymnasium 无界面测试 ===")

    try:
        # 创建环境，不启用渲染
        env = gym.make('CartPole-v1')

        # 重置环境
        observation, info = env.reset()
        print(f"✓ 环境创建和重置成功")
        print(f"  观察值形状: {observation.shape}")
        print(f"  观察值范围: [{observation.min():.3f}, {observation.max():.3f}]")

        # 测试几步交互
        total_reward = 0
        for i in range(10):  # 只测试10步
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"  步骤 {i}: 动作={action}, 奖励={reward}, 终止={terminated}")

            if terminated or truncated:
                print("  回合提前结束")
                break

        print(f"✓ 交互测试完成，总奖励: {total_reward}")
        env.close()

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

    print("Gymnasium 安装成功！")
    return True


if __name__ == "__main__":
    test_gymnasium_no_render()