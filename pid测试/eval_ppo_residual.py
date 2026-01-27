# eval_ppo_residual.py
"""
PPO+PID 对比评估脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_env_f16_formation import FormationEnvFixed
from config import TRAIN_CONFIG_FIXED


class Evaluator:
    """评估器"""

    def __init__(self, model_path, vec_normalize_path=None):
        self.model_path = model_path
        self.vec_normalize_path = vec_normalize_path

        # 加载模型
        config = TRAIN_CONFIG_FIXED.to_dict()

        env = DummyVecEnv([lambda: FormationEnvFixed(config)])

        if vec_normalize_path:
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False

        self.model = PPO.load(model_path, env=env)
        self.env = env

    def evaluate_episode(self, seed=42):
        """评估单个episode"""
        obs = self.env.reset()

        history = {
            'time': [],
            'positions': [[] for _ in range(4)],
            'track_errors': [[] for _ in range(4)],
            'min_distance': [],
            'rewards': []
        }

        done = False
        t = 0

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)

            # 记录数据
            history['time'].append(t * 0.05)
            history['rewards'].append(reward[0])

            if 'min_distance' in info[0]:
                history['min_distance'].append(info[0]['min_distance'])

            t += 1

            if done[0]:
                break

        return history

    def compare_methods(self, n_episodes=5):
        """对比三种方法"""

        results = {
            'baseline': [],
            'adaptive_pid': [],
            'adaptive_pid_ppo': []
        }

        for ep in range(n_episodes):
            print(f"\nEpisode {ep + 1}/{n_episodes}")

            # PPO+PID
            history_ppo = self.evaluate_episode(seed=ep)
            results['adaptive_pid_ppo'].append(history_ppo)

        return results


def visualize_comparison(baseline_hist, adaptive_hist, ppo_hist):
    """可视化对比结果"""

    fig = plt.figure(figsize=(20, 12))

    # 1. 最小距离对比
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(baseline_hist['time'], baseline_hist['min_distance'],
             'r-', linewidth=2, label='Baseline')
    ax1.plot(adaptive_hist['time'], adaptive_hist['min_distance'],
             'g-', linewidth=2, label='Adaptive+PID')
    ax1.plot(ppo_hist['time'], ppo_hist['min_distance'],
             'b-', linewidth=2, label='Adaptive+(PID+PPO)')

    ax1.axhspan(0, 100, alpha=0.3, color='red', label='Collision')
    ax1.axhspan(100, 160, alpha=0.3, color='yellow', label='Danger')

    ax1.set_title('Minimum Distance Comparison')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (ft)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 跟踪误差对比
    ax2 = fig.add_subplot(2, 3, 2)

    for i in range(4):
        if baseline_hist['track_errors'][i]:
            ax2.plot(baseline_hist['time'], baseline_hist['track_errors'][i],
                     '--', alpha=0.5, label=f'Baseline Agent {i + 1}')

    for i in range(4):
        if ppo_hist['track_errors'][i]:
            ax2.plot(ppo_hist['time'], ppo_hist['track_errors'][i],
                     '-', linewidth=2, label=f'PPO Agent {i + 1}')

    ax2.set_title('Tracking Error Comparison')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (ft)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. 奖励曲线
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(ppo_hist['time'], ppo_hist['rewards'],
             'b-', linewidth=2, label='PPO Reward')
    ax3.set_title('PPO Reward Over Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 统计对比
    ax4 = fig.add_subplot(2, 3, 4)

    metrics = ['Min Dist\n(ft)', 'Avg Track\nError (ft)', 'Final Form\nError (ft)']

    baseline_vals = [
        min(baseline_hist['min_distance']),
        np.mean([np.mean(errs) for errs in baseline_hist['track_errors'] if errs]),
        0  # placeholder
    ]

    ppo_vals = [
        min(ppo_hist['min_distance']),
        np.mean([np.mean(errs) for errs in ppo_hist['track_errors'] if errs]),
        0  # placeholder
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax4.bar(x - width / 2, baseline_vals, width, label='Baseline', color='lightcoral')
    ax4.bar(x + width / 2, ppo_vals, width, label='PPO+PID', color='lightblue')

    ax4.set_ylabel('Value')
    ax4.set_title('Key Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('ppo_pid_comparison.png', dpi=150)
    print("\n✅ Comparison saved to 'ppo_pid_comparison.png'")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained PPO model')
    parser.add_argument('--vec-normalize', type=str, default=None,
                        help='Path to VecNormalize stats')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of evaluation episodes')

    args = parser.parse_args()

    evaluator = Evaluator(args.model, args.vec_normalize)
    results = evaluator.compare_methods(n_episodes=args.episodes)

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
