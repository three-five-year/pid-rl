# train_ppo_fixed.py - 修复版训练脚本

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
from rl_env_f16_formation import FormationEnvFixed
from config import TRAIN_CONFIG_FIXED


class CompactLoggerCallback(BaseCallback):
    """紧凑型日志回调"""

    def __init__(self, log_freq=4000):
        super().__init__()
        self.log_freq = log_freq

        # 分别存储每个完整episode的数据
        self.episode_rewards = []
        self.episode_avg_rewards = []
        self.episode_lengths = []
        self.episode_track_errors = []
        self.episode_max_track_errors = []  # 🔥 新增: 最大误差
        self.episode_min_distances = []
        self.episode_collisions = []

    def _on_step(self) -> bool:
        # 收集完整episode的统计数据
        infos = self.locals.get('infos', [])
        for info in infos:
            # 从Monitor提取episode奖励
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode'].get('l', 1)
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_avg_rewards.append(ep_reward / max(ep_length, 1))

            # 从environment的episode_stats提取其他指标
            if 'episode_stats' in info and info['episode_stats']:
                stats = info['episode_stats']

                self.episode_track_errors.append(stats.get('avg_tracking_error', 0))
                self.episode_max_track_errors.append(stats.get('max_tracking_error', 0))
                self.episode_min_distances.append(stats.get('min_distance_ever', 0))
                self.episode_collisions.append(1 if stats.get('collision') else 0)

        # 每log_freq步打印
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            n_recent = min(10, len(self.episode_rewards))

            avg_reward = np.mean(self.episode_avg_rewards[-n_recent:])
            avg_track_err = np.mean(self.episode_track_errors[-n_recent:]) if len(
                self.episode_track_errors) >= n_recent else 0
            max_track_err = np.mean(self.episode_max_track_errors[-n_recent:]) if len(
                self.episode_max_track_errors) >= n_recent else 0
            avg_min_dist = np.mean(self.episode_min_distances[-n_recent:]) if len(
                self.episode_min_distances) >= n_recent else 0
            collision_rate = np.mean(self.episode_collisions[-n_recent:]) if len(
                self.episode_collisions) >= n_recent else 0

            print(f"Steps:{self.num_timesteps:>7} | "
                  f"AvgRwd:{avg_reward:>7.3f} | "
                  f"AvgErr:{avg_track_err:>5.0f}ft | "
                  f"MaxErr:{max_track_err:>5.0f}ft | "  # 🔥 显示最大误差
                  f"MinDist:{avg_min_dist:>5.0f}ft | "
                  f"Collision:{collision_rate * 100:>3.0f}%")

        return True


def make_env(config_dict):
    def _init():
        env = FormationEnvFixed(config_dict)
        env = Monitor(env)
        return env

    return _init


def train():
    """修复版训练"""
    config = TRAIN_CONFIG_FIXED.to_dict()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/ppo_fixed_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 80)
    print("🔥 PPO+PID FIXED Training - Focus on Tracking Performance")
    print("=" * 80)
    print(f"Log Directory: {log_dir}")
    print(f"Training Steps: {config['total_timesteps']}")
    print(f"Warm-Start: {config['warmstart_steps']} steps ({config['warmstart_steps'] * 0.05:.0f}s)")
    print(f"RL Activation: TrackErr>{config['rl_threshold']}ft & Dist>{config['distance_safety_margin']}ft")
    print("\n🔥 KEY FIXES:")
    print(f"  - Elevator Limit: ±{config['delta_elevator_limit']}° (was ±2.0°)")
    print("=" * 80)

    # 创建环境
    n_envs = config.get('n_envs', 8)
    env = DummyVecEnv([make_env(config) for _ in range(n_envs)])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=20.0,
        gamma=config.get('gamma', 0.99)
    )

    # PPO模型
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],
            vf=[256, 256, 128]
        ),
        activation_fn=torch.nn.Tanh
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        learning_rate=config.get('learning_rate'),
        n_steps=config.get('n_steps', 4096),
        batch_size=config.get('batch_size', 128),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.1),
        ent_coef=config.get('ent_coef'),
        vf_coef=config.get('vf_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=log_dir
    )

    # 回调
    logger_callback = CompactLoggerCallback(log_freq=4000)
    eval_env = DummyVecEnv([make_env(config)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=20.0,
        gamma=config.get('gamma', 0.99),
        training=False
    )
    eval_env.obs_rms = env.obs_rms
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=config.get('eval_freq', 20000),
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )
    callbacks = CallbackList([logger_callback, eval_callback])

    print("\n🔥 Training Start...")
    print("-" * 80)
    print("Steps:   | AvgRwd: | AvgErr: | MaxErr: | MinDist: | Collision:")
    print("-" * 80)

    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callbacks,
        progress_bar=False
    )

    # 保存
    final_model_path = f"{log_dir}/final_model"
    model.save(final_model_path)
    env.save(f"{log_dir}/vec_normalize.pkl")

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print(f"Model: {final_model_path}")
    print(f"Best Model: {os.path.join(log_dir, 'best_model.zip')}")
    print("=" * 80)

    return model, env


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    train()
