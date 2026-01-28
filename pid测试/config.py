# config.py - 修复版配置

from dataclasses import dataclass
import numpy as np


@dataclass
class PPOConfigFixed:
    """修复版PPO配置 - 解决奖励尺度、高度控制、初始条件问题"""

    # 环境参数
    num_agents: int = 4
    dt: float = 0.05
    max_steps: int = 2400  # 120s

    # 激活策略
    warmstart_steps: int = 600      # 30s warm-up
    rl_threshold: float = 120.0       # 误差>120ft时激活
    distance_safety_margin: float = 300.0  # 距离<300ft时缓慢削弱RL
    rl_activation_ramp_sec: float = 5.0  # RL介入渐变时间

    # 🔥 修复1: 调整奖励权重与尺度，增强平滑引导
    w_track_h: float = 2.0            # 水平跟踪权重
    w_track_v: float = 3.0            # 🔥 高度跟踪权重（单独控制）
    w_safe: float = 0.2               # 安全权重
    w_ctrl: float = 0.05              # 控制惩罚（降低）
    w_smooth: float = 0.1             # 平滑惩罚（降低）

    # 观测归一化参数
    euler_norm: float = np.pi
    pqr_norm: float = 5.0

    # 安全参数
    d_collision: float = 100.0
    d_danger: float = 160.0
    d_safe: float = 350.0

    # 🔥 修复2: 增大电梯舵面限幅以改善高度控制
    delta_throttle_limit: float = 0.03
    delta_elevator_limit: float = 6.0  # 提升到6.0 (原2.0)
    delta_aileron_limit: float = 2.0
    delta_rudder_limit: float = 2.0

    # 🔥 修复3: 标准初始位置（与main.py完全一致）
    standard_initial_offsets: np.ndarray = None
    planner_gain: float = 3.0
    planner_steps_turn: int = 4
    planner_steps_straight: int = 10

    def __post_init__(self):
        """初始化标准初始偏移量"""
        self.standard_initial_offsets = np.array([
            [0.0, 0.0, 0.0],
            [-300.0, -150.0, 0.0],
            [-500.0, -500.0, 0.0],
            [-1000.0, 0.0, 0.0],
        ])

    # PPO超参数
    learning_rate: float = 1e-4
    n_steps: int = 4096
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # 训练参数
    total_timesteps: int = 1000000
    n_envs: int = 8
    eval_freq: int = 20000
    save_freq: int = 50000

    def to_dict(self):
        result = self.__dict__.copy()
        # 将numpy array转换为list以便序列化
        if isinstance(result.get('standard_initial_offsets'), np.ndarray):
            result['standard_initial_offsets'] = result['standard_initial_offsets'].tolist()
        return result


# 训练配置
TRAIN_CONFIG_FIXED = PPOConfigFixed(
    total_timesteps=1000000,
    n_envs=8,
    warmstart_steps=600,
    rl_threshold=120.0,
    w_track_h=2.0,
    w_track_v=3.0,  # 🔥 关键: 单独的高度权重
    w_safe=0.2
)

# 调试配置
DEBUG_CONFIG_FIXED = PPOConfigFixed(
    total_timesteps=50000,
    n_envs=2,
    warmstart_steps=600,
    eval_freq=5000
)
