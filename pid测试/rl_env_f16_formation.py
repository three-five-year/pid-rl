# rl_env_f16_formation.py - 完全修复版

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple
from f16Model import F16
from pid_tracker import PIDTracker
from adaptive_negotiation_trajectory import AdaptiveNegotiationTrajectory


class FormationEnvFixed(gym.Env):
    """
    完全修复版环境:
    1. 奖励函数归一化，限制单步范围在[-10, +5]
    2. 分离水平/高度误差，独立惩罚
    3. 初始条件与main.py完全一致
    4. 记录三类误差用于可视化
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.N = 4
        self.dt = config.get('dt', 0.05)
        self.max_steps = config.get('max_steps', 2400)

        # 激活参数
        self.warmstart_steps = config.get('warmstart_steps', 1200)
        self.rl_threshold = config.get('rl_threshold', 150.0)
        self.distance_safety_margin = config.get('distance_safety_margin', 180.0)

        # 🔥 修复1: 分离水平/高度权重
        self.w_track_h = config.get('w_track_h', 3.0)
        self.w_track_v = config.get('w_track_v', 2.0)
        self.w_safe = config.get('w_safe', 2.0)
        self.w_ctrl = config.get('w_ctrl', 0.05)
        self.w_smooth = config.get('w_smooth', 0.1)

        # 安全参数
        self.d_collision = 100.0
        self.d_danger = 160.0
        self.d_safe = 350.0

        # 🔥 修复2: 增大电梯舵面限幅
        self.delta_throttle_limit = config.get('delta_throttle_limit', 0.03)
        self.delta_elevator_limit = config.get('delta_elevator_limit', 5.0)
        self.delta_aileron_limit = config.get('delta_aileron_limit', 2.0)
        self.delta_rudder_limit = config.get('delta_rudder_limit', 2.0)

        # 🔥 修复3: 标准初始偏移量
        standard_offsets = config.get('standard_initial_offsets')
        if standard_offsets is not None:
            if isinstance(standard_offsets, list):
                self.standard_initial_offsets = np.array(standard_offsets)
            else:
                self.standard_initial_offsets = standard_offsets
        else:
            # 默认值（与main.py一致）
            self.standard_initial_offsets = np.array([
                [0.0, 0.0, 0.0],
                [-300.0, -150.0, 0.0],
                [-500.0, -500.0, 0.0],
                [-1000.0, 0.0, 0.0],
            ])

        # 动作/观测空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.N * 4,), dtype=np.float32
        )
        obs_dim = 18 * self.N
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 环境组件
        self.agents = [F16(time_step=self.dt) for _ in range(self.N)]
        self.trackers = [
            PIDTracker(dt=self.dt, agent_role="leader"),
            PIDTracker(dt=self.dt, agent_role="follower_direct"),
            PIDTracker(dt=self.dt, agent_role="follower_direct"),
            PIDTracker(dt=self.dt, agent_role="follower_indirect")
        ]

        A = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
        leader_access = np.array([1, 0, 0, 0])
        self.desired_offsets = [
            np.array([0.0, 0.0, 0.0]),
            np.array([-500.0, -500.0, 0.0]),
            np.array([-500.0, 500.0, 0.0]),
            np.array([-1000.0, 0.0, 0.0]),
        ]

        self.planner = AdaptiveNegotiationTrajectory(
            N=self.N, adjacency_matrix=A, leader_access=leader_access,
            formation_offsets=self.desired_offsets, k_gain=2.0,
            sensing_radius=350.0, safety_radius=100.0
        )

        self.leader_start_pos = np.array([1000.0, 0.0, -5000.0])
        self.leader_velocity = 350.0

        # 状态变量
        self.step_count = 0
        self.current_time = 0.0
        self.leader_pos = None
        self.leader_vel = None
        self.leader_heading = 0.0
        self.turn_rate = 0.0
        self.prev_actions = np.zeros(self.N * 4)
        self.rl_active = False

        # 🔥 修复4: 扩展统计变量，记录三类误差
        self.episode_stats = {
            'total_reward': 0.0,
            'min_distance_ever': float('inf'),
            'sum_tracking_error': 0.0,
            'sum_error_total': 0.0,  # ε₁: 真实跟踪误差
            'sum_error_planning': 0.0,  # ε₂: 轨迹重构误差
            'sum_error_tracking': 0.0,  # ε₃: 控制器跟踪误差
            'sum_error_horizontal': 0.0,  # 水平误差
            'sum_error_vertical': 0.0,  # 高度误差
            'sum_rl_activation': 0,
            'step_count': 0,
            'collision': False,
            'max_tracking_error': 0.0
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.current_time = 0.0
        self.leader_heading = 0.0
        self.turn_rate = 0.0
        self.prev_actions = np.zeros(self.N * 4)
        self.rl_active = False

        # 重置统计
        self.episode_stats = {
            'total_reward': 0.0,
            'min_distance_ever': float('inf'),
            'sum_tracking_error': 0.0,
            'sum_error_total': 0.0,
            'sum_error_planning': 0.0,
            'sum_error_tracking': 0.0,
            'sum_error_horizontal': 0.0,
            'sum_error_vertical': 0.0,
            'sum_rl_activation': 0,
            'step_count': 0,
            'collision': False,
            'max_tracking_error': 0.0
        }

        # 🔥 修复3: 使用标准初始位置（与main.py完全一致）
        initial_positions = self.leader_start_pos + self.standard_initial_offsets

        # ❌ 移除随机噪声（保证公平性）
        # noise = np.random.uniform(-30, 30, (self.N, 3))
        # noise[:, 2] = 0
        # initial_positions[1:] += noise[1:]

        self.planner.initialize(initial_positions)
        self.leader_pos = self.leader_start_pos.copy()
        self.leader_vel = np.array([self.leader_velocity, 0.0, 0.0])

        for i, agent in enumerate(self.agents):
            agent.reset(
                position=initial_positions[i],
                velocity_body=np.array([350.0, 0.0, 0.0]),
                mach=0.35
            )

        # 🔥 重置PID积分器
        for tracker in self.trackers:
            tracker.int_lat = 0.0
            tracker.int_v = 0.0
            tracker.alpha_int = 0.0

        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        self.current_time = self.step_count * self.dt

        # Warm-Start阶段
        if self.step_count <= self.warmstart_steps:
            return self._pure_pid_step()

        # 更新领机轨迹
        self._update_leader_trajectory()

        c, s = np.cos(self.leader_heading), np.sin(self.leader_heading)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        current_offsets = [R_z @ off for off in self.desired_offsets]
        self.planner.update_offsets(current_offsets)

        is_turning = abs(self.turn_rate) > 1e-3
        if is_turning:
            for _ in range(3):
                self.planner.step(self.leader_pos, self.leader_vel, self.dt / 3)
        else:
            for _ in range(8):
                self.planner.step(self.leader_pos, self.leader_vel, self.dt / 8)

        target_pos_all, target_vel_all = self.planner.get_target_trajectories()

        # 计算当前状态
        current_positions = np.array([agent.position for agent in self.agents])

        # 🔥 修复4: 计算三类误差
        error_total_list = []
        error_planning_list = []
        error_tracking_list = []
        error_horizontal_list = []
        error_vertical_list = []

        for i in range(self.N):
            # 理想编队位置
            ideal_formation = self.leader_pos + current_offsets[i]

            # ε₁: 真实跟踪误差 (Agent vs 理想编队)
            e_total = current_positions[i] - ideal_formation
            error_total = np.linalg.norm(e_total)
            error_total_list.append(error_total)

            # 水平/高度分解
            error_horizontal = np.linalg.norm(e_total[0:2])
            error_vertical = abs(e_total[2])
            error_horizontal_list.append(error_horizontal)
            error_vertical_list.append(error_vertical)

            # ε₂: 轨迹重构误差 (协商轨迹 vs 理想编队)
            error_planning = np.linalg.norm(target_pos_all[i] - ideal_formation)
            error_planning_list.append(error_planning)

            # ε₃: 控制器跟踪误差 (Agent vs 协商轨迹)
            error_tracking = np.linalg.norm(current_positions[i] - target_pos_all[i])
            error_tracking_list.append(error_tracking)

        avg_error_total = np.mean(error_total_list)
        avg_error_horizontal = np.mean(error_horizontal_list)
        avg_error_vertical = np.mean(error_vertical_list)
        max_track_err = max(error_tracking_list)

        # 🔥 修复: 正确计算最小距离
        min_dist = float('inf')
        for i in range(self.N):
            for j in range(i + 1, self.N):
                d = np.linalg.norm(current_positions[i] - current_positions[j])
                if d > 1.0:
                    min_dist = min(min_dist, d)

        if min_dist == float('inf'):
            min_dist = 500.0

        # RL激活逻辑
        self.rl_active = (
                max_track_err > self.rl_threshold and
                min_dist > self.distance_safety_margin
        )

        # 解析动作
        action = np.clip(action, -1.0, 1.0)
        delta_u_all = np.zeros((self.N, 4))

        if self.rl_active:
            for i in range(self.N):
                delta_u_all[i] = np.array([
                    action[i * 4 + 0] * self.delta_throttle_limit,
                    action[i * 4 + 1] * self.delta_elevator_limit,
                    action[i * 4 + 2] * self.delta_aileron_limit,
                    action[i * 4 + 3] * self.delta_rudder_limit
                ])

        # 执行控制
        for i in range(self.N):
            u_pid = self.trackers[i].compute_control(
                target_pos=target_pos_all[i],
                target_vel=target_vel_all[i],
                current_pos=self.agents[i].position,
                vel_earth=self.agents[i].velocity_earth,
                euler_rad=self.agents[i].euler,
                alpha_rad=self.agents[i].alpha,
                beta_rad=self.agents[i].beta,
                rot_body2earth=self.agents[i].rotation_body2earth,
                feedforward_turn_rate=self.turn_rate,
                leader_pos=self.leader_pos
            )

            u_total = u_pid + delta_u_all[i]

            u_total[0] = np.clip(u_total[0], 0.0, 1.0)
            u_total[1] = np.clip(u_total[1], -25.0, 25.0)
            u_total[2] = np.clip(u_total[2], -21.5, 21.5)
            u_total[3] = np.clip(u_total[3], -30.0, 30.0)

            self.agents[i].step(u_total)

        # 🔥 修复1: 新的奖励计算（分离水平/高度，限制范围）
        reward, reward_info = self._compute_reward_fixed(
            avg_error_horizontal, avg_error_vertical, min_dist, action
        )

        # 终止条件
        terminated = min_dist < self.d_collision
        truncated = self.step_count >= self.max_steps

        if terminated:
            self.episode_stats['collision'] = True

        # 统计累加
        self.episode_stats['total_reward'] += reward
        self.episode_stats['min_distance_ever'] = min(
            self.episode_stats['min_distance_ever'], min_dist
        )
        self.episode_stats['sum_tracking_error'] += avg_error_total
        self.episode_stats['sum_error_total'] += avg_error_total
        self.episode_stats['sum_error_planning'] += np.mean(error_planning_list)
        self.episode_stats['sum_error_tracking'] += np.mean(error_tracking_list)
        self.episode_stats['sum_error_horizontal'] += avg_error_horizontal
        self.episode_stats['sum_error_vertical'] += avg_error_vertical
        self.episode_stats['sum_rl_activation'] += (1 if self.rl_active else 0)
        self.episode_stats['max_tracking_error'] = max(
            self.episode_stats['max_tracking_error'], max_track_err
        )
        self.episode_stats['step_count'] += 1

        # 在episode结束时计算平均值
        final_stats = {}
        if terminated or truncated:
            n_steps = self.episode_stats['step_count']
            if n_steps > 0:
                final_stats = {
                    'total_reward': self.episode_stats['total_reward'],
                    'min_distance_ever': self.episode_stats['min_distance_ever'],
                    'avg_tracking_error': self.episode_stats['sum_tracking_error'] / n_steps,
                    'avg_error_total': self.episode_stats['sum_error_total'] / n_steps,
                    'avg_error_planning': self.episode_stats['sum_error_planning'] / n_steps,
                    'avg_error_tracking': self.episode_stats['sum_error_tracking'] / n_steps,
                    'avg_error_horizontal': self.episode_stats['sum_error_horizontal'] / n_steps,
                    'avg_error_vertical': self.episode_stats['sum_error_vertical'] / n_steps,
                    'max_tracking_error': self.episode_stats['max_tracking_error'],
                    'rl_activation_ratio': self.episode_stats['sum_rl_activation'] / n_steps,
                    'collision': self.episode_stats['collision']
                }

        obs = self._get_observation()
        info = {
            **reward_info,
            'rl_active': self.rl_active,
            'collision': terminated,
            'episode_stats': final_stats,
            # 🔥 添加三类误差到info
            'error_total': avg_error_total,
            'error_horizontal': avg_error_horizontal,
            'error_vertical': avg_error_vertical,
        }

        self.prev_actions = action.copy()

        return obs, reward, terminated, truncated, info

    def _pure_pid_step(self):
        """Warm-Start阶段"""
        self._update_leader_trajectory()

        c, s = np.cos(self.leader_heading), np.sin(self.leader_heading)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        current_offsets = [R_z @ off for off in self.desired_offsets]
        self.planner.update_offsets(current_offsets)

        is_turning = abs(self.turn_rate) > 1e-3
        if is_turning:
            for _ in range(3):
                self.planner.step(self.leader_pos, self.leader_vel, self.dt / 3)
        else:
            for _ in range(8):
                self.planner.step(self.leader_pos, self.leader_vel, self.dt / 8)

        target_pos_all, target_vel_all = self.planner.get_target_trajectories()

        for i in range(self.N):
            u_pid = self.trackers[i].compute_control(
                target_pos=target_pos_all[i],
                target_vel=target_vel_all[i],
                current_pos=self.agents[i].position,
                vel_earth=self.agents[i].velocity_earth,
                euler_rad=self.agents[i].euler,
                alpha_rad=self.agents[i].alpha,
                beta_rad=self.agents[i].beta,
                rot_body2earth=self.agents[i].rotation_body2earth,
                feedforward_turn_rate=self.turn_rate,
                leader_pos=self.leader_pos
            )
            self.agents[i].step(u_pid)

        obs = self._get_observation()
        return obs, 0.0, False, self.step_count >= self.max_steps, {'warmstart': True}

    def _compute_reward_fixed(self, avg_error_h, avg_error_v, min_dist, action):
        """
        🔥 修复版奖励函数:
        1. 分离水平/高度误差
        2. 限制单步奖励范围在[-10, +5]
        3. 使用clip而非tanh以避免梯度消失
        """

        # 1. 水平跟踪奖励: [-1, 0] × w_track_h(3.0) = [-3, 0]
        r_track_h_raw = -np.clip(avg_error_h / 100.0, 0.0, 1.0)
        r_track_h = r_track_h_raw * self.w_track_h

        # 2. 高度跟踪奖励: [-1, 0] × w_track_v(2.0) = [-2, 0]
        r_track_v_raw = -np.clip(avg_error_v / 50.0, 0.0, 1.0)
        r_track_v = r_track_v_raw * self.w_track_v

        # 3. 安全奖励: [-1, +0.2] × w_safe(2.0) = [-2, 0.4]
        if min_dist < self.d_collision:
            r_safe_raw = -1.0
        elif min_dist < self.d_danger:
            alpha = (min_dist - self.d_collision) / (self.d_danger - self.d_collision)
            r_safe_raw = -1.0 + alpha * 0.5
        elif min_dist < self.d_safe:
            alpha = (min_dist - self.d_danger) / (self.d_safe - self.d_danger)
            r_safe_raw = -0.5 + alpha * 0.5
        else:
            bonus = min(1.0, (min_dist - self.d_safe) / 200.0)
            r_safe_raw = bonus * 0.2

        r_safe = np.clip(r_safe_raw, -1.0, 0.2) * self.w_safe

        # 4. 控制惩罚: [-1, 0] × w_ctrl(0.05) = [-0.05, 0]
        if self.rl_active:
            action_norm = np.linalg.norm(action) / np.sqrt(self.N * 4)
            r_ctrl = -np.clip(action_norm, 0.0, 1.0) * self.w_ctrl

            action_change = np.linalg.norm(action - self.prev_actions) / np.sqrt(self.N * 4)
            r_smooth = -np.clip(action_change, 0.0, 1.0) * self.w_smooth
        else:
            r_ctrl = 0.0
            r_smooth = 0.0

        # 5. Bonus: [0, 1.5]
        r_bonus = 0.0
        if avg_error_h < 50.0 and avg_error_v < 25.0 and min_dist > 300.0:
            r_bonus += 0.5
        if self.step_count >= self.max_steps - 10 and min_dist > 200.0:
            r_bonus += 1.0

        # 总奖励: 理论范围 [-7.05, 1.9]
        reward = r_track_h + r_track_v + r_safe + r_ctrl + r_smooth + r_bonus

        # 🔥 额外保护：clip到[-10, +5]
        reward = np.clip(reward, -10.0, 5.0)

        reward_info = {
            'r_track_h': r_track_h,
            'r_track_v': r_track_v,
            'r_safe': r_safe,
            'r_ctrl': r_ctrl,
            'r_smooth': r_smooth,
            'r_bonus': r_bonus,
            'avg_error_h': avg_error_h,
            'avg_error_v': avg_error_v,
            'min_distance': min_dist
        }

        return reward, reward_info

    def _update_leader_trajectory(self):
        """领机轨迹(包含90°转弯)"""
        t = self.current_time
        turn_start = 20.0
        turn_end = 70.0
        transition_time = 5.0
        turn_rate_max = np.deg2rad(90.0 / 50.0)

        if t < turn_start:
            omega = 0.0
        elif t < turn_start + transition_time:
            progress = (t - turn_start) / transition_time
            smooth = 3 * progress ** 2 - 2 * progress ** 3
            omega = turn_rate_max * smooth
        elif t < turn_end - transition_time:
            omega = turn_rate_max
        elif t < turn_end:
            progress = (turn_end - t) / transition_time
            smooth = 3 * progress ** 2 - 2 * progress ** 3
            omega = turn_rate_max * smooth
        else:
            omega = 0.0

        self.turn_rate = omega
        self.leader_heading += omega * self.dt

        vel = np.array([
            self.leader_velocity * np.cos(self.leader_heading),
            self.leader_velocity * np.sin(self.leader_heading),
            0.0
        ])

        self.leader_pos += vel * self.dt
        self.leader_vel = vel

    def _get_observation(self):
        """观测"""
        obs = []

        c, s = np.cos(self.leader_heading), np.sin(self.leader_heading)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        current_offsets = [R_z @ off for off in self.desired_offsets]

        for i in range(self.N):
            agent = self.agents[i]
            ref_pos = self.leader_pos + current_offsets[i]
            ref_vel = self.leader_vel

            e_p = ref_pos - agent.position
            e_v = ref_vel - agent.velocity_earth

            leader_relative = agent.position - self.agents[0].position
            desired_relative = current_offsets[i] - current_offsets[0]
            e_form = leader_relative - desired_relative

            euler = agent.euler
            pqr = agent.angular_velocity

            min_d = float('inf')
            for j in range(self.N):
                if j != i:
                    d = np.linalg.norm(agent.position - self.agents[j].position)
                    if d > 1.0:
                        min_d = min(min_d, d)

            if min_d == float('inf'):
                min_d = 500.0

            danger_flag = 1.0 if min_d < self.d_danger else 0.0

            agent_obs = np.concatenate([
                e_p / 1000.0,
                e_v / 100.0,
                e_form / 500.0,
                euler,
                pqr,
                [min_d / 1000.0],
                [danger_flag],
                [self.turn_rate]
            ])

            obs.append(agent_obs)

        return np.concatenate(obs).astype(np.float32)