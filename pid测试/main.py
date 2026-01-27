# main.py - 修正版,与 quick_validation.py 对应

import numpy as np
import matplotlib.pyplot as plt
from adaptive_negotiation_trajectory import AdaptiveNegotiationTrajectory
from f16Model import F16
from pid_tracker import PIDTracker


class CoordinatedTurnLeader:
    """领机参考轨迹生成器"""

    def __init__(self, start_pos, velocity, dt):
        self.pos = np.array(start_pos, dtype=float)
        self.v_mag = velocity
        self.heading = 0.0
        self.dt = dt
        self.turn_start = 20.0
        self.turn_end = 70.0
        self.turn_rate_max = np.deg2rad(90.0 / 50.0)
        self.transition_time = 5.0

    def step(self, t):
        if t < self.turn_start:
            omega = 0.0
        elif t < self.turn_start + self.transition_time:
            progress = (t - self.turn_start) / self.transition_time
            smooth_factor = 3 * progress ** 2 - 2 * progress ** 3
            omega = self.turn_rate_max * smooth_factor
        elif t < self.turn_end - self.transition_time:
            omega = self.turn_rate_max
        elif t < self.turn_end:
            progress = (self.turn_end - t) / self.transition_time
            smooth_factor = 3 * progress ** 2 - 2 * progress ** 3
            omega = self.turn_rate_max * smooth_factor
        else:
            omega = 0.0

        self.heading += omega * self.dt
        vel = np.array([
            self.v_mag * np.cos(self.heading),
            self.v_mag * np.sin(self.heading),
            0.0
        ])
        self.pos += vel * self.dt
        return self.pos.copy(), vel, self.heading, omega


def create_conflict_scenario():
    """
    构造可控冲突场景 - 扩大编队版本
    """
    leader_start = np.array([1000.0, 0.0, -5000.0])

    # 扩大编队偏移量到500ft以上
    desired_offsets = [
        np.array([0.0, 0.0, 0.0]),              # Agent 1: leader
        np.array([-500.0, -500.0, 0.0]),        # Agent 2: back-left
        np.array([-500.0, 500.0, 0.0]),         # Agent 3: back-right
        np.array([-1000.0, 0.0, 0.0]),          # Agent 4: rear
    ]

    # 初始位置仍然有一定偏差,用于测试自适应轨迹规划
    initial_positions = np.array([
        leader_start,
        leader_start + np.array([-300.0, -150.0, 0.0]),  # Agent 2
        leader_start + np.array([-500.0, -500.0, 0.0]),  # Agent 3
        leader_start + np.array([-1000.0, 0.0, 0.0]),  # Agent 4
    ])

    return initial_positions, desired_offsets



def main():
    N = 4
    dt = 0.05
    T_sim = 120.0

    # ✅ 使用冲突场景初始化
    initial_positions, desired_offsets = create_conflict_scenario()
    leader_start_pos = initial_positions[0]

    A = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    leader_access = np.array([1, 0, 0, 0])

    # ✅ 使用 desired_offsets 而非 base_offsets
    planner = AdaptiveNegotiationTrajectory(
        N=N,
        adjacency_matrix=A,
        leader_access=leader_access,
        formation_offsets=desired_offsets,  # 关键修改
        k_gain=2.0,  # 与 quick_validation 一致
        sensing_radius=350.0,
        safety_radius=100.0
    )
    planner.initialize(initial_positions)

    agents = [F16(time_step=dt) for _ in range(N)]
    trackers = [
        PIDTracker(dt=dt, agent_role="leader"),
        PIDTracker(dt=dt, agent_role="follower_direct"),
        PIDTracker(dt=dt, agent_role="follower_direct"),
        PIDTracker(dt=dt, agent_role="follower_indirect")
    ]

    for i, agent in enumerate(agents):
        agent.reset(
            position=initial_positions[i],
            velocity_body=np.array([350., 0, 0]),
            mach=0.35
        )

    leader = CoordinatedTurnLeader(leader_start_pos, velocity=350.0, dt=dt)

    history = {
        'time': [],
        'track_err': [[] for _ in range(N)],
        'pos_x': [[] for _ in range(N)],
        'pos_y': [[] for _ in range(N)],
        'min_distance': [],  # ✅ 添加最小距离记录
        'leader_x': [],
        'leader_y': []
    }

    print(f"Simulating Adaptive Formation Control (T={T_sim}s)...")
    print("=" * 70)

    steps = int(T_sim / dt)
    for step in range(steps):
        t = step * dt

        y_r, y_r_dot, leader_heading, turn_rate = leader.step(t)

        # ✅ 更新编队偏移量 (使用旋转矩阵)
        c, s = np.cos(leader_heading), np.sin(leader_heading)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        current_offsets = [R_z @ off for off in desired_offsets]
        planner.update_offsets(current_offsets)

        # ✅ 与 quick_validation 一致的更新策略
        is_turning = abs(turn_rate) > 1e-3
        if is_turning:
            for _ in range(3):
                planner.step(y_r, y_r_dot, dt / 3)
        else:
            for _ in range(8):
                planner.step(y_r, y_r_dot, dt / 8)

        target_pos_all, target_vel_all = planner.get_target_trajectories()

        history['time'].append(t)
        history['leader_x'].append(y_r[0])
        history['leader_y'].append(y_r[1])

        current_positions = np.zeros((N, 3))
        for i in range(N):
            u_cmd = trackers[i].compute_control(
                target_pos=target_pos_all[i],
                target_vel=target_vel_all[i],
                current_pos=agents[i].position,
                vel_earth=agents[i].velocity_earth,
                euler_rad=agents[i].euler,
                alpha_rad=agents[i].alpha,
                beta_rad=agents[i].beta,
                rot_body2earth=agents[i].rotation_body2earth,
                feedforward_turn_rate=turn_rate,
                leader_pos=y_r
            )
            agents[i].step(u_cmd)
            current_positions[i] = agents[i].position

            track_err = np.linalg.norm(agents[i].position - target_pos_all[i])
            history['track_err'][i].append(track_err)
            history['pos_x'][i].append(agents[i].position[0])
            history['pos_y'][i].append(agents[i].position[1])

        # ✅ 计算最小距离
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(current_positions[i] - current_positions[j])
                min_dist = min(min_dist, d)
        history['min_distance'].append(min_dist)

        if step % (int(10.0 / dt)) == 0:
            max_err = max([history['track_err'][i][-1] for i in range(N)])
            avg_err = np.mean([history['track_err'][i][-1] for i in range(N)])
            print(f"t={t:.1f}s | φ={np.rad2deg(leader_heading):.1f}° | "
                  f"MinDist={min_dist:.0f}ft | "
                  f"Max={max_err:.0f}ft | Avg={avg_err:.0f}ft")

    print("\n" + "=" * 70)
    print("Simulation Complete!")

    # 分析最终收敛性
    final_errors = [history['track_err'][i][-1] for i in range(N)]
    min_distance_ever = min(history['min_distance'])

    print(f"Final Errors: {[f'{e:.0f}ft' for e in final_errors]}")
    print(f"Average Final Error: {np.mean(final_errors):.0f}ft")
    print(f"Minimum Distance Ever: {min_distance_ever:.1f}ft")
    print("=" * 70)

    # 可视化
    fig = plt.figure(figsize=(18, 6))

    # 轨迹图
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(history['leader_x'], history['leader_y'], 'k--', linewidth=2, label='Leader')
    colors = ['red', 'green', 'blue', 'orange']
    labels = ['Agent 1 (Leader)', 'Agent 2', 'Agent 3', 'Agent 4']

    for i in range(N):
        ax1.plot(history['pos_x'][i], history['pos_y'][i],
                 color=colors[i], label=labels[i], alpha=0.8)

    ax1.set_title('Formation Flight - Adaptive Control')
    ax1.set_xlabel('North (ft)')
    ax1.set_ylabel('East (ft)')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()

    # 跟踪误差图
    ax2 = fig.add_subplot(1, 3, 2)
    for i in range(N):
        ax2.plot(history['time'], history['track_err'][i],
                 color=colors[i], label=labels[i])

    ax2.axvspan(20, 70, alpha=0.2, color='gray', label='Turning Phase')
    ax2.set_title('Tracking Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (ft)')
    ax2.grid(True)
    ax2.legend()

    # ✅ 最小距离图
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(history['time'], history['min_distance'], 'b-', linewidth=2)
    ax3.axhspan(0, 100, alpha=0.3, color='red', label='Collision (<100ft)')
    ax3.axhspan(100, 160, alpha=0.3, color='yellow', label='Danger (100-160ft)')
    ax3.set_title('Minimum Inter-Agent Distance')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance (ft)')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('adaptive_control_result.png', dpi=150)
    print("Graph saved to 'adaptive_control_result.png'")
    plt.show()


if __name__ == "__main__":
    main()
