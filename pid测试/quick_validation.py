# comparative_experiment_revised.py - 修正版

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


class OriginalReferenceTrajectory:
    """原始参考轨迹（不考虑避碰）"""

    def __init__(self, formation_offsets):
        self.base_offsets = np.array(formation_offsets)
        self.N = len(formation_offsets)

    def get_reference_trajectories(self, leader_pos, leader_vel, leader_heading):
        c, s = np.cos(leader_heading), np.sin(leader_heading)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        ref_positions = np.zeros((self.N, 3))
        ref_velocities = np.zeros((self.N, 3))

        for i in range(self.N):
            rotated_offset = R_z @ self.base_offsets[i]
            ref_positions[i] = leader_pos + rotated_offset
            ref_velocities[i] = leader_vel

        return ref_positions, ref_velocities


class SafetyMetrics:
    """安全性能与误差指标统计"""

    def __init__(self, N, collision_threshold=100.0, danger_threshold=160.0):
        self.N = N
        self.r_collision = collision_threshold
        self.r_danger = danger_threshold

        self.min_distance_history = []
        self.collision_count = 0
        self.danger_count = 0
        self.safe_count = 0
        self.pairwise_min_distances = {(i, j): float('inf')
                                       for i in range(N) for j in range(i + 1, N)}

        self.tracking_errors = [[] for _ in range(N)]
        self.formation_errors = []

    def update_safety(self, positions):
        min_dist = float('inf')

        for i in range(self.N):
            for j in range(i + 1, self.N):
                d = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, d)

                pair_key = (i, j)
                self.pairwise_min_distances[pair_key] = min(
                    self.pairwise_min_distances[pair_key], d
                )

        self.min_distance_history.append(min_dist)

        if min_dist < self.r_collision:
            self.collision_count += 1
        elif min_dist < self.r_danger:
            self.danger_count += 1
        else:
            self.safe_count += 1

    def update_tracking_errors(self, actual_positions, reference_positions):
        for i in range(self.N):
            error = np.linalg.norm(actual_positions[i] - reference_positions[i])
            self.tracking_errors[i].append(error)

    def update_formation_error(self, actual_positions, desired_offsets):
        actual_relative = [actual_positions[i] - actual_positions[0]
                           for i in range(1, self.N)]
        desired_relative = [desired_offsets[i] for i in range(1, self.N)]

        formation_error = np.mean([
            np.linalg.norm(actual - desired)
            for actual, desired in zip(actual_relative, desired_relative)
        ])
        self.formation_errors.append(formation_error)

    def get_summary(self):
        total_steps = len(self.min_distance_history)

        return {
            'min_distance_ever': min(self.min_distance_history) if self.min_distance_history else float('inf'),
            'avg_min_distance': np.mean(self.min_distance_history) if self.min_distance_history else 0,
            'collision_rate': self.collision_count / total_steps if total_steps > 0 else 0,
            'danger_rate': self.danger_count / total_steps if total_steps > 0 else 0,
            'safe_rate': self.safe_count / total_steps if total_steps > 0 else 0,
            'final_tracking_errors': [self.tracking_errors[i][-1] if self.tracking_errors[i] else 0
                                      for i in range(self.N)],
            'avg_tracking_error': np.mean([np.mean(errors) if errors else 0
                                           for errors in self.tracking_errors]),
            'max_tracking_error': max([max(errors) if errors else 0
                                       for errors in self.tracking_errors]),
            'final_formation_error': self.formation_errors[-1] if self.formation_errors else 0,
            'avg_formation_error': np.mean(self.formation_errors) if self.formation_errors else 0,
            'pairwise_mins': self.pairwise_min_distances,
            'min_distance_history': self.min_distance_history,
            'tracking_error_history': self.tracking_errors,
            'formation_error_history': self.formation_errors
        }


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
        leader_start + np.array([-1000.0, 0.0, 0.0]),    # Agent 4
    ])

    return initial_positions, desired_offsets



def run_simulation(use_adaptive: bool, T_sim=200.0, dt=0.05, verbose=True):
    """运行对照实验"""
    N = 4

    initial_positions, desired_offsets = create_conflict_scenario()
    leader_start_pos = initial_positions[0]

    original_ref = OriginalReferenceTrajectory(desired_offsets)
    leader = CoordinatedTurnLeader(leader_start_pos, velocity=350.0, dt=dt)

    if use_adaptive:
        A = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
        leader_access = np.array([1, 0, 0, 0])

        planner = AdaptiveNegotiationTrajectory(
            N=N, adjacency_matrix=A, leader_access=leader_access,
            formation_offsets=desired_offsets,
            k_gain=2.0,  # ✅ 提高增益，增强避碰响应
            sensing_radius=350.0,  # ✅ 扩大感知范围
            safety_radius=100.0
        )
        planner.initialize(initial_positions)
        method_name = "Adaptive (Collision Avoidance)"
    else:
        planner = None
        method_name = "Baseline (Direct Tracking)"

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

    history = {
        'time': [],
        'actual_positions': [[] for _ in range(N)],
        'target_positions': [[] for _ in range(N)],
        'reference_positions': [[] for _ in range(N)],
        'leader_pos': [],
        'leader_heading': []
    }

    metrics = SafetyMetrics(N, collision_threshold=100.0, danger_threshold=160.0)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Running {method_name} Simulation")
        print(f"Duration: {T_sim}s | Initial Conflict Scenario Loaded")
        print(f"{'=' * 70}\n")

    steps = int(T_sim / dt)
    for step in range(steps):
        t = step * dt

        y_r, y_r_dot, leader_heading, turn_rate = leader.step(t)
        ref_pos, ref_vel = original_ref.get_reference_trajectories(y_r, y_r_dot, leader_heading)

        if use_adaptive:
            c, s = np.cos(leader_heading), np.sin(leader_heading)
            R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            current_offsets = [R_z @ off for off in desired_offsets]
            planner.update_offsets(current_offsets)

            is_turning = abs(turn_rate) > 1e-3
            if is_turning:
                for _ in range(3):  # ✅ 增加更新频率
                    planner.step(y_r, y_r_dot, dt / 3)
            else:
                for _ in range(8):
                    planner.step(y_r, y_r_dot, dt / 8)

            target_pos, target_vel = planner.get_target_trajectories()
        else:
            target_pos, target_vel = ref_pos, ref_vel

        current_positions = np.zeros((N, 3))
        for i in range(N):
            u_cmd = trackers[i].compute_control(
                target_pos=target_pos[i],
                target_vel=target_vel[i],
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

        history['time'].append(t)
        history['leader_pos'].append(y_r.copy())
        history['leader_heading'].append(leader_heading)

        for i in range(N):
            history['actual_positions'][i].append(current_positions[i].copy())
            history['target_positions'][i].append(target_pos[i].copy())
            history['reference_positions'][i].append(ref_pos[i].copy())

        metrics.update_safety(current_positions)
        metrics.update_tracking_errors(current_positions, ref_pos)

        c, s = np.cos(leader_heading), np.sin(leader_heading)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        current_desired_offsets = [R_z @ off for off in desired_offsets]
        metrics.update_formation_error(current_positions, current_desired_offsets)

        if verbose and step % (int(10.0 / dt)) == 0:
            summary = metrics.get_summary()
            print(f"t={t:.1f}s | "
                  f"MinDist={summary['min_distance_ever']:.0f}ft | "
                  f"AvgTrackErr={summary['avg_tracking_error']:.0f}ft | "
                  f"FormErr={summary['avg_formation_error']:.0f}ft")

    summary = metrics.get_summary()

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"{method_name} - Final Report")
        print(f"{'=' * 70}")
        print(f"Safety Performance:")
        print(f"  Min Distance Ever:  {summary['min_distance_ever']:.1f} ft")
        print(f"  Collision Rate:     {summary['collision_rate'] * 100:.2f}% (<100ft)")
        print(f"  Danger Rate:        {summary['danger_rate'] * 100:.2f}% (100-160ft)")
        print(f"  Safe Rate:          {summary['safe_rate'] * 100:.2f}% (>160ft)")

        print(f"\nTracking Performance (vs Original Reference):")
        print(f"  Final Errors: {[f'{e:.0f}ft' for e in summary['final_tracking_errors']]}")
        print(f"  Average Error:      {summary['avg_tracking_error']:.0f}ft")
        print(f"  Maximum Error:      {summary['max_tracking_error']:.0f}ft")

        print(f"\nFormation Maintenance:")
        print(f"  Final Formation Error: {summary['final_formation_error']:.0f}ft")
        print(f"  Average Formation Error: {summary['avg_formation_error']:.0f}ft")
        print(f"{'=' * 70}\n")

    return {
        'history': history,
        'metrics': summary,
        'method': method_name
    }




def comparative_visualization(baseline_result, adaptive_result):
    """生成对比可视化"""

    fig = plt.figure(figsize=(20, 12))

    colors = ['red', 'green', 'blue', 'orange']
    labels = ['Agent 1 (Leader)', 'Agent 2', 'Agent 3', 'Agent 4']

    # Row 1: 轨迹对比
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2)

    for ax, result in zip([ax1, ax2], [baseline_result, adaptive_result]):
        hist = result['history']

        for i in range(4):
            ref_x = [p[0] for p in hist['reference_positions'][i]]
            ref_y = [p[1] for p in hist['reference_positions'][i]]
            ax.plot(ref_x, ref_y, '--', color=colors[i], linewidth=1, alpha=0.3)

        for i in range(4):
            pos_x = [p[0] for p in hist['actual_positions'][i]]
            pos_y = [p[1] for p in hist['actual_positions'][i]]
            ax.plot(pos_x, pos_y, color=colors[i], label=labels[i], linewidth=2)

        ax.set_title(f'{result["method"]} - Trajectory')
        ax.set_xlabel('North (ft)')
        ax.set_ylabel('East (ft)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    # 最小距离时序对比
    ax3 = fig.add_subplot(3, 3, 3)

    time = baseline_result['history']['time']
    ax3.plot(time, baseline_result['metrics']['min_distance_history'],
             'r-', linewidth=2, label=f'Baseline')
    ax3.plot(time, adaptive_result['metrics']['min_distance_history'],
             'g-', linewidth=2, label=f'Adaptive')

    ax3.axhspan(0, 100, alpha=0.3, color='red', label='Collision (<100ft)')
    ax3.axhspan(100, 160, alpha=0.3, color='yellow', label='Danger (100-160ft)')

    ax3.set_title('Minimum Inter-Agent Distance Over Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance (ft)')
    ax3.set_ylim([0, 400])
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Row 2: 跟踪误差对比
    ax4 = fig.add_subplot(3, 3, 4)
    ax5 = fig.add_subplot(3, 3, 5)

    for ax, result in zip([ax4, ax5], [baseline_result, adaptive_result]):
        time = result['history']['time']

        for i in range(4):
            errors = result['metrics']['tracking_error_history'][i]
            ax.plot(time, errors, color=colors[i], label=labels[i])

        ax.axvspan(20, 70, alpha=0.15, color='gray', label='Turning Phase')
        ax.set_title(f'{result["method"]} - Tracking Error')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error vs Reference (ft)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    # 编队误差对比
    ax6 = fig.add_subplot(3, 3, 6)

    time = baseline_result['history']['time']
    ax6.plot(time, baseline_result['metrics']['formation_error_history'],
             'r-', linewidth=2, label='Baseline')
    ax6.plot(time, adaptive_result['metrics']['formation_error_history'],
             'g-', linewidth=2, label='Adaptive')

    ax6.set_title('Formation Maintenance Error Over Time')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Formation Error (ft)')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # Row 3: 综合指标对比
    ax7 = fig.add_subplot(3, 3, 7)
    metrics_names = ['Min Dist\n(ft)', 'Collision\nRate (%)', 'Avg Track\nError (ft)']
    baseline_vals = [
        baseline_result['metrics']['min_distance_ever'],
        baseline_result['metrics']['collision_rate'] * 100,
        baseline_result['metrics']['avg_tracking_error']
    ]
    adaptive_vals = [
        adaptive_result['metrics']['min_distance_ever'],
        adaptive_result['metrics']['collision_rate'] * 100,
        adaptive_result['metrics']['avg_tracking_error']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax7.bar(x - width / 2, baseline_vals, width, label='Baseline', color='lightcoral')
    ax7.bar(x + width / 2, adaptive_vals, width, label='Adaptive', color='lightgreen')

    ax7.set_ylabel('Value')
    ax7.set_title('Key Performance Metrics')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics_names)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    for i, (b, a) in enumerate(zip(baseline_vals, adaptive_vals)):
        ax7.text(i - width / 2, b + max(baseline_vals) * 0.02, f'{b:.1f}',
                 ha='center', va='bottom', fontsize=9)
        ax7.text(i + width / 2, a + max(adaptive_vals) * 0.02, f'{a:.1f}',
                 ha='center', va='bottom', fontsize=9)

    # 初始冲突场景可视化
    ax8 = fig.add_subplot(3, 3, 8)
    initial_pos, desired_offsets = create_conflict_scenario()

    for i in range(4):
        ax8.scatter(initial_pos[i, 0], initial_pos[i, 1],
                    s=200, color=colors[i], marker='o', label=f'Agent {i + 1} Start')

    for i in range(4):
        desired_pos = initial_pos[0] + desired_offsets[i]
        ax8.scatter(desired_pos[0], desired_pos[1],
                    s=200, color=colors[i], marker='x', alpha=0.5)

        ax8.arrow(initial_pos[i, 0], initial_pos[i, 1],
                  desired_pos[0] - initial_pos[i, 0],
                  desired_pos[1] - initial_pos[i, 1],
                  head_width=20, head_length=30, fc=colors[i], ec=colors[i], alpha=0.3)

    ax8.set_title('Initial Conflict Scenario')
    ax8.set_xlabel('North (ft)')
    ax8.set_ylabel('East (ft)')
    ax8.axis('equal')
    ax8.grid(True, alpha=0.3)
    ax8.legend(loc='best', fontsize=8)

    # 改进百分比
    ax9 = fig.add_subplot(3, 3, 9)

    improvements = {
        'Min Distance': ((adaptive_result['metrics']['min_distance_ever'] -
                          baseline_result['metrics']['min_distance_ever']) /
                         baseline_result['metrics']['min_distance_ever'] * 100),
        'Collision Rate': ((baseline_result['metrics']['collision_rate'] -
                            adaptive_result['metrics']['collision_rate']) /
                           (baseline_result['metrics']['collision_rate'] + 1e-6) * 100),
        'Tracking Error': ((baseline_result['metrics']['avg_tracking_error'] -
                            adaptive_result['metrics']['avg_tracking_error']) /
                           baseline_result['metrics']['avg_tracking_error'] * 100)
    }

    colors_bar = ['green' if v > 0 else 'red' for v in improvements.values()]
    ax9.barh(list(improvements.keys()), list(improvements.values()), color=colors_bar)
    ax9.set_xlabel('Improvement (%)')
    ax9.set_title('Adaptive vs Baseline Performance Gain')
    ax9.axvline(x=0, color='black', linewidth=1)
    ax9.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('comparative_experiment_results.png', dpi=150)
    print("✅ Visualization saved to 'comparative_experiment_results.png'")
    plt.show()


def main():
    print("\n" + "=" * 80)
    print("COMPARATIVE EXPERIMENT: Baseline vs Adaptive Negotiation Trajectory")
    print("Experimental Design:")
    print("  ✓ Identical initial positions for both groups")
    print("  ✓ Adaptive initialized with actual agent positions")
    print("  ✓ Baseline tracks original reference directly")
    print("  ✓ Error measured against original reference trajectory")
    print("  ✓ Initial conflict scenario intentionally designed")
    print("=" * 80)

    np.random.seed(42)

    T_sim = 120.0
    dt = 0.05

    print("\n[1/2] Running BASELINE (Direct Reference Tracking)...")
    baseline_result = run_simulation(use_adaptive=False, T_sim=T_sim, dt=dt, verbose=True)

    print("\n[2/2] Running ADAPTIVE (Collision Avoidance)...")
    adaptive_result = run_simulation(use_adaptive=True, T_sim=T_sim, dt=dt, verbose=True)

    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    baseline_m = baseline_result['metrics']
    adaptive_m = adaptive_result['metrics']

    print(f"\n{'Metric':<35} {'Baseline':>15} {'Adaptive':>15} {'Improvement':>15}")
    print("-" * 80)

    min_dist_improve = ((adaptive_m['min_distance_ever'] - baseline_m['min_distance_ever'])
                        / baseline_m['min_distance_ever'] * 100)
    print(f"{'Min Distance (ft)':<35} {baseline_m['min_distance_ever']:>15.1f} "
          f"{adaptive_m['min_distance_ever']:>15.1f} {min_dist_improve:>14.1f}%")

    collision_reduce = (baseline_m['collision_rate'] - adaptive_m['collision_rate']) * 100
    print(f"{'Collision Rate Reduction':<35} {baseline_m['collision_rate'] * 100:>15.2f}% "
          f"{adaptive_m['collision_rate'] * 100:>15.2f}% {collision_reduce:>14.2f}pp")

    danger_reduce = (baseline_m['danger_rate'] - adaptive_m['danger_rate']) * 100
    print(f"{'Danger Rate Reduction':<35} {baseline_m['danger_rate'] * 100:>15.2f}% "
          f"{adaptive_m['danger_rate'] * 100:>15.2f}% {danger_reduce:>14.2f}pp")

    track_err_change = ((baseline_m['avg_tracking_error'] - adaptive_m['avg_tracking_error'])
                        / baseline_m['avg_tracking_error'] * 100)
    print(f"{'Avg Tracking Error (ft)':<35} {baseline_m['avg_tracking_error']:>15.1f} "
          f"{adaptive_m['avg_tracking_error']:>15.1f} {track_err_change:>14.1f}%")

    form_err_change = ((baseline_m['avg_formation_error'] - adaptive_m['avg_formation_error'])
                       / baseline_m['avg_formation_error'] * 100)
    print(f"{'Avg Formation Error (ft)':<35} {baseline_m['avg_formation_error']:>15.1f} "
          f"{adaptive_m['avg_formation_error']:>15.1f} {form_err_change:>14.1f}%")

    print("=" * 80)

    print(f"\n{'VALIDATION SUMMARY':^80}")
    print("=" * 80)

    if adaptive_m['min_distance_ever'] > baseline_m['min_distance_ever']:
        print("✅ Adaptive maintains LARGER minimum safe distance")
    else:
        print("❌ WARNING: Adaptive failed to improve minimum distance")

    if adaptive_m['collision_rate'] < baseline_m['collision_rate']:
        print("✅ Adaptive achieves LOWER collision rate")
    else:
        print("❌ WARNING: Adaptive did not reduce collisions")

    if adaptive_m['min_distance_ever'] > 100.0:
        print("✅ Adaptive successfully AVOIDS collisions (>100ft)")
    else:
        print("⚠️  Adaptive has collision events (<100ft)")

    if adaptive_m['avg_tracking_error'] > baseline_m['avg_tracking_error']:
        print(f"ℹ️  Adaptive trades {abs(track_err_change):.1f}% tracking error for safety")
    else:
        print("✅ Adaptive achieves BOTH better safety AND tracking")

    print("=" * 80)

    print("\nGenerating comparative visualization...")
    comparative_visualization(baseline_result, adaptive_result)

    print("\n✅ Experiment Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
