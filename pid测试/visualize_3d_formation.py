#!/usr/bin/env python3
# visualize_3d_formation.py - 完全修复版可视化工具

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_env_f16_formation import FormationEnvFixed
from config import TRAIN_CONFIG_FIXED
import os


class PPOVisualizer:
    """
    PPO训练结果可视化器 - 完全修复版

    修复内容:
    1. 正确访问VecNormalize包装的环境
    2. 记录三类误差: ε₁(真实), ε₂(规划), ε₃(控制)
    3. 分离水平/高度误差
    4. 时间对齐和数据维度保护
    """

    def __init__(self, model_path, vec_normalize_path=None):
        """
        Args:
            model_path: 训练好的模型路径 (例如: "./logs/ppo_fixed/best_model")
            vec_normalize_path: VecNormalize统计文件路径 (例如: "./logs/ppo_fixed/vec_normalize.pkl")
        """
        self.model_path = model_path
        self.vec_normalize_path = vec_normalize_path

        # 加载环境配置
        config = TRAIN_CONFIG_FIXED.to_dict()
        self.config = config

        # 创建环境
        env = DummyVecEnv([lambda: FormationEnvFixed(config)])

        # 加载归一化参数
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
            print(f"✅ Loaded VecNormalize from {vec_normalize_path}")
        else:
            print("⚠️  No VecNormalize file found, using raw observations")

        # 加载模型
        self.model = PPO.load(model_path, env=env)
        self.env = env

        # 🔥 修复: 正确获取底层环境
        if isinstance(env, VecNormalize):
            self.base_env = env.venv.envs[0]
        else:
            self.base_env = env.envs[0]

        print(f"✅ Loaded model from {model_path}")

    def evaluate_episode(self, seed=42, render_mode='full'):
        """
        评估单个episode并收集数据

        Args:
            seed: 随机种子
            render_mode: 'full' 或 'compact'

        Returns:
            history: 包含完整轨迹数据的字典
        """
        obs = self.env.reset()
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)

        history = {
            'time': [],
            'positions': [[] for _ in range(4)],
            'target_positions': [[] for _ in range(4)],
            'ideal_positions': [[] for _ in range(4)],  # 🔥 新增: 理想编队位置

            # 🔥 修复: 三类误差
            'error_total': [[] for _ in range(4)],  # ε₁: Agent vs 理想编队
            'error_planning': [[] for _ in range(4)],  # ε₂: 协商轨迹 vs 理想编队
            'error_tracking': [[] for _ in range(4)],  # ε₃: Agent vs 协商轨迹

            # 🔥 新增: 分维度误差
            'error_horizontal': [[] for _ in range(4)],
            'error_vertical': [[] for _ in range(4)],

            'rewards': [],
            'r_track_h': [],
            'r_track_v': [],
            'r_safe': [],
            'r_ctrl': [],
            'r_smooth': [],
            'min_distance': [],
            'rl_active': [],
            'actions': [],
            'leader_pos': []
        }

        done = False
        step_count = 0
        dt = 0.05

        print("\n" + "=" * 70)
        print("Running Episode Evaluation...")
        print("=" * 70)

        while not done:
            # 获取动作
            action, _ = self.model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, done, info = self.env.step(action)
            if done[0] and len(info) > 0 and isinstance(info[0], dict):
                terminal_obs = info[0].get('terminal_observation')
                if terminal_obs is not None:
                    obs = terminal_obs

            # 🔥 修复: 正确访问底层环境状态
            env_state = self.base_env

            # 记录数据
            t = step_count * dt
            history['time'].append(t)
            history['rewards'].append(reward[0])
            history['actions'].append(action[0].copy())

            # 🔥 获取当前旋转矩阵和编队偏移
            c, s = np.cos(env_state.leader_heading), np.sin(env_state.leader_heading)
            R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            current_offsets = [R_z @ off for off in env_state.desired_offsets]

            # 记录各智能体位置和误差
            for i in range(4):
                agent = env_state.agents[i]
                history['positions'][i].append(agent.position.copy())

                # 获取协商轨迹目标位置
                target_pos_all, _ = env_state.planner.get_target_trajectories()
                history['target_positions'][i].append(target_pos_all[i].copy())

                # 🔥 计算理想编队位置
                ideal_pos = env_state.leader_pos + current_offsets[i]
                history['ideal_positions'][i].append(ideal_pos.copy())

                # 🔥 计算三类误差
                e_total = agent.position - ideal_pos
                error_total = np.linalg.norm(e_total)
                history['error_total'][i].append(error_total)

                error_planning = np.linalg.norm(target_pos_all[i] - ideal_pos)
                history['error_planning'][i].append(error_planning)

                error_tracking = np.linalg.norm(agent.position - target_pos_all[i])
                history['error_tracking'][i].append(error_tracking)

                # 🔥 分维度误差
                error_horizontal = np.linalg.norm(e_total[0:2])
                error_vertical = abs(e_total[2])
                history['error_horizontal'][i].append(error_horizontal)
                history['error_vertical'][i].append(error_vertical)

            # 记录leader位置
            history['leader_pos'].append(env_state.leader_pos.copy())

            # 🔥 修复: 提取info中的奖励分解（带防御性检查）
            if len(info) > 0 and isinstance(info[0], dict):
                info_dict = info[0]
                history['r_track_h'].append(info_dict.get('r_track_h', 0))
                history['r_track_v'].append(info_dict.get('r_track_v', 0))
                history['r_safe'].append(info_dict.get('r_safe', 0))
                history['r_ctrl'].append(info_dict.get('r_ctrl', 0))
                history['r_smooth'].append(info_dict.get('r_smooth', 0))
                history['min_distance'].append(info_dict.get('min_distance', 0))
                history['rl_active'].append(info_dict.get('rl_active', False))
            else:
                # Warmstart阶段或信息缺失
                history['r_track_h'].append(0)
                history['r_track_v'].append(0)
                history['r_safe'].append(0)
                history['r_ctrl'].append(0)
                history['r_smooth'].append(0)
                history['min_distance'].append(500.0)
                history['rl_active'].append(False)

            step_count += 1

            # 进度显示
            if step_count % 400 == 0:
                avg_err_total = np.mean([history['error_total'][i][-1] for i in range(4)])
                avg_err_h = np.mean([history['error_horizontal'][i][-1] for i in range(4)])
                avg_err_v = np.mean([history['error_vertical'][i][-1] for i in range(4)])
                min_dist = history['min_distance'][-1] if history['min_distance'] else 0
                print(f"t={t:.1f}s | Total={avg_err_total:.0f}ft | H={avg_err_h:.0f}ft | "
                      f"V={avg_err_v:.0f}ft | MinDist={min_dist:.0f}ft | Reward={reward[0]:.2f}")

            if done[0]:
                break

        print("\n" + "=" * 70)
        print("Episode Complete!")
        print("=" * 70)

        # 统计信息
        final_errors_total = [history['error_total'][i][-1] for i in range(4)]
        final_errors_h = [history['error_horizontal'][i][-1] for i in range(4)]
        final_errors_v = [history['error_vertical'][i][-1] for i in range(4)]
        min_distance_ever = min(history['min_distance']) if history['min_distance'] else 0

        print(f"Total Steps: {step_count}")
        print(f"Total Time: {step_count * dt:.1f}s")
        print(f"\nFinal Errors (Total): {[f'{e:.0f}ft' for e in final_errors_total]}")
        print(f"Average Final Error (Total): {np.mean(final_errors_total):.0f}ft")
        print(f"Average Final Error (Horizontal): {np.mean(final_errors_h):.0f}ft")
        print(f"Average Final Error (Vertical): {np.mean(final_errors_v):.0f}ft")
        print(f"Minimum Distance Ever: {min_distance_ever:.1f}ft")
        print(f"Total Reward: {sum(history['rewards']):.1f}")
        print("=" * 70)

        return history

    def plot_comprehensive_analysis(self, history, save_path='ppo_analysis_fixed.png'):
        """
        绘制综合分析图 - 修复版

        包含:
        1. 3D轨迹图
        2. 三类误差对比
        3. 水平/高度误差分解
        4. 最小距离
        5. 奖励分解
        6. RL激活状态
        """
        history = self._trim_history(history)
        fig = plt.figure(figsize=(24, 14))

        colors = ['red', 'green', 'blue', 'orange']
        labels = ['Agent 1 (Leader)', 'Agent 2', 'Agent 3', 'Agent 4']

        # ==================== 1. 3D轨迹图 ====================
        ax1 = fig.add_subplot(2, 4, 1, projection='3d')

        for i in range(4):
            pos_array = np.array(history['positions'][i])
            ax1.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2],
                     color=colors[i], label=labels[i], linewidth=2, alpha=0.8)

        leader_array = np.array(history['leader_pos'])
        ax1.plot(leader_array[:, 0], leader_array[:, 1], leader_array[:, 2],
                 'k--', linewidth=2, label='Leader Reference', alpha=0.6)

        ax1.set_xlabel('X (North, ft)')
        ax1.set_ylabel('Y (East, ft)')
        ax1.set_zlabel('Z (Down, ft)')
        ax1.set_title('3D Flight Trajectories')
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # ==================== 2. 三类误差对比 (Agent 1) ====================
        ax2 = fig.add_subplot(2, 4, 2)

        i = 0  # 只显示Agent 1作为示例
        ax2.plot(history['time'], history['error_total'][i],
                 'r-', linewidth=2, label=f'ε₁: Total (真实跟踪)')
        ax2.plot(history['time'], history['error_planning'][i],
                 'b--', linewidth=1.5, label=f'ε₂: Planning (轨迹重构)')
        ax2.plot(history['time'], history['error_tracking'][i],
                 'g:', linewidth=1.5, label=f'ε₃: Tracking (控制器)')

        ax2.axvspan(20, 70, alpha=0.15, color='gray', label='Turn Phase')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (ft)')
        ax2.set_title(f'Three Error Types - {labels[0]}')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # ==================== 3. 总体真实跟踪误差 (所有Agent) ====================
        ax3 = fig.add_subplot(2, 4, 3)

        for i in range(4):
            ax3.plot(history['time'], history['error_total'][i],
                     color=colors[i], label=labels[i], linewidth=1.5)

        ax3.axhline(
            y=self.config.get('rl_threshold', 150.0),
            color='orange',
            linestyle='--',
            linewidth=1,
            label='RL Threshold'
        )
        ax3.axvspan(20, 70, alpha=0.15, color='gray')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Error (ft)')
        ax3.set_title('ε₁: True Tracking Error (All Agents)')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        # ==================== 4. 水平/高度误差分解 ====================
        ax4 = fig.add_subplot(2, 4, 4)

        avg_error_h = [np.mean([history['error_horizontal'][i][t] for i in range(4)])
                       for t in range(len(history['time']))]
        avg_error_v = [np.mean([history['error_vertical'][i][t] for i in range(4)])
                       for t in range(len(history['time']))]

        ax4.plot(history['time'], avg_error_h, 'b-', linewidth=2, label='Horizontal Error')
        ax4.plot(history['time'], avg_error_v, 'r-', linewidth=2, label='Vertical Error')
        ax4.axvspan(20, 70, alpha=0.15, color='gray', label='Turn Phase')

        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Error (ft)')
        ax4.set_title('Horizontal vs Vertical Error (Average)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # ==================== 5. 最小距离 ====================
        ax5 = fig.add_subplot(2, 4, 5)

        ax5.plot(history['time'], history['min_distance'], 'b-', linewidth=2)
        ax5.axhspan(0, 100, alpha=0.3, color='red', label='Collision Zone (<100ft)')
        ax5.axhspan(100, 160, alpha=0.3, color='yellow', label='Danger Zone (100-160ft)')
        ax5.axhspan(160, 350, alpha=0.15, color='orange', label='Warning Zone (160-350ft)')
        safety_margin = self.config.get('distance_safety_margin', 300.0)
        ax5.axhline(
            y=safety_margin,
            color='purple',
            linestyle='--',
            linewidth=1.5,
            label=f'Safety Margin ({safety_margin:.0f}ft)'
        )

        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Minimum Inter-Agent Distance (ft)')
        ax5.set_title('Safety: Minimum Distance')
        ax5.legend(fontsize=7, loc='lower right')
        ax5.grid(True, alpha=0.3)

        # ==================== 6. 奖励分解 ====================
        ax6 = fig.add_subplot(2, 4, 6)

        ax6.plot(history['time'], history['r_track_h'], 'b-', label='Track (H)', alpha=0.7)
        ax6.plot(history['time'], history['r_track_v'], 'c-', label='Track (V)', alpha=0.7)
        ax6.plot(history['time'], history['r_safe'], 'g-', label='Safety', alpha=0.7)
        ax6.plot(history['time'], history['r_ctrl'], 'r-', label='Control', alpha=0.7)
        ax6.plot(history['time'], history['r_smooth'], 'm-', label='Smoothness', alpha=0.7)
        ax6.plot(history['time'], history['rewards'], 'k-', label='Total', linewidth=2)

        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Reward')
        ax6.set_title('Reward Decomposition (Fixed)')
        ax6.legend(fontsize=7)
        ax6.grid(True, alpha=0.3)

        # ==================== 7. RL激活状态 ====================
        ax7 = fig.add_subplot(2, 4, 7)

        rl_active_int = [1 if x else 0 for x in history['rl_active']]
        ax7.fill_between(history['time'], 0, rl_active_int, alpha=0.5, color='cyan', label='RL Active')

        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('RL Status')
        ax7.set_ylim(-0.1, 1.1)
        ax7.set_yticks([0, 1])
        ax7.set_yticklabels(['PID Only', 'PID+RL'])
        ax7.set_title('RL Activation Status')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # ==================== 8. 动作幅度 ====================
        ax8 = fig.add_subplot(2, 4, 8)

        if len(history['actions']) > 0:
            actions_array = np.array(history['actions'])
            action_norms = np.linalg.norm(actions_array, axis=1)
            ax8.plot(history['time'], action_norms, 'purple', linewidth=1.5)

        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Action L2 Norm')
        ax8.set_title('Control Action Magnitude')
        ax8.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✅ Comprehensive analysis saved to '{save_path}'")
        plt.show()

    def plot_error_decomposition(self, history, save_path='ppo_error_decomposition.png'):
        """
        🔥 新增: 误差分解详细图

        为每个Agent单独显示三类误差
        """
        history = self._trim_history(history)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        colors_err = {'total': 'red', 'planning': 'blue', 'tracking': 'green'}
        labels_agent = ['Agent 1 (Leader)', 'Agent 2', 'Agent 3', 'Agent 4']

        for i in range(4):
            ax = axes[i]

            ax.plot(history['time'], history['error_total'][i],
                    color=colors_err['total'], linewidth=2,
                    label='ε₁: Total (真实跟踪)')
            ax.plot(history['time'], history['error_planning'][i],
                    color=colors_err['planning'], linewidth=1.5, linestyle='--',
                    label='ε₂: Planning (轨迹重构)')
            ax.plot(history['time'], history['error_tracking'][i],
                    color=colors_err['tracking'], linewidth=1.5, linestyle=':',
                    label='ε₃: Tracking (控制器)')

            ax.axvspan(20, 70, alpha=0.15, color='gray', label='Turn Phase')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Error (ft)')
            ax.set_title(f'{labels_agent[i]} - Error Decomposition')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✅ Error decomposition saved to '{save_path}'")
        plt.show()

    def plot_top_view_comparison(self, history, save_path='ppo_topview_fixed.png'):
        """
        绘制俯视图对比 - 修复版

        左图: 实际轨迹 vs 理想编队
        右图: 实际轨迹 vs 协商轨迹 vs 理想编队
        """
        history = self._trim_history(history)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        colors = ['red', 'green', 'blue', 'orange']
        labels = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4']

        # ==================== 左图: 实际 vs 理想 ====================
        for i in range(4):
            pos_array = np.array(history['positions'][i])
            ideal_array = np.array(history['ideal_positions'][i])

            ax1.plot(pos_array[:, 0], pos_array[:, 1],
                     color=colors[i], label=f'{labels[i]} (Actual)',
                     linewidth=2, alpha=0.8)
            ax1.plot(ideal_array[:, 0], ideal_array[:, 1],
                     color=colors[i], linestyle=':',
                     label=f'{labels[i]} (Ideal)',
                     linewidth=1, alpha=0.5)

        leader_array = np.array(history['leader_pos'])
        ax1.plot(leader_array[:, 0], leader_array[:, 1],
                 'k--', linewidth=2, label='Leader Ref', alpha=0.6)

        ax1.set_xlabel('X (North, ft)')
        ax1.set_ylabel('Y (East, ft)')
        ax1.set_title('Actual vs Ideal Formation')
        ax1.axis('equal')
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        # ==================== 右图: 实际 vs 协商 vs 理想 ====================
        for i in range(4):
            pos_array = np.array(history['positions'][i])
            target_array = np.array(history['target_positions'][i])
            ideal_array = np.array(history['ideal_positions'][i])

            ax2.plot(ideal_array[:, 0], ideal_array[:, 1],
                     color=colors[i], linestyle=':',
                     linewidth=1, alpha=0.3, label=f'{labels[i]} (Ideal)')
            ax2.plot(target_array[:, 0], target_array[:, 1],
                     color=colors[i], linestyle='--',
                     linewidth=1.5, alpha=0.6, label=f'{labels[i]} (Negotiated)')
            ax2.plot(pos_array[:, 0], pos_array[:, 1],
                     color=colors[i], linewidth=2,
                     alpha=0.8, label=f'{labels[i]} (Actual)')

        ax2.set_xlabel('X (North, ft)')
        ax2.set_ylabel('Y (East, ft)')
        ax2.set_title('Actual vs Negotiated vs Ideal')
        ax2.axis('equal')
        ax2.legend(fontsize=6, ncol=3)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✅ Top-view comparison saved to '{save_path}'")
        plt.show()

    @staticmethod
    def _trim_history(history):
        if not history.get('time'):
            return history
        n_steps = len(history['time'])
        trimmed = {}
        for key, value in history.items():
            if isinstance(value, list):
                if len(value) == n_steps + 1:
                    trimmed[key] = value[:n_steps]
                elif len(value) > n_steps:
                    trimmed[key] = value[:n_steps]
                else:
                    trimmed[key] = value
            else:
                trimmed[key] = value
        return trimmed


def main():
    """主函数: 评估并可视化训练结果"""

    # ==================== 配置路径 ====================
    # 请根据您的实际训练输出路径修改
    LOG_DIR = "./logs/ppo_fixed"
    MODEL_PATH = f"{LOG_DIR}/best_model"
    VEC_NORMALIZE_PATH = f"{LOG_DIR}/vec_normalize.pkl"

    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"❌ Model not found: {MODEL_PATH}.zip")
        print("Please update MODEL_PATH in the script!")
        return

    # ==================== 创建可视化器 ====================
    visualizer = PPOVisualizer(
        model_path=MODEL_PATH,
        vec_normalize_path=VEC_NORMALIZE_PATH
    )

    # ==================== 运行评估 ====================
    history = visualizer.evaluate_episode(seed=42)

    # ==================== 生成可视化 ====================
    print("\nGenerating visualizations...")

    # 1. 综合分析图 (8个子图)
    visualizer.plot_comprehensive_analysis(
        history,
        save_path='ppo_comprehensive_analysis_fixed.png'
    )

    # 2. 误差分解详细图 (4个Agent分别显示)
    visualizer.plot_error_decomposition(
        history,
        save_path='ppo_error_decomposition_fixed.png'
    )

    # 3. 俯视图对比
    visualizer.plot_top_view_comparison(
        history,
        save_path='ppo_topview_comparison_fixed.png'
    )

    print("\n" + "=" * 70)
    print("✅ All visualizations complete!")
    print("=" * 70)
    print("Generated files:")
    print("  - ppo_comprehensive_analysis_fixed.png  (8-panel analysis)")
    print("  - ppo_error_decomposition_fixed.png     (error breakdown per agent)")
    print("  - ppo_topview_comparison_fixed.png      (trajectory comparison)")
    print("=" * 70)


if __name__ == "__main__":
    main()
