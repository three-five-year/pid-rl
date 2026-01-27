# pid_tracker.py - 分层参数版本

import numpy as np

FT_G = 32.174


class PIDTracker:
    def __init__(self, dt: float = 0.05, agent_role: str = "leader"):
        """
        Args:
            dt: 时间步长
            agent_role: 飞机角色
                - "leader": 有Leader访问权限(Agent 1)
                - "follower_direct": 直接跟随Leader的僚机(Agent 2, 3)
                - "follower_indirect": 间接跟随的尾翼机(Agent 4)
        """
        self.dt = dt
        self.role = agent_role

        # ==================== 根据角色选择参数 ====================
        if agent_role == "leader":
            # Agent 1: 标准参数
            self.Kp_y = 0.003
            self.Ki_y = 0.000005
            self.Kd_y = 0.20
            self.Kp_phi = 0.8
            self.phi_limit_cruise = np.deg2rad(50)

        elif agent_role == "follower_direct":
            # Agent 2, 3: 提高响应速度
            self.Kp_y = 0.0065  # 提高50%
            self.Ki_y = 0.000008  # 稍微提高积分
            self.Kd_y = 0.25  # 增强阻尼
            self.Kp_phi = 1.0  # 更激进的航向跟踪
            self.phi_limit_cruise = np.deg2rad(55)

        else:  # "follower_indirect"
            # Agent 4: 最激进的参数,补偿信息延迟
            self.Kp_y = 0.006  # 提高100%
            self.Ki_y = 0.00001  # 提高积分以消除稳态误差
            self.Kd_y = 0.30  # 最强阻尼防止震荡
            self.Kp_phi = 1.2  # 最激进的航向跟踪
            self.phi_limit_cruise = np.deg2rad(60)

        # ==================== 通用参数 ====================
        self.int_lat = 0.0
        self.int_lat_limit = 50.0
        self.prev_err_lateral = 0.0
        self.phi_limit_turn = np.deg2rad(65)
        self.Kp_roll = 0.25

        # 滚转速率限制
        self.prev_phi_cmd = 0.0
        self.max_phi_dot = np.deg2rad(30)

        # ==================== 速度控制 ====================
        self.K_pos_v = 0.8
        self.max_catch_up_speed = 50.0
        self.Kp_v = 0.18
        self.Ki_v = 0.008
        self.Kd_v = 0.12
        self.int_v = 0.0
        self.prev_ev = 0.0

        # ==================== 纵向控制 ====================
        self.Kp_h = 0.008
        self.Kp_gamma = 12.0
        self.alpha_last = 0.0
        self.alpha_int = 0.0
        self.Kp_alpha = 3.2
        self.Kd_alpha = 0.5
        self.Ki_alpha = 0.25
        self.alpha_int_limit = 4.0

        self.beta_last = 0.0
        self.Kp_beta = 5.5
        self.Kd_beta = 0.5

        self._debug_counter = 0
        self._debug_enabled = False
    @staticmethod
    def _sat(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def wrap_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _alpha_control(self, alpha_deg, alpha_des_deg):
        e = alpha_des_deg - alpha_deg
        alpha_rate = (alpha_deg - self.alpha_last) / self.dt
        self.alpha_last = alpha_deg

        if abs(e) < 3.0:
            self.alpha_int = self._sat(self.alpha_int + e * self.dt, -4.0, 4.0)
        else:
            self.alpha_int *= 0.9

        u = self.Kp_alpha * e - self.Kd_alpha * alpha_rate + self.Ki_alpha * self.alpha_int
        return self._sat(-u, -25.0, 25.0)

    def _roll_control(self, roll_deg, roll_des_deg):
        e = roll_des_deg - roll_deg
        u = self.Kp_roll * e
        return self._sat(-u, -21.5, 21.5)

    def _beta_control(self, beta_deg):
        beta_rate = (beta_deg - self.beta_last) / self.dt
        self.beta_last = beta_deg
        u = self.Kp_beta * (0 - beta_deg) - self.Kd_beta * beta_rate
        return self._sat(u, -30.0, 30.0)

    def compute_control(
            self,
            target_pos: np.ndarray,
            target_vel: np.ndarray,
            current_pos: np.ndarray,
            vel_earth: np.ndarray,
            euler_rad: np.ndarray,
            alpha_rad: float,
            beta_rad: float,
            rot_body2earth: np.ndarray,
            feedforward_turn_rate: float = 0.0,
            leader_pos: np.ndarray = None
    ) -> np.ndarray:

        # ==================== 状态提取 ====================
        v_now = np.linalg.norm(vel_earth) + 1e-4
        psi_now = np.arctan2(vel_earth[1], vel_earth[0])
        v_h = np.linalg.norm(vel_earth[0:2]) + 1e-4
        gamma_now = np.arctan2(-vel_earth[2], v_h)

        # ==================== 参考系 ====================
        if np.linalg.norm(target_vel[0:2]) > 10:
            psi_reference = np.arctan2(target_vel[1], target_vel[0])
        else:
            delta_pos = target_pos - current_pos
            psi_reference = np.arctan2(delta_pos[1], delta_pos[0])

        cos_psi = np.cos(psi_reference)
        sin_psi = np.sin(psi_reference)

        pos_err = target_pos - current_pos
        err_longitudinal = pos_err[0] * cos_psi + pos_err[1] * sin_psi
        err_lateral = -pos_err[0] * sin_psi + pos_err[1] * cos_psi
        err_vertical = pos_err[2]

        # ==================== 油门控制 ====================
        if v_now < 180.0:
            throttle = 1.0
            v_cmd = 400.0
            self.int_v = 0.0
        else:
            v_target_norm = np.linalg.norm(target_vel)
            desired_vel_diff = self.K_pos_v * err_longitudinal
            desired_vel_diff = np.clip(desired_vel_diff, -self.max_catch_up_speed, self.max_catch_up_speed)
            v_cmd = v_target_norm + desired_vel_diff

            ev = v_cmd - v_now

            if abs(ev) < 25:
                self.int_v = self._sat(self.int_v + ev * self.dt, -12.0, 12.0)
            else:
                self.int_v *= 0.95

            dev = (ev - self.prev_ev) / self.dt
            self.prev_ev = ev
            throttle = 0.55 + self.Kp_v * ev + self.Ki_v * self.int_v + self.Kd_v * dev

        throttle = self._sat(throttle, 0.0, 1.0)

        # ==================== 横向控制 ====================
        is_turning = abs(feedforward_turn_rate) > 1e-6

        if is_turning:
            a_centripetal = v_now * abs(feedforward_turn_rate)
            tan_phi = a_centripetal / FT_G
            phi_feedforward = np.arctan(tan_phi) * np.sign(feedforward_turn_rate)
            phi_feedforward *= 0.6
            phi_feedforward = np.clip(phi_feedforward, -np.deg2rad(40), np.deg2rad(40))
            phi_limit = self.phi_limit_turn
        else:
            phi_feedforward = 0.0
            phi_limit = self.phi_limit_cruise

        # 反馈项 - 使用角色相关的增益
        d_err_lat = (err_lateral - self.prev_err_lateral) / self.dt
        self.prev_err_lateral = err_lateral

        # 条件积分
        if (abs(err_lateral) < 200 and
                abs(self.int_lat) < self.int_lat_limit * 0.8 and
                err_lateral * self.int_lat >= 0):
            self.int_lat += err_lateral * self.dt * 0.3
            self.int_lat = np.clip(self.int_lat, -self.int_lat_limit, self.int_lat_limit)
        else:
            self.int_lat *= 0.85

        lateral_correction = (
                self.Kp_y * err_lateral +
                self.Ki_y * self.int_lat +
                self.Kd_y * d_err_lat
        )
        lateral_correction = np.clip(lateral_correction, -np.deg2rad(18), np.deg2rad(18))

        psi_cmd = psi_reference + lateral_correction
        psi_err = self.wrap_pi(psi_cmd - psi_now)
        phi_feedback = self.Kp_phi * psi_err

        phi_cmd_total = phi_feedforward + phi_feedback
        phi_cmd = np.clip(phi_cmd_total, -phi_limit, phi_limit)

        # 滚转速率限制
        phi_rate = (phi_cmd - self.prev_phi_cmd) / self.dt
        if abs(phi_rate) > self.max_phi_dot:
            phi_cmd = self.prev_phi_cmd + np.sign(phi_rate) * self.max_phi_dot * self.dt
        self.prev_phi_cmd = phi_cmd

        # ==================== 纵向控制 ====================
        h_err = err_vertical
        gamma_cmd = -self.Kp_h * h_err
        gamma_cmd = np.clip(gamma_cmd, -np.deg2rad(18), np.deg2rad(18))

        gamma_err = gamma_cmd - gamma_now
        nz_cmd = self.Kp_gamma * gamma_err + 1.0
        nz_cmd = np.clip(nz_cmd, 0.2, 5.5)

        alpha_des_deg = 2.0 + (nz_cmd - 1.0) * 3.2
        max_alpha = 14.0 if v_now < 300.0 else 20.0
        alpha_des_deg = np.clip(alpha_des_deg, -2.0, max_alpha)

        # ==================== 执行器 ====================
        alpha_deg = np.rad2deg(alpha_rad)
        beta_deg = np.rad2deg(beta_rad)
        roll_deg = np.rad2deg(euler_rad[0])
        roll_des_deg = np.rad2deg(phi_cmd)

        elevator = self._alpha_control(alpha_deg, alpha_des_deg)
        aileron = self._roll_control(roll_deg, roll_des_deg)
        rudder = self._beta_control(beta_deg)

        # ==================== 调试 ====================
        self._debug_counter += 1
        if self._debug_enabled and self._debug_counter % 20 == 0:
            print(f"[PID-{self.role}] t={self._debug_counter * self.dt:.1f}s | "
                  f"err_lat={err_lateral:.0f}ft | "
                  f"φ_cmd={np.rad2deg(phi_cmd):.1f}° | "
                  f"Kp={self.Kp_y:.5f}")

        return np.array([throttle, elevator, aileron, rudder])
