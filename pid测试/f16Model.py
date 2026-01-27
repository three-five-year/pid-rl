from scipy.spatial.transform import Rotation as R
import numpy as np
from fForward import fForward
from adc import adc


class F16:
    def __init__(self, time_step=0.05):
        self.time_step = time_step
        self.sim_dt = 0.01
        self.sub_steps = int(self.time_step / self.sim_dt)

        # 状态变量
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.euler = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.power = 0

        # 气动数据
        self.alpha = 0
        self.beta = 0
        self.mach = 0
        self.qbar = 0

        # 舵面状态
        self.throttle = 0
        self.elevator = 0
        self.aileron = 0
        self.rudder = 0

        self.rotation_body2earth = np.eye(3)
        self.velocity_earth = np.zeros(3)

    def reset(self, position=np.array([0., 0., 0.]),
              euler=np.array([0., 0., 0.]),
              velocity_body=None,
              mach=0.5):
        self.position = position
        self.height = -self.position[2]
        self.euler = euler
        self.angular_velocity = np.zeros(3)

        if velocity_body is not None:
            self.velocity = velocity_body
            self.ground_speed = np.linalg.norm(velocity_body)
        else:
            temperature = 390 if self.height >= 35000 else 519 * (1 - .703e-5 * self.height)
            sound_speed = (1.4 * 1716.3 * temperature) ** 0.5
            target_speed = sound_speed * mach
            self.velocity = np.array([target_speed, 0, 0])
            self.ground_speed = target_speed

        self.power = 50.0
        self._update_state()

    def step(self, u_cmd):
        """
        u_cmd: [throttle(0-1), elevator(deg), aileron(deg), rudder(deg)]
        直接接收控制器计算好的物理舵面指令
        """
        t_cmd, e_cmd, a_cmd, r_cmd = u_cmd

        # 简单的执行机构动态 (Actuator Dynamics) - 模拟舵机限速
        self.throttle = np.clip(t_cmd, 0, 1)
        self.elevator = np.clip(e_cmd, -25, 25)
        self.aileron = np.clip(a_cmd, -21.5, 21.5)
        self.rudder = np.clip(r_cmd, -30, 30)

        state = np.concatenate([
            self.position, self.velocity, self.euler,
            self.angular_velocity, [self.power]
        ])

        # 物理积分循环 (RK4)
        for _ in range(self.sub_steps):
            u_act = [self.throttle, self.elevator, self.aileron, self.rudder]

            k1, _, _ = fForward(state=state, u=u_act)
            k2, _, _ = fForward(state=state + k1 * self.sim_dt / 2, u=u_act)
            k3, _, _ = fForward(state=state + k2 * self.sim_dt / 2, u=u_act)
            k4, _, _ = fForward(state=state + k3 * self.sim_dt, u=u_act)

            state = state + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * self.sim_dt

        # 更新状态
        self.position = state[0:3]
        self.velocity = state[3:6]
        self.euler = state[6:9]
        self.angular_velocity = state[9:12]
        self.power = state[12]

        self._update_state()

    def _update_state(self):
        self.height = -self.position[2]
        self.rotation_body2earth = R.from_euler('ZYX', self.euler[::-1]).as_matrix()
        self.velocity_earth = np.matmul(self.rotation_body2earth, self.velocity)
        self.ground_speed = np.linalg.norm(self.velocity)

        if self.ground_speed < 1e-3:
            self.alpha = 0
            self.beta = 0
        else:
            self.alpha = np.arctan2(self.velocity[2], self.velocity[0])
            self.beta = np.arcsin(np.clip(self.velocity[1] / self.ground_speed, -1, 1))

        self.mach, self.qbar = adc(groundSpeed=self.ground_speed, height=self.height)