from scipy.spatial.transform import Rotation as R
import numpy as np
from math import sin, cos, tan

from adc import adc
from thrust import calculate_thrust
from coefficient import dampp, cx, cy, cz, cl, cm, cn, dlda, dldr, dnda, dndr
from util import rad2degree, s, b, cbar, mass, he, gravity, J, Jinv


def fForward(state, u):
    sd = np.zeros(shape=[13])

    position = state[0: 3]  # feet
    velocity = state[3: 6]  # feet / s
    euler = state[6: 9]  # rad
    angular_velocity = state[9: 12]  # rad / s
    power = state[12]  # 0-100
    throttle, elevator, aileron, rudder = u  # [0, 1], [-25, 25], [-21.5, 21.5], [30, -30]

    # [修正] 计算地速
    ground_speed = np.linalg.norm(velocity)

    height = - position[2]

    # [修正] 增加除零保护和数值限幅
    if ground_speed < 1e-3:
        # 速度过小，认为迎角和侧滑角为0
        alpha = 0.0
        beta = 0.0
    else:
        alpha = np.arctan2(velocity[2], velocity[0])
        # 限制 arcsin 的输入在 [-1, 1] 之间，防止浮点误差导致 NaN
        beta_arg = np.clip(velocity[1] / ground_speed, -1.0, 1.0)
        beta = np.arcsin(beta_arg)

    rotation_body2earth = R.from_euler(seq='ZYX', angles=euler[::-1]).as_matrix()
    phi, theta, psi = euler
    tan_theta = tan(theta)
    cos_theta = cos(theta)
    sin_phi = sin(phi)
    cos_phi = cos(phi)

    # 奇异点保护 (Theta 接近 90度时 tan_theta 会爆炸)
    if abs(cos_theta) < 1e-4:
        cos_theta = 1e-4 * np.sign(cos_theta)

    rotation_h = np.array([[1, tan_theta * sin_phi, tan_theta * cos_phi],
                           [0, cos_phi, -sin_phi],
                           [0, sin_phi / cos_theta, cos_phi / cos_theta]])

    mach, qbar = adc(groundSpeed=ground_speed, height=height)  # qbar: dynamic pressure
    sd[12], thrust = calculate_thrust(throttle=throttle, power=power, height=height, mach=mach)

    # the look-up table in C calculation needs angle in degree
    alpha_deg = alpha * rad2degree
    beta_deg = beta * rad2degree

    # [修正] 将计算出的角度传入
    cxt = cx(alpha_deg, elevator)
    cyt = cy(beta_deg, aileron, rudder)
    czt = cz(alpha_deg, beta_deg, elevator)
    daileron = aileron / 20
    drudder = rudder / 30
    clt = cl(alpha_deg, beta_deg) + dlda(alpha_deg, beta_deg) * daileron + dldr(alpha_deg, beta_deg) * drudder
    cmt = cm(alpha_deg, elevator)
    cnt = cn(alpha_deg, beta_deg) + dnda(alpha_deg, beta_deg) * daileron + dndr(alpha_deg, beta_deg) * drudder

    # add damping derivatives
    p, q, r = angular_velocity
    # [修正] 防止 ground_speed 为 0 导致 tvt 无穷大
    if ground_speed < 1.0:
        tvt = 0.5 / 1.0
    else:
        tvt = 0.5 / ground_speed

    b2v = b * tvt
    cq = cbar * q * tvt
    d = dampp(alpha_deg)

    cxt = cxt + cq * d[0]
    cyt = cyt + b2v * (d[1] * r + d[2] * p)
    czt = czt + cq * d[3]
    clt = clt + b2v * (d[4] * r + d[5] * p)
    cmt = cmt + cq * d[6]
    cnt = cnt + b2v * (d[7] * r + d[8] * p)

    # \dot{position} = velocity
    sd[0: 3] = np.matmul(rotation_body2earth, velocity)

    # \dot{velocity} = acceleration
    force = qbar * s * np.array([cxt, cyt, czt])
    sd[3: 6] = (force + np.array([thrust, 0, 0])) / mass + \
               np.matmul(rotation_body2earth.T, gravity) + \
               np.cross(velocity, angular_velocity)

    # \dot{euler} = h * angular_velocity
    sd[6: 9] = np.matmul(rotation_h, angular_velocity)

    # \dot{angular velocity} = moment
    torque = qbar * s * np.array([b * clt, cbar * cmt, b * cnt]) + np.array([0, -r * he, q * he])
    sd[9: 12] = np.matmul(Jinv, torque - np.cross(angular_velocity, np.matmul(J, angular_velocity)))

    _, overloady, overloadz = force / mass / 32.17
    return sd, overloady, -overloadz