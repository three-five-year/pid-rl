import numpy as np
#--- 慢车推力 (Idle Thrust) 表 -
a = np.array([[1060, 670, 880, 1140, 1500, 1860],
        [635, 425, 690, 1010, 1330, 1700],
        [60, 25, 345, 755, 1130, 1525],
        [-1020, -170, -300, 350, 910, 1360],
        [-2700, -1900, -1300, -247, 600, 1100],
        [-3600, -1400, -595, -342, -200, 700]], dtype=float).T#行代表高度 (i)，列代表马赫数 (m)。
# 这里的负数（如 -3600）表示在高空高速下收油门，发动机会产生巨大的气动阻力。
# --- 军用推力 (Military Thrust) 表 ---
# 军用推力指不开加力时的最大推力 (Power = 50)。
# 数据随高度增加而显著降低（从12680磅降到1400磅）
b = np.array([[12680, 9150, 6200, 3950, 2450, 1400],
    [12680, 9150, 6313, 4040, 2470, 1400],
    [12610, 9312, 6610, 4290, 2600, 1560],
    [12640, 9839, 7090, 4660, 2840, 1660],
    [12390, 10176, 7750, 5320, 3250, 1930],
    [11680, 9848, 8050, 6100, 3800, 2310]], dtype=float).T
# --- 最大推力 (Maximum Thrust) 表 ---
# 全加力状态下的最大推力 (Power = 100)。
# 数值通常是军用推力的1.5到2倍。
c = np.array([[20000, 15000, 10800, 7000, 4000, 2500],
    [21420, 15700, 11225, 7323, 4435, 2600],
    [22700, 16860, 12250, 8154, 5000, 2835],
    [24240, 18910, 13760, 9285, 5700, 3215],
    [26070, 21075, 15975, 11115, 6860, 3950],
    [28886, 23319, 18300, 13484, 8642, 5057]], dtype=float).T


def tgear(throttle):
    '''
    Accelerator transmission device
    0 < throttle < 1
    0 < power < 100
    throttle = 0.77 ---> power = 50
    '''
    # 如果油门小于 0.77（非加力区）：
    # 线性映射：0 -> 0 (慢车), 0.77 -> 50 (军用推力)
    # 64.94 * 0.77 ≈ 50
    # 如果油门大于 0.77（加力区）：
    # 线性映射：0.77 -> 50, 1.0 -> 100 (全加力)
    tgear = 64.94 * throttle if throttle < 0.77 else 217.38 * throttle - 117.38#油门杆映射 tgear(throttle)将物理油门杆的位置（0.0 到 1.0）映射到内部的功率指令 power（0 到 100）
    return tgear

#喷气式发动机不能瞬时改变推力，有一个“进气量建立（Spool up/down）”的过程。这两个函数模拟了一阶滞后响应。
def rtau(delta_power):
    '''rtau function''' #计算时间常数的倒数 (1/tau)
    if delta_power <= 25:#功率的变化值
        rt = 1.0  # reciprocal time constance
    elif delta_power >= 50:
        rt = 0.1
    else:
        rt = 1.9 - .036 * delta_power
    return rt


def pdot(p3, p1):
    '''
    power dot function
    p3: actual power
    p1: power command
    0 < p1, p3 < 100
    when power ≈ 50, pd becomes very small, power slowly change
    '''
    if p1 >= 50: #
        if p3 >= 50:
            t = 5 # 响应很快 (tau = 0.2s)，因为只是调节喷油量，不需要改变核心机转速
            p2 = p1# 目标就是指令值
        else:
            p2 = 60
            t = rtau(p2 - p3)# 使用标准滞后计算
    else:
        if p3 >= 50:
            t = 5
            p2 = 40
        else:
            p2 = p1
            t = rtau(p2 - p3)
    pd = t * (p2 - p3)
    return pd#计算功率状态的导数 模拟发动机响应滞后


def thrust(power, height, mach):
    '''Engine thrust model'''
    if height < 0:
        height = 0.01

    h = .0001 * height

    i = int(h)

    if i >= 5:
        i = 4

    dh = h - i
    rm = 5 * mach
    m = int(rm)

    if m >= 5:
        m = 4
    elif m <= 0:
        m = 0

    dm = rm - m
    cdh = 1 - dh
## 计算当前的军用推力 (Power=50时的推力)
    # 这是一个标准的双线性插值公式：f(x,y) ≈ ...
    # 先在高度维度插值，再在马赫维度插值？
    s = b[i, m] * cdh + b[i + 1, m] * dh
    t = b[i, m + 1] * cdh + b[i + 1, m + 1] * dh
    tmil = s + (t - s) * dm
# # 非加力区：在 慢车推力(a) 和 军用推力(tmil) 之间插值
    if power < 50:
        s = a[i, m] * cdh + a[i + 1, m] * dh
        t = a[i, m + 1] * cdh + a[i + 1, m + 1] * dh
        tidl = s + (t - s) * dm # 当前状态下的慢车推力
        thrst = tidl + (tmil - tidl) * power * 0.02# 最终推力 = 慢车 + (军用-慢车) * 比例
        # power * 0.02 相当于 power / 50，将 0~50 归一化为 0~1
    else:
        # 加力区：在 军用推力(tmil) 和 最大推力(c) 之间插值
        s = c[i, m] * cdh + c[i + 1, m] * dh
        t = c[i, m + 1] * cdh + c[i + 1, m + 1] * dh
        tmax = s + (t - s) * dm
        thrst = tmil + (tmax - tmil) * (power - 50) * 0.02

    return thrst


def calculate_thrust(throttle, power, height, mach):
    power_command = tgear(throttle=throttle)
    power_dot = pdot(p3=power, p1=power_command)## 2. 计算功率变化率 (用于更新下一时刻的 power 状态)
    thr = thrust(power=power, height=height, mach=mach) # 3. 计算当前时刻产生的推力 (用于动力学方程 f=ma)
    return power_dot, thr
