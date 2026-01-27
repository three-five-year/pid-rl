from util import rho0


def adc(groundSpeed, height):
    '''air data computation'''
    tfac = 1 - .703e-5 * height#计算温度因子，随高度增加温度因子减小。
    temperature = 390 if height >= 35000 else 519 * tfac#如果高度 ≥ 35000英尺(平流层):温度恒定为 390°R，否则(对流层):519°R × 温度因子

    rho = rho0 * tfac ** 4.14  # air mass density，空气密度
    soundSpeed = (1.4 * 1716.3 * temperature) ** 0.5  # sqrt(adiabatic exponent * gas constant * temperature)，计算音速
    mach = groundSpeed / soundSpeed#计算马赫数:地速除以音速
    qbar = rho * groundSpeed ** 2 / 2  # dynamic pressure动压 q = ½ρv²
    # ps = 1715 * rho * temperature # static pressure
    return mach, qbar#根据飞行高度和地速,计算大气参数(温度、密度、音速)并返回马赫数和动压