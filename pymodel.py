import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 定义模型参数
ms = 1.0  # 内滚道质量
ks = 1.0  # 内滚道刚度
cs = 0.1  # 内滚道阻尼
mp = 1.0  # 外滚道质量
kp = 1.0  # 外滚道刚度
cp = 0.1  # 外滚道阻尼
mr = 0.1  # 单元谐振器质量
kr = 10.0  # 单元谐振器刚度
cr = 0.5  # 单元谐振器阻尼
e = 0.1  # 偏心距离
omega_s = 1.0  # 轴的角频率
Fx = 0.0  # X轴方向的外部力
Fy = 0.0  # Y轴方向的外部力
nb = 10  # 滚动元素数量
g = 9.81  # 重力加速度（地球）
# 定义动力学模型
def model(y, t):
    xs, xsp, ys, ysp, xp, xpp, yp, ypp, yr, yrp = y

    # 计算接触力
    Fx_contact = -ks * (xs - xp) - cs * (xsp - xpp) + e * ms * omega_s ** 2 * np.cos(omega_s * t)
    Fy_contact = -ks * (ys - yp) - cs * (ysp - ypp) + e * ms * omega_s ** 2 * np.sin(omega_s * t)

    # 计算滚动元素的角度
    theta = 2 * np.pi * np.arange(0, nb) / nb + omega_s * t

    # 计算滚动体的位置
    xp_new = xs - e * np.cos(omega_s * t)
    yp_new = ys - e * np.sin(omega_s * t)

    # 计算单元谐振器的位移
    xrp = yr - yp
    yrp_dot = (Fy_contact - mp * g) / (mp * cr)

    # 计算yr的二阶导数
    yrpp = 0.0  # 在这个简化模型中，yrpp设为0

    # 构建动力学方程
    dxsdt = xsp
    dxspdt = (Fx_contact - mp * g) / (ms - mp) - cs * xsp - ks * (xs - xp_new)
    dysdt = ysp
    dyspdt = (Fy_contact + mp * omega_s ** 2 * xp_new) / (ms - mp) - cs * ysp - ks * (ys - yp_new)
    dxpdt = xpp
    dxp = (Fx_contact - mp * g) / mp - cp * xpp - kp * (xp - xs)
    dypdt = ypp
    dyp = (Fy_contact - mp * omega_s ** 2 * ys) / mp - cp * ypp - kp * (yp - ys)
    dyrdt = yrp
    dyrdt = yrp_dot
    dyrdt = yrpp

    return [dxsdt, dxspdt, dysdt, dyspdt, dxpdt, dxp, dypdt, dyp, dyrdt, dyrdt]


# 初始化初始条件
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
t = np.linspace(0, 10, 1000)  # 时间范围

# 解动力学方程
sol = odeint(model, y0, t)

# 提取位移信息
xs, _, ys, _, xp, _, yp, _, yr, _ = sol.T

# 绘制位移随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(t, xs, label='Inner Race X Displacement')
plt.plot(t, ys, label='Inner Race Y Displacement')
plt.plot(t, xp, label='Outer Race X Displacement')
plt.plot(t, yp, label='Outer Race Y Displacement')
plt.plot(t, yr, label='Unit Resonator Y Displacement')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()
plt.title('Rolling Bearing Dynamic Model')
plt.grid()
plt.show()
