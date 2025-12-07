import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 參數設定
n = 200  # 解析度
frames = 60  # 幀數
duration = 5  # 秒
fps = frames / duration

x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, n)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# 建立圖形
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_axis_off()
im = ax.imshow(np.zeros_like(R), extent=(-2, 2, -2, 2),
               cmap="bone", vmin=-1, vmax=1)  # 柔和色調

def update(frame):
    t = frame / frames * duration
    omega_twist = 2 * np.pi * 1 / duration   # 扭轉角速度（2圈）
    rotation_speed = - np.pi / 15 / duration    # 整體旋轉速度（5秒轉45度）
    twist_rate = 0.5                         # 外圈延遲旋轉程度
    ripple_strength = 1                      # 皺摺強度
    progress = (t / duration)**1.5           # 平面逐漸起皺

    # 整體旋轉角度（布整體轉動）
    Theta_global = Theta + rotation_speed * t

    # 局部扭轉：外圈延遲 → 螺旋皺摺
    Theta_twist = Theta_global + omega_twist * t * (1 - np.exp(-twist_rate * R))

    # 高度場（皺摺）
    Z = progress * np.sin(ripple_strength * R + 2 * Theta_twist) * np.exp(-R**2 / 2)

    im.set_data(Z)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)

# 儲存 GIF
gif_path = "twisting.gif"
ani.save(gif_path, writer="pillow", fps=fps)
gif_path