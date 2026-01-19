import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree

# 你的模型：確保 PSSpy.py 在同一層，或自行改 sys.path
import PSSpy as pss


def main():
    # -------------------------
    # 使用者可改的參數
    # -------------------------
    frames = 29
    fps = 10
    interval_ms = 2900

    Theta_zero = np.deg2rad(42)
    Phi_zero = np.deg2rad(240)
    Inclination = np.deg2rad(0)

    T = 0.7                 # Myr（你 PSS_model 裡用 Myr）
    omega_max = 0.56         # 最終 omega（t*omega 會變成 T*omega_max）
    solar_mass = 5

    r_in_au = 1.6e3
    r_out_au = 7e3
    resolution = 1200

    lim = 1.1 * r_out_au

    # ---- 線段周圍的 Gaussian halo（離線越遠越暗）----
    halo_sigma_au = 150.0          # halo 的 1-sigma 寬度（AU）
    halo_base_alpha = 1         # 中心線最大透明度（之後乘 Gaussian 權重）
    # -------------------------
    # 畫布（2D top-down field）
    # -------------------------
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.set_axis_off()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])

    grid_n = 520  # 2D 灰階面解析度：越大越細但越慢
    xs = np.linspace(-lim, lim, grid_n)
    ys = np.linspace(-lim, lim, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

    def draw_background_2d():
        # th = np.linspace(0, 2*np.pi, 400)
        # ax.plot(1.2 * r_out_au * np.cos(th), 1.2 * r_out_au * np.sin(th),
        #         color='gray', linewidth=1.0, alpha=0.7)
        # ax.plot(0.4 * r_out_au * np.cos(th), 0.4 * r_out_au * np.sin(th),
        #         color='gray', linewidth=1.0, alpha=0.5)
        # ax.plot([-lim, lim], [0, 0], color='black', linewidth=1.0)
        # ax.plot([0, 0], [-lim, lim], color='black', linewidth=1.0)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_aspect('equal', adjustable='box')

    # 先建立 imshow（之後每幀只更新 data）
    im = ax.imshow(
        np.zeros((grid_n, grid_n), dtype=float),
        extent=[-lim, lim, -lim, lim],
        origin='lower',
        cmap=plt.get_cmap('Greys').copy(),        
        vmin=0.0, vmax=1.0,
        interpolation='bilinear'
    )
    im.cmap.set_bad('white')
    draw_background_2d()
    txt = ax.text(-0.65*lim, 0.65*lim, '', fontsize=12, color='k')

    def update(frame):
        omega = omega_max * (frame / (frames - 1))

        x, y, z, _, _, _ = pss.PSS_model(
            Theta_zero, Phi_zero, Inclination, T, omega,
            solar_mass, r_in_au, r_out_au, resolution,
            scale='log', log_power=1.5
        )

        # 用 KDTree 近似到「線」距離：對 model line 取樣成點集合
        stride = max(1, int(len(x) / 1200))
        pts2 = np.column_stack([x[::stride], y[::stride]])
        tree = cKDTree(pts2)

        d, _ = tree.query(grid_pts, k=1)
        D = d.reshape(grid_n, grid_n)

        # Gaussian halo：線附近亮、遠離線暗（連續面）
        I = np.exp(-0.5 * (D / halo_sigma_au) ** 2)
        Z = np.clip(halo_base_alpha * I, 0.0, 1.0)

        Zm = np.ma.masked_less(Z, 0.02)  # 0.02 可調：越大背景越乾淨
        im.set_data(Zm)
        txt.set_text(r'$t_{\rm s}\omega$ = ' + f'{T*omega:.3f}')

        return [im, txt]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=True)    
    out = "Phi_schematic.gif"
    with tqdm(total=frames, desc="Rendering GIF", unit="frame") as pbar:
        def _cb(i, n):
            # i: 目前完成的 frame index (0-based)
            # n: 總 frames（matplotlib 傳進來的）
            pbar.n = i + 1
            pbar.refresh()

        ani.save(
            out,
            writer="pillow",
            fps=fps,
            progress_callback=_cb,
            savefig_kwargs=dict(facecolor='white', pad_inches=0)
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()