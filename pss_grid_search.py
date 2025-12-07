# ============================================================
# pss_grid_search.py
# 通用 5D grid search + 自動 prior 決定模組
# ============================================================

import numpy as np
from itertools import product
from tqdm.auto import tqdm
import scipy.constants as spc
import PSSpy as pss


def run_grid_search(
    streamer_x_AU, streamer_z_AU, streamer_v_LS_km,
    v_weight_phys, M_star, scale, log_power,
    radius_ref_au, n_grid=10,
    T_factor_range=(130.0, 300.0),
    verbose=True
):
    """
    通用 grid search 函式。
    回傳:
      - best_params: dict with (Theta, Phi, Incl, T, Omega)
      - grid: dict of arrays (theta_grid, phi_grid, ...)
      - error: 5D array
    """
    Omega_ref = pss.Omega_ref(radius_ref_au, M_star)
    P_half_Myr = np.pi / Omega_ref / 1e6 / spc.year
    T_range = [T_factor_range[0]*P_half_Myr, T_factor_range[1]*P_half_Myr]

    n_theta = n_phi = n_inc = n_T = n_Omega = n_grid
    theta_grid = np.linspace(0.0, 0.5*np.pi, n_theta + 2)[1:-1]
    phi_grid   = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    inc_grid   = np.linspace(-0.5*np.pi, 0.5*np.pi, n_inc + 2)[1:-1]
    T_grid     = np.logspace(np.log10(T_range[0]), np.log10(T_range[1]), n_T)
    omega_grid = np.linspace(0.0, 1.0, n_Omega + 1)[1:]

    error = np.zeros((n_theta, n_phi, n_inc, n_T, n_Omega), dtype=float)
    best_val, best_idx = np.inf, None

    total = n_theta * n_phi * n_inc * n_T * n_Omega
    if verbose:
        print(f"\n[Grid fitting] Start 5D grid search ({total:,} points) ...")

    for i_th, i_ph, i_I, i_T, i_O in tqdm(
        product(range(n_theta), range(n_phi), range(n_inc), range(n_T), range(n_Omega)),
        total=total, desc="Grid search", ncols=80
    ):
        val = pss.error_function(
            [theta_grid[i_th], phi_grid[i_ph]],
            streamer_x_AU, streamer_z_AU, streamer_v_LS_km,
            v_weight_phys, T_grid[i_T], omega_grid[i_O],
            inc_grid[i_I], M_star, scale, log_power
        )
        error[i_th, i_ph, i_I, i_T, i_O] = val
        if val < best_val:
            best_val = val

    min_th, min_ph, min_I, min_T, min_O = np.unravel_index(np.argmin(error), error.shape)
    best_params = dict(
        Theta=theta_grid[min_th],
        Phi=phi_grid[min_ph],
        Incl=inc_grid[min_I],
        T=T_grid[min_T],
        Omega=omega_grid[min_O],
        best_val=best_val
    )
    grid = dict(
        theta_grid=theta_grid,
        phi_grid=phi_grid,
        inc_grid=inc_grid,
        T_grid=T_grid,
        omega_grid=omega_grid
    )
    return best_params, grid, error


def compute_priors_from_grid(error, grid, best_val, frac=0.05):
    """
    根據 grid 結果自動決定每個參數的 prior 範圍，
    並且回傳一個代表性的 sigma_like（用在 fast likelihood 裡）。
    """
    theta_grid = grid["theta_grid"]
    phi_grid   = grid["phi_grid"]
    inc_grid   = grid["inc_grid"]
    T_grid     = grid["T_grid"]
    omega_grid = grid["omega_grid"]

    Theta_flat, Phi_flat, Incl_flat, T_flat, Omega_flat = np.meshgrid(
        theta_grid, phi_grid, inc_grid, T_grid, omega_grid, indexing="ij"
    )
    Theta_flat = Theta_flat.ravel()
    Phi_flat   = Phi_flat.ravel()
    Incl_flat  = Incl_flat.ravel()
    T_flat     = T_flat.ravel()
    Omega_flat = Omega_flat.ravel()
    err_flat   = error.ravel()

    # ---------- 1) 選出 "好" 的 grid 模型 ----------
    mask_good = err_flat <= best_val * (1.0 + frac)

    # 如果好模型太少，放寬 frac
    if np.sum(mask_good) < 50:
        frac = 0.10
        mask_good = err_flat <= best_val * (1.0 + frac)

    # 如果還是太少，就直接挑誤差最小的前 k 個
    if np.sum(mask_good) < 50:
        k = max(50, int(0.01 * err_flat.size))
        idx_sort = np.argsort(err_flat)
        mask_good = np.zeros_like(err_flat, dtype=bool)
        mask_good[idx_sort[:k]] = True

    # 確保至少有一個
    if np.sum(mask_good) == 0:
        raise RuntimeError("compute_priors_from_grid: no good models selected.")

    # 這一群就是「grid 中還不錯的模型」
    E_good = err_flat[mask_good]

    # ---------- 2) 用 E_good 的中位數當作 sigma_like ----------
    sigma_like = np.median(E_good)

    # ---------- 3) 用這群好模型決定 prior 範圍 ----------
    Theta_good = Theta_flat[mask_good]
    Phi_good   = Phi_flat[mask_good]
    Incl_good  = Incl_flat[mask_good]
    T_good     = T_flat[mask_good]
    Omega_good = Omega_flat[mask_good]

    def padded_range(arr, pad_frac=0.3, abs_min=None, abs_max=None):
        q5, q95 = np.percentile(arr, [5, 95])
        width = q95 - q5
        pad = pad_frac * width
        lo = max(q5 - pad, abs_min) if abs_min is not None else q5 - pad
        hi = min(q95 + pad, abs_max) if abs_max is not None else q95 + pad
        return lo, hi

    priors = {
        "Theta zero":  padded_range(Theta_good, abs_min=0.0,     abs_max=0.5*np.pi),
        "Phi zero":    padded_range(Phi_good,   abs_min=-np.pi,  abs_max=np.pi),
        "Inclination": padded_range(Incl_good,  abs_min=-0.5*np.pi, abs_max=0.5*np.pi),
        "Time":        padded_range(T_good,     abs_min=0.0),
        "Omega":       padded_range(Omega_good, abs_min=0.0,     abs_max=1.0),
    }

    # ⭐ 重點：現在回傳 (priors, sigma_like)
    return priors, sigma_like