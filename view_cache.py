#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
view_cache.py

簡單檢視 .npz cache 檔內容的小工具。

用法例子：
    python view_cache.py --file Per50_results/cache/Per-emb-50_grid_results.npz
    python view_cache.py --file cache/Per-emb-2_cache.npz

功能：
  - 顯示所有 key、shape、dtype
  - 對常見參數 (Theta, Phi, Incl, Time, Omega, Mdot, RMSE...) 做比較好讀的輸出
  - 對陣列型資料，顯示前幾個元素做 sanity check
"""

import argparse
import os
import numpy as np


# ------------------ 小工具函式 ------------------ #

def is_angle_key(name: str) -> bool:
    """判斷這個 key 是否代表「角度（radian）」參數."""
    s = name.lower()
    return (
        "theta" in s
        or "phi" in s
        or "incl" in s
        or "pa" in s
    )


def is_time_key(name: str) -> bool:
    s = name.lower()
    return "time" in s or "t_myr" in s or s.endswith("_t")


def is_omega_key(name: str) -> bool:
    return "omega" in name.lower()


def is_rmse_key(name: str) -> bool:
    s = name.lower()
    return ("rmse" in s) or ("error" in s and "best" in s)


def pretty_print_scalar(name: str, value):
    """針對 scalar 做比較好讀的輸出格式."""
    if np.isnan(value):
        print(f"  {name:24s}: NaN")
        return

    if is_angle_key(name):
        deg = np.rad2deg(value)
        print(f"  {name:24s}: {value:.6e} rad  ({deg:.3f} deg)")
    elif is_time_key(name):
        print(f"  {name:24s}: {value:.6e} Myr")
    elif is_omega_key(name):
        print(f"  {name:24s}: {value:.6e} (Omega)")
    elif "mdot" in name.lower():
        print(f"  {name:24s}: {value:.6e} M_sun/yr")
    elif is_rmse_key(name):
        print(f"  {name:24s}: {value:.6e}  (RMSE / error-like)")
    else:
        print(f"  {name:24s}: {value:.6e}")


def summarize_array(name: str, arr, max_show: int = 5):
    """輸出陣列的基本資訊與前幾個元素."""
    arr = np.asarray(arr)
    print(f"  {name:24s}: array, shape={arr.shape}, dtype={arr.dtype}")

    # 只顯示 1D 前幾個元素，避免炸掉畫面
    if arr.ndim == 0:
        pretty_print_scalar(name, float(arr))
        return

    flat = arr.ravel()
    n = flat.size
    if n == 0:
        print("    (empty array)")
        return

    if is_angle_key(name):
        # 對角度陣列：印出前幾個元素的 rad & deg
        n_show = min(max_show, n)
        print(f"    first {n_show} values (rad -> deg):")
        for i in range(n_show):
            v = flat[i]
            d = np.rad2deg(v)
            print(f"      [{i:2d}] {v:.6e} rad  ({d:.3f} deg)")
    else:
        # 一般陣列
        n_show = min(max_show, n)
        print(f"    first {n_show} values:")
        for i in range(n_show):
            v = flat[i]
            if isinstance(v, (float, np.floating)):
                print(f"      [{i:2d}] {v:.6e}")
            else:
                # e.g. object array 裡面又是 array
                print(f"      [{i:2d}] type={type(v)}, repr={repr(v)[:80]}...")


def view_cache(path: str, verbose: bool = True):
    if not os.path.isfile(path):
        print(f"[Error] File not found: {path}")
        return

    print(f"[view_cache] Loading: {path}")
    c = np.load(path, allow_pickle=True)

    print("\n=== Keys in cache ===")
    for k in c.files:
        v = c[k]
        print(f"- {k:24s}  shape={np.shape(v)!r}, dtype={getattr(v, 'dtype', type(v))}")

    print("\n=== Detailed summary ===")
    for k in c.files:
        v = c[k]
        print(f"\n--- {k} ---")
        # scalar
        if np.isscalar(v) or (isinstance(v, np.ndarray) and v.shape == ()):
            try:
                val = float(v)
                pretty_print_scalar(k, val)
            except Exception:
                print(f"  {k:24s}: scalar, type={type(v)}, value={repr(v)}")
        else:
            summarize_array(k, v)


# ------------------ main ------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Inspect .npz cache files (Per-emb-2 / Per-emb-50 / SCrA results)"
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to .npz cache file (e.g. Per50_results/cache/Per-emb-50_grid_results.npz)",
    )
    args = parser.parse_args()

    view_cache(args.file)


if __name__ == "__main__":
    main()