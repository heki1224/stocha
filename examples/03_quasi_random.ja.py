"""
stocha チュートリアル 3: 準乱数列の生成
========================================

Sobol 列と Halton 列（低不一致列）の使い方と、
数値積分における通常の擬似乱数モンテカルロに対する優位性を示す。

金融工学的背景:
    - 準モンテカルロ（QMC）は O(1/N) の収束速度を達成する（通常 MC は O(1/sqrt(N))）。
      滑らかな低〜中次元の被積分関数において、サンプル数を大幅に削減できる。
    - Sobol 列（Joe & Kuo 2008）は金融シミュレーション（オプション定価、CVA、
      シナリオ生成）で事実上の標準。
    - Halton 列は連続する素数を基底とする。実装は単純だが高次元では Sobol に劣る。
"""

import math
import time

import numpy as np
import stocha

# --- 1. Sobol 列 ---
print("=" * 55)
print("1. Sobol 低不一致列")
print("=" * 55)

pts = stocha.sobol(dim=2, n_samples=1024)
print(f"形状: {pts.shape}")
print(f"値域: [{pts.min():.6f}, {pts.max():.6f}]  (期待値: [0, 1))")
print(f"次元ごとの平均: {pts.mean(axis=0)}")  # ~0.5 になるはず
print(f"次元ごとの標準偏差: {pts.std(axis=0)}")   # Uniform[0,1] → ~0.289

# --- 2. Halton 列 ---
print("\n" + "=" * 55)
print("2. Halton 低不一致列")
print("=" * 55)

pts_h = stocha.halton(dim=2, n_samples=1024)
print(f"形状: {pts_h.shape}")
print(f"値域: [{pts_h.min():.6f}, {pts_h.max():.6f}]  (期待値: (0, 1))")
print(f"次元ごとの平均: {pts_h.mean(axis=0)}")
print(f"次元ごとの標準偏差: {pts_h.std(axis=0)}")

# skip パラメータ: ランダム化 QMC やバーンイン回避に有用
pts_h_skip = stocha.halton(dim=2, n_samples=100, skip=1000)
print(f"Halton (skip=1000), 形状: {pts_h_skip.shape}")

# --- 3. 収束比較: 円周率の推定 ---
print("\n" + "=" * 55)
print("3. 収束比較: 円周率の推定")
print("   (単位円内に入る点の割合 × 4 ≈ π)")
print("=" * 55)

rng = stocha.RNG(seed=42)

print(f"\n{'N':>10}  {'MC 誤差':>12}  {'Sobol 誤差':>12}  {'Halton 誤差':>12}")
print("-" * 52)

for n in [64, 256, 1024, 4096, 16384, 65536]:
    # 通常の擬似乱数モンテカルロ
    u_mc = rng.uniform(size=n * 2).reshape(n, 2)
    pi_mc = 4.0 * np.sum(u_mc[:, 0]**2 + u_mc[:, 1]**2 <= 1.0) / n
    err_mc = abs(pi_mc - math.pi)

    # Sobol QMC
    u_sb = stocha.sobol(dim=2, n_samples=n)
    pi_sb = 4.0 * np.sum(u_sb[:, 0]**2 + u_sb[:, 1]**2 <= 1.0) / n
    err_sb = abs(pi_sb - math.pi)

    # Halton QMC
    u_ht = stocha.halton(dim=2, n_samples=n)
    pi_ht = 4.0 * np.sum(u_ht[:, 0]**2 + u_ht[:, 1]**2 <= 1.0) / n
    err_ht = abs(pi_ht - math.pi)

    print(f"{n:>10,}  {err_mc:>12.6f}  {err_sb:>12.6f}  {err_ht:>12.6f}")

print(f"\n真の π = {math.pi:.6f}")
print("QMC 列は一般に擬似乱数 MC より速く収束する。")

# --- 4. 高次元 Sobol ---
print("\n" + "=" * 55)
print("4. 高次元 Sobol（最大 1000 次元）")
print("=" * 55)

pts_hd = stocha.sobol(dim=100, n_samples=2048)
print(f"100 次元 Sobol 形状: {pts_hd.shape}")
print(f"次元ごとの平均の範囲: [{pts_hd.mean(axis=0).min():.4f}, "
      f"{pts_hd.mean(axis=0).max():.4f}]  (全て ~0.5 になるはず)")

# --- 5. スループットベンチマーク ---
print("\n" + "=" * 55)
print("5. スループットベンチマーク")
print("=" * 55)

print("\nSobol:")
for dim, n in [(2, 65536), (10, 65536), (50, 16384), (100, 8192)]:
    t0 = time.time()
    _ = stocha.sobol(dim=dim, n_samples=n)
    elapsed = time.time() - t0
    total = dim * n
    print(f"  dim={dim:>4}, n={n:>6,}: {elapsed*1000:6.1f}ms  "
          f"({total/elapsed/1e6:.1f}M 値/秒)")

print("\nHalton:")
for dim, n in [(2, 65536), (10, 65536), (30, 16384), (40, 8192)]:
    t0 = time.time()
    _ = stocha.halton(dim=dim, n_samples=n)
    elapsed = time.time() - t0
    total = dim * n
    print(f"  dim={dim:>4}, n={n:>6,}: {elapsed*1000:6.1f}ms  "
          f"({total/elapsed/1e6:.1f}M 値/秒)")

print("\n✅ 準乱数生成 完了")
