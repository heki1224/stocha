"""
stocha チュートリアル 5: リスク指標とコピュラ
=============================================

Value-at-Risk（VaR）、Conditional VaR（CVaR / 期待ショートフォール）、
およびコピュラによる多変量依存構造のモデリングを示す。

金融工学的背景:
    VaR と CVaR:
        VaR(alpha) = 信頼水準 alpha でのパーセンタイル損失。
        CVaR(alpha) = E[損失 | 損失 > VaR(alpha)] — 最悪 (1-alpha) 分位の
        平均損失。CVaR はコヒーレント（凸・劣加法的）であり、
        Basel III / FRTB において推奨されるリスク指標。

    コピュラ:
        コピュラは周辺分布と依存構造を分離してモデル化する。
        ガウスコピュラは線形相関を表現するがテール依存性がない
        — 極端な損失が同時に起きる確率は独立の場合と変わらない。
        Student-t コピュラはテール依存性（lambda_L = lambda_U > 0）を持ち、
        「資産が同時に暴落する」という経験則を表現できる。
        2008年金融危機では、CDO トランシェにガウスコピュラを使った
        危険性が露呈した（"ウォール街を殺した数式"）。
"""

import math
import time

import numpy as np
import stocha

# --- 1. ポートフォリオリターンのシミュレーション ---
print("=" * 60)
print("1. ポートフォリオリターンのシミュレーション（2資産 GBM）")
print("=" * 60)

S0     = 100.0
r      = 0.05
sigma1 = 0.20   # 資産1のボラティリティ
sigma2 = 0.30   # 資産2のボラティリティ
T      = 1.0 / 252  # 1営業日ホライズン
steps  = 1
n_paths = 100_000

paths1 = stocha.gbm(s0=S0, mu=r, sigma=sigma1, t=T, steps=steps, n_paths=n_paths, seed=1)
paths2 = stocha.gbm(s0=S0, mu=r, sigma=sigma2, t=T, steps=steps, n_paths=n_paths, seed=2)

ret1 = paths1[:, -1] / paths1[:, 0] - 1.0
ret2 = paths2[:, -1] / paths2[:, 0] - 1.0

# 等加重ポートフォリオ
port_returns = 0.5 * ret1 + 0.5 * ret2

print(f"n_paths={n_paths:,}, ホライズン=1営業日")
print(f"資産1: 平均={ret1.mean():.6f}, 標準偏差={ret1.std():.4f}")
print(f"資産2: 平均={ret2.mean():.6f}, 標準偏差={ret2.std():.4f}")
print(f"ポートフォリオ: 平均={port_returns.mean():.6f}, 標準偏差={port_returns.std():.4f}")

# --- 2. VaR と CVaR ---
print("\n" + "=" * 60)
print("2. Value-at-Risk (VaR) と Conditional VaR (CVaR)")
print("   損失は正値。両指標とも損失の大きさを表す。")
print("=" * 60)

print(f"\n{'信頼水準':>12}  {'VaR':>10}  {'CVaR':>10}  {'VaR/CVaR 比':>14}")
print("-" * 54)
for conf in [0.90, 0.95, 0.99, 0.999]:
    var, cvar = stocha.var_cvar(port_returns, confidence=conf)
    print(f"{conf:>12.1%}  {var:>10.6f}  {cvar:>10.6f}  {var/cvar:>14.4f}")

# 正規近似との比較
port_std = port_returns.std()

def norm_ppf(p: float) -> float:
    """正規分布の逆 CDF（有理近似）"""
    if p < 0.5:
        return -norm_ppf(1 - p)
    t = math.sqrt(-2 * math.log(1 - p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1]*t + c[2]*t**2) / (1 + d[0]*t + d[1]*t**2 + d[2]*t**3)

print(f"\n正規近似（標準偏差={port_std:.4f}）との比較:")
for conf in [0.95, 0.99]:
    var_norm = -port_returns.mean() + port_std * norm_ppf(conf)
    var_sim, cvar_sim = stocha.var_cvar(port_returns, confidence=conf)
    print(f"  {conf:.0%}: 正規 VaR={var_norm:.6f}, シミュレーション VaR={var_sim:.6f}")

# --- 3. ガウスコピュラ ---
print("\n" + "=" * 60)
print("3. ガウスコピュラのサンプリング")
print("=" * 60)

corr = np.array([[1.0, 0.8],
                 [0.8, 1.0]])
n_cop = 100_000

t0 = time.time()
u_gauss = stocha.gaussian_copula(corr, n_samples=n_cop, seed=42)
elapsed = time.time() - t0

print(f"相関行列 rho=0.8, n_samples={n_cop:,}")
print(f"生成時間: {elapsed*1000:.1f}ms")
print(f"出力形状: {u_gauss.shape}, 値域=[{u_gauss.min():.4f}, {u_gauss.max():.4f}]")

# Spearman 順位相関（sin(pi/2 * rho) に近いはず）
from_rank = np.corrcoef(u_gauss[:, 0].argsort().argsort(),
                        u_gauss[:, 1].argsort().argsort())[0, 1]
spearman_theory = (6 / math.pi) * math.asin(0.8 / 2)
print(f"Spearman 順位相関: シミュレーション={from_rank:.4f}, "
      f"理論値≈{spearman_theory:.4f}")

# --- 4. Student-t コピュラ ---
print("\n" + "=" * 60)
print("4. Student-t コピュラのサンプリング（ヘビーテール・テール依存性）")
print("=" * 60)

nu = 4.0  # 自由度（小さいほどテールが厚い）

t0 = time.time()
u_t = stocha.student_t_copula(corr, nu=nu, n_samples=n_cop, seed=42)
elapsed = time.time() - t0

print(f"相関行列 rho=0.8, nu={nu}, n_samples={n_cop:,}")
print(f"生成時間: {elapsed*1000:.1f}ms")
print(f"出力形状: {u_t.shape}, 値域=[{u_t.min():.4f}, {u_t.max():.4f}]")

# --- 5. テール依存性の比較: ガウス vs Student-t ---
print("\n" + "=" * 60)
print("5. テール依存性比較: ガウスコピュラ vs Student-t コピュラ")
print("   （両変数が 99 パーセンタイルを超える同時確率）")
print("=" * 60)

for threshold in [0.95, 0.99, 0.999]:
    joint_gauss = np.mean((u_gauss[:, 0] > threshold) & (u_gauss[:, 1] > threshold))
    joint_t     = np.mean((u_t[:, 0] > threshold)     & (u_t[:, 1] > threshold))
    independent  = (1 - threshold)**2
    print(f"  P(U1>{threshold:.1%}, U2>{threshold:.1%}): "
          f"ガウス={joint_gauss:.6f}, "
          f"Student-t={joint_t:.6f}, "
          f"独立={independent:.6f}")

print("\n（Student-t コピュラはガウスより多くの同時極端値が発生 → テール依存性）")

# --- 6. スループットベンチマーク ---
print("\n" + "=" * 60)
print("6. スループットベンチマーク")
print("=" * 60)

for name, fn in [
    ("ガウスコピュラ（2次元）", lambda n: stocha.gaussian_copula(corr, n_samples=n, seed=0)),
    ("Student-t コピュラ（2次元）", lambda n: stocha.student_t_copula(corr, nu=nu, n_samples=n, seed=0)),
]:
    print(f"\n{name}:")
    for n in [10_000, 100_000, 500_000]:
        t0 = time.time()
        _ = fn(n)
        elapsed = time.time() - t0
        print(f"  n={n:>8,}: {elapsed*1000:6.1f}ms  ({n/elapsed/1e6:.2f}M サンプル/秒)")

print("\n✅ リスク指標とコピュラ 完了")
