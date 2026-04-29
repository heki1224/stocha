"""
stocha チュートリアル 8: マルチアセット相関シミュレーション
==========================================================

Cholesky 分解による相関マルチアセット GBM、ポートフォリオリスク分析、
相関構造の検証を示す。

金融工学的背景:
    実務では資産間のリターンは相関を持つ。��ック株のポートフォリオは
    連動して動き、株式と債券はしばしば逆相関を示す。
    クロスアセットの相関を無視するとテールリスクを過小評価する
    — 2008年の金融危機では、市場ストレス���に相関が急上昇することが示された。

    Cholesky アプローチ:
        相関行列 Σ に対し Σ = LL^T と分解する。
        独立な Z ~ N(0, I_n) を生成し、ε = LZ で所望の相関構造を与える。
        各資産は以下で発展:
            S_i(t+dt) = S_i(t) * exp((μ_i - 0.5σ_i²)dt
                                      + σ_i * √dt * ε_i)
"""

import math
import time

import numpy as np
import stocha

# --- 1. マルチアセットシミュレーション ---
print("=" * 60)
print("1. 相関 3 資産 GBM シミュレーション")
print("=" * 60)

s0 = [100.0, 50.0, 200.0]
mu = [0.05, 0.08, 0.03]
sigma = [0.20, 0.30, 0.15]
corr = np.array([
    [1.00, 0.60, 0.30],
    [0.60, 1.00, 0.50],
    [0.30, 0.50, 1.00],
])
n_paths = 50_000

t0 = time.time()
paths = stocha.multi_gbm(
    s0=s0, mu=mu, sigma=sigma, corr=corr,
    t=1.0, steps=252, n_paths=n_paths, seed=42,
)
elapsed = time.time() - t0

print(f"形状: {paths.shape}  (パス数, ステップ+1, 資産数)")
print(f"生成時間: {elapsed*1000:.1f}ms")
print(f"全価格が正: {(paths > 0).all()}")
print(f"\n初期価格: {paths[0, 0, :]}")
print(f"終端平均:  {paths[:, -1, :].mean(axis=0)}")
for i in range(3):
    expected = s0[i] * math.exp(mu[i])
    actual = paths[:, -1, i].mean()
    print(f"  資産 {i}: E[S(T)]={expected:.2f}, シミュレーション={actual:.2f}, "
          f"相対誤差={abs(actual-expected)/expected:.4f}")

# --- 2. 相関構造の検証 ---
print("\n" + "=" * 60)
print("2. 相関構造の検証")
print("   （対数リターンのサンプル相関 vs 入力相関行列）")
print("=" * 60)

log_ret = np.log(paths[:, 1:, :] / paths[:, :-1, :])
flat = log_ret.reshape(-1, 3)
sample_corr = np.corrcoef(flat.T)

print(f"\n入力相関行列:")
for row in corr:
    print(f"  [{', '.join(f'{v:6.2f}' for v in row)}]")

print(f"\nサンプル相関（{flat.shape[0]:,} 個の対数リターンから）:")
for row in sample_corr:
    print(f"  [{', '.join(f'{v:6.4f}' for v in row)}]")

max_err = np.max(np.abs(sample_corr - corr))
print(f"\n最大絶対誤差: {max_err:.6f}")

# --- 3. ポートフォリオ VaR / CVaR ---
print("\n" + "=" * 60)
print("3. ポートフォリオ VaR / CVaR（等加重ポートフォリオ）")
print("=" * 60)

weights = np.array([1/3, 1/3, 1/3])
initial_value = (np.array(s0) * weights).sum()
terminal_values = (paths[:, -1, :] * weights).sum(axis=1)
portfolio_returns = terminal_values / initial_value - 1.0

print(f"初期ポートフォリオ価値: {initial_value:.2f}")
print(f"終端平均: {terminal_values.mean():.2f}")
print(f"ポートフォリオリターン: 平均={portfolio_returns.mean():.4f}, "
      f"標準偏差={portfolio_returns.std():.4f}")

print(f"\n{'信頼水準':>12}  {'VaR':>10}  {'CVaR':>10}")
print("-" * 38)
for conf in [0.90, 0.95, 0.99]:
    var, cvar = stocha.var_cvar(portfolio_returns, confidence=conf)
    print(f"{conf:>12.0%}  {var:>10.4f}  {cvar:>10.4f}")

# --- 4. 対称変量法（Antithetic Variates） ---
print("\n" + "=" * 60)
print("4. 対称変量法: 分散削減効果")
print("=" * 60)

kw = dict(s0=s0, mu=mu, sigma=sigma, corr=corr,
          t=1.0, steps=252, n_paths=n_paths, seed=42)

plain = stocha.multi_gbm(**kw, antithetic=False)
anti = stocha.multi_gbm(**kw, antithetic=True)

for i in range(3):
    std_plain = plain[:, -1, i].std()
    std_anti = anti[:, -1, i].std()
    reduction = (1 - std_anti / std_plain) * 100
    print(f"資産 {i}: 通常 std={std_plain:.4f}, 対称 std={std_anti:.4f}, "
          f"削減率={reduction:.1f}%")

port_plain = (plain[:, -1, :] * weights).sum(axis=1)
port_anti = (anti[:, -1, :] * weights).sum(axis=1)
reduction = (1 - port_anti.std() / port_plain.std()) * 100
print(f"ポートフォリオ: 通常 std={port_plain.std():.4f}, 対称 std={port_anti.std():.4f}, "
      f"削減率={reduction:.1f}%")

# --- 5. スループットベンチマーク ---
print("\n" + "=" * 60)
print("5. スループットベンチマーク")
print("=" * 60)

for n_assets, label in [(2, "2資産"), (3, "3資産"), (5, "5資産")]:
    c = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            c[i, j] = c[j, i] = 0.5
    t0 = time.time()
    _ = stocha.multi_gbm(
        s0=[100.0]*n_assets, mu=[0.05]*n_assets, sigma=[0.2]*n_assets,
        corr=c, t=1.0, steps=252, n_paths=100_000, seed=0,
    )
    elapsed = time.time() - t0
    total_steps = 100_000 * 252 * n_assets
    print(f"  {label}: {elapsed*1000:.0f}ms  ({total_steps/elapsed/1e6:.1f}M ステップ/秒)")

print("\n✅ マルチアセット相関シミュレーション 完了")
