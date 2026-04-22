"""
stocha チュートリアル 2: GBM による株価シミュレーション
=======================================================

幾何ブラウン運動 (GBM) を使った株価パス生成、
モンテカルロ法によるヨーロッパン・コールオプション価格計算、
対称変量法による分散低減を示します。

対応する金融理論:
    - Black-Scholes モデル
    - リスク中立評価 (risk-neutral pricing)
    - モンテカルロ法による期待値計算
"""

import math
import time

import numpy as np
import stocha

# --- 1. GBM パスの生成 ---
print("=" * 55)
print("1. GBM パスの生成")
print("=" * 55)

S0 = 100.0    # 初期株価
mu = 0.05     # ドリフト（年率 5%）
sigma = 0.20  # ボラティリティ（年率 20%）
T = 1.0       # 満期 1 年
steps = 252   # 252 営業日
n_paths = 10_000

t0 = time.time()
paths = stocha.gbm(s0=S0, mu=mu, sigma=sigma, t=T, steps=steps, n_paths=n_paths, seed=42)
elapsed = time.time() - t0

print(f"パス数: {n_paths:,}, ステップ数: {steps}")
print(f"配列 shape: {paths.shape}")
print(f"生成時間: {elapsed*1000:.1f}ms")
print(f"初期価格（平均）: {paths[:, 0].mean():.4f}")
print(f"満期価格: 平均={paths[:, -1].mean():.4f}, 標準偏差={paths[:, -1].std():.4f}")

# --- 2. 理論値との比較 ---
print("\n" + "=" * 55)
print("2. 理論値との比較: E[S(T)] = S0 * exp(mu * T)")
print("=" * 55)

expected_mean = S0 * math.exp(mu * T)
simulated_mean = paths[:, -1].mean()
rel_err = abs(simulated_mean - expected_mean) / expected_mean

print(f"理論値 E[S(T)]  = {expected_mean:.4f}")
print(f"シミュレーション = {simulated_mean:.4f}")
print(f"相対誤差         = {rel_err:.4f} ({rel_err*100:.2f}%)")

# --- 3. ヨーロッパン・コールオプション価格（モンテカルロ法） ---
print("\n" + "=" * 55)
print("3. ヨーロッパン・コール価格（モンテカルロ法）")
print("=" * 55)

K = 105.0   # 行使価格
r = 0.02    # 無リスク金利（年率）

# リスク中立測度（ドリフトを無リスク金利に設定）でパス生成
paths_rn = stocha.gbm(s0=S0, mu=r, sigma=sigma, t=T, steps=steps, n_paths=100_000, seed=0)

# コールオプションのペイオフ: max(S_T - K, 0)
payoffs = np.maximum(paths_rn[:, -1] - K, 0.0)
mc_price = math.exp(-r * T) * payoffs.mean()
mc_stderr = payoffs.std() / math.sqrt(len(payoffs))

# Black-Scholes 解析解
d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

bs_price = S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

print(f"行使価格 K = {K}, 無リスク金利 r = {r}")
print(f"モンテカルロ価格: {mc_price:.4f} ± {mc_stderr:.4f}  (95% CI: ±{1.96*mc_stderr:.4f})")
print(f"Black-Scholes   : {bs_price:.4f}")
print(f"差              : {abs(mc_price - bs_price):.4f}")

# --- 4. 対称変量法による分散低減 ---
print("\n" + "=" * 55)
print("4. 対称変量法（Antithetic Variates）による分散低減")
print("=" * 55)

n_compare = 10_000

paths_std = stocha.gbm(s0=S0, mu=r, sigma=sigma, t=T, steps=steps, n_paths=n_compare, seed=0)
payoffs_std = np.maximum(paths_std[:, -1] - K, 0.0)
price_std = math.exp(-r * T) * payoffs_std.mean()
stderr_std = payoffs_std.std() / math.sqrt(n_compare)

paths_anti = stocha.gbm(
    s0=S0, mu=r, sigma=sigma, t=T, steps=steps,
    n_paths=n_compare, seed=0, antithetic=True
)
payoffs_anti = np.maximum(paths_anti[:, -1] - K, 0.0)
price_anti = math.exp(-r * T) * payoffs_anti.mean()
stderr_anti = payoffs_anti.std() / math.sqrt(n_compare)

print(f"標準モンテカルロ: 価格={price_std:.4f}, 標準誤差={stderr_std:.4f}")
print(f"対称変量法      : 価格={price_anti:.4f}, 標準誤差={stderr_anti:.4f}")
print(f"Black-Scholes   : {bs_price:.4f}")

# --- 5. スループット計測 ---
print("\n" + "=" * 55)
print("5. スループット計測")
print("=" * 55)

for n in [1_000, 10_000, 100_000]:
    t0 = time.time()
    _ = stocha.gbm(s0=S0, mu=mu, sigma=sigma, t=T, steps=252, n_paths=n, seed=42)
    elapsed = time.time() - t0
    rate = n / elapsed
    print(f"n_paths={n:>8,}: {elapsed*1000:6.1f}ms  ({rate:>10,.0f} パス/秒)")

print("\n✅ GBM シミュレーション完了")
