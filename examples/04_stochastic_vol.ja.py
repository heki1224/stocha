"""
stocha チュートリアル 4: 確率ボラティリティとジャンプ拡散モデル
=================================================================

Heston 確率ボラティリティモデルと Merton ジャンプ拡散モデルを示す。
いずれも標準的な GBM（幾何ブラウン運動）の拡張。

金融工学的背景:
    - Black-Scholes はボラティリティを定数と仮定するが、現実の市場では
      ボラティリティは時変かつ確率的（ボラティリティ・クラスタリング、
      ボラティリティ・スマイル）。
    - Heston (1993): 分散 v(t) が平均回帰する確率過程（CIR）に従う。
      パラメータ: kappa（回帰速度）、theta（長期分散）、xi（ボル・オブ・ボル）、
      rho（資産価格とボラティリティの相関。通常負）。
    - Merton (1976): GBM に複合ポアソン過程（ジャンプ）を追加。
      決算発表・マクロショックなど突発的な価格変動を表現。対数正規分布より
      ファット・テールになる。

stocha の実装:
    - Heston: Full Truncation（FT）スキームによるオイラー・丸山法
      （ステップ間で分散が負になっても計算エラーを防ぐ）
    - Merton: 対数正規ジャンプサイズ＋ドリフト補正項
      （リスク中立マルチンゲール条件 E[S(T)] = S0 * exp(mu * T) を保持）
"""

import math
import time

import numpy as np
import stocha

S0    = 100.0   # 初期株価
mu    = 0.05    # ドリフト（リスク中立測度下では無リスク金利）
sigma = 0.20    # GBM の基準ボラティリティ
T     = 1.0     # 満期: 1年
steps = 252     # 日次ステップ
K     = 100.0   # 行使価格（ATM）
r     = mu      # 無リスク金利

# --- 1. Heston パス生成 ---
print("=" * 60)
print("1. Heston 確率ボラティリティ — パス生成")
print("=" * 60)

v0    = 0.04   # 初期分散（初期ボラティリティ = 20%）
kappa = 2.0    # 平均回帰速度
theta = 0.04   # 長期分散（長期ボラティリティ = 20%）
xi    = 0.3    # ボル・オブ・ボル
rho   = -0.7   # 資産価格とボラティリティの相関（負 → レバレッジ効果）
n_paths = 50_000

t0 = time.time()
paths = stocha.heston(
    s0=S0, v0=v0, mu=mu,
    kappa=kappa, theta=theta, xi=xi, rho=rho,
    t=T, steps=steps, n_paths=n_paths, seed=42,
)
elapsed = time.time() - t0

print(f"n_paths={n_paths:,}, steps={steps}")
print(f"出力形状: {paths.shape}")
print(f"生成時間: {elapsed*1000:.1f}ms")
print(f"満期株価: 平均={paths[:, -1].mean():.4f}, 標準偏差={paths[:, -1].std():.4f}")

feller = 2 * kappa * theta / xi**2
print(f"Feller 条件: 2*kappa*theta/xi^2 = {feller:.2f}  "
      f"({'満たされている' if feller > 1 else '満たされていない — 分散が 0 に触れる可能性あり'})")

# --- 2. Heston による欧州コール定価 ---
print("\n" + "=" * 60)
print("2. Heston 欧州コールオプション定価（モンテカルロ）")
print("=" * 60)

payoffs = np.maximum(paths[:, -1] - K, 0.0)
mc_price = math.exp(-r * T) * payoffs.mean()
mc_stderr = payoffs.std() / math.sqrt(n_paths)

# GBM（定数ボラティリティ）の Black-Scholes 参照値
d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

bs_price = S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

print(f"行使価格 K={K}, 無リスク金利 r={r}")
print(f"Heston MC:        {mc_price:.4f} ± {mc_stderr:.4f}")
print(f"Black-Scholes:    {bs_price:.4f}  (定数ボラティリティ={sigma:.0%})")
print(f"差:               {abs(mc_price - bs_price):.4f}")
print("（Heston 価格は rho<0 によるボラティリティ・スキューの影響で異なる）")

# --- 3. Merton ジャンプ拡散 パス生成 ---
print("\n" + "=" * 60)
print("3. Merton ジャンプ拡散 — パス生成")
print("=" * 60)

lambda_ = 1.0    # 平均ジャンプ回数: 年 1 回
mu_j    = -0.05  # 対数ジャンプサイズの平均（負 → 下落方向のジャンプ）
sigma_j = 0.10   # 対数ジャンプサイズの標準偏差

t0 = time.time()
paths_jump = stocha.merton_jump_diffusion(
    s0=S0, mu=mu, sigma=sigma,
    lambda_=lambda_, mu_j=mu_j, sigma_j=sigma_j,
    t=T, steps=steps, n_paths=n_paths, seed=42,
)
elapsed = time.time() - t0

print(f"n_paths={n_paths:,}, lambda={lambda_}, mu_j={mu_j}, sigma_j={sigma_j}")
print(f"出力形状: {paths_jump.shape}")
print(f"生成時間: {elapsed*1000:.1f}ms")

# --- 4. テール分布の比較: GBM vs Merton ---
print("\n" + "=" * 60)
print("4. テール分布比較: GBM vs Merton ジャンプ拡散")
print("   （満期価格の対数リターン）")
print("=" * 60)

paths_gbm = stocha.gbm(
    s0=S0, mu=mu, sigma=sigma, t=T, steps=steps,
    n_paths=n_paths, seed=42,
)

log_ret_gbm  = np.log(paths_gbm[:, -1] / S0)
log_ret_jump = np.log(paths_jump[:, -1] / S0)

def skewness(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    return float(((x - m)**3).mean() / s**3)

def kurtosis(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    return float(((x - m)**4).mean() / s**4)

print(f"\n{'指標':<22} {'GBM':>12} {'Merton':>12}")
print("-" * 48)
print(f"{'平均対数リターン':<22} {log_ret_gbm.mean():>12.4f} {log_ret_jump.mean():>12.4f}")
print(f"{'標準偏差':<22} {log_ret_gbm.std():>12.4f} {log_ret_jump.std():>12.4f}")
print(f"{'歪度':<22} {skewness(log_ret_gbm):>12.4f} {skewness(log_ret_jump):>12.4f}")
print(f"{'超過尖度':<22} {kurtosis(log_ret_gbm)-3:>12.4f} {kurtosis(log_ret_jump)-3:>12.4f}")

for pct in [1, 5]:
    q_gbm  = float(np.percentile(log_ret_gbm, pct))
    q_jump = float(np.percentile(log_ret_jump, pct))
    print(f"{'VaR '+str(pct)+'%（対数リターン）':<22} {q_gbm:>12.4f} {q_jump:>12.4f}")

print("\n（Merton: 左テールが厚い → 超過尖度が大きく、歪度がより負）")

# --- 5. スループットベンチマーク ---
print("\n" + "=" * 60)
print("5. スループットベンチマーク")
print("=" * 60)

models = [
    ("Heston", lambda n: stocha.heston(
        s0=S0, v0=v0, mu=mu, kappa=kappa, theta=theta,
        xi=xi, rho=rho, t=T, steps=steps, n_paths=n, seed=0,
    )),
    ("Merton", lambda n: stocha.merton_jump_diffusion(
        s0=S0, mu=mu, sigma=sigma, lambda_=lambda_,
        mu_j=mu_j, sigma_j=sigma_j, t=T, steps=steps, n_paths=n, seed=0,
    )),
]

for name, fn in models:
    print(f"\n{name}:")
    for n in [1_000, 10_000, 50_000]:
        t0 = time.time()
        _ = fn(n)
        elapsed = time.time() - t0
        print(f"  n_paths={n:>7,}: {elapsed*1000:6.1f}ms  ({n/elapsed:>10,.0f} パス/秒)")

print("\n✅ 確率ボラティリティ・ジャンプ拡散モデル 完了")
