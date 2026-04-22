"""
stocha チュートリアル 7: LSMC によるアメリカンオプション定価
============================================================

最小二乗モンテカルロ（Longstaff-Schwartz LSMC）を使った
アメリカンオプションの定価を示す。

金融工学的背景:
    欧州オプションは満期のみ行使可能。アメリカンオプションは
    任意の時点で早期行使が可能で、より高い価値を持つ（早期行使プレミアム）。
    ただしほとんどのモデルで解析解は存在しない。

    Longstaff-Schwartz (2001) LSMC アルゴリズム:
        1. N 本のリスク中立 GBM パスをシミュレート。
        2. 各タイムステップで（後ろ向き帰納法）、割引済み将来キャッシュフローを
           現在価格の基底関数（多項式）に回帰して継続価値を推定。
        3. 本源的価値 > 継続価値推定値 なら早期行使。
        4. 価格 = 最適行使キャッシュフローの割引期待値。

    stocha は次数 1〜4 の多項式基底と faer ライブラリの QR 分解で
    最小二乗回帰を実装。

    早期行使プレミアム = アメリカン価格 − 欧州 Black-Scholes 価格。
    プット: 深くイン・ザ・マネーになると、行使価格分の金利収入が
    待機の保険価値を上回り、早期行使が最適になる。
"""

import math
import time

import numpy as np
import stocha

# オプションパラメータ
S0    = 100.0   # 初期株価
K     = 100.0   # 行使価格（ATM）
r     = 0.05    # 無リスク金利
sigma = 0.20    # ボラティリティ
T     = 1.0     # 満期: 1年
steps = 50      # 行使機会の数

# --- 1. LSMC アメリカンプット定価 ---
print("=" * 60)
print("1. LSMC アメリカンプット・オプション定価")
print("=" * 60)

n_paths = 100_000
t0 = time.time()
price, stderr = stocha.lsmc_american_option(
    s0=S0, k=K, r=r, sigma=sigma, t=T,
    steps=steps, n_paths=n_paths, is_put=True, poly_degree=3, seed=42,
)
elapsed = time.time() - t0

print(f"S0={S0}, K={K}, r={r:.0%}, sigma={sigma:.0%}, T={T}年")
print(f"n_paths={n_paths:,}, steps={steps}, poly_degree=3")
print(f"アメリカンプット: {price:.4f} ± {stderr:.4f}  (95% CI: ±{1.96*stderr:.4f})")
print(f"定価時間: {elapsed*1000:.1f}ms")

# --- 2. 欧州プット（Black-Scholes）との比較 ---
print("\n" + "=" * 60)
print("2. 欧州プット（Black-Scholes）vs アメリカンプット")
print("=" * 60)

d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

bs_put = K * math.exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)

early_exercise_premium = price - bs_put

print(f"欧州プット（Black-Scholes）: {bs_put:.4f}")
print(f"アメリカンプット（LSMC）:    {price:.4f} ± {stderr:.4f}")
print(f"早期行使プレミアム:           {early_exercise_premium:.4f}  "
      f"（欧州価格比 {early_exercise_premium/bs_put*100:.2f}%）")
print("\nアメリカン ≥ 欧州（裁定なし条件）; 正のプレミアムで正確性を確認。")

# --- 3. アメリカンコール vs 欧州コール ---
print("\n" + "=" * 60)
print("3. アメリカンコール（無配当 — 早期行使は最適でない）")
print("=" * 60)

price_call, stderr_call = stocha.lsmc_american_option(
    s0=S0, k=K, r=r, sigma=sigma, t=T,
    steps=steps, n_paths=n_paths, is_put=False, poly_degree=3, seed=42,
)

bs_call = S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
eep_call = price_call - bs_call

print(f"欧州コール（Black-Scholes）:  {bs_call:.4f}")
print(f"アメリカンコール（LSMC）:     {price_call:.4f} ± {stderr_call:.4f}")
print(f"早期行使プレミアム:            {eep_call:.4f}  （~0 のはず）")
print("（無配当コールの早期行使は常に非最適 — Merton 1973）")

# --- 4. 多項式次数の感度分析 ---
print("\n" + "=" * 60)
print("4. 多項式次数の感度分析")
print("   （次数が高いほど継続価値の近似が豊か）")
print("=" * 60)

print(f"\n{'次数':>8}  {'価格':>10}  {'標準誤差':>10}  {'vs BS':>10}")
print("-" * 44)
for deg in [1, 2, 3, 4]:
    p, se = stocha.lsmc_american_option(
        s0=S0, k=K, r=r, sigma=sigma, t=T,
        steps=steps, n_paths=50_000, is_put=True,
        poly_degree=deg, seed=42,
    )
    print(f"{deg:>8}  {p:>10.4f}  {se:>10.4f}  {p-bs_put:>+10.4f}")
print(f"{'（BS参照）':>8}  {bs_put:>10.4f}")

# --- 5. マネーネス別の感度分析 ---
print("\n" + "=" * 60)
print("5. マネーネス別 アメリカンプット価格")
print("   （深くイン・ザ・マネーほど早期行使プレミアムが大きい）")
print("=" * 60)

print(f"\n{'S0':>8}  {'S/K':>6}  {'欧州 BS':>10}  {'LSMC':>10}  {'EEP':>10}")
print("-" * 50)
for s0_test in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]:
    d1_t = (math.log(s0_test/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2_t = d1_t - sigma * math.sqrt(T)
    bs_t = K * math.exp(-r*T) * norm_cdf(-d2_t) - s0_test * norm_cdf(-d1_t)

    p_t, _ = stocha.lsmc_american_option(
        s0=s0_test, k=K, r=r, sigma=sigma, t=T,
        steps=steps, n_paths=50_000, is_put=True, poly_degree=3, seed=42,
    )
    eep_t = p_t - bs_t
    print(f"{s0_test:>8.0f}  {s0_test/K:>6.2f}  {bs_t:>10.4f}  {p_t:>10.4f}  {eep_t:>+10.4f}")

print("\n（深くイン・ザ・マネーのプットほど EEP が大きい — 行使価格の金利収入が大きいため）")

# --- 6. スループットベンチマーク ---
print("\n" + "=" * 60)
print("6. スループットベンチマーク")
print("=" * 60)

for n in [10_000, 50_000, 100_000]:
    t0 = time.time()
    _ = stocha.lsmc_american_option(
        s0=S0, k=K, r=r, sigma=sigma, t=T,
        steps=steps, n_paths=n, is_put=True, poly_degree=3, seed=42,
    )
    elapsed = time.time() - t0
    print(f"  n_paths={n:>8,}: {elapsed*1000:6.1f}ms  ({n/elapsed:>10,.0f} パス/秒)")

print("\n✅ アメリカンオプション定価（LSMC）完了")
