"""
stocha チュートリアル 6: 金利モデルとボラティリティ・スマイル
=============================================================

Hull-White 1ファクター短期金利モデルと、スワップション市場で使われる
SABR インプライドボラティリティモデルを示す。

金融工学的背景:
    Hull-White (1990):
        dr(t) = (theta - a*r(t))*dt + sigma*dW(t)
        平均回帰型のガウス短期金利モデル。長期平均 = theta/a。
        stocha はガウス条件分布を使った厳密シミュレーション
        （離散化誤差ゼロ）を実装。

    SABR (Hagan et al. 2002):
        dF = alpha * F^beta * dW
        dalpha = nu * alpha * dZ
        corr(dW, dZ) = rho
        SABR モデルはブラックインプライドボラティリティの解析近似式を与え、
        スワップション・キャップ/フロア市場で観察されるボラティリティ・
        スマイル（スキュー）を表現できる。
        Shifted SABR（shift > 0）はネガティブフォワードレートに対応。
"""

import math
import time

import numpy as np
import stocha

# ──────────────────────────────────────────────
# Hull-White パラメータ
# ──────────────────────────────────────────────
r0      = 0.05   # 初期短期金利（5%）
a       = 0.10   # 平均回帰速度
lr_mean = 0.05   # 長期平均金利（5%）
theta   = a * lr_mean  # Hull-White の theta = a * 長期平均 = 0.005
sigma_hw = 0.01  # 金利ボラティリティ（1%）
T       = 10.0   # シミュレーション期間: 10年
steps   = 120    # 月次ステップ（10年 × 12ヶ月）
n_paths = 50_000

# --- 1. Hull-White パス生成 ---
print("=" * 60)
print("1. Hull-White 短期金利モデル — パス生成")
print("=" * 60)

t0 = time.time()
rates = stocha.hull_white(
    r0=r0, a=a, theta=theta, sigma=sigma_hw,
    t=T, steps=steps, n_paths=n_paths, seed=42,
)
elapsed = time.time() - t0

print(f"n_paths={n_paths:,}, steps={steps}（{T:.0f}年間の月次）")
print(f"出力形状: {rates.shape}")
print(f"生成時間: {elapsed*1000:.1f}ms")
print(f"初期金利（列 0）: 平均={rates[:, 0].mean():.4f}  (= r0={r0})")
print(f"満期金利:         平均={rates[:, -1].mean():.4f}")

# --- 2. 理論値 vs シミュレーション平均 ---
print("\n" + "=" * 60)
print("2. 理論値 vs シミュレーション E[r(t)]")
print("   E[r(t)] = theta/a + (r0 - theta/a) * exp(-a*t)")
print("=" * 60)

times = np.linspace(0, T, steps + 1)
print(f"\n{'t（年）':>10}  {'理論 E[r]':>14}  {'シミュ平均':>14}  {'誤差':>10}")
print("-" * 54)
for i in [0, 12, 24, 60, 120]:  # 0, 1, 2, 5, 10 年
    t_val = times[i]
    theory = lr_mean + (r0 - lr_mean) * math.exp(-a * t_val)
    simulated = float(rates[:, i].mean())
    err = abs(simulated - theory)
    print(f"{t_val:>10.1f}  {theory:>14.6f}  {simulated:>14.6f}  {err:>10.6f}")

print(f"\n長期平均（theta/a） = {lr_mean:.4f}")
print(f"金利の分散  t=0:   標準偏差={rates[:, 0].std():.4f}")
print(f"金利の分散  t=5y:  標準偏差={rates[:, 60].std():.4f}")
print(f"金利の分散  t=10y: 標準偏差={rates[:, 120].std():.4f}")

# --- 3. SABR ボラティリティ・スマイル ---
print("\n" + "=" * 60)
print("3. SABR ボラティリティ・スマイル")
print("   5年スワップションのストライク別インプライドボラティリティ")
print("=" * 60)

F     = 0.05   # ATM フォワード・スワップレート
T_sab = 5.0    # スワップション満期
alpha = 0.20   # 初期ボラティリティ（SABR）
beta  = 0.5    # CEV 指数（0.5 = 平方根過程、金利市場で一般的）
rho   = -0.30  # 負の相関（金利上昇 → ボラティリティ低下の傾向）
nu    = 0.40   # ボル・オブ・ボル

strikes = [0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.10]

print(f"\nフォワード F={F:.0%}, T={T_sab}年, alpha={alpha}, beta={beta}, "
      f"rho={rho}, nu={nu}")
print(f"\n{'ストライク K':>12}  {'マネーネス':>10}  {'SABR インプライドボル':>20}")
print("-" * 48)

for k in strikes:
    iv = stocha.sabr_implied_vol(f=F, k=k, t=T_sab,
                                  alpha=alpha, beta=beta, rho=rho, nu=nu)
    moneyness = k / F
    marker = " ← ATM" if abs(k - F) < 1e-10 else ""
    print(f"{k:>12.2%}  {moneyness:>10.2f}x  {iv:>20.4%}{marker}")

# --- 4. Shifted SABR（ネガティブレート対応）---
print("\n" + "=" * 60)
print("4. Shifted SABR — ネガティブレート対応")
print("   shift=2% でストライク -2% まで対応")
print("=" * 60)

shift = 0.02  # 2% シフト: 全金利を shift だけ上方シフト
F_neg = 0.00  # フォワードが 0%（超低金利環境）

strikes_neg = [-0.015, -0.01, -0.005, 0.00, 0.005, 0.01, 0.02, 0.03]

print(f"\nフォワード F={F_neg:.1%}（ゼロ金利近傍）, shift={shift:.0%}")
print(f"\n{'ストライク K':>12}  {'Shifted SABR インプライドボル':>30}")
print("-" * 46)

for k in strikes_neg:
    if k + shift <= 0:
        print(f"{k:>12.2%}  {'シフト下限以下':>30}")
        continue
    iv = stocha.sabr_implied_vol(f=F_neg, k=k, t=T_sab,
                                  alpha=alpha, beta=beta, rho=rho, nu=nu,
                                  shift=shift)
    print(f"{k:>12.2%}  {iv:>30.4%}")

# --- 5. スループットベンチマーク ---
print("\n" + "=" * 60)
print("5. スループットベンチマーク")
print("=" * 60)

print("\nHull-White:")
for n in [1_000, 10_000, 50_000]:
    t0 = time.time()
    _ = stocha.hull_white(r0=r0, a=a, theta=theta, sigma=sigma_hw,
                          t=T, steps=steps, n_paths=n, seed=0)
    elapsed = time.time() - t0
    print(f"  n_paths={n:>7,}: {elapsed*1000:6.1f}ms  ({n/elapsed:>10,.0f} パス/秒)")

print("\nSABR（スカラー出力）:")
n_eval = 10_000
t0 = time.time()
for _ in range(n_eval):
    stocha.sabr_implied_vol(f=F, k=F, t=T_sab, alpha=alpha, beta=beta, rho=rho, nu=nu)
elapsed = time.time() - t0
print(f"  {n_eval:,} 回評価: {elapsed*1000:.1f}ms  ({n_eval/elapsed:,.0f} 評価/秒)")

print("\n✅ 金利モデル・ボラティリティスマイル 完了")
