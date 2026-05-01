"""
stocha チュートリアル 9: Heston 解析プライシングとキャリブレーション
====================================================================

COS 法による高速 Heston オプション価格計算と、
Levenberg-Marquardt 法による Heston 確率ボラティリティモデルの
キャリブレーションを実演します。

金融工学の背景:
    Heston (1993) モデル:
        dS = r*S*dt + sqrt(V)*S*dW_S
        dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_V
        corr(dW_S, dW_V) = rho

    Black-Scholes と異なり、Heston モデルは確率的分散を通じて
    ボラティリティスマイル/スキューを捉えます。半解析的な特性関数を持ち、
    COS 法（Fang & Oosterlee 2008）によりモンテカルロなしで
    高速にオプション価格を計算できます。

    キャリブレーションは (v0, kappa, theta, xi, rho) をマーケットの
    オプション価格にフィットさせます。射影 Levenberg-Marquardt 法を使用し、
    Vega 加重残差で最適化します。

内容:
    1. COS 法プライシング — 解析的 Heston コール/プット価格
    2. Heston 価格からのインプライドボラティリティスマイル
    3. キャリブレーション — 合成マーケットデータからのパラメータ復元
    4. マルチ満期キャリブレーション — ボラティリティ曲面の期間構造
"""

import math
import time

import numpy as np
import stocha


# ──────────────────────────────────────────────
# Heston モデルパラメータ（典型的な株式インデックス）
# ──────────────────────────────────────────────
S0    = 100.0
V0    = 0.04    # 初期分散（ボラティリティ 20%）
R     = 0.05    # 無リスク金利
KAPPA = 2.0     # 平均回帰速度
THETA = 0.04    # 長期分散
XI    = 0.3     # ボラティリティ・オブ・ボラティリティ
RHO   = -0.7    # 株価-分散の相関（負 = レバレッジ効果）
T     = 1.0     # 満期 1 年


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. COS 法プライシング
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("1. Heston COS 法 — 解析的オプションプライシング")
print("=" * 60)

strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)

t0 = time.time()
call_prices = stocha.heston_price(
    strikes=strikes, is_call=[True] * len(strikes),
    s0=S0, v0=V0, r=R, kappa=KAPPA, theta=THETA, xi=XI, rho=RHO, t=T,
)
put_prices = stocha.heston_price(
    strikes=strikes, is_call=[False] * len(strikes),
    s0=S0, v0=V0, r=R, kappa=KAPPA, theta=THETA, xi=XI, rho=RHO, t=T,
)
elapsed = time.time() - t0

print(f"\nパラメータ: S0={S0}, V0={V0}, r={R}, kappa={KAPPA}, theta={THETA}")
print(f"            xi={XI}, rho={RHO}, T={T}")
print(f"計算時間: {elapsed*1000:.1f} ミリ秒 ({len(strikes)} ストライク × 2)")
print(f"\n{'ストライク':>10} {'コール':>10} {'プット':>10} {'パリティ誤差':>14}")
print("-" * 48)
for i, k in enumerate(strikes):
    parity_err = call_prices[i] - put_prices[i] - S0 + k * math.exp(-R * T)
    print(f"{k:10.0f} {call_prices[i]:10.4f} {put_prices[i]:10.4f} {parity_err:14.2e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. インプライドボラティリティスマイル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("2. Heston インプライドボラティリティスマイル")
print("=" * 60)


def bs_iv_newton(price, s, k, r, t, is_call, tol=1e-10):
    """Newton 法で BS 価格からインプライドボラティリティを逆算"""
    def _ncdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    sigma = 0.2
    for _ in range(100):
        d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        if is_call:
            bs = s * _ncdf(d1) - k * math.exp(-r * t) * _ncdf(d2)
        else:
            bs = k * math.exp(-r * t) * _ncdf(-d2) - s * _ncdf(-d1)
        vega = s * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) * math.sqrt(t)
        if vega < 1e-20:
            break
        diff = bs - price
        if abs(diff) < tol:
            break
        sigma -= diff / vega
        sigma = max(sigma, 0.001)
    return sigma


print(f"\n{'ストライク':>10} {'コール価格':>12} {'IV':>12}")
print("-" * 38)
for i, k in enumerate(strikes):
    iv = bs_iv_newton(call_prices[i], S0, k, R, T, True)
    print(f"{k:10.0f} {call_prices[i]:12.4f} {iv:12.4f}")

print("\n注: 負の rho (-0.7) により右下がりのスキューが発生します")
print("   （低ストライクほど IV が高い — レバレッジ効果）。")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. キャリブレーション — 単一満期
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("3. Heston キャリブレーション — 単一満期")
print("=" * 60)

# 既知のパラメータで合成「マーケット」価格を生成
true_params = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
calib_strikes = np.array([85, 90, 95, 100, 105, 110, 115], dtype=float)
market_prices = stocha.heston_price(
    strikes=calib_strikes, is_call=[True] * 7,
    s0=S0, r=R, t=T, n_cos=256, **true_params,
)

t0 = time.time()
result = stocha.heston_calibrate(
    strikes=calib_strikes,
    maturities=np.full(7, T),
    market_prices=market_prices,
    is_call=[True] * 7,
    s0=S0, r=R,
)
calib_time = time.time() - t0

print(f"\nキャリブレーション時間: {calib_time*1000:.0f} ミリ秒")
print(f"収束: {result['converged']}, 反復回数: {result['iterations']}")
print(f"RMSE: {result['rmse']:.2e}")
print(f"Feller 条件 (2κθ > ξ²): {result['feller_satisfied']}")
print(f"\n{'パラメータ':>10} {'真値':>10} {'キャリブ値':>12} {'誤差':>10}")
print("-" * 46)
for p in ['v0', 'kappa', 'theta', 'xi', 'rho']:
    err = abs(result[p] - true_params[p])
    print(f"{p:>10} {true_params[p]:10.4f} {result[p]:12.6f} {err:10.2e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. マルチ満期キャリブレーション
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("4. Heston キャリブレーション — マルチ満期（ボラティリティ曲面）")
print("=" * 60)

true2 = dict(v0=0.06, kappa=1.5, theta=0.06, xi=0.5, rho=-0.8)
maturities_set = [0.25, 0.5, 1.0]
strikes_per_mat = [90.0, 95.0, 100.0, 105.0, 110.0]

all_strikes = []
all_mats = []
all_prices = []
all_calls = []

for tau in maturities_set:
    ks = np.array(strikes_per_mat, dtype=float)
    ps = stocha.heston_price(
        strikes=ks, is_call=[True] * len(ks),
        s0=S0, r=R, t=tau, n_cos=256, **true2,
    )
    all_strikes.extend(strikes_per_mat)
    all_mats.extend([tau] * len(ks))
    all_prices.extend(ps.tolist())
    all_calls.extend([True] * len(ks))

t0 = time.time()
result2 = stocha.heston_calibrate(
    strikes=np.array(all_strikes),
    maturities=np.array(all_mats),
    market_prices=np.array(all_prices),
    is_call=all_calls,
    s0=S0, r=R, max_iter=300,
)
calib_time2 = time.time() - t0

print(f"\nデータ: {len(all_strikes)} オプション（満期 {maturities_set}）")
print(f"キャリブレーション時間: {calib_time2*1000:.0f} ミリ秒")
print(f"収束: {result2['converged']}, 反復回数: {result2['iterations']}")
print(f"RMSE: {result2['rmse']:.2e}")
print(f"Feller 条件: {result2['feller_satisfied']}")
print(f"\n{'パラメータ':>10} {'真値':>10} {'キャリブ値':>12} {'誤差':>10}")
print("-" * 46)
for p in ['v0', 'kappa', 'theta', 'xi', 'rho']:
    err = abs(result2[p] - true2[p])
    print(f"{p:>10} {true2[p]:10.4f} {result2[p]:12.6f} {err:10.2e}")

print("\n" + "=" * 60)
print("チュートリアル完了。")
print("=" * 60)
