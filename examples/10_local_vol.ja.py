"""チュートリアル 10: SSVI 曲面 & Dupire 局所ボラティリティ

内容:
1. SSVI によるインプライド・ボラティリティ曲面の構築
2. SSVI パラメータのキャリブレーション
3. Dupire 局所ボラティリティ曲面の計算（解析的）
4. 連続配当利回り付き GBM シミュレーション
"""

import numpy as np

import stocha

# --- 1. SSVI インプライド・ボラティリティ曲面 ---
print("=" * 60)
print("1. SSVI インプライド・ボラティリティ曲面")
print("=" * 60)

# SSVI パラメータ（典型的な株価指数の値）
eta, gamma, rho = 1.0, 0.5, -0.4

# 複数の満期
maturities = [0.25, 0.5, 1.0, 2.0]
atm_vols = [0.22, 0.20, 0.18, 0.17]  # ATM ボラの期間構造
log_strikes = np.linspace(-0.3, 0.3, 7)

print(f"パラメータ: η={eta}, γ={gamma}, ρ={rho}")
print(f"対数マネーネス: {log_strikes}")
print()

for T, atm_vol in zip(maturities, atm_vols):
    theta = atm_vol**2 * T  # ATM トータル分散
    vols = stocha.ssvi_implied_vol(log_strikes, theta=theta, t=T, eta=eta, gamma=gamma, rho=rho)
    print(f"T={T:.2f}年 (θ={theta:.4f}): IV = [{', '.join(f'{v:.1%}' for v in vols)}]")

# --- 2. SSVI キャリブレーション ---
print()
print("=" * 60)
print("2. SSVI キャリブレーション（ラウンドトリップ検証）")
print("=" * 60)

# 既知パラメータから合成市場データを生成
true_eta, true_gamma, true_rho = 1.2, 0.4, -0.35
thetas = np.array([0.01, 0.02, 0.04, 0.06, 0.09])
ks = np.linspace(-0.3, 0.3, 9)

log_m, theta_v, market_w = [], [], []
for th in thetas:
    for k in ks:
        log_m.append(k)
        theta_v.append(th)

log_m = np.array(log_m)
theta_v = np.array(theta_v)
market_w = np.array([
    stocha.ssvi_implied_vol(np.array([k]), theta=th, t=1.0, eta=true_eta, gamma=true_gamma, rho=true_rho)[0]**2
    for k, th in zip(log_m, theta_v)
])

result = stocha.ssvi_calibrate(log_m, theta_v, market_w, max_iter=200, tol=1e-12)
print(f"真のパラメータ:     η={true_eta}, γ={true_gamma}, ρ={true_rho}")
print(f"キャリブレーション: η={result['eta']:.4f}, γ={result['gamma']:.4f}, ρ={result['rho']:.4f}")
print(f"RMSE={result['rmse']:.2e}, 収束={result['converged']}, 反復={result['iterations']}回")

# --- 3. Dupire 局所ボラティリティ曲面 ---
print()
print("=" * 60)
print("3. Dupire 局所ボラティリティ曲面（SSVI 解析微分）")
print("=" * 60)

theta_values = np.array([0.01, 0.02, 0.04, 0.06, 0.09])
t_values = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
k_grid = np.linspace(-0.2, 0.2, 5)

lv = stocha.ssvi_local_vol(k_grid, theta_values, t_values, eta=1.0, gamma=0.5, rho=-0.4)
print(f"局所ボラ曲面サイズ: {lv.shape}（スライス × ストライク）")
print(f"ATM 局所ボラ（満期別）:")
atm_idx = len(k_grid) // 2
for i, T in enumerate(t_values):
    print(f"  T={T:.2f}年: σ_loc = {lv[i, atm_idx]:.2%}")

# --- 4. 連続配当付き GBM ---
print()
print("=" * 60)
print("4. 連続配当利回り付き GBM シミュレーション")
print("=" * 60)

s0, mu, sigma, q = 100.0, 0.08, 0.2, 0.03
paths = stocha.gbm(s0=s0, mu=mu, sigma=sigma, t=1.0, steps=252, n_paths=50000, seed=42, q=q)

expected = s0 * np.exp((mu - q) * 1.0)
realized = paths[:, -1].mean()
print(f"S0={s0}, μ={mu}, σ={sigma}, q={q}")
print(f"E[S(1)] 理論値   = {expected:.2f}")
print(f"E[S(1)] シミュ値 = {realized:.2f}")
print(f"相対誤差          = {abs(realized - expected) / expected:.4%}")
