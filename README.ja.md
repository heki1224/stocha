# stocha

> Rust 製の高速乱数・金融シミュレーションライブラリ（Python 向け）

[![PyPI](https://img.shields.io/pypi/v/stocha)](https://pypi.org/project/stocha/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[English README](README.md)

## 特長

- **高速 PRNG**: PCG64DXSM（NumPy デフォルト実装）
- **準乱数列**: Sobol（Joe & Kuo 2008）・Halton 列
- **確率過程モデル**: GBM、マルチアセット相関 GBM、Heston、Merton Jump-Diffusion、Hull-White
- **リスク指標**: VaR・CVaR（Expected Shortfall）
- **コピュラ**: ガウスコピュラ・Student-t コピュラ（多変量依存構造モデリング）
- **ボラティリティ**: SABR インプライドボラティリティ（Hagan 2002、マイナス金利対応）
- **オプション価格付け**: Longstaff-Schwartz LSMC（アメリカンオプション）
- **グリークス**: バンピング有限差分（全モデル） + Pathwise IPA（GBM）
- **キャリブレーション**: SABR（射影 LM）・Heston（COS 法 + 射影 LM）
- **並列処理**: Rayon によるパス生成の並列化
- **完全再現性**: ブロック分割 RNG ストリームによりスレッド数に依存せず同一結果を保証

## インストール

**PyPI（近日公開予定）:**

```bash
pip install stocha
```

**ソースからビルド（現在はこちら）:**

```bash
git clone https://github.com/heki1224/stocha.git
cd stocha
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin
maturin develop --release
```

PyPI 公開後は `pip install stocha` のみで利用できるようになります。Rust コンパイラは不要です。

## クイックスタート

```python
import stocha
import numpy as np

# ── 乱数生成 ──────────────────────────────────────────────────────────────
rng = stocha.RNG(seed=42)
samples = rng.normal(size=10_000, loc=0.0, scale=1.0)

# ── GBM による株価シミュレーション ────────────────────────────────────────
paths = stocha.gbm(
    s0=100.0, mu=0.05, sigma=0.20,
    t=1.0, steps=252, n_paths=100_000, seed=42,
)
# paths.shape == (100_000, 253)

# ── マルチアセット相関 GBM ────────────────────────────────────────────────
corr = np.array([[1.0, 0.6, 0.3],
                 [0.6, 1.0, 0.5],
                 [0.3, 0.5, 1.0]])
multi_paths = stocha.multi_gbm(
    s0=[100.0, 50.0, 200.0], mu=[0.05, 0.08, 0.03],
    sigma=[0.2, 0.3, 0.15], corr=corr,
    t=1.0, steps=252, n_paths=10_000, seed=42,
)
# multi_paths.shape == (10_000, 253, 3)
# ポートフォリオ終端価値: (multi_paths[:, -1, :] * weights).sum(axis=1)

# ── 準乱数列（低差異列） ──────────────────────────────────────────────────
pts = stocha.sobol(dim=2, n_samples=1024)    # (1024, 2), 値域 [0, 1)
pts = stocha.halton(dim=2, n_samples=1024)   # (1024, 2), 値域 (0, 1)

# ── Heston 確率ボラティリティ ─────────────────────────────────────────────
paths_h = stocha.heston(
    s0=100.0, v0=0.04, mu=0.05,
    kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    t=1.0, steps=252, n_paths=10_000, seed=42,
)
# paths_h.shape == (10_000, 253)

# ── Merton ジャンプ拡散 ───────────────────────────────────────────────────
paths_m = stocha.merton_jump_diffusion(
    s0=100.0, mu=0.05, sigma=0.20,
    lambda_=1.0, mu_j=-0.05, sigma_j=0.10,
    t=1.0, steps=252, n_paths=10_000, seed=42,
)
# paths_m.shape == (10_000, 253)

# ── VaR / CVaR ────────────────────────────────────────────────────────────
returns = paths[:, -1] / paths[:, 0] - 1
var, cvar = stocha.var_cvar(returns, confidence=0.95)
print(f"95% VaR={var:.4f}  CVaR={cvar:.4f}")

# ── ガウスコピュラ ─────────────────────────────────────────────────────────
corr = np.array([[1.0, 0.8], [0.8, 1.0]])
u = stocha.gaussian_copula(corr, n_samples=10_000)
# u.shape == (10_000, 2),  値域 (0, 1)

# ── Student-t コピュラ（テール依存性あり） ────────────────────────────────
u_t = stocha.student_t_copula(corr, nu=5.0, n_samples=10_000)

# ── Hull-White 短期金利モデル ─────────────────────────────────────────────
rates = stocha.hull_white(
    r0=0.05, a=0.1, theta=0.005, sigma=0.01,
    t=1.0, steps=252, n_paths=10_000,
)
# rates.shape == (10_000, 253)

# ── SABR インプライドボラティリティ ──────────────────────────────────────
iv = stocha.sabr_implied_vol(
    f=0.05, k=0.05, t=1.0,
    alpha=0.20, beta=0.5, rho=-0.3, nu=0.4,
)
print(f"SABR ATM implied vol: {iv:.4f}")

# ── SABR キャリブレーション: 市場スマイルから (α, ρ, ν) を復元 ──────────
strikes = np.array([0.04, 0.045, 0.05, 0.055, 0.06])
market_vols = np.array([0.244, 0.218, 0.201, 0.190, 0.184])
fit = stocha.sabr_calibrate(strikes, market_vols, f=0.05, t=1.0, beta=0.5)
print(f"calibrated: alpha={fit['alpha']:.4f}, rho={fit['rho']:.4f}, nu={fit['nu']:.4f}")

# ── LSMC によるアメリカンオプション価格付け ───────────────────────────────
price, std_err = stocha.lsmc_american_option(
    s0=100.0, k=100.0, r=0.05, sigma=0.20,
    t=1.0, steps=50, n_paths=50_000,
)
print(f"アメリカンプット: {price:.4f} ± {std_err:.4f}")

# ── Heston 解析プライシング（COS 法） ────────────────────────────────────
prices = stocha.heston_price(
    strikes=np.array([90.0, 100.0, 110.0]),
    is_call=[True, True, True],
    s0=100.0, v0=0.04, r=0.05,
    kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, t=1.0,
)
print(f"Heston コール価格: {prices}")

# ── Heston キャリブレーション ─────────────────────────────────────────────
fit = stocha.heston_calibrate(
    strikes=np.array([90.0, 95.0, 100.0, 105.0, 110.0]),
    maturities=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    market_prices=prices_from_market,  # 観測価格
    is_call=[True]*5,
    s0=100.0, r=0.05,
)
print(f"v0={fit['v0']:.4f}, kappa={fit['kappa']:.2f}, rho={fit['rho']:.2f}")

# ── モンテカルロ・グリークス ──────────────────────────────────────────────
greeks = stocha.greeks_fd(
    model="gbm",
    params={"s0": 100.0, "r": 0.05, "sigma": 0.2, "t": 1.0},
    payoff="call", strike=100.0,
    n_paths=100_000, n_steps=252,
    greeks=["delta", "gamma", "vega", "theta", "rho"],
)
print(f"Delta={greeks['delta']:.4f}  Gamma={greeks['gamma']:.4f}  Vega={greeks['vega']:.2f}")

# Pathwise 法（GBM 限定、高精度）
pw = stocha.greeks_pathwise(
    s0=100.0, r=0.05, sigma=0.2, t=1.0,
    strike=100.0, is_call=True,
    n_paths=100_000, n_steps=252,
    greeks=["delta", "vega"],
)
print(f"Pathwise Delta={pw['delta']:.4f}  Vega={pw['vega']:.2f}")
```

## API リファレンス

### 乱数生成

| 関数 / クラス | 説明 |
|---|---|
| `RNG(seed)` | PCG64DXSM 擬似乱数生成器（`seed` は 128 ビット整数まで対応） |
| `RNG.standard_normal(size)` | N(0, 1) からサンプリング |
| `RNG.normal(size, loc, scale)` | N(loc, scale²) からサンプリング |
| `RNG.uniform(size)` | Uniform[0, 1) からサンプリング |
| `RNG.save_state()` | RNG の完全な内部状態を JSON にシリアライズ（途中位置からのチェックポイント対応） |
| `RNG.from_state(json)` | JSON から RNG を復元。full-state（v1.2+）と旧 seed-only 形式の両方に対応 |
| `sobol(dim, n_samples)` | Sobol 低差異列（Joe & Kuo 2008）。高次元・大量サンプル用途では `scipy.stats.qmc.Sobol` の方が大幅に高速。 |
| `halton(dim, n_samples, skip)` | Halton 低差異列 |

### 確率過程モデル

| 関数 | 説明 |
|---|---|
| `gbm(s0, mu, sigma, t, steps, n_paths, ...)` | 幾何ブラウン運動（Euler-Maruyama、Rayon 並列） |
| `multi_gbm(s0, mu, sigma, corr, t, steps, n_paths, ...)` | マルチアセット相関 GBM（Cholesky 分解）; 戻り値 `(n_paths, steps+1, n_assets)` |
| `heston(s0, v0, mu, kappa, theta, xi, rho, ..., scheme)` | Heston ストキャスティックボラティリティ（`"euler"` Full Truncation / `"qe"` Andersen QE） |
| `merton_jump_diffusion(s0, mu, sigma, lambda_, ...)` | Merton ジャンプ拡散（対数正規ジャンプ） |
| `hull_white(r0, a, theta, sigma, t, steps, n_paths)` | Hull-White 1因子短期金利（Exact Simulation） |

### リスク・デリバティブ

| 関数 | 説明 |
|---|---|
| `var_cvar(returns, confidence)` | Value-at-Risk と Conditional VaR |
| `gaussian_copula(corr, n_samples)` | ガウスコピュラ サンプル |
| `student_t_copula(corr, nu, n_samples)` | Student-t コピュラ サンプル |
| `sabr_implied_vol(f, k, t, alpha, beta, rho, nu, shift)` | SABR Black インプライドボラティリティ |
| `sabr_calibrate(strikes, market_vols, f, t, beta, shift, ...)` | 観測 IV スマイルへ SABR `(α, ρ, ν)` をフィット（射影 LM ＋ ATM α の 1 次元 Brent） |
| `lsmc_american_option(s0, k, r, sigma, t, steps, n_paths, ...)` | LSMC によるアメリカンオプション価格付け |
| `greeks_fd(model, params, payoff, strike, n_paths, n_steps, greeks, ...)` | バンピング有限差分によるMCグリークス（GBM/Heston/Merton） |
| `greeks_pathwise(s0, r, sigma, t, strike, is_call, n_paths, n_steps, greeks)` | Pathwise IPA によるMCグリークス（GBM 限定; delta, vega） |
| `heston_price(strikes, is_call, s0, v0, r, kappa, theta, xi, rho, t, n_cos)` | COS 法による Heston 解析プライシング（Fang & Oosterlee 2008） |
| `heston_calibrate(strikes, maturities, market_prices, is_call, s0, r, ...)` | Heston `(v0, κ, θ, ξ, ρ)` のキャリブレーション（射影 LM + Vega 加重 COS 再評価） |
| `ssvi_calibrate(log_moneyness, theta, market_total_var, ...)` | SSVI 曲面 `(η, γ, ρ)` のキャリブレーション — 設計上カレンダー裁定フリー |
| `ssvi_implied_vol(log_moneyness, theta, t, eta, gamma, rho)` | SSVI 曲面からインプライド・ボラティリティを計算 |
| `ssvi_local_vol(log_moneyness, theta_values, t_values, eta, gamma, rho)` | SSVI 解析微分による Dupire 局所ボラティリティ（有限差分不使用） |

## パフォーマンス（Apple M シリーズ、リリースビルド）

| 処理 | 速度 | NumPy 比 |
|---|---|---|
| 正規分布サンプリング（Ziggurat） | 約 300M サンプル/秒 | 1.0× |
| GBM（252 ステップ、10 万パス） | 約 360M ステップ/秒 | 3.0× |
| Halton（dim=4） | 約 106M サンプル/秒 | 2.9× |

## チュートリアル

| ファイル | 内容 |
|---|---|
| `examples/01_basic_rng.ja.py` | RNG 基本操作・再現性・パフォーマンス計測 |
| `examples/02_stock_gbm.ja.py` | GBM による株価シミュレーション・オプション価格計算 |
| `examples/03_quasi_random.ja.py` | Sobol/Halton 列・QMC vs MC 収束比較 |
| `examples/04_stochastic_vol.ja.py` | Heston 確率ボラティリティ・Merton ジャンプ拡散 |
| `examples/05_risk_copula.ja.py` | VaR/CVaR・ガウス/Student-t コピュラのテール依存比較 |
| `examples/06_interest_rate.ja.py` | Hull-White 金利モデル・SABR ボラティリティスマイル |
| `examples/07_american_option.ja.py` | LSMC アメリカンオプション・早期行使プレミアム |
| `examples/08_multi_asset.ja.py` | マルチアセット相関 GBM・ポートフォリオ VaR・相関検証 |
| `examples/09_heston_calibration.ja.py` | Heston COS 法プライシング・IV スマイル・単一/マルチ満期キャリブレーション |
| `examples/10_local_vol.ja.py` | SSVI 曲面・Dupire 局所ボラ・連続配当利回り |

## 対象ユーザー

- 金融工学を学ぶ学部生・大学院生
- クオンツアナリスト・リスク管理部門
- AI/ML エンジニア（拡散モデル・MCMC）

## ロードマップ

| バージョン | 機能 |
|---|---|
| **v0.1** | PCG64DXSM、正規分布、GBM、対称変量法 |
| **v0.2** | Sobol 列（Joe & Kuo 2008）、Halton 列、Heston モデル、Merton Jump-Diffusion |
| **v0.3** | VaR/CVaR、ガウス/Student-t コピュラ、Hull-White、SABR、LSMC |
| **v1.0** ✅ | Ziggurat サンプラー（正規分布サンプリング約 3 倍高速化） |
| **v1.1** ✅ | SABR キャリブレーション（`sabr_calibrate`） |
| **v1.2** ✅ | 完全 RNG 状態シリアライズ、Heston QE スキーム |
| **v1.3** ✅ | マルチアセット相関シミュレーション |
| **v1.4** ✅ | グリークス（バンピング有限差分 + Pathwise IPA） |
| **v1.5** ✅ | Heston キャリブレーション（COS 法プライシング + 射影 LM） |
| **v1.6** ✅ | **局所ボラティリティ**: SSVI 曲面、Dupire 局所ボラ（解析微分）、連続配当 |
| **v1.7** | **エキゾチック・オプション**: バリア、アジア、ルックバック・オプション |
| **v1.8** | **ハイブリッド・モデル**: Heston-Hull-White（株価＋金利の同時確率化） |
| **v1.9** | **高度な感応度**: 尤度比法（LRM）による不連続ペイオフのグリークス計算 |
| **v2.0** | **AI エコシステム**: DLPack ゼロコピー連携（PyTorch/JAX）による Deep Hedging 対応 |

## ライセンス

[MIT ライセンス](LICENSE) のもとで公開しています。商用利用可能です。

Copyright (c) 2026 Shigeki Yamato
