# stocha

> Rust 製の高速乱数・金融シミュレーションライブラリ（Python 向け）

[![PyPI](https://img.shields.io/pypi/v/stocha)](https://pypi.org/project/stocha/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[English README](README.md)

## 特長

- **高速 PRNG**: PCG64DXSM（NumPy デフォルト実装）
- **準乱数列**: Sobol（Joe & Kuo 2008）・Halton 列
- **確率過程モデル**: GBM、Heston、Merton Jump-Diffusion、Hull-White
- **リスク指標**: VaR・CVaR（Expected Shortfall）
- **コピュラ**: ガウスコピュラ・Student-t コピュラ（多変量依存構造モデリング）
- **ボラティリティ**: SABR インプライドボラティリティ（Hagan 2002、マイナス金利対応）
- **オプション価格付け**: Longstaff-Schwartz LSMC（アメリカンオプション）
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

# ── LSMC によるアメリカンオプション価格付け ───────────────────────────────
price, std_err = stocha.lsmc_american_option(
    s0=100.0, k=100.0, r=0.05, sigma=0.20,
    t=1.0, steps=50, n_paths=50_000,
)
print(f"アメリカンプット: {price:.4f} ± {std_err:.4f}")
```

## API リファレンス

### 乱数生成

| 関数 / クラス | 説明 |
|---|---|
| `RNG(seed)` | PCG64DXSM 擬似乱数生成器（`seed` は 128 ビット整数まで対応） |
| `RNG.standard_normal(size)` | N(0, 1) からサンプリング |
| `RNG.normal(size, loc, scale)` | N(loc, scale²) からサンプリング |
| `RNG.uniform(size)` | Uniform[0, 1) からサンプリング |
| `RNG.save_state()` | シードを JSON 文字列にシリアライズ（ストリーム位置ではなくシード値のみ記録） |
| `RNG.from_state(json)` | `save_state` が返した JSON から RNG を復元（`RNG(seed=original_seed)` と等価） |
| `sobol(dim, n_samples)` | Sobol 低差異列（Joe & Kuo 2008）。高次元・大量サンプル用途では `scipy.stats.qmc.Sobol` の方が大幅に高速。 |
| `halton(dim, n_samples, skip)` | Halton 低差異列 |

### 確率過程モデル

| 関数 | 説明 |
|---|---|
| `gbm(s0, mu, sigma, t, steps, n_paths, ...)` | 幾何ブラウン運動（Euler-Maruyama、Rayon 並列） |
| `heston(s0, v0, mu, kappa, theta, xi, rho, ...)` | Heston ストキャスティックボラティリティ（Full Truncation スキーム） |
| `merton_jump_diffusion(s0, mu, sigma, lambda_, ...)` | Merton ジャンプ拡散（対数正規ジャンプ） |
| `hull_white(r0, a, theta, sigma, t, steps, n_paths)` | Hull-White 1因子短期金利（Exact Simulation） |

### リスク・デリバティブ

| 関数 | 説明 |
|---|---|
| `var_cvar(returns, confidence)` | Value-at-Risk と Conditional VaR |
| `gaussian_copula(corr, n_samples)` | ガウスコピュラ サンプル |
| `student_t_copula(corr, nu, n_samples)` | Student-t コピュラ サンプル |
| `sabr_implied_vol(f, k, t, alpha, beta, rho, nu, shift)` | SABR Black インプライドボラティリティ |
| `lsmc_american_option(s0, k, r, sigma, t, steps, n_paths, ...)` | LSMC によるアメリカンオプション価格付け |

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
| **v1.1** | DLPack ゼロコピー、SABR/Heston キャリブレーション |

## ライセンス

[MIT ライセンス](LICENSE) のもとで公開しています。商用利用可能です。

Copyright (c) 2026 Shigeki Yamato
