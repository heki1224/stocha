# stocha

> Rust 製の高速乱数・金融シミュレーションライブラリ（Python 向け）

[![PyPI](https://img.shields.io/pypi/v/stocha)](https://pypi.org/project/stocha/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[English README](README.md)

## 特長

- **高速 PRNG**: PCG64DXSM（NumPy デフォルト実装）、Xoshiro256++、MT19937
- **正規分布**: Marsaglia 極座標法（v0.2 で Ziggurat 法に移行予定）
- **GBM シミュレーション**: Euler-Maruyama × Rayon 並列処理
- **対称変量法**: 組み込み分散低減オプション
- **完全再現性**: ブロック分割 RNG ストリームによりスレッド数に依存せず同一結果を保証
- **PyTorch / JAX 連携**: DLPack ゼロコピー出力（v0.2 対応予定）

## インストール

```bash
pip install stocha
```

Rust コンパイラは不要です。`pip install` のみで使えます。

## クイックスタート

```python
import stocha

# 乱数生成
rng = stocha.RNG(seed=42)
samples = rng.normal(size=10_000, loc=0.0, scale=1.0)

# GBM による株価シミュレーション
paths = stocha.gbm(
    s0=100.0,    # 初期株価
    mu=0.05,     # ドリフト（年率 5%）
    sigma=0.20,  # ボラティリティ（年率 20%）
    t=1.0,       # 満期 1 年
    steps=252,   # 営業日ステップ
    n_paths=100_000,
    seed=42,
)
# paths.shape == (100_000, 253)
```

## パフォーマンス（Apple M シリーズ、リリースビルド）

| 処理 | 速度 |
|---|---|
| 正規分布サンプリング | 約 155M サンプル/秒 |
| GBM（252 ステップ） | 約 68万 パス/秒 |

## チュートリアル

| ファイル | 内容 |
|---|---|
| `examples/01_basic_rng.ja.py` | RNG 基本操作・再現性・パフォーマンス計測 |
| `examples/02_stock_gbm.ja.py` | GBM による株価シミュレーション・オプション価格計算 |

## 対象ユーザー

- 金融工学を学ぶ学部生・大学院生
- クオンツアナリスト・リスク管理部門
- AI/ML エンジニア（拡散モデル・MCMC）

## ロードマップ

| バージョン | 機能 |
|---|---|
| **v0.1** | PCG64DXSM、正規分布、GBM、対称変量法 |
| **v0.2** | Sobol 列（Joe & Kuo 2008 + Owen スクランブル）、Halton 列、Heston モデル、Jump-Diffusion |
| **v1.0** | VaR/CVaR、コピュラ、Hull-White、SABR、LSMC、DLPack ゼロコピー |

## ライセンス

[MIT ライセンス](LICENSE) のもとで公開しています。商用利用可能です。

Copyright (c) 2026 Shigeki Yamato
