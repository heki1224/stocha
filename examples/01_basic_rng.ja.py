"""
stocha チュートリアル 1: 乱数生成の基本
========================================

RNG の基本操作・再現性・パフォーマンス計測を示します。
"""

import time

import numpy as np
import stocha

# --- 1. RNG の基本操作 ---
print("=" * 50)
print("1. RNG 基本操作")
print("=" * 50)

rng = stocha.RNG(seed=42)
print(f"RNG: {rng}")

u = rng.uniform(size=10)
print(f"uniform(10): min={u.min():.4f}, max={u.max():.4f}")

z = rng.standard_normal(size=100_000)
print(f"standard_normal(10万): 平均={z.mean():.4f}, 標準偏差={z.std():.4f}")

x = rng.normal(size=100_000, loc=5.0, scale=2.0)
print(f"normal(10万, loc=5, scale=2): 平均={x.mean():.4f}, 標準偏差={x.std():.4f}")

# --- 2. 再現性の確認 ---
print("\n" + "=" * 50)
print("2. 再現性の確認（同じシード → 同じ結果）")
print("=" * 50)

rng1 = stocha.RNG(seed=123)
rng2 = stocha.RNG(seed=123)
s1 = rng1.normal(size=5)
s2 = rng2.normal(size=5)
print(f"rng1: {s1}")
print(f"rng2: {s2}")
print(f"完全一致: {np.allclose(s1, s2)}")

# --- 3. 状態の保存（チェックポイント） ---
print("\n" + "=" * 50)
print("3. 状態の保存（チェックポイント）")
print("=" * 50)

rng = stocha.RNG(seed=999)
state_json = rng.save_state()
print(f"保存された状態: {state_json}")

# --- 4. アルゴリズム選択 ---
print("\n" + "=" * 50)
print("4. アルゴリズム選択")
print("=" * 50)

rng_pcg = stocha.RNG(seed=42, algorithm="pcg64dxsm")
print(f"PCG64DXSM（デフォルト）: {rng_pcg}")

# --- 5. パフォーマンス計測 ---
print("\n" + "=" * 50)
print("5. パフォーマンス計測")
print("=" * 50)

rng = stocha.RNG(seed=0)
for n in [10_000, 100_000, 1_000_000]:
    t0 = time.time()
    _ = rng.standard_normal(size=n)
    elapsed = time.time() - t0
    rate = n / elapsed / 1e6
    print(f"standard_normal({n:>10,}): {elapsed*1000:.1f}ms  ({rate:.1f}M サンプル/秒)")

print("\n✅ 全テスト完了")
