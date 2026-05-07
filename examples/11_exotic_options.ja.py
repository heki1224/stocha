"""チュートリアル 11: エキゾチック・オプション — バリア・アジア・ルックバック

デモ内容:
1. バリアオプション（8タイプ、解析解 vs MC、イン・アウト・パリティ）
2. アジアンオプション（幾何平均 解析解、算術平均 MC + CV）
3. ルックバックオプション（浮動・固定ストライク、解析解 vs MC）
4. 手法選択: auto, analytical, mc
"""

import stocha

# === 1. バリアオプション ====================================================
print("=" * 60)
print("1. バリアオプション")
print("=" * 60)

# アップ・アンド・アウト・コール: S >= 120 でノックアウト
uo_call = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=120.0, barrier_type="up-and-out",
)
print(f"  アップ・アンド・アウト・コール (H=120): {uo_call:.4f}")

# アップ・アンド・イン・コール: S >= 120 で有効化
ui_call = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=120.0, barrier_type="up-and-in",
)
print(f"  アップ・アンド・イン・コール  (H=120): {ui_call:.4f}")
print(f"  イン + アウト = バニラ:                {uo_call + ui_call:.4f}")

# ダウン・アンド・アウト・プット: S <= 80 でノックアウト
do_put = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=80.0, barrier_type="down-and-out", option_type="put",
)
print(f"  ダウン・アンド・アウト・プット (H=80):  {do_put:.4f}")

# ダウン・アンド・イン・プット
di_put = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=80.0, barrier_type="down-and-in", option_type="put",
)
print(f"  ダウン・アンド・イン・プット  (H=80):   {di_put:.4f}")
print(f"  イン + アウト = バニラ:                 {do_put + di_put:.4f}")

# 解析解 vs MC の比較
print("\n  解析解 vs MC:")
analytical = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=120.0, barrier_type="up-and-out", method="analytical",
)
mc = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=120.0, barrier_type="up-and-out",
    method="mc", n_paths=500_000, n_steps=1000,
)
print(f"  解析解: {analytical:.4f}")
print(f"  MC:     {mc:.4f}")

# === 2. アジアンオプション ==================================================
print("\n" + "=" * 60)
print("2. アジアンオプション")
print("=" * 60)

# 幾何平均コール（解析解、Kemna-Vorst）
geo_call = stocha.asian_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    average_type="geometric", method="analytical",
)
print(f"  幾何平均コール（解析解）:      {geo_call:.4f}")

# 算術平均コール（MC + 幾何 CV）
arith_call = stocha.asian_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    average_type="arithmetic", n_paths=200_000,
)
print(f"  算術平均コール（MC + CV）:     {arith_call:.4f}")

# 算術平均 >= 幾何平均（AM-GM 不等式）
print(f"  算術 >= 幾何? {arith_call >= geo_call}")

# 浮動ストライク
float_asian = stocha.asian_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    strike_type="floating", n_paths=200_000,
)
print(f"  浮動ストライク・コール（MC）:  {float_asian:.4f}")

# アジアン・プット
asian_put = stocha.asian_price(
    s=100.0, k=105.0, r=0.05, sigma=0.2, t=1.0,
    option_type="put", n_paths=200_000,
)
print(f"  算術プット (K=105, MC):        {asian_put:.4f}")

# === 3. ルックバックオプション ==============================================
print("\n" + "=" * 60)
print("3. ルックバックオプション")
print("=" * 60)

# 浮動ストライク・コール: ペイオフ = S_T - S_min（解析解）
float_call = stocha.lookback_price(
    s=100.0, r=0.05, sigma=0.2, t=1.0,
    strike_type="floating", method="analytical",
)
print(f"  浮動コール（解析解）:            {float_call:.4f}")

# 浮動ストライク・プット: ペイオフ = S_max - S_T
float_put = stocha.lookback_price(
    s=100.0, r=0.05, sigma=0.2, t=1.0,
    strike_type="floating", option_type="put", method="analytical",
)
print(f"  浮動プット（解析解）:            {float_put:.4f}")

# 固定ストライク・コール: ペイオフ = (S_max - K)+
fixed_call = stocha.lookback_price(
    s=100.0, r=0.05, sigma=0.2, t=1.0,
    strike_type="fixed", k=100.0, method="analytical",
)
print(f"  固定コール K=100（解析解）:      {fixed_call:.4f}")

# 固定ストライク・プット: ペイオフ = (K - S_min)+
fixed_put = stocha.lookback_price(
    s=100.0, r=0.05, sigma=0.2, t=1.0,
    strike_type="fixed", option_type="put", k=100.0, method="analytical",
)
print(f"  固定プット K=100（解析解）:      {fixed_put:.4f}")

# 連続（解析解）vs 離散（MC）の比較
mc_float = stocha.lookback_price(
    s=100.0, r=0.05, sigma=0.2, t=1.0,
    method="mc", n_paths=500_000,
)
print(f"\n  連続監視 vs 離散監視:")
print(f"  解析解（連続）:              {float_call:.4f}")
print(f"  MC（離散、252ステップ）:     {mc_float:.4f}")
print(f"  離散の過小評価率: {(1 - mc_float / float_call) * 100:.1f}%")

# === 4. 配当の影響 ==========================================================
print("\n" + "=" * 60)
print("4. 配当の影響 (q=3%)")
print("=" * 60)

barrier_no_q = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
)
barrier_q = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out", q=0.03,
)
print(f"  バリア (q=0): {barrier_no_q:.4f}  (q=3%): {barrier_q:.4f}")

lookback_no_q = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0)
lookback_q = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0, q=0.03)
print(f"  ルックバック (q=0): {lookback_no_q:.4f}  (q=3%): {lookback_q:.4f}")
