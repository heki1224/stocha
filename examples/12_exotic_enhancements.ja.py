"""チュートリアル 12: エキゾチック拡張 — BGK、期中評価、リベート (v1.7.1)

v1.7.1 で追加されたエキゾチック・オプション関連機能:
1. Broadie-Glasserman-Kou (1997) 離散モニタリング連続性補正 (バリア)
2. アジア／ルックバックの期中評価 (running_avg / time_elapsed,
   running_max / running_min 引数)
3. バリア・オプションのリベート (Haug §4.17) — paid_at_hit / paid_at_expiry
"""

import math

import stocha

# === 1. Broadie-Glasserman-Kou 連続性補正 ==================================
print("=" * 64)
print("1. BGK 離散モニタリング補正 (バリア・オプション)")
print("=" * 64)

# 連続モニタリング (教科書の解析解)
cont = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
)
# 日次モニタリング (年 252 回): バリアを H · exp(+β·σ·√(T/n)) で外側へシフト
# β = 0.5826 (Riemann ζ(1/2) より導出)
daily = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out", n_monitoring=252,
)
# 月次モニタリング — 補正効果がより大きい
monthly = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out", n_monitoring=12,
)
print(f"  UO コール (連続):         {cont:.4f}")
print(f"  UO コール (日次, n=252):  {daily:.4f}  (Δ={daily-cont:+.4f})")
print(f"  UO コール (月次, n=12):   {monthly:.4f}  (Δ={monthly-cont:+.4f})")
print("  → 離散モニタリングはノックアウト機会が減る分、価格が上がる")

# === 2a. アジア期中評価 ====================================================
print()
print("=" * 64)
print("2a. アジア期中評価 — 経過平均からの再評価")
print("=" * 64)

# 半年経過時、経過平均が 102 (ストライクをわずかに上回る)
fresh = stocha.asian_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    average_type="geometric", method="analytical",
)
seasoned_low = stocha.asian_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    average_type="geometric", method="analytical",
    running_avg=98, time_elapsed=0.5,
)
seasoned_high = stocha.asian_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    average_type="geometric", method="analytical",
    running_avg=105, time_elapsed=0.5,
)
print(f"  期中評価なし (新規):       {fresh:.4f}")
print(f"  期中評価 A=98,  t1=0.5:    {seasoned_low:.4f}")
print(f"  期中評価 A=105, t1=0.5:    {seasoned_high:.4f}")
print("  → 経過平均が高いほどコールの価値が上昇")

# Deep-ITM: 経過平均が高すぎて K* = (T·K - t1·A)/(T-t1) ≤ 0 となるケース
deep_itm_call = stocha.asian_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    average_type="geometric", method="analytical",
    running_avg=200, time_elapsed=0.9,
)
deep_itm_put = stocha.asian_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    option_type="put", average_type="geometric", method="analytical",
    running_avg=200, time_elapsed=0.9,
)
print(f"  Deep-ITM コール (A=200):   {deep_itm_call:.4f}  (確定 PV)")
print(f"  Deep-ITM プット (A=200):   {deep_itm_put:.4f}  (常に OTM)")

# === 2b. ルックバック期中評価 ==============================================
print()
print("=" * 64)
print("2b. ルックバック期中評価 — running_max / running_min")
print("=" * 64)

# 浮動ストライク・プット: 履歴最大値が上がるほど (max - S_T) が広がり価格上昇
fresh_put = stocha.lookback_price(
    s=100, r=0.05, sigma=0.2, t=1.0, option_type="put",
)
seasoned_put = stocha.lookback_price(
    s=100, r=0.05, sigma=0.2, t=1.0, option_type="put",
    running_max=120,
)
print(f"  浮動プット (新規):          {fresh_put:.4f}")
print(f"  浮動プット (max=120):       {seasoned_put:.4f}")
print("  → 履歴最大値が高いほど浮動プットのペイオフは広がる")

# 固定コール期中評価: 履歴最大値が既にストライクを超えている → 確定本源価値
fixed_call = stocha.lookback_price(
    s=100, r=0.05, sigma=0.2, t=1.0,
    strike_type="fixed", k=100, running_max=120,
)
intrinsic = (120 - 100) * math.exp(-0.05)
print(f"  固定コール (M=120, K=100):  {fixed_call:.4f}")
print(f"  本源価値の割引現在価値:     {intrinsic:.4f}  ⇒ 残期間オプション価値が上乗せ")

# === 3. バリア・リベート ===================================================
print()
print("=" * 64)
print("3. バリア・リベート (Haug §4.17)")
print("=" * 64)

base = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
)
# KO paid at hit: バリア接触の瞬間に R を支払う
hit = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
    rebate=5, rebate_at_hit=True,
)
# KO paid at expiry: 接触したことを条件に満期 T で R を支払う
expiry = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
    rebate=5, rebate_at_hit=False,
)
print(f"  UO コール (リベートなし):           {base:.4f}")
print(f"  UO コール + R=5 paid-at-hit:        {hit:.4f}  (リベート PV {hit-base:.4f})")
print(f"  UO コール + R=5 paid-at-expiry:     {expiry:.4f}  (リベート PV {expiry-base:.4f})")
print("  → paid-at-hit ≥ paid-at-expiry (早く受け取るほど価値が高い)")

# ノックイン・リベート: バリアが「触れなかった」場合に満期で支払い
di_base = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=80, barrier_type="down-and-in",
)
di_reb = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=80, barrier_type="down-and-in", rebate=5,
)
print(f"  DI プット (リベートなし):           {di_base:.4f}")
print(f"  DI プット + R=5 (未接触時に満期支払): {di_reb:.4f}  (リベート PV {di_reb-di_base:.4f})")

# エッジケース: スポットが既にバリアを突破済み
breached = stocha.barrier_price(
    s=125, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
    rebate=5, rebate_at_hit=True,
)
print(f"  UO コール, S0=125 (突破済), R=5 @hit: {breached:.4f}  (= R を即時支払い)")
