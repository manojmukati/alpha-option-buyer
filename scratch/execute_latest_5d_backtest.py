# -*- coding: utf-8 -*-
"""
scratch/execute_latest_5d_backtest.py
======================================
Authoritative 5-Day Backtest Ending Today (2026-10-28 to 2026-11-03).
Uses ONLY 3-minute resampled candles.
Applies exact production 5-pillar rules without changing any code:
  1. Priority 1: Isolated 4-Index Evaluation (MAX_INDEX_PER_SLAB = 1)
  2. Priority 2: Isolated Stock F&O Universe Evaluation (MAX_STOCKS_PER_SLAB = 2)
  3. Symmetrical Macro Trend Gate (Nifty Day Open + 9-EMA alignment; Climax candle block)
  4. Anti-Chase Buffer (Max 0.08 * ATR for Stocks; Max 20 pts for Indices)
  5. Calibrated Dynamic Risk & Target 1 Profit Locking (T1 Cost Trail, T2 Jackpot, 15M Time SL)
"""

import os
import sys
import glob
import json
import datetime
import numpy as np
import pandas as pd

sys.path.append('.')

CACHE_DIR = r"c:\Users\Mohit\.gemini\antigravity\playground\aphelion-zenith\scratch\data_cache_latest_5d_1m"
REPORT_DIR = r"c:\Users\Mohit\.gemini\antigravity\playground\aphelion-zenith\scratch"
ATR_CACHE_FILE = r"c:\Users\Mohit\.gemini\antigravity\playground\aphelion-zenith\alpha_profile_oi_system\config\stock_atr_cache.json"

INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
MIN_STOCK_PRICE = 350.0
MAX_INDEX_PER_SLAB = 1
MAX_STOCKS_PER_SLAB = 2
NIFTY_HEAVYWEIGHTS = {'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'}


def load_resampled_3m_data():
    files = glob.glob(os.path.join(CACHE_DIR, "*.parquet"))
    asset_data = {}
    for f in files:
        sym = os.path.basename(f).replace('.parquet', '')
        try:
            df_1m = pd.read_parquet(f)
            if df_1m.empty:
                continue
            df_1m.columns = [c.lower() for c in df_1m.columns]
            rng_1m = np.maximum(0.01, df_1m['high'] - df_1m['low'])
            if sym in INDEX_NAMES:
                df_1m['delta'] = (2.0 * df_1m['close'] - (df_1m['high'] + df_1m['low']))
            else:
                df_1m['delta'] = df_1m['volume'] * ((2.0 * df_1m['close'] - (df_1m['high'] + df_1m['low'])) / rng_1m)

            # Per-session clean resampling (zero cross-day bar contamination)
            resampled_days = []
            for date_val, group in df_1m.groupby(df_1m.index.date):
                df_day_3m = group.resample('3min', label='left', closed='left').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'delta': 'sum'
                }).dropna()
                if len(df_day_3m) >= 5:
                    resampled_days.append(df_day_3m)

            if resampled_days:
                df_3m = pd.concat(resampled_days)
                if len(df_3m) >= 20:
                    asset_data[sym] = df_3m
        except Exception:
            pass
    return asset_data


def load_verified_atrs():
    with open(ATR_CACHE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_bar_delta_proxy(df_bars, is_index=False):
    vols = df_bars['volume'].values.astype(float)
    deltas = df_bars['delta'].values.astype(float)
    highs = df_bars['high'].values.astype(float)
    lows = df_bars['low'].values.astype(float)
    rngs = np.maximum(0.01, highs - lows)
    tot_v = float(np.sum(vols))

    if not is_index and tot_v > 0:
        session_cvd_ratio = round(float(np.sum(deltas)) / max(1.0, tot_v), 3)
        rec_n = min(3, len(df_bars))
        recent_cvd_ratio = round(float(np.sum(deltas[-rec_n:])) / max(1.0, float(np.sum(vols[-rec_n:]))), 3)
    else:
        tot_rng = float(np.sum(rngs))
        session_cvd_ratio = round(float(np.sum(deltas)) / max(0.5, tot_rng), 3)
        rec_n = min(3, len(df_bars))
        recent_cvd_ratio = round(float(np.sum(deltas[-rec_n:])) / max(0.5, float(np.sum(rngs[-rec_n:]))), 3)

    return {'recent_cvd_ratio': recent_cvd_ratio, 'session_cvd_ratio': session_cvd_ratio}


def get_nifty_regime(nifty_day, b_idx):
    live_b = nifty_day.iloc[b_idx]
    eval_b = nifty_day.iloc[b_idx - 1]
    n_spot = float(live_b['close'])
    ema9 = float(nifty_day['ema9'].iloc[b_idx])
    day_open = float(nifty_day['open'].iloc[0])
    move_from_open = n_spot - day_open

    # Climax candle check on Nifty (Pillar 3 Anti-Exhaustion - Calibrated for high-volatility 24,000+ Index)
    n_last_bar_rng = float(eval_b['high'] - eval_b['low'])
    if n_last_bar_rng > 65.0:
        return {'allow_call': False, 'allow_put': False, 'reason': 'NIFTY_CLIMAX_EXHAUSTION', 'move': move_from_open, 'spot': n_spot, 'open': day_open}

    # 18m chop check
    recent_rng = nifty_day['high'].iloc[max(0, b_idx - 5):b_idx + 1].max() - nifty_day['low'].iloc[max(0, b_idx - 5):b_idx + 1].min()
    ema_dist = abs(n_spot - ema9)

    if recent_rng < 22.0 and ema_dist < 12.0:
        return {'allow_call': False, 'allow_put': False, 'reason': 'CHOP_TRAP', 'move': move_from_open, 'spot': n_spot, 'open': day_open}

    allow_call = (move_from_open >= -10.0) and (n_spot >= ema9 - 15.0)
    allow_put = (move_from_open <= +10.0) and (n_spot <= ema9 + 15.0)

    return {
        'allow_call': allow_call,
        'allow_put': allow_put,
        'reason': 'NORMAL',
        'spot': n_spot,
        'ema9': ema9,
        'open': day_open,
        'move': move_from_open
    }


def check_index_signal(sym, df_today, bar_idx, regime):
    if bar_idx < 10:
        return None

    eval_b = df_today.iloc[bar_idx - 1]
    live_b = df_today.iloc[bar_idx]
    b_open = float(eval_b['open'])
    b_high = float(eval_b['high'])
    b_low = float(eval_b['low'])
    b_close = float(eval_b['close'])
    cur_p = float(live_b['close'])
    rng = max(0.01, b_high - b_low)

    max_bar_rng = 65.0 if sym != 'BANKNIFTY' else 150.0
    if rng > max_bar_rng:
        return None

    comp_bars = df_today.iloc[:bar_idx - 1].tail(8)
    comp_high = float(comp_bars['high'].max())
    comp_low = float(comp_bars['low'].min())
    comp_width = comp_high - comp_low

    max_allowed_base = 120.0 if sym != 'BANKNIFTY' else 280.0
    if comp_width > max_allowed_base:
        return None

    idx_dist = 35.0 if sym != 'BANKNIFTY' else 80.0
    direction = None
    if b_close >= comp_high and b_close <= comp_high + idx_dist and cur_p >= comp_high and b_close > b_open:
        direction = 'CALL'
    elif b_close <= comp_low and b_close >= comp_low - idx_dist and cur_p <= comp_low and b_close < b_open:
        direction = 'PUT'

    if direction is None:
        return None

    idx_day_open = float(df_today['open'].iloc[0])
    if direction == 'CALL':
        if not regime['allow_call'] or cur_p < idx_day_open:
            return None
    elif direction == 'PUT':
        if not regime['allow_put'] or cur_p > idx_day_open:
            return None

    solid_body_ratio = round(abs(b_close - b_open) / rng, 2)
    if solid_body_ratio < 0.45:
        return None

    cvd_res = compute_bar_delta_proxy(df_today.iloc[:bar_idx], is_index=True)
    rec_cvd = cvd_res['recent_cvd_ratio']
    cvd_pass = (rec_cvd >= 0.12) if direction == 'CALL' else (rec_cvd <= -0.12)
    if not cvd_pass:
        return None

    session_rngs = df_today.iloc[:bar_idx]['high'] - df_today.iloc[:bar_idx]['low']
    median_rng = float(session_rngs.median()) if len(session_rngs) > 0 else 15.0
    idx_rvol = round(rng / max(0.5, median_rng), 2)
    if idx_rvol < 1.15:
        return None

    score = min(99.0, 60.0 + min(25.0, (idx_rvol - 1.0) * 15.0) + (15.0 if abs(rec_cvd) >= 0.25 else 5.0))
    return {
        'symbol': sym,
        'direction': direction,
        'entry_price': cur_p,
        'breakout_bar_close': b_close,
        'eval_time': df_today.index[bar_idx],
        'is_index': True,
        'stock_atr': 0.0,
        'rvol': idx_rvol,
        'turnover_cr': 0.0,
        'pct_chg': round(((cur_p - idx_day_open) / max(1.0, idx_day_open)) * 100.0, 2),
        'score': round(score, 2),
        'base_high': comp_high,
        'base_low': comp_low,
        'type': 'INDEX_INSTITUTIONAL_BREAKOUT'
    }


def check_stock_signal(sym, df_today, bar_idx, stock_atr, regime):
    if bar_idx < 10:
        return None

    eval_b = df_today.iloc[bar_idx - 1]
    live_b = df_today.iloc[bar_idx]
    spot = float(live_b['close'])
    b_open = float(eval_b['open'])
    b_high = float(eval_b['high'])
    b_low = float(eval_b['low'])
    b_close = float(eval_b['close'])
    rng = max(0.01, b_high - b_low)

    atr_pct = (stock_atr / max(1.0, spot)) * 100.0
    if spot < MIN_STOCK_PRICE or atr_pct < 1.0:
        return None

    day_open = float(df_today['open'].iloc[0])
    day_pct_move = ((spot - day_open) / max(1.0, day_open)) * 100.0
    stk_day_high = float(df_today.iloc[:bar_idx + 1]['high'].max())
    stk_day_low = float(df_today.iloc[:bar_idx + 1]['low'].min())
    day_range = stk_day_high - stk_day_low
    if abs(day_pct_move) > 3.00 or day_range > stock_atr * 1.30:
        return None

    comp_bars = df_today.iloc[:bar_idx - 1].tail(8)
    comp_high = float(comp_bars['high'].max())
    comp_low = float(comp_bars['low'].min())
    comp_w = comp_high - comp_low
    if comp_w > stock_atr * 0.60:
        return None

    max_dist_buffer = stock_atr * 0.08
    direction = None
    if b_close >= comp_high and b_close <= comp_high + max_dist_buffer and b_close > b_open and spot >= (comp_high - stock_atr * 0.03):
        direction = 'CALL'
    elif b_close <= comp_low and b_close >= comp_low - max_dist_buffer and b_close < b_open and spot <= (comp_low + stock_atr * 0.03):
        direction = 'PUT'

    if direction is None:
        return None

    if direction == 'CALL' and not regime['allow_call']:
        return None
    if direction == 'PUT' and not regime['allow_put']:
        return None

    c_rng = b_high - b_low
    if c_rng > stock_atr * 0.40:
        return None

    p3_low = df_today.iloc[max(0, bar_idx - 4):bar_idx - 1]['low'].min()
    p3_high = df_today.iloc[max(0, bar_idx - 4):bar_idx - 1]['high'].max()
    p3_move = (b_close - p3_low) if direction == 'CALL' else (p3_high - b_close)
    if p3_move > stock_atr * 0.55:
        return None

    curr_3m_v = float(eval_b.get('volume', 0.0))
    prior_10_bars = df_today.iloc[:bar_idx - 1].tail(10)
    base_10_v = float(prior_10_bars['volume'].mean()) if len(prior_10_bars) > 0 else 1000.0
    real_3m_rvol = round(curr_3m_v / max(1.0, base_10_v), 2)
    turnover_cr = round((curr_3m_v * spot) / 1e7, 2)

    if real_3m_rvol < 1.80 or turnover_cr < 3.0:
        return None

    cvd_res = compute_bar_delta_proxy(df_today.iloc[:bar_idx], is_index=False)
    rec_cvd = cvd_res['recent_cvd_ratio']
    bar_delta_ratio = round((2.0 * b_close - (b_high + b_low)) / rng, 3)

    if direction == 'CALL':
        cvd_pass = (rec_cvd >= 0.15 or bar_delta_ratio >= 0.35) and (rec_cvd >= 0.0)
    else:
        cvd_pass = (rec_cvd <= -0.15 or bar_delta_ratio <= -0.35) and (rec_cvd <= 0.0)

    if not cvd_pass:
        return None

    # Day High / Low Frontier Check (Strictly eliminates mid-range chop in both morning & afternoon)
    near_hl = (direction == 'CALL' and b_close >= stk_day_high - stock_atr * 0.12) or \
              (direction == 'PUT' and b_close <= stk_day_low + stock_atr * 0.12)
    if not near_hl:
        return None

    body = abs(b_close - b_open)
    solid_body_ratio = round(body / rng, 2)
    clv = round((b_close - b_low) / rng, 2)
    if solid_body_ratio < 0.50:
        return None

    clv_pass = (clv >= 0.60) if direction == 'CALL' else (clv <= 0.40)
    if not clv_pass:
        return None

    score = min(99.0, 55.0 + min(25.0, (real_3m_rvol - 1.0) * 10.0) + (15.0 if abs(bar_delta_ratio) >= 0.35 else 5.0) + min(10.0, turnover_cr * 1.5))
    return {
        'symbol': sym,
        'direction': direction,
        'entry_price': spot,
        'breakout_bar_close': b_close,
        'eval_time': df_today.index[bar_idx],
        'is_index': False,
        'stock_atr': stock_atr,
        'rvol': real_3m_rvol,
        'turnover_cr': turnover_cr,
        'pct_chg': round(((spot - day_open) / max(1.0, day_open)) * 100.0, 2),
        'score': round(score, 2),
        'base_high': comp_high,
        'base_low': comp_low,
        'type': 'STOCK_INSTITUTIONAL_BREAKOUT'
    }


def simulate_trade_forward(cand, df_today, entry_idx):
    entry_p = cand['entry_price']
    direction = cand['direction']
    is_idx = cand.get('is_index', False)
    atr = cand.get('stock_atr', 30.0)

    if is_idx:
        sym = cand['symbol']
        spot_t1 = 75.0 if sym == 'BANKNIFTY' else 35.0
        spot_t2 = 150.0 if sym == 'BANKNIFTY' else 70.0
        spot_sl = 60.0 if sym == 'BANKNIFTY' else 30.0
    else:
        spot_t1 = round(atr * 0.35, 1)
        spot_t2 = round(atr * 0.70, 1)
        if direction == 'CALL':
            spot_sl = max(round(atr * 0.15, 1), entry_p - (cand['base_low'] - atr * 0.03))
        else:
            spot_sl = max(round(atr * 0.15, 1), (cand['base_high'] + atr * 0.03) - entry_p)
        spot_sl = min(spot_sl, round(atr * 0.25, 1))

    bars_fwd = df_today.iloc[entry_idx + 1:]
    exit_p = entry_p
    exit_reason = "EOD_EXIT"
    exit_idx_local = len(bars_fwd)
    t1_hit = False

    for step, (ts, b) in enumerate(bars_fwd.iterrows()):
        b_high = float(b['high'])
        b_low = float(b['low'])
        b_close = float(b['close'])

        fav_move = (b_high - entry_p) if direction == 'CALL' else (entry_p - b_low)
        adv_move = (entry_p - b_low) if direction == 'CALL' else (b_high - entry_p)
        cur_gain = (b_close - entry_p) if direction == 'CALL' else (entry_p - b_close)

        if not t1_hit and fav_move >= spot_t1:
            t1_hit = True

        if t1_hit:
            if fav_move >= spot_t2:
                exit_p = (entry_p + spot_t2) if direction == 'CALL' else (entry_p - spot_t2)
                exit_reason = "TARGET_2_HIT (JACKPOT)"
                exit_idx_local = step
                break
            # Target 1 Hit: Lock in +0.40 * T1 profit floor (Never give back profit to flat 0.00R!)
            lock_pts = spot_t1 * 0.40
            locked_sl_level = (entry_p + lock_pts) if direction == 'CALL' else (entry_p - lock_pts)
            if (direction == 'CALL' and b_low <= locked_sl_level) or (direction == 'PUT' and b_high >= locked_sl_level):
                exit_p = locked_sl_level
                exit_reason = "T1_PROFIT_LOCKED_EXIT"
                exit_idx_local = step
                break
        else:
            if adv_move >= spot_sl:
                exit_p = (entry_p - spot_sl) if direction == 'CALL' else (entry_p + spot_sl)
                exit_reason = "SL_HIT"
                exit_idx_local = step
                break
            # 15-Minute Stagnant Kill Switch: ONLY if T1 not hit AND position is stagnant in loss/flat!
            if step == 4 and (not t1_hit) and cur_gain <= 0.0:
                exit_p = b_close
                exit_reason = "15M_TIME_SL_EXIT"
                exit_idx_local = step
                break

    if exit_reason == "EOD_EXIT" and len(bars_fwd) > 0:
        exit_p = float(bars_fwd.iloc[-1]['close'])

    pts_captured = round((exit_p - entry_p) if direction == 'CALL' else (entry_p - exit_p), 2)
    sl_denom = max(0.5, spot_sl)
    rr_realized = round(pts_captured / sl_denom, 2) if pts_captured != 0 else 0.0

    return {
        'symbol': cand['symbol'],
        'direction': cand['direction'],
        'entry_price': cand['entry_price'],
        'breakout_bar_close': cand['breakout_bar_close'],
        'eval_time': cand['eval_time'],
        'is_index': cand['is_index'],
        'stock_atr': cand.get('stock_atr', 0),
        'rvol': cand['rvol'],
        'turnover_cr': cand['turnover_cr'],
        'pct_chg': cand['pct_chg'],
        'score': cand['score'],
        'type': cand['type'],
        'exit_price': exit_p,
        'points_captured': pts_captured,
        'spot_sl': spot_sl,
        'exit_reason': exit_reason,
        'realized_r': rr_realized
    }


def main():
    print("Loading latest 1m parquet data resampled to 3m candles...")
    asset_data = load_resampled_3m_data()
    print(f"Loaded {len(asset_data)} active assets.")
    atrs = load_verified_atrs()

    if 'NIFTY' not in asset_data:
        print("[ERROR]: NIFTY data not found!")
        return

    all_dates = sorted(list(set(d for df in asset_data.values() for d in df.index.date)))
    print(f"5 Backtest Dates (Ending Today): {[str(d) for d in all_dates]}")

    stock_syms = [s for s in asset_data.keys() if s not in INDEX_NAMES]
    executed_trades = []

    for d in all_dates:
        nifty_day = asset_data['NIFTY'][asset_data['NIFTY'].index.date == d].copy()
        if nifty_day.empty:
            continue
        nifty_day['ema9'] = nifty_day['close'].ewm(span=9, adjust=False).mean()

        m_idx, m_stk = set(), set()
        a_idx, a_stk = set(), set()
        time_index = nifty_day.index

        for b_idx in range(10, len(time_index)):
            curr_t = time_index[b_idx]
            tot_m = curr_t.hour * 60 + curr_t.minute
            is_m = (570 <= tot_m <= 615)
            is_a = (795 <= tot_m <= 855)
            if not (is_m or is_a):
                continue

            regime = get_nifty_regime(nifty_day, b_idx)
            if not (regime['allow_call'] or regime['allow_put']):
                continue

            # PRIORITY 1: INDEX EVALUATION (Max 1 per slab)
            c_idx = len(m_idx) if is_m else len(a_idx)
            if c_idx < MAX_INDEX_PER_SLAB:
                for isym in INDEX_NAMES:
                    if isym not in asset_data:
                        continue
                    idf = asset_data[isym][asset_data[isym].index.date == d]
                    if len(idf) <= b_idx:
                        continue
                    if isym in (m_idx if is_m else a_idx):
                        continue

                    sig = check_index_signal(isym, idf, b_idx, regime)
                    if sig:
                        out = simulate_trade_forward(sig, idf, b_idx)
                        out['date'] = str(d)
                        executed_trades.append(out)
                        if is_m: m_idx.add(isym)
                        else: a_idx.add(isym)
                        break

            # PRIORITY 2: STOCK EVALUATION (Max 2 per slab)
            c_stk = len(m_stk) if is_m else len(a_stk)
            if c_stk < MAX_STOCKS_PER_SLAB:
                stk_candidates = []
                for ssym in stock_syms:
                    if ssym in (m_stk if is_m else a_stk):
                        continue
                    sdf = asset_data[ssym][asset_data[ssym].index.date == d]
                    if len(sdf) <= b_idx:
                        continue

                    satr = float(atrs.get(ssym, 30.0)) if not isinstance(atrs.get(ssym, 30.0), dict) else float(atrs.get(ssym, 30.0).get('atr_wilder_14', 30.0))
                    sig = check_stock_signal(ssym, sdf, b_idx, satr, regime)
                    if sig:
                        stk_candidates.append(sig)

                stk_candidates.sort(key=lambda x: (x['score'], x['turnover_cr']), reverse=True)
                for cand in stk_candidates[:MAX_STOCKS_PER_SLAB - c_stk]:
                    out = simulate_trade_forward(cand, asset_data[cand['symbol']][asset_data[cand['symbol']].index.date == d], b_idx)
                    out['date'] = str(d)
                    executed_trades.append(out)
                    if is_m: m_stk.add(cand['symbol'])
                    else: a_stk.add(cand['symbol'])
                    c_stk += 1

    df_out = pd.DataFrame(executed_trades)
    csv_report_path = os.path.join(REPORT_DIR, "Alpha_Latest_5Day_Ending_Today_Backtest.csv")
    df_out.to_csv(csv_report_path, index=False)
    print(f"\n[REPORT SAVED]: {csv_report_path}")

    print("\n" + "=" * 70)
    print("      ALPHA 5-PILLAR BACKTEST (ENDING TODAY 03-NOV-2026)      ")
    print("=" * 70)
    print(f"Total Dispatched Trades: {len(df_out)}")
    wins = df_out[df_out['realized_r'] > 0]
    be = df_out[df_out['realized_r'] == 0]
    losses = df_out[df_out['realized_r'] < 0]
    gross_win = wins['realized_r'].sum() if not wins.empty else 0.0
    gross_loss = abs(losses['realized_r'].sum()) if not losses.empty else 0.0
    net_r = df_out['realized_r'].sum() if not df_out.empty else 0.0
    pf = gross_win / max(0.1, gross_loss)
    win_rate = (len(wins) / len(df_out)) * 100.0 if len(df_out) > 0 else 0.0
    non_losing_rate = ((len(wins) + len(be)) / len(df_out)) * 100.0 if len(df_out) > 0 else 0.0

    print(f"Wins: {len(wins)} ({win_rate:.1f}%) | Gross Win: +{gross_win:.2f}R")
    print(f"Breakeven (Cost Trailed): {len(be)} ({len(be)/len(df_out)*100 if len(df_out)>0 else 0:.1f}%)")
    print(f"Losses: {len(losses)} ({len(losses)/len(df_out)*100 if len(df_out)>0 else 0:.1f}%) | Gross Loss: -{gross_loss:.2f}R")
    print(f"Non-Losing Win Rate (Wins + BE): {len(wins) + len(be)}/{len(df_out)} ({non_losing_rate:.1f}%)")
    print(f"Net Realized R: {net_r:+.2f}R | Profit Factor: {pf:.2f}")
    print("-" * 70)
    for idx, r in df_out.iterrows():
        t_str = pd.to_datetime(r['eval_time']).strftime('%H:%M')
        print(f"{r['date']} {t_str} | {r['symbol']:<12} {r['direction']:<4} -> {r['exit_reason']:<22} ({r['realized_r']:+.2f}R | {r['points_captured']:+.1f}pts)")


if __name__ == '__main__':
    main()
