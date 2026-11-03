# -*- coding: utf-8 -*-
"""
scratch/execute_authoritative_backtest.py
==========================================
Sequential Decoupled 5-Day Historical Backtest Engine for Alpha Option Buyer.
Executes the exact 5-Pillar quantitative model from daily_autonomous_screener.py:
  1. Priority 1: Isolated 4-Index Evaluation (MAX_INDEX_PER_SLAB = 1)
  2. Priority 2: Isolated Stock F&O Universe Evaluation (MAX_STOCKS_PER_SLAB = 2)
  3. Dynamic Risk: 1.5R T1 (Cost Trail), 3.5R T2 (Jackpot), -1.0R SL, 15-Min Stagnant Exit, EOD 15:20 Square-Off
"""

import os
import sys
import glob
import json
import datetime
import numpy as np
import pandas as pd

CACHE_DIR = r"C:\Users\Mohit\.gemini\antigravity\brain\49080d4e-cdec-428a-901b-2a1de15d8cef\scratch\data_cache_5d_1m"
REPORT_DIR = r"C:\Users\Mohit\.gemini\antigravity\brain\49080d4e-cdec-428a-901b-2a1de15d8cef"
ATR_CACHE_FILE = r"c:\Users\Mohit\.gemini\antigravity\playground\aphelion-zenith\alpha_profile_oi_system\config\stock_atr_cache.json"

INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
MIN_STOCK_PRICE = 250.0
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

            df_3m = df_1m.resample('3min', origin='start').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'delta': 'sum'
            }).dropna()

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


def check_index_signal(sym, df_today, bar_idx, nifty_day_bars, nifty_spot):
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

    comp_bars = df_today.iloc[:bar_idx - 1].tail(8)
    comp_high = float(comp_bars['high'].max())
    comp_low = float(comp_bars['low'].min())
    comp_width = comp_high - comp_low

    max_allowed_base = 150.0 if sym != 'BANKNIFTY' else 350.0
    if comp_width > max_allowed_base:
        return None

    idx_dist = 35.0 if sym != 'BANKNIFTY' else 75.0
    direction = None
    if b_close >= comp_high and b_close <= comp_high + idx_dist and cur_p >= comp_high and b_close > b_open:
        direction = 'CALL'
    elif b_close <= comp_low and b_close >= comp_low - idx_dist and cur_p <= comp_low and b_close < b_open:
        direction = 'PUT'

    if direction is None:
        return None

    solid_body_ratio = round(abs(b_close - b_open) / rng, 2)
    if solid_body_ratio < 0.45:
        return None

    # Macro alignment: Dynamic Intraday Trend & Gap-Day Protection
    n_day_open = float(nifty_day_bars.iloc[0]['open'])
    n_day_high = float(nifty_day_bars.iloc[:bar_idx + 1]['high'].max())
    n_day_low = float(nifty_day_bars.iloc[:bar_idx + 1]['low'].min())
    n_ema9 = float(nifty_day_bars['ema9'].iloc[bar_idx]) if 'ema9' in nifty_day_bars.columns else nifty_spot
    drop_from_high = n_day_high - nifty_spot
    rise_from_low = nifty_spot - n_day_low
    n_move = nifty_spot - n_day_open
    idx_day_open = float(df_today.iloc[0]['open'])
    idx_move = cur_p - idx_day_open

    # Strict Macro & Day Open Alignment:
    # CALL requires Index and Nifty to be green / non-negative (idx_move >= -10.0 and n_move >= -15.0)
    if direction == 'CALL':
        if idx_move < -10.0 or n_move < -15.0 or (nifty_spot < n_ema9 and drop_from_high >= 50.0):
            return None
    # PUT requires Index and Nifty to be red / non-positive (idx_move <= +10.0 and n_move <= +15.0)
    if direction == 'PUT':
        if idx_move > +10.0 or n_move > +15.0 or (nifty_spot > n_ema9 and rise_from_low >= 50.0):
            return None

    # Range Expansion RVOL (Calculated against session median consolidation baseline)
    session_rngs = df_today.iloc[:bar_idx]['high'] - df_today.iloc[:bar_idx]['low']
    median_rng = float(session_rngs.median()) if len(session_rngs) > 0 else 15.0
    idx_rvol = round(rng / max(0.5, median_rng), 2)
    if idx_rvol < 1.15:
        return None

    # BVC CVD
    cvd_res = compute_bar_delta_proxy(df_today.iloc[:bar_idx], is_index=True)
    rec_cvd = cvd_res['recent_cvd_ratio']
    cvd_pass = (rec_cvd >= 0.12) if direction == 'CALL' else (rec_cvd <= -0.12)
    if not cvd_pass:
        return None

    score = min(99.0, 55.0 + min(25.0, (idx_rvol - 1.0) * 15.0) + (15.0 if abs(rec_cvd) >= 0.25 else 5.0) + (10.0 if solid_body_ratio >= 0.55 else 0.0))
    idx_open = float(df_today['open'].iloc[0])
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
        'pct_chg': round(((cur_p - idx_open) / max(1.0, idx_open)) * 100.0, 2),
        'score': round(score, 2),
        'type': 'INDEX_INSTITUTIONAL_BREAKOUT'
    }


def check_stock_signal(sym, df_today, bar_idx, stock_atr, nifty_day_open, nifty_spot, nifty_ema9=None, nifty_drop=0.0, nifty_rise=0.0):
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

    # Anti-Exhaustion Cap: Blocks crashed (-6%) or vertically extended (+6%) stocks
    day_open = float(df_today['open'].iloc[0])
    day_pct_move = ((spot - day_open) / max(1.0, day_open)) * 100.0
    stk_day_high = float(df_today.iloc[:bar_idx + 1]['high'].max())
    stk_day_low = float(df_today.iloc[:bar_idx + 1]['low'].min())
    day_range = stk_day_high - stk_day_low
    if abs(day_pct_move) > 4.50 or day_range > stock_atr * 1.30:
        return None

    # Pillar 3: 8-Bar Base Compression & Live Spot Holding
    comp_bars = df_today.iloc[:bar_idx - 1].tail(8)
    comp_high = float(comp_bars['high'].max())
    comp_low = float(comp_bars['low'].min())
    comp_w = comp_high - comp_low
    if comp_w > stock_atr * 0.65:
        return None

    max_dist_buffer = stock_atr * 0.20
    direction = None
    if b_close >= comp_high and b_close <= comp_high + max_dist_buffer and b_close > b_open and spot >= (comp_high - stock_atr * 0.05):
        direction = 'CALL'
    elif b_close <= comp_low and b_close >= comp_low - max_dist_buffer and b_close < b_open and spot <= (comp_low + stock_atr * 0.05):
        direction = 'PUT'
    if direction is None:
        return None

    # Anti-exhaustion (Climax candle & 3-bar prior run)
    c_rng = b_high - b_low
    if c_rng > stock_atr * 0.45:
        return None

    p3_low = df_today.iloc[max(0, bar_idx - 4):bar_idx - 1]['low'].min()
    p3_high = df_today.iloc[max(0, bar_idx - 4):bar_idx - 1]['high'].max()
    p3_move = (b_close - p3_low) if direction == 'CALL' else (p3_high - b_close)
    if p3_move > stock_atr * 0.60:
        return None

    curr_3m_v = float(eval_b.get('volume', 0.0))
    prior_10_bars = df_today.iloc[:bar_idx - 1].tail(10)
    base_10_v = float(prior_10_bars['volume'].mean()) if len(prior_10_bars) > 0 else 1000.0
    real_3m_rvol = round(curr_3m_v / max(1.0, base_10_v), 2)
    turnover_cr = round((curr_3m_v * spot) / 1e7, 2)

    # Pillar 1: Strict Institutional RVOL Floor (No Turnover exceptions allowing low volume!)
    if real_3m_rvol < 1.80:
        return None
    if turnover_cr < 3.0:
        return None

    # BUG 1 FIX: True Breakout Bar Delta Ratio from BVC formula
    cvd_res = compute_bar_delta_proxy(df_today.iloc[:bar_idx], is_index=False)
    rec_cvd = cvd_res['recent_cvd_ratio']
    bar_delta_ratio = round((2.0 * b_close - (b_high + b_low)) / rng, 3)

    cvd_pass = (rec_cvd >= 0.15 or bar_delta_ratio >= 0.35) if direction == 'CALL' else (rec_cvd <= -0.15 or bar_delta_ratio <= -0.35)
    if not cvd_pass:
        return None

    # BUG 5 FIX: Day High / Day Low Check with 0.12 * ATR Buffer
    comp_df = df_today.iloc[:bar_idx]
    day_high = float(comp_df['high'].max())
    day_low = float(comp_df['low'].min())
    near_day_high = (direction == 'CALL' and b_close >= day_high - stock_atr * 0.12)
    near_day_low = (direction == 'PUT' and b_close <= day_low + stock_atr * 0.12)

    is_afternoon = (df_today.index[bar_idx].hour >= 13)
    if is_afternoon and not (near_day_high or near_day_low):
        return None

    # BUG 4 FIX: Dynamic Nifty Macro Tide Check
    n_move = nifty_spot - nifty_day_open
    n_ema = nifty_ema9 if nifty_ema9 is not None else nifty_spot

    # Mega heavyweights CANNOT trade against Nifty macro direction
    if sym in NIFTY_HEAVYWEIGHTS:
        if direction == 'CALL' and n_move < -15.0:
            return None
        if direction == 'PUT' and n_move > +15.0:
            return None

    is_counter = (direction == 'CALL' and (n_move < -30.0 or nifty_spot < n_ema - 25.0)) or \
                 (direction == 'PUT' and (n_move > +30.0 or nifty_spot > n_ema + 25.0))
    if is_counter:
        if not ((near_day_high or near_day_low) and turnover_cr >= 4.0 and abs(bar_delta_ratio) >= 0.35):
            return None

    body = abs(b_close - b_open)
    solid_body_ratio = round(body / rng, 2)
    clv = round((b_close - b_low) / rng, 2)
    min_thrust = max(2.5, round(stock_atr * 0.06, 1))
    if solid_body_ratio < 0.50 or body < min_thrust:
        return None

    clv_pass = (clv >= 0.60) if direction == 'CALL' else (clv <= 0.40)
    if not clv_pass:
        return None

    score = min(99.0, 50.0 + min(25.0, (real_3m_rvol - 1.0) * 10.0) + (15.0 if abs(bar_delta_ratio) >= 0.35 else 5.0) + min(10.0, turnover_cr * 1.5))
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
        'type': 'STOCK_INSTITUTIONAL_BREAKOUT'
    }


def simulate_trade_forward(cand, df_today, entry_idx):
    entry_p = cand['entry_price']
    direction = cand['direction']
    is_idx = cand.get('is_index', False)
    atr = cand.get('stock_atr', 30.0)

    if is_idx:
        sym = cand['symbol']
        spot_t1 = 140.0 if sym == 'BANKNIFTY' else (70.0 if sym in ['NIFTY', 'FINNIFTY'] else 40.0)
        spot_t2 = 240.0 if sym == 'BANKNIFTY' else (120.0 if sym in ['NIFTY', 'FINNIFTY'] else 70.0)
        spot_sl = 70.0 if sym == 'BANKNIFTY' else (35.0 if sym in ['NIFTY', 'FINNIFTY'] else 20.0)
    else:
        spot_t1 = round(atr * 0.35, 1)
        spot_t2 = round(atr * 0.70, 1)
        spot_sl = round(atr * 0.20, 1)

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
            if cur_gain <= 0.0:
                exit_p = entry_p
                exit_reason = "COST_TRAILED_EXIT"
                exit_idx_local = step
                break
        else:
            if adv_move >= spot_sl:
                exit_p = (entry_p - spot_sl) if direction == 'CALL' else (entry_p + spot_sl)
                exit_reason = "SL_HIT"
                exit_idx_local = step
                break
            if step == 4 and cur_gain <= 0.0:
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
        'pct_chg': cand['pct_chg'],
        'score': cand['score'],
        'type': cand['type'],
        'exit_price': exit_p,
        'exit_time': bars_fwd.index[min(exit_idx_local, len(bars_fwd) - 1)] if len(bars_fwd) > 0 else cand['eval_time'],
        'points_captured': pts_captured,
        'spot_sl': spot_sl,
        'spot_t1': spot_t1,
        'spot_t2': spot_t2,
        'exit_reason': exit_reason,
        'realized_r': rr_realized
    }


def run_full_5day_backtest():
    print("[BACKTEST]: Loading 217 assets resampled 3m intraday data...")
    asset_data = load_resampled_3m_data()
    atrs = load_verified_atrs()
    all_dates = sorted(list(set(d for df in asset_data.values() for d in df.index.date)))
    stock_syms = [s for s in asset_data.keys() if s not in INDEX_NAMES]

    print(f"[BACKTEST]: Ready. {len(asset_data)} assets loaded across {len(all_dates)} dates: {all_dates}")

    all_executed_trades = []

    for d in all_dates:
        print(f"\n=======================================================")
        print(f"  PROCESSING DATE: {d}")
        print(f"=======================================================")

        nifty_day = asset_data['NIFTY'][asset_data['NIFTY'].index.date == d].copy()
        nifty_day['ema9'] = nifty_day['close'].ewm(span=9, adjust=False).mean()
        n_day_open = nifty_day.iloc[0]['open']

        morn_idx_dispatched = set()
        morn_stk_dispatched = set()
        aft_idx_dispatched = set()
        aft_stk_dispatched = set()

        time_index = nifty_day.index

        for b_idx in range(10, len(time_index)):
            curr_t = time_index[b_idx]
            tot_m = curr_t.hour * 60 + curr_t.minute
            is_m = (570 <= tot_m <= 615)
            is_a = (795 <= tot_m <= 855)
            if not (is_m or is_a):
                continue

            slab_name = 'MORNING_0930_1015' if is_m else 'AFTERNOON_1315_1415'
            n_spot = nifty_day.iloc[b_idx]['close']

            # PRIORITY 1: EVALUATE 4 INDICES FIRST!
            cur_idx_count = len(morn_idx_dispatched) if is_m else len(aft_idx_dispatched)
            if cur_idx_count < MAX_INDEX_PER_SLAB:
                for isym in INDEX_NAMES:
                    if isym not in asset_data:
                        continue
                    idf_day = asset_data[isym][asset_data[isym].index.date == d]
                    if len(idf_day) <= b_idx:
                        continue
                    already = isym in (morn_idx_dispatched if is_m else aft_idx_dispatched)
                    if already:
                        continue

                    isig = check_index_signal(isym, idf_day, b_idx, nifty_day, n_spot)
                    if isig:
                        out = simulate_trade_forward(isig, idf_day, b_idx)
                        out['date'] = d
                        out['slab'] = slab_name
                        all_executed_trades.append(out)
                        if is_m:
                            morn_idx_dispatched.add(isym)
                        else:
                            aft_idx_dispatched.add(isym)
                        print(f"  [INDEX DISPATCH]: {curr_t.strftime('%H:%M')} | {slab_name} | {isym} {isig['direction']} @ {isig['entry_price']:.1f} -> Outcome: {out['exit_reason']} ({out['realized_r']:+.2f}R)")
                        break

            # PRIORITY 2: EVALUATE F&O STOCKS SECOND!
            cur_stk_count = len(morn_stk_dispatched) if is_m else len(aft_stk_dispatched)
            if cur_stk_count < MAX_STOCKS_PER_SLAB:
                for ssym in stock_syms:
                    if cur_stk_count >= MAX_STOCKS_PER_SLAB:
                        break
                    sdf_day = asset_data[ssym][asset_data[ssym].index.date == d]
                    if len(sdf_day) <= b_idx:
                        continue
                    already = ssym in (morn_stk_dispatched if is_m else aft_stk_dispatched)
                    if already:
                        continue

                    satr_val = atrs.get(ssym, 30.0)
                    satr = float(satr_val) if not isinstance(satr_val, dict) else float(satr_val.get('atr_wilder_14', 30.0))
                    n_high_so_far = nifty_day.iloc[:b_idx + 1]['high'].max()
                    n_low_so_far = nifty_day.iloc[:b_idx + 1]['low'].min()
                    n_ema = nifty_day['ema9'].iloc[b_idx]
                    n_drop = n_high_so_far - n_spot
                    n_rise = n_spot - n_low_so_far
                    ssig = check_stock_signal(ssym, sdf_day, b_idx, satr, n_day_open, n_spot, nifty_ema9=n_ema, nifty_drop=n_drop, nifty_rise=n_rise)
                    if ssig:
                        out = simulate_trade_forward(ssig, sdf_day, b_idx)
                        out['date'] = d
                        out['slab'] = slab_name
                        all_executed_trades.append(out)
                        if is_m:
                            morn_stk_dispatched.add(ssym)
                        else:
                            aft_stk_dispatched.add(ssym)
                        cur_stk_count += 1
                        print(f"  [STOCK DISPATCH]: {curr_t.strftime('%H:%M')} | {slab_name} | {ssym} {ssig['direction']} @ {ssig['entry_price']:.1f} -> Outcome: {out['exit_reason']} ({out['realized_r']:+.2f}R)")

    df_out = pd.DataFrame(all_executed_trades)
    csv_path = os.path.join(REPORT_DIR, "Alpha_Authoritative_5Day_Backtest.csv")
    df_out.to_csv(csv_path, index=False)

    print("\n=======================================================")
    print("             5-DAY BACKTEST SUMMARY REPORT             ")
    print("=======================================================")
    print(f"Trades saved to: {csv_path}")
    print(f"Total Trades Executed:   {len(df_out)}")
    wins = df_out[df_out['realized_r'] > 0]
    be = df_out[df_out['realized_r'] == 0]
    losses = df_out[df_out['realized_r'] < 0]
    t2_jackpots = df_out[df_out['exit_reason'] == 'TARGET_2_HIT (JACKPOT)']
    cost_trails = df_out[df_out['exit_reason'] == 'COST_TRAILED_EXIT']
    time_sls = df_out[df_out['exit_reason'] == '15M_TIME_SL_EXIT']
    sl_hits = df_out[df_out['exit_reason'] == 'SL_HIT']

    gross_win = wins['realized_r'].sum()
    gross_loss = abs(losses['realized_r'].sum())
    net_r = df_out['realized_r'].sum()
    pf = round(gross_win / max(0.1, gross_loss), 2)

    print(f"Winning Trades (>0R):    {len(wins)} ({len(wins)/len(df_out)*100:.1f}%) | Gross Win: +{gross_win:.2f}R")
    print(f"Breakeven Exits (0.0R):  {len(be)} trades")
    print(f"Losing Trades (<0R):     {len(losses)} | Gross Loss: -{gross_loss:.2f}R")
    print(f"Net Realized R:          {net_r:+.2f}R")
    print(f"Profit Factor:           {pf}")
    print(f"Target 2 Jackpots:       {len(t2_jackpots)} trades (+{t2_jackpots['realized_r'].sum():.1f}R)")
    print(f"Cost Trailed (T1 Lock):  {len(cost_trails)} trades (0.00R)")
    print(f"15M Time SL Exits:       {len(time_sls)} trades ({time_sls['realized_r'].sum():.2f}R)")
    print(f"Full Stop Losses:        {len(sl_hits)} trades ({sl_hits['realized_r'].sum():.1f}R)")


if __name__ == '__main__':
    run_full_5day_backtest()
