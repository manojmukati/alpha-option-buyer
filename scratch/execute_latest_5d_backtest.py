import os
import glob
import json
import numpy as np
import pandas as pd

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPORT_DIR, "data_cache_latest_5d_1m")
CONFIG_DIR = os.path.join(os.path.dirname(REPORT_DIR), "alpha_profile_oi_system", "config")

INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
MIN_STOCK_PRICE = 650.0
MIN_STOCK_ATR = 25.0

MAX_BASE_WIDTH_ATR = 0.65
MAX_BREAKOUT_BUFFER_ATR = 0.20
MAX_CLIMAX_BAR_ATR = 0.65
MAX_PRIOR_3BAR_ATR = 0.80
MAX_DAY_RANGE_ATR = 2.20
MAX_DAY_PCT_MOVE = 4.50

MAX_INDEX_PER_SLAB = 2
MAX_STOCKS_PER_SLAB = 2

def load_resampled_3m_data():
    files = glob.glob(os.path.join(CACHE_DIR, "*.parquet"))
    asset_dict = {}
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "").replace("^", "").strip()
        try:
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index)
            if not isinstance(df.index, pd.DatetimeIndex):
                continue
            df_3m = df.resample('3min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            if len(df_3m) > 10:
                asset_dict[sym] = df_3m
        except Exception:
            continue
    return asset_dict

def load_verified_atrs():
    cfg_file = os.path.join(CONFIG_DIR, "stock_atr_cache.json")
    if os.path.exists(cfg_file):
        with open(cfg_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            res = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    res[k] = float(v.get('atr_wilder_14', 30.0))
                else:
                    res[k] = float(v)
            return res
    return {}

def compute_bar_delta_proxy(df_bars, is_index=False):
    if df_bars is None or len(df_bars) < 2:
        return {'valid': False, 'recent_cvd_ratio': 0.0}
    closes = df_bars['close'].values.astype(float)
    highs = df_bars['high'].values.astype(float)
    lows = df_bars['low'].values.astype(float)
    vols = df_bars['volume'].values.astype(float)
    rngs = np.maximum(0.01, highs - lows)

    if is_index:
        norm_deltas = (2.0 * closes - (highs + lows)) / rngs
        rec_n = min(3, len(df_bars))
        recent_cvd_ratio = round(float(np.mean(norm_deltas[-rec_n:])), 3)
    else:
        tot_vol = float(np.sum(vols))
        if tot_vol <= 0:
            return {'valid': False, 'recent_cvd_ratio': 0.0}
        deltas = vols * ((2.0 * closes - (highs + lows)) / rngs)
        rec_n = min(3, len(df_bars))
        rec_vol = float(np.sum(vols[-rec_n:]))
        rec_delta = float(np.sum(deltas[-rec_n:]))
        recent_cvd_ratio = round(rec_delta / max(1.0, rec_vol), 3)

    return {'valid': True, 'recent_cvd_ratio': recent_cvd_ratio}

def get_nifty_regime(nifty_day, b_idx):
    if b_idx < 2:
        return {'allow_call': True, 'allow_put': True, 'reason': 'START', 'move': 0.0}
    eval_b = nifty_day.iloc[b_idx - 1]
    live_b = nifty_day.iloc[b_idx]
    n_spot = float(live_b['close'])
    ema9 = float(nifty_day['ema9'].iloc[b_idx])
    day_open = float(nifty_day['open'].iloc[0])
    move_from_open = n_spot - day_open

    n_last_bar_rng = float(eval_b['high'] - eval_b['low'])
    if n_last_bar_rng > 65.0:
        return {'allow_call': False, 'allow_put': False, 'reason': 'NIFTY_CLIMAX_EXHAUSTION', 'move': move_from_open}

    recent_rng = nifty_day['high'].iloc[max(0, b_idx - 5):b_idx + 1].max() - nifty_day['low'].iloc[max(0, b_idx - 5):b_idx + 1].min()
    ema_dist = abs(n_spot - ema9)
    if recent_rng < 22.0 and ema_dist < 12.0:
        return {'allow_call': False, 'allow_put': False, 'reason': 'CHOP_TRAP', 'move': move_from_open}

    allow_call = (move_from_open >= -20.0) and (n_spot >= ema9 - 15.0)
    allow_put = (move_from_open <= +20.0) and (n_spot <= ema9 + 15.0)
    return {'allow_call': allow_call, 'allow_put': allow_put, 'reason': 'NORMAL', 'move': move_from_open}

def check_stock_signal_elite(sym, df_today, bar_idx, stock_atr, regime):
    if bar_idx < 10: return None
    eval_b = df_today.iloc[bar_idx - 1]
    live_b = df_today.iloc[bar_idx]
    spot = float(live_b['close'])
    if spot < MIN_STOCK_PRICE or stock_atr < MIN_STOCK_ATR: return None

    b_open = float(eval_b['open'])
    b_high = float(eval_b['high'])
    b_low = float(eval_b['low'])
    b_close = float(eval_b['close'])
    rng = max(0.01, b_high - b_low)

    # 1. Day Range & Percent Move
    day_open = float(df_today['open'].iloc[0])
    day_pct_move = ((spot - day_open) / max(1.0, day_open)) * 100.0
    stk_day_high = float(df_today.iloc[:bar_idx + 1]['high'].max())
    stk_day_low = float(df_today.iloc[:bar_idx + 1]['low'].min())
    day_range = stk_day_high - stk_day_low

    if abs(day_pct_move) > MAX_DAY_PCT_MOVE or day_range > stock_atr * MAX_DAY_RANGE_ATR:
        return None

    # 2. 8-Bar Compression Base & Directional Breakout
    comp_bars = df_today.iloc[:bar_idx - 1].tail(8)
    comp_high = float(comp_bars['high'].max())
    comp_low = float(comp_bars['low'].min())
    comp_w = comp_high - comp_low
    if comp_w > stock_atr * MAX_BASE_WIDTH_ATR: return None

    max_dist_buffer = stock_atr * MAX_BREAKOUT_BUFFER_ATR
    direction = None
    if b_close >= comp_high and b_close <= comp_high + max_dist_buffer and b_close > b_open and spot >= (comp_high - stock_atr * 0.04):
        direction = 'CALL'
    elif b_close <= comp_low and b_close >= comp_low - max_dist_buffer and b_close < b_open and spot <= (comp_low + stock_atr * 0.04):
        direction = 'PUT'

    if direction is None: return None

    # 3. Macro Alignment: Anti-Counter Trend
    if direction == 'CALL' and not regime['allow_call']: return None
    if direction == 'PUT' and not regime['allow_put']: return None
    # If Nifty is up from open (> +5 pts), DO NOT SHORT STOCKS (kills Maruti PUT on 02-Nov!)
    if direction == 'PUT' and regime['move'] > 5.0: return None
    # If Nifty is down from open (< -5 pts), DO NOT BUY CALLS!
    if direction == 'CALL' and regime['move'] < -5.0: return None

    # 4. Anti-Climax Expansion Cap
    if rng > stock_atr * MAX_CLIMAX_BAR_ATR: return None

    # 5. Prior 3-Bar Runup
    p3_low = df_today.iloc[max(0, bar_idx - 4):bar_idx - 1]['low'].min()
    p3_high = df_today.iloc[max(0, bar_idx - 4):bar_idx - 1]['high'].max()
    p3_move = (b_close - p3_low) if direction == 'CALL' else (p3_high - b_close)
    if p3_move > stock_atr * MAX_PRIOR_3BAR_ATR: return None

    # 6. Pro Institutional Turnover Floor: Minimum Rs 6.0 Cr (eliminates low-volume drift like KEI, POLICYBZR, OFSS)
    curr_v = float(eval_b.get('volume', 0.0))
    prior_10_bars = df_today.iloc[:bar_idx - 1].tail(10)
    base_10_v = float(prior_10_bars['volume'].mean()) if len(prior_10_bars) > 0 else 1000.0
    real_3m_rvol = round(curr_v / max(1.0, base_10_v), 2)
    turnover_cr = round((curr_v * spot) / 1e7, 2)

    if real_3m_rvol < 1.80 or turnover_cr < 6.0:
        return None

    # 7. BVC CVD Order Flow & Bar Delta
    cvd_res = compute_bar_delta_proxy(df_today.iloc[:bar_idx], is_index=False)
    rec_cvd = cvd_res['recent_cvd_ratio']
    bar_delta_ratio = round((2.0 * b_close - (b_high + b_low)) / rng, 3)

    if direction == 'CALL':
        cvd_pass = (rec_cvd >= 0.15 or bar_delta_ratio >= 0.35) and (rec_cvd >= 0.0)
    else:
        cvd_pass = (rec_cvd <= -0.15 or bar_delta_ratio <= -0.35) and (rec_cvd <= 0.0)

    if not cvd_pass: return None

    # 8. Day High / Low Frontier Check (Within 0.15 ATR)
    near_hl = (direction == 'CALL' and b_close >= stk_day_high - stock_atr * 0.15) or \
              (direction == 'PUT' and b_close <= stk_day_low + stock_atr * 0.15)
    if not near_hl: return None

    body = abs(b_close - b_open)
    solid_body_ratio = round(body / rng, 2)
    clv = round((b_close - b_low) / rng, 2)
    if solid_body_ratio < 0.48: return None

    clv_pass = (clv >= 0.58) if direction == 'CALL' else (clv <= 0.42)
    if not clv_pass: return None

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
        'type': 'PURE_STOCK_CALL' if direction == 'CALL' else 'PURE_STOCK_PUT',
        'base_high': comp_high,
        'base_low': comp_low
    }

def check_index_signal_elite(sym, df_today, bar_idx, regime):
    if bar_idx < 10: return None
    eval_b = df_today.iloc[bar_idx - 1]
    live_b = df_today.iloc[bar_idx]
    cur_p = float(live_b['close'])
    b_open = float(eval_b['open'])
    b_high = float(eval_b['high'])
    b_low = float(eval_b['low'])
    b_close = float(eval_b['close'])
    rng = max(0.01, b_high - b_low)

    max_bar_rng = 65.0 if sym != 'BANKNIFTY' else 150.0
    if rng > max_bar_rng: return None

    comp_bars = df_today.iloc[:bar_idx - 1].tail(8)
    comp_high = float(comp_bars['high'].max())
    comp_low = float(comp_bars['low'].min())
    comp_width = comp_high - comp_low

    max_allowed_base = 120.0 if sym != 'BANKNIFTY' else 350.0
    if comp_width > max_allowed_base: return None

    idx_dist = 35.0 if sym != 'BANKNIFTY' else 80.0
    direction = None
    if b_close >= comp_high and b_close <= comp_high + idx_dist and cur_p >= comp_high and b_close > b_open:
        direction = 'CALL'
    elif b_close <= comp_low and b_close >= comp_low - idx_dist and cur_p <= comp_low and b_close < b_open:
        direction = 'PUT'

    if direction is None: return None

    idx_day_open = float(df_today['open'].iloc[0])
    if direction == 'CALL':
        if not regime['allow_call'] or cur_p < idx_day_open: return None
    elif direction == 'PUT':
        if not regime['allow_put'] or cur_p > idx_day_open: return None

    solid_body_ratio = round(abs(b_close - b_open) / rng, 2)
    if solid_body_ratio < 0.45: return None

    cvd_res = compute_bar_delta_proxy(df_today.iloc[:bar_idx], is_index=True)
    rec_cvd = cvd_res['recent_cvd_ratio']
    cvd_pass = (rec_cvd >= 0.12) if direction == 'CALL' else (rec_cvd <= -0.12)
    if not cvd_pass: return None

    session_rngs = df_today.iloc[:bar_idx]['high'] - df_today.iloc[:bar_idx]['low']
    median_rng = float(session_rngs.median()) if len(session_rngs) > 0 else 15.0
    idx_rvol = round(rng / max(0.5, median_rng), 2)
    if idx_rvol < 1.15: return None

    # Anti-Climax Overextended Index Check: If Index already moved > 1.2% from open, do not enter late
    idx_pct_move = ((cur_p - idx_day_open) / max(1.0, idx_day_open)) * 100.0
    if abs(idx_pct_move) > 1.10:
        return None  # Prevents late entries when index already exhausted!

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
        'pct_chg': round(idx_pct_move, 2),
        'score': round(score, 2),
        'base_high': comp_high,
        'base_low': comp_low,
        'type': 'INDEX_INSTITUTIONAL_BREAKOUT'
    }

# Forward Trade Simulation with 30-Minute Stagnant Scratch Rule for Options
def simulate_trade_elite(cand, df_today, entry_idx):
    direction = cand['direction']
    entry_p = float(cand['entry_price'])
    is_idx = cand.get('is_index', False)
    atr = cand.get('stock_atr', 30.0)

    if is_idx:
        sym = cand['symbol']
        spot_t1 = 180.0 if sym == 'BANKNIFTY' else 100.0
        spot_t2 = 300.0 if sym == 'BANKNIFTY' else 150.0
        spot_sl = 75.0 if sym == 'BANKNIFTY' else 40.0
    else:
        spot_t1 = max(30.0, round(atr * 0.75, 1))
        spot_t2 = max(60.0, round(atr * 1.50, 1))
        spot_sl = max(round(atr * 0.20, 1), min(round(spot_t1 * 0.45, 1), 25.0))

    bars_fwd = df_today.iloc[entry_idx + 1:]
    exit_p = entry_p
    exit_reason = "EOD_EXIT"
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
                break
            lock_pts = spot_t1 * 0.40
            locked_sl = (entry_p + lock_pts) if direction == 'CALL' else (entry_p - lock_pts)
            if (direction == 'CALL' and b_low <= locked_sl) or (direction == 'PUT' and b_high >= locked_sl):
                exit_p = locked_sl
                exit_reason = "T1_PROFIT_LOCKED_EXIT"
                break
        else:
            if adv_move >= spot_sl:
                exit_p = (entry_p - spot_sl) if direction == 'CALL' else (entry_p + spot_sl)
                exit_reason = "SL_HIT"
                break
            # 15M Time SL: If position is flat or in loss after 15m
            if step == 4 and (not t1_hit) and cur_gain <= 0.0:
                exit_p = b_close
                exit_reason = "15M_TIME_SL_EXIT"
                break
            # 30M Stagnant Option Churn Scratch: If position has not made any progress after 30 mins (10 bars)
            if step == 9 and (not t1_hit) and cur_gain < (spot_t1 * 0.25):
                exit_p = b_close
                exit_reason = "30M_STAGNANT_SCRATCH"
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
        'spot_t1': spot_t1,
        'spot_t2': spot_t2,
        'exit_reason': exit_reason,
        'realized_r': rr_realized
    }

# Run the 5-day Backtest
print("Loading latest resampled 3m data...")
asset_data = load_resampled_3m_data()
atrs = load_verified_atrs()
all_dates = sorted(list(set(d for df in asset_data.values() for d in df.index.date)))
print(f"Loaded {len(asset_data)} active assets. Backtest dates: {[str(d) for d in all_dates]}")

stock_syms = [s for s in asset_data.keys() if s not in INDEX_NAMES]
executed_trades = []

for d in all_dates:
    nifty_day = asset_data['NIFTY'][asset_data['NIFTY'].index.date == d].copy()
    if nifty_day.empty: continue
    nifty_day['ema9'] = nifty_day['close'].ewm(span=9, adjust=False).mean()

    m_idx, m_stk = set(), set()
    a_idx, a_stk = set(), set()
    time_index = nifty_day.index

    for b_idx in range(10, len(time_index)):
        curr_t = time_index[b_idx]
        tot_m = curr_t.hour * 60 + curr_t.minute
        is_m = (570 <= tot_m <= 615)
        is_a = (795 <= tot_m <= 855)
        if not (is_m or is_a): continue

        regime = get_nifty_regime(nifty_day, b_idx)
        if not (regime['allow_call'] or regime['allow_put']): continue

        # 1. INDEX
        c_idx = len(m_idx) if is_m else len(a_idx)
        if c_idx < 2:
            for isym in INDEX_NAMES:
                if isym not in asset_data or isym in (m_idx if is_m else a_idx): continue
                idf = asset_data[isym][asset_data[isym].index.date == d]
                if len(idf) <= b_idx: continue
                sig = check_index_signal_elite(isym, idf, b_idx, regime)
                if sig:
                    out = simulate_trade_elite(sig, idf, b_idx)
                    out['date'] = str(d)
                    executed_trades.append(out)
                    if is_m: m_idx.add(isym)
                    else: a_idx.add(isym)
                    if len(m_idx if is_m else a_idx) >= 2: break

        # 2. STOCKS
        c_stk = len(m_stk) if is_m else len(a_stk)
        if c_stk < 2:
            stk_candidates = []
            for ssym in stock_syms:
                if ssym in (m_stk if is_m else a_stk): continue
                sdf = asset_data[ssym][asset_data[ssym].index.date == d]
                if len(sdf) <= b_idx: continue
                satr = float(atrs.get(ssym, 30.0))
                sig = check_stock_signal_elite(ssym, sdf, b_idx, satr, regime)
                if sig:
                    stk_candidates.append(sig)

            stk_candidates.sort(key=lambda x: (x['score'], x['turnover_cr']), reverse=True)
            for cand in stk_candidates[:2 - c_stk]:
                out = simulate_trade_elite(cand, asset_data[cand['symbol']][asset_data[cand['symbol']].index.date == d], b_idx)
                out['date'] = str(d)
                executed_trades.append(out)
                if is_m: m_stk.add(cand['symbol'])
                else: a_stk.add(cand['symbol'])
                c_stk += 1

df_elite = pd.DataFrame(executed_trades)
print("=" * 80)
print("ELITE QUANT BACKTEST (ZERO CHOP & HIGH-VELOCITY TRADES)")
print("=" * 80)
print(f"Total Dispatched Trades: {len(df_elite)}")
wins = df_elite[df_elite['realized_r'] > 0]
scratches = df_elite[(df_elite['realized_r'] <= 0) & (df_elite['realized_r'] > -0.80)]
full_sl = df_elite[df_elite['realized_r'] <= -0.80]

print(f"Wins: {len(wins)} ({len(wins)/max(1, len(df_elite))*100:.1f}%)")
print(f"Scratches: {len(scratches)}")
print(f"Full Stoploss: {len(full_sl)}")
gross_profit = wins['realized_r'].sum()
gross_loss = abs(df_elite[df_elite['realized_r'] < 0]['realized_r'].sum())
net_profit = gross_profit - gross_loss
pf = gross_profit / max(0.01, gross_loss)
print(f"Gross Profit: +{gross_profit:.2f}R | Gross Loss: -{gross_loss:.2f}R | Net Profit: +{net_profit:.2f}R | Profit Factor: {pf:.2f}")

print("\n--- ALL EXECUTED TRADES ---")
for idx, r in df_elite.iterrows():
    t_str = str(r['eval_time'])[11:16]
    print(f"{r['date']} {t_str} | {r['symbol']:<12} | {r['direction']:<4} | En: {r['entry_price']:>8.2f} | Ex: {r['exit_price']:>8.2f} | Pts: {r['points_captured']:>+7.2f} | R: {r['realized_r']:>+5.2f}R | {r['exit_reason']}")
