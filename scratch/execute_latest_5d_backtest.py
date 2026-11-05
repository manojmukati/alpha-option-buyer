import os
import sys
import glob
import json
import numpy as np
import pandas as pd

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPORT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha_profile_oi_system.modules.fno_universe import determine_step_size

CACHE_DIR = os.path.join(REPORT_DIR, "data_cache_latest_5d_1m")
CONFIG_DIR = os.path.join(REPO_ROOT, "alpha_profile_oi_system", "config")

INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
MIN_STOCK_PRICE = 650.0
MIN_STOCK_ATR = 18.0

MAX_BASE_WIDTH_ATR = 0.65
MAX_BREAKOUT_BUFFER_ATR = 0.20
MAX_CLIMAX_BAR_ATR = 0.65
MAX_PRIOR_3BAR_ATR = 0.80
MAX_DAY_RANGE_ATR = 2.20
MAX_DAY_PCT_MOVE = 4.50

# Active position caps matching daily_autonomous_screener.py
MAX_ACTIVE_STOCKS = 2
MAX_ACTIVE_BROAD_INDEX = 1   # NIFTY, MIDCPNIFTY, FINNIFTY
MAX_ACTIVE_BANK_INDEX = 1    # BANKNIFTY

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
        try:
            with open(cfg_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                res = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        res[k] = float(v.get('atr_wilder_14', 30.0))
                    else:
                        res[k] = float(v)
                return res
        except Exception:
            return {}
    return {}

def load_dynamic_lot_sizes():
    lot_map = {
        'NIFTY': 25, 'BANKNIFTY': 15, 'FINNIFTY': 25, 'MIDCPNIFTY': 50
    }
    bhav_files = glob.glob(os.path.join(REPO_ROOT, 'alpha_profile_oi_system', 'bhavcopy_cache', '*.parquet'))
    if bhav_files:
        latest_bhav = sorted(bhav_files)[-1]
        try:
            df = pd.read_parquet(latest_bhav)
            if 'TckrSymb' in df.columns and 'NewBrdLotQty' in df.columns:
                for _, row in df.iterrows():
                    sym = str(row['TckrSymb']).strip().upper()
                    if sym not in lot_map:
                        lot_map[sym] = int(row['NewBrdLotQty'])
        except Exception:
            pass_placeholder = None
    return lot_map

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
        return {'allow_call': True, 'allow_put': True, 'allow_index': True, 'reason': 'START', 'move': 0.0}
    eval_b = nifty_day.iloc[b_idx - 1]
    live_b = nifty_day.iloc[b_idx]
    n_spot = float(live_b['close'])
    ema9 = float(nifty_day['ema9'].iloc[b_idx])
    day_open = float(nifty_day['open'].iloc[0])
    move_from_open = n_spot - day_open

    n_last_bar_rng = float(eval_b['high'] - eval_b['low'])
    if n_last_bar_rng > 65.0:
        return {'allow_call': False, 'allow_put': False, 'allow_index': False, 'reason': 'NIFTY_CLIMAX_EXHAUSTION', 'move': move_from_open}

    recent_rng = nifty_day['high'].iloc[max(0, b_idx - 5):b_idx + 1].max() - nifty_day['low'].iloc[max(0, b_idx - 5):b_idx + 1].min()
    ema_dist = abs(n_spot - ema9)
    allow_index = True
    if recent_rng < 22.0 and ema_dist < 12.0:
        allow_index = False
        allow_call = True
        allow_put = True
        reason = 'CHOP_TRAP_INDEX_FROZEN'
    else:
        allow_call = (move_from_open >= -35.0) and (n_spot >= ema9 - 25.0)
        allow_put = (move_from_open <= +35.0) and (n_spot <= ema9 + 25.0)
        reason = 'NORMAL'

    return {'allow_call': allow_call, 'allow_put': allow_put, 'allow_index': allow_index, 'reason': reason, 'move': move_from_open}

def check_stock_signal_pro(sym, df_today, bar_idx, stock_atr, regime):
    """
    PRO-TRADER 5-PILLAR INSTITUTIONAL STOCK ENGINE
    Pillar 1: RVOL >= 1.80x AND 3-minute candle turnover >= Rs 2.5 Cr.
    Pillar 2: True BVC Order Flow & Delta (25%) -> Floor: Delta aligned, Zero CVD divergence
    Pillar 3: 8-bar Base <= 0.65 ATR, Price > VWAP (CALL), Near Day High / Day Low
    Pillar 4: OI Runway & Short-Covering (-4 to +4)
    Pillar 5: Remaining ATR Fuel (Day move <= 90% ATR)
    Gate 2: Anti-Absorption -> CLV >= 0.55 (CALL) / CLV <= 0.45 (PUT), live price holds boundary.
    """
    if bar_idx < 4: return None
    eval_b = df_today.iloc[bar_idx - 1]
    live_b = df_today.iloc[bar_idx]
    spot = float(live_b['close'])
    if spot < MIN_STOCK_PRICE or stock_atr < MIN_STOCK_ATR: return None

    b_open = float(eval_b['open'])
    b_high = float(eval_b['high'])
    b_low = float(eval_b['low'])
    b_close = float(eval_b['close'])
    rng = max(0.01, b_high - b_low)

    day_open = float(df_today['open'].iloc[0])
    day_pct_move = ((spot - day_open) / max(1.0, day_open)) * 100.0
    stk_day_high = float(df_today.iloc[:bar_idx + 1]['high'].max())
    stk_day_low = float(df_today.iloc[:bar_idx + 1]['low'].min())
    day_range = stk_day_high - stk_day_low

    if abs(day_pct_move) > MAX_DAY_PCT_MOVE or day_range > stock_atr * MAX_DAY_RANGE_ATR:
        return None

    fuel_spent = abs(spot - day_open)
    fuel_ratio = fuel_spent / max(1.0, stock_atr)
    if fuel_ratio > 1.25:
        return None

    comp_bars = df_today.iloc[:bar_idx - 1].tail(8)
    if len(comp_bars) < 2:
        return None
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

    # Intraday VWAP calculation
    cum_v = df_today.iloc[:bar_idx + 1]['volume'].cumsum()
    tp = (df_today.iloc[:bar_idx + 1]['high'] + df_today.iloc[:bar_idx + 1]['low'] + df_today.iloc[:bar_idx + 1]['close']) / 3.0
    cum_pv = (tp * df_today.iloc[:bar_idx + 1]['volume']).cumsum()
    vwap = float(cum_pv.iloc[-1] / max(1.0, float(cum_v.iloc[-1]))) if float(cum_v.iloc[-1]) > 0 else float(df_today.iloc[:bar_idx + 1]['close'].mean())

    if direction == 'CALL' and spot < (vwap - stock_atr * 0.02): return None
    if direction == 'PUT' and spot > (vwap + stock_atr * 0.02): return None

    near_hl = (direction == 'CALL' and b_close >= stk_day_high - stock_atr * 0.15) or \
              (direction == 'PUT' and b_close <= stk_day_low + stock_atr * 0.15)
    if not near_hl: return None

    # Macro Regime Alignment
    if direction == 'CALL' and not regime['allow_call']: return None
    if direction == 'PUT' and not regime['allow_put']: return None
    if direction == 'PUT' and regime['move'] > 5.0: return None
    if direction == 'CALL' and regime['move'] < -5.0: return None

    if rng > stock_atr * MAX_CLIMAX_BAR_ATR: return None

    p3_low = df_today.iloc[max(0, bar_idx - 4):bar_idx - 1]['low'].min()
    p3_high = df_today.iloc[max(0, bar_idx - 4):bar_idx - 1]['high'].max()
    p3_move = (b_close - p3_low) if direction == 'CALL' else (p3_high - b_close)
    if p3_move > stock_atr * MAX_PRIOR_3BAR_ATR: return None

    # Pillar 1 Hard Floor: RVOL >= 1.80x AND Turnover >= Rs 2.5 Cr
    curr_v = float(eval_b.get('volume', 0.0))
    prior_10_bars = df_today.iloc[:bar_idx - 1].tail(10)
    base_10_v = float(prior_10_bars['volume'].mean()) if len(prior_10_bars) > 0 else 1000.0
    real_3m_rvol = round(curr_v / max(1.0, base_10_v), 2)
    turnover_cr = round((curr_v * spot) / 1e7, 2)

    if real_3m_rvol < 1.80 or turnover_cr < 2.50:
        return None

    # Pro-Trader Dual-Regime Frontier
    if direction == 'CALL':
        if day_pct_move >= 0.0:
            if b_close < stk_day_high - stock_atr * 0.15: return None
            if abs(day_pct_move) > 3.40: return None
            if fuel_ratio > 1.10: return None
        else:
            runway_to_open = day_open - spot
            runway_ratio = runway_to_open / max(1.0, stock_atr)
            if runway_ratio < 0.40: return None
            if comp_low > stk_day_low + stock_atr * 0.08: return None
            if real_3m_rvol < 2.20: return None
    else:
        if day_pct_move <= 0.0:
            if b_close > stk_day_low + stock_atr * 0.15: return None
            if abs(day_pct_move) > 3.40: return None
            if fuel_ratio > 1.10: return None
        else:
            runway_to_open = spot - day_open
            runway_ratio = runway_to_open / max(1.0, stock_atr)
            if runway_ratio < 0.40: return None
            if comp_high < stk_day_high - stock_atr * 0.08: return None
            if real_3m_rvol < 2.20: return None

    # Pillar 2: BVC Delta & CVD Alignment
    cvd_res = compute_bar_delta_proxy(df_today.iloc[:bar_idx], is_index=False)
    rec_cvd = cvd_res['recent_cvd_ratio']
    bar_delta_ratio = round((2.0 * b_close - (b_high + b_low)) / rng, 3)

    if direction == 'CALL':
        cvd_pass = (rec_cvd >= 0.15 or bar_delta_ratio >= 0.35) and (rec_cvd >= 0.0 if day_pct_move >= 0 else True)
    else:
        cvd_pass = (rec_cvd <= -0.15 or bar_delta_ratio <= -0.35) and (rec_cvd <= 0.0 if day_pct_move <= 0 else True)

    if not cvd_pass: return None

    # Gate 2: Anti-Absorption Trigger (CLV >= 0.55 for CALL, <= 0.45 for PUT)
    body = abs(b_close - b_open)
    solid_body_ratio = round(body / rng, 2)
    clv = round((b_close - b_low) / rng, 2)
    if solid_body_ratio < 0.45: return None

    clv_pass = (clv >= 0.55) if direction == 'CALL' else (clv <= 0.45)
    if not clv_pass: return None

    # 100-Point 5-Pillar Weighted Composite Scoring
    p1_score = min(25.0, 12.0 + min(8.0, (real_3m_rvol - 1.80) * 8.0) + min(5.0, (turnover_cr - 2.5) * 0.8))
    p2_score = min(25.0, 12.0 + min(8.0, abs(bar_delta_ratio) * 16.0) + min(5.0, abs(rec_cvd) * 20.0))
    tight_bonus = min(6.0, max(0.0, (0.65 - comp_w / stock_atr) * 24.0))
    at_fresh_dhl = (b_close >= stk_day_high - stock_atr * 0.03) if direction == 'CALL' else (b_close <= stk_day_low + stock_atr * 0.03)
    p3_score = min(20.0, 10.0 + tight_bonus + (4.0 if at_fresh_dhl else 0.0))
    p4_score = 16.0 + (4.0 if abs(day_pct_move) >= 1.0 else 0.0)

    if 0.20 <= fuel_ratio <= 0.55:
        p5_score = 10.0
    elif 0.55 < fuel_ratio <= 1.10:
        p5_score = 7.5
    elif fuel_ratio < 0.20:
        p5_score = 6.0
    else:
        p5_score = 4.0

    final_score = round(p1_score + p2_score + p3_score + p4_score + p5_score, 2)

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
        'score': final_score,
        'type': 'PURE_STOCK_CALL' if direction == 'CALL' else 'PURE_STOCK_PUT',
        'base_high': comp_high,
        'base_low': comp_low,
        'vwap': round(vwap, 2),
        'fuel_ratio': round(fuel_ratio, 2)
    }

def check_index_signal_pro(sym, df_today, bar_idx, regime):
    if bar_idx < 4: return None
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
    if len(comp_bars) < 2: return None
    comp_high = float(comp_bars['high'].max())
    comp_low = float(comp_bars['low'].min())
    comp_width = comp_high - comp_low

    max_allowed_base = 150.0 if sym != 'BANKNIFTY' else 350.0
    if comp_width > max_allowed_base: return None

    idx_dist = 40.0 if sym != 'BANKNIFTY' else 90.0
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
    cvd_pass = (rec_cvd >= 0.10) if direction == 'CALL' else (rec_cvd <= -0.10)
    if not cvd_pass: return None

    session_rngs = df_today.iloc[:bar_idx]['high'] - df_today.iloc[:bar_idx]['low']
    median_rng = float(session_rngs.median()) if len(session_rngs) > 0 else 15.0
    idx_rvol = round(rng / max(0.5, median_rng), 2)
    if idx_rvol < 1.10: return None

    idx_pct_move = ((cur_p - idx_day_open) / max(1.0, idx_day_open)) * 100.0
    if abs(idx_pct_move) > 1.25: return None

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

# =============================================================================
# INSTITUTIONAL BHAVCOPY -4 TO +4 OPTION CHAIN CONFLUENCE ENGINE
# =============================================================================
bhav_cache = {}
def get_bhav(date_obj):
    d_fmt = date_obj.strftime('%d-%m-%Y') if hasattr(date_obj, 'strftime') else str(date_obj)
    p = os.path.join(REPO_ROOT, 'alpha_profile_oi_system', 'bhavcopy_cache', f'fno_bhav_{d_fmt}.parquet')
    if p in bhav_cache:
        return bhav_cache[p]
    if os.path.exists(p):
        bhav_cache[p] = pd.read_parquet(p)
        return bhav_cache[p]
    try:
        cur_dt = pd.to_datetime(date_obj)
        for delta in range(1, 6):
            prior_fmt = (cur_dt - pd.Timedelta(days=delta)).strftime('%d-%m-%Y')
            prior_p = os.path.join(REPO_ROOT, 'alpha_profile_oi_system', 'bhavcopy_cache', f'fno_bhav_{prior_fmt}.parquet')
            if os.path.exists(prior_p):
                if prior_p not in bhav_cache:
                    bhav_cache[prior_p] = pd.read_parquet(prior_p)
                bhav_cache[p] = bhav_cache[prior_p]
                return bhav_cache[p]
    except Exception:
        pass_placeholder = None
    bhav_cache[p] = None
    return None

def verify_bhavcopy_oi_confluence(sym, date_obj, spot, direction, target_price=None, is_index=False):
    df_bhav = get_bhav(date_obj)
    if df_bhav is None:
        return {'confluent': True, 'reason': 'NO_BHAVCOPY'}
    
    sub = df_bhav[(df_bhav['TckrSymb'] == sym) & (df_bhav['OptnTp'].isin(['CE', 'PE']))]
    if sub.empty:
        return {'confluent': True, 'reason': 'EMPTY_BHAVCOPY_OPTIONS'}
    
    cur_exp = sub['XpryDt'].min()
    exp_df = sub[sub['XpryDt'] == cur_exp]
    lot_size = float(exp_df['NewBrdLotQty'].iloc[0]) if ('NewBrdLotQty' in exp_df.columns and float(exp_df['NewBrdLotQty'].iloc[0]) > 0) else (25.0 if is_index else 250.0)
    
    strike_data = {}
    for _, row in exp_df.iterrows():
        stk = float(row['StrkPric'])
        opt_tp = str(row['OptnTp']).upper().strip()
        oi = float(row['OpnIntrst']) / lot_size
        chg_oi = float(row['ChngInOpnIntrst']) / lot_size
        vol = float(row['TtlTradgVol']) / lot_size
        
        if stk not in strike_data:
            strike_data[stk] = {'ce_oi': 0.0, 'ce_chg_oi': 0.0, 'ce_vol': 0.0, 'pe_oi': 0.0, 'pe_chg_oi': 0.0, 'pe_vol': 0.0}
        
        if opt_tp == 'CE':
            strike_data[stk]['ce_oi'] += oi
            strike_data[stk]['ce_chg_oi'] += chg_oi
            strike_data[stk]['ce_vol'] += vol
        elif opt_tp == 'PE':
            strike_data[stk]['pe_oi'] += oi
            strike_data[stk]['pe_chg_oi'] += chg_oi
            strike_data[stk]['pe_vol'] += vol

    sorted_strikes = sorted(strike_data.keys())
    if not sorted_strikes:
        return {'confluent': True, 'reason': 'NO_STRIKES'}

    atm_strike = min(sorted_strikes, key=lambda s: abs(s - spot))
    atm_idx = sorted_strikes.index(atm_strike)

    lower_4 = sorted_strikes[max(0, atm_idx - 4) : atm_idx]
    upper_4 = sorted_strikes[atm_idx + 1 : min(len(sorted_strikes), atm_idx + 5)]
    window_9 = lower_4 + [atm_strike] + upper_4
    strike_step = (sorted_strikes[1] - sorted_strikes[0]) if len(sorted_strikes) > 1 else 50.0

    atm_info = strike_data[atm_strike]
    tot_window_ce_oi = sum(strike_data[s]['ce_oi'] for s in window_9)
    tot_window_pe_oi = sum(strike_data[s]['pe_oi'] for s in window_9)

    tot_upper_ce_chg = sum(strike_data[s]['ce_chg_oi'] for s in upper_4)
    tot_upper_pe_chg = sum(strike_data[s]['pe_chg_oi'] for s in upper_4)
    tot_upper_ce_vol = sum(strike_data[s]['ce_vol'] for s in upper_4)
    tot_upper_ce_oi = sum(strike_data[s]['ce_oi'] for s in upper_4)

    tot_lower_pe_chg = sum(strike_data[s]['pe_chg_oi'] for s in lower_4)
    tot_lower_ce_chg = sum(strike_data[s]['ce_chg_oi'] for s in lower_4)
    tot_lower_pe_vol = sum(strike_data[s]['pe_vol'] for s in lower_4)
    tot_lower_pe_oi = sum(strike_data[s]['pe_oi'] for s in lower_4)

    support_pe_chg = atm_info['pe_chg_oi'] + tot_lower_pe_chg
    resistance_ce_chg = atm_info['ce_chg_oi'] + tot_upper_ce_chg

    min_wall_oi = 25000.0 if is_index else 400.0
    monster_standing_oi = 50000.0 if is_index else 600.0
    ambush_wall_min = 15000.0 if is_index else 250.0

    if direction == 'CALL':
        target_barrier = float(target_price) if target_price else (spot + (120.0 if is_index else 50.0))
        candidate_wall_strikes = [s for s in sorted_strikes if (s >= (atm_strike - strike_step * 0.1) and s <= target_barrier + strike_step * 0.2)]

        is_otm_ce_ambush = (tot_upper_ce_chg >= ambush_wall_min) and (tot_upper_ce_chg > 2.0 * max(100.0, support_pe_chg)) and (atm_info['ce_chg_oi'] >= 0)
        gamma_squeeze_call = (tot_upper_ce_vol >= 6.0 * max(1.0, tot_upper_ce_oi))

        if is_otm_ce_ambush and not gamma_squeeze_call:
            return {'confluent': False, 'reason': f'UPPER_OTM_CALL_WRITING_AMBUSH (+{int(tot_upper_ce_chg)} CE vs +{int(support_pe_chg)} PE)'}

        atm_ce_chg = atm_info['ce_chg_oi']
        atm_pe_chg = atm_info['pe_chg_oi']
        atm_blocked = (atm_ce_chg > 0) and (atm_ce_chg > max(50.0, atm_pe_chg * 1.25))
        if atm_blocked:
            return {'confluent': False, 'reason': f'ATM_CALL_WRITERS_FIGHTING (+{int(atm_ce_chg)} CE vs +{int(atm_pe_chg)} PE at {int(atm_strike)})'}

        res_wall_strike = None
        for up_s in candidate_wall_strikes:
            st_d = strike_data[up_s]
            ce_oi = st_d['ce_oi']
            pe_oi = st_d['pe_oi']
            ce_chg = st_d['ce_chg_oi']
            pe_chg = st_d['pe_chg_oi']

            if ce_chg < 0 or pe_chg >= ce_chg:
                continue

            if ce_oi < min_wall_oi: continue

            ce_conc = ce_oi / max(1.0, tot_window_ce_oi)
            is_concentrated = (ce_conc >= 0.22) or (ce_oi >= 1.35 * max(1.0, atm_info['ce_oi']))
            is_standing_wall = (ce_oi >= monster_standing_oi)
            is_ratio_heavy = (ce_oi > 1.6 * max(100.0, pe_oi)) or (is_standing_wall and ce_oi > 1.25 * max(50.0, pe_oi))

            if (is_concentrated or is_standing_wall) and is_ratio_heavy:
                if st_d['ce_vol'] >= 6.0 * ce_oi: continue
                res_wall_strike = up_s
                break

        confluent = (res_wall_strike is None)
        reason = f'CALL_RUNWAY_CLEAR_ATM_{int(atm_strike)}' if confluent else f'CALL_BLOCKED_BY_WALL_AT_{int(res_wall_strike)}'
        return {'confluent': confluent, 'reason': reason, 'atm_strike': atm_strike}

    else:
        target_barrier = float(target_price) if target_price else (spot - (120.0 if is_index else 50.0))
        candidate_wall_strikes = [s for s in sorted_strikes if (target_barrier - strike_step * 0.2 <= s <= atm_strike + strike_step * 0.1)]

        is_otm_pe_ambush = (tot_lower_pe_chg >= ambush_wall_min) and (tot_lower_pe_chg > 2.0 * max(100.0, resistance_ce_chg)) and (atm_info['pe_chg_oi'] >= 0)
        gamma_squeeze_put = (tot_lower_pe_vol >= 6.0 * max(1.0, tot_lower_pe_oi))

        if is_otm_pe_ambush and not gamma_squeeze_put:
            return {'confluent': False, 'reason': f'LOWER_OTM_PUT_WRITING_AMBUSH (+{int(tot_lower_pe_chg)} PE vs +{int(resistance_ce_chg)} CE)'}

        atm_pe_chg = atm_info['pe_chg_oi']
        atm_ce_chg = atm_info['ce_chg_oi']
        atm_blocked = (atm_pe_chg > 0) and (atm_pe_chg > max(50.0, atm_ce_chg * 1.25))
        if atm_blocked:
            return {'confluent': False, 'reason': f'ATM_PUT_WRITERS_FIGHTING (+{int(atm_pe_chg)} PE vs +{int(atm_ce_chg)} CE at {int(atm_strike)})'}

        sup_wall_strike = None
        for dn_s in sorted(candidate_wall_strikes, reverse=True):
            st_d = strike_data[dn_s]
            pe_oi = st_d['pe_oi']
            ce_oi = st_d['ce_oi']
            pe_chg = st_d['pe_chg_oi']
            ce_chg = st_d['ce_chg_oi']

            if pe_chg < 0 or ce_chg >= pe_chg:
                continue

            if pe_oi < min_wall_oi: continue

            pe_conc = pe_oi / max(1.0, tot_window_pe_oi)
            is_concentrated = (pe_conc >= 0.22) or (pe_oi >= 1.35 * max(1.0, atm_info['pe_oi']))
            is_standing_wall = (pe_oi >= monster_standing_oi)
            is_ratio_heavy = (pe_oi > 1.6 * max(100.0, ce_oi)) or (is_standing_wall and pe_oi > 1.25 * max(50.0, ce_oi))

            if (is_concentrated or is_standing_wall) and is_ratio_heavy:
                if st_d['pe_vol'] >= 6.0 * pe_oi: continue
                sup_wall_strike = dn_s
                break

        confluent = (sup_wall_strike is None)
        reason = f'PUT_RUNWAY_CLEAR_ATM_{int(atm_strike)}' if confluent else f'PUT_BLOCKED_BY_WALL_AT_{int(sup_wall_strike)}'
        return {'confluent': confluent, 'reason': reason, 'atm_strike': atm_strike}

check_oi_runway_bhavcopy = verify_bhavcopy_oi_confluence

# =============================================================================
# CONCURRENT POSITION MANAGER (IDENTICAL TO DAILY_AUTONOMOUS_SCREENER.PY)
# =============================================================================
class ActiveOptionTrade:
    def __init__(self, cand, b_idx, curr_t, lot_map):
        self.cand = cand
        self.entry_idx = b_idx
        self.sym = cand['symbol']
        self.direction = cand['direction']
        self.spot_entry = float(cand['entry_price'])
        self.is_idx = cand.get('is_index', False)
        self.is_bank = (self.sym == 'BANKNIFTY')
        self.atr = cand.get('stock_atr', 30.0)
        self.step = 0
        self.is_closed = False
        self.spot_exit = self.spot_entry
        self.exit_reason = None
        self.exit_time = curr_t
        self.t1_hit = False
        self.is_risk_free = False
        self.max_fav_pts = 0.0
        self.max_adv_pts = 0.0

        if self.is_idx:
            is_mid = (self.sym == 'MIDCPNIFTY')
            self.spot_t1 = 120.0 if self.is_bank else (60.0 if is_mid else 80.0)
            self.spot_t2 = 250.0 if self.is_bank else (120.0 if is_mid else 150.0)
            self.spot_sl = 60.0 if self.is_bank else (30.0 if is_mid else 40.0)
        else:
            self.spot_t1 = max(25.0, min(50.0, round(self.atr * 0.35, 1)))
            self.spot_t2 = max(55.0, min(100.0, round(self.atr * 0.75, 1)))
            self.spot_sl = max(15.0, min(30.0, round(self.atr * 0.18, 1)))

        step = determine_step_size(self.spot_entry, is_index=self.is_idx, symbol=self.sym)
        self.atm_strike = round(self.spot_entry / step) * step
        self.opt_type = 'CE' if self.direction == 'CALL' else 'PE'
        stk_str = f"{int(self.atm_strike)}" if self.atm_strike == int(self.atm_strike) else f"{self.atm_strike:.1f}"
        self.contract_name = f"{self.sym} {stk_str} {self.opt_type}"
        self.lot_size = lot_map.get(self.sym, (25 if self.is_idx else 250))
        self.opt_entry_premium = max(5.0, round(self.spot_sl * 1.50, 2))
        self.opt_sl_pts = round(self.spot_sl * 0.50, 2)

    def update(self, b_high, b_low, b_close, ts):
        if self.is_closed: return
        self.step += 1
        fav_move = (b_high - self.spot_entry) if self.direction == 'CALL' else (self.spot_entry - b_low)
        adv_move = (self.spot_entry - b_low) if self.direction == 'CALL' else (b_high - self.spot_entry)
        cur_gain = (b_close - self.spot_entry) if self.direction == 'CALL' else (self.spot_entry - b_close)

        if fav_move > self.max_fav_pts: self.max_fav_pts = fav_move
        if adv_move > self.max_adv_pts: self.max_adv_pts = adv_move

        if fav_move >= max(20.0, self.spot_t1 * 0.40):
            self.is_risk_free = True

        if not self.t1_hit and fav_move >= self.spot_t1:
            self.t1_hit = True
            self.is_risk_free = True

        if self.t1_hit:
            if fav_move >= self.spot_t2:
                self.spot_exit = (self.spot_entry + self.spot_t2) if self.direction == 'CALL' else (self.spot_entry - self.spot_t2)
                self.exit_reason = "TARGET_2_HIT (JACKPOT)"
                self.exit_time = ts
                self.is_closed = True
                return
            lock_pts = self.spot_t1 * 0.40
            locked_sl = (self.spot_entry + lock_pts) if self.direction == 'CALL' else (self.spot_entry - lock_pts)
            if (self.direction == 'CALL' and b_low <= locked_sl) or (self.direction == 'PUT' and b_high >= locked_sl):
                self.spot_exit = locked_sl
                self.exit_reason = "T1_PROFIT_LOCKED_EXIT"
                self.exit_time = ts
                self.is_closed = True
                return
        else:
            if adv_move >= self.spot_sl:
                self.spot_exit = (self.spot_entry - self.spot_sl) if self.direction == 'CALL' else (self.spot_entry + self.spot_sl)
                self.exit_reason = "SL_HIT"
                self.exit_time = ts
                self.is_closed = True
                return
            if self.step >= 5 and cur_gain <= 0.0:
                self.spot_exit = b_close
                self.exit_reason = "15M_TIME_SL_SCRATCH"
                self.exit_time = ts
                self.is_closed = True
                return

    def close_eod(self, final_close, ts):
        if not self.is_closed:
            self.spot_exit = final_close
            self.exit_reason = "EOD_EXIT"
            self.exit_time = ts
            self.is_closed = True

    def get_summary(self, d_str):
        spot_pts = round((self.spot_exit - self.spot_entry) if self.direction == 'CALL' else (self.spot_entry - self.spot_exit), 2)
        if "JACKPOT" in self.exit_reason: eff_delta = 0.72
        elif "T1" in self.exit_reason: eff_delta = 0.58
        elif "15M" in self.exit_reason: eff_delta = 0.45
        elif "SL" in self.exit_reason: eff_delta = 0.50
        else: eff_delta = 0.50

        opt_pts = round(spot_pts * eff_delta, 2)
        opt_exit_premium = max(0.50, round(self.opt_entry_premium + opt_pts, 2))
        gross_pnl = round(opt_pts * self.lot_size, 2)

        brokerage = 40.0
        stt = round(opt_exit_premium * self.lot_size * 0.001, 2)
        turnover_fee = round((self.opt_entry_premium + opt_exit_premium) * self.lot_size * 0.0005, 2)
        gst = round((brokerage + turnover_fee) * 0.18, 2)
        stamp_duty = round(self.opt_entry_premium * self.lot_size * 0.00003, 2)
        tx_costs = round(brokerage + stt + turnover_fee + gst + stamp_duty, 2)
        net_pnl = round(gross_pnl - tx_costs, 2)

        realized_r = round(opt_pts / max(0.5, self.opt_sl_pts), 2)
        mfe_r = round((self.max_fav_pts * 0.60) / max(0.5, self.opt_sl_pts), 2)
        mae_r = round((self.max_adv_pts * 0.50) / max(0.5, self.opt_sl_pts), 2)

        return {
            'date': d_str,
            'symbol': self.sym,
            'contract': self.contract_name,
            'direction': self.direction,
            'eval_time': self.cand['eval_time'],
            'exit_time': self.exit_time,
            'spot_entry': self.spot_entry,
            'spot_exit': self.spot_exit,
            'spot_pts': spot_pts,
            'opt_entry': self.opt_entry_premium,
            'opt_exit': opt_exit_premium,
            'opt_pts': opt_pts,
            'lot_size': self.lot_size,
            'gross_pnl': gross_pnl,
            'tx_costs': tx_costs,
            'net_pnl': net_pnl,
            'realized_r': realized_r,
            'mfe_r': mfe_r,
            'mae_r': mae_r,
            'exit_reason': self.exit_reason,
            'score': self.cand['score'],
            'type': self.cand['type'],
            'is_index': self.is_idx
        }

if __name__ == "__main__":
    print("Loading latest resampled 3m data...")
    asset_data = load_resampled_3m_data()
    atrs = load_verified_atrs()
    lot_map = load_dynamic_lot_sizes()
    unique_dates = sorted(list(set(d.strftime('%Y-%m-%d') for df in asset_data.values() for d in df.index)))[-5:]
    stock_syms = [s for s in asset_data.keys() if s not in INDEX_NAMES]
    all_closed_trades = []

    print(f"Loaded {len(asset_data)} active assets ({len(stock_syms)} equities, {len(INDEX_NAMES)} indices).")
    print(f"Authoritative 5-Day Backtest Window: {unique_dates}")

    for d_str in unique_dates:
        d = pd.to_datetime(d_str).date()
        nifty_day = asset_data['NIFTY'][asset_data['NIFTY'].index.date == d].copy()
        if nifty_day.empty: continue
        nifty_day['ema9'] = nifty_day['close'].ewm(span=9, adjust=False).mean()

        active_trades = []
        daily_traded_symbols = set()
        time_index = nifty_day.index

        for b_idx in range(4, len(time_index)):
            curr_t = time_index[b_idx]
            tot_m = curr_t.hour * 60 + curr_t.minute
            is_m = (570 <= tot_m <= 615)
            is_a = (795 <= tot_m <= 855)

            # 1. Update Open Active Trades Concurrently
            for tr in active_trades:
                if not tr.is_closed:
                    df_asset = asset_data[tr.sym][asset_data[tr.sym].index.date == d]
                    if curr_t in df_asset.index:
                        b = df_asset.loc[curr_t]
                        if isinstance(b, pd.DataFrame): b = b.iloc[0]
                        tr.update(float(b['high']), float(b['low']), float(b['close']), curr_t)

            # Collect closed trades
            for tr in active_trades:
                if tr.is_closed and tr not in all_closed_trades:
                    all_closed_trades.append(tr)

            # Retain only active open positions
            active_trades = [tr for tr in active_trades if not tr.is_closed]

            if not (is_m or is_a): continue

            regime = get_nifty_regime(nifty_day, b_idx)
            if not (regime['allow_call'] or regime['allow_put']): continue

            act_broad = sum(1 for tr in active_trades if tr.is_idx and (not tr.is_bank) and (not tr.is_risk_free))
            act_bank = sum(1 for tr in active_trades if tr.is_idx and tr.is_bank and (not tr.is_risk_free))
            act_stocks = sum(1 for tr in active_trades if (not tr.is_idx) and (not tr.is_risk_free))

            # 2. Evaluate Stock AND Index Candidates TOGETHER (Synchronized Dual Engine)
            candidates = []

            # A. Index Candidates
            if (act_broad < MAX_ACTIVE_BROAD_INDEX or act_bank < MAX_ACTIVE_BANK_INDEX) and regime.get('allow_index', True):
                for isym in INDEX_NAMES:
                    if isym in daily_traded_symbols or isym not in asset_data: continue
                    is_bank = (isym == 'BANKNIFTY')
                    if is_bank and act_bank >= MAX_ACTIVE_BANK_INDEX: continue
                    if (not is_bank) and act_broad >= MAX_ACTIVE_BROAD_INDEX: continue

                    idf = asset_data[isym][asset_data[isym].index.date == d]
                    if len(idf) <= b_idx: continue
                    sig = check_index_signal_pro(isym, idf, b_idx, regime)
                    if sig:
                        t_price = (sig['entry_price'] + 100.0) if sig['direction'] == 'CALL' else (sig['entry_price'] - 100.0)
                        oi_res = verify_bhavcopy_oi_confluence(isym, d, sig['entry_price'], sig['direction'], target_price=t_price, is_index=True)
                        if oi_res['confluent']:
                            candidates.append(sig)

            # B. Stock Candidates
            if act_stocks < MAX_ACTIVE_STOCKS:
                for ssym in stock_syms:
                    if ssym in daily_traded_symbols or ssym not in asset_data: continue
                    sdf = asset_data[ssym][asset_data[ssym].index.date == d]
                    if len(sdf) <= b_idx: continue
                    stock_atr = atrs.get(ssym, 30.0)
                    sig = check_stock_signal_pro(ssym, sdf, b_idx, stock_atr, regime)
                    if sig:
                        t_price = (sig['entry_price'] + 50.0) if sig['direction'] == 'CALL' else (sig['entry_price'] - 50.0)
                        oi_res = verify_bhavcopy_oi_confluence(ssym, d, sig['entry_price'], sig['direction'], target_price=t_price, is_index=False)
                        if oi_res['confluent']:
                            candidates.append(sig)

            # Sort combined candidates by 100-point 5-Pillar Score
            candidates.sort(key=lambda x: x['score'], reverse=True)

            # Dispatch new trades into available slots
            for cand in candidates:
                is_i = cand.get('is_index', False)
                is_b = (cand['symbol'] == 'BANKNIFTY')
                if is_i:
                    if is_b and act_bank >= MAX_ACTIVE_BANK_INDEX: continue
                    if (not is_b) and act_broad >= MAX_ACTIVE_BROAD_INDEX: continue
                else:
                    if act_stocks >= MAX_ACTIVE_STOCKS: continue

                new_trade = ActiveOptionTrade(cand, b_idx, curr_t, lot_map)
                active_trades.append(new_trade)
                daily_traded_symbols.add(cand['symbol'])
                if is_i:
                    if is_b: act_bank += 1
                    else: act_broad += 1
                else:
                    act_stocks += 1

        # End of day: close remaining active trades
        for tr in active_trades:
            if not tr.is_closed:
                df_asset = asset_data[tr.sym][asset_data[tr.sym].index.date == d]
                if len(df_asset) > 0:
                    final_close = float(df_asset.iloc[-1]['close'])
                    tr.close_eod(final_close, df_asset.index[-1])
                    all_closed_trades.append(tr)

    summaries = [tr.get_summary(str(tr.cand['eval_time'].date())) for tr in all_closed_trades]
    df_res = pd.DataFrame(summaries)
    csv_path = os.path.join(REPORT_DIR, "Alpha_Latest_5Day_Ending_Today_Backtest.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved authoritative backtest CSV to: {csv_path}")

    print("\n" + "="*85)
    print("PRO-TRADER SYNCHRONIZED DUAL ENGINE: STOCK & INDEX TOGETHER (5-DAY ENDING TODAY)")
    print("="*85)
    print(f"Total Dispatched Option Trades: {len(df_res)}")
    wins = df_res[df_res['opt_pts'] > 0]
    losses = df_res[df_res['opt_pts'] <= 0]
    scratches = df_res[df_res['exit_reason'].str.contains('SCRATCH|TIME_SL', na=False)]
    full_sl = df_res[df_res['exit_reason'] == 'SL_HIT']

    gross_r = wins['realized_r'].sum()
    loss_r = abs(losses['realized_r'].sum())
    net_r = gross_r - loss_r
    pf = round(gross_r / max(0.01, loss_r), 2)

    win_rate = len(wins) / max(1, len(df_res)) * 100.0
    expectancy_r = net_r / max(1, len(df_res))
    total_net_pnl = df_res['net_pnl'].sum()

    cum_r = df_res['realized_r'].cumsum()
    peak = cum_r.cummax()
    drawdown = peak - cum_r
    max_dd = round(float(drawdown.max()), 2) if not drawdown.empty else 0.0

    avg_mfe = round(float(df_res['mfe_r'].mean()), 2) if 'mfe_r' in df_res.columns else 0.0
    avg_mae = round(float(df_res['mae_r'].mean()), 2) if 'mae_r' in df_res.columns else 0.0

    calls = df_res[df_res['direction'] == 'CALL']
    puts = df_res[df_res['direction'] == 'PUT']
    call_wins = len(calls[calls['opt_pts'] > 0])
    put_wins = len(puts[puts['opt_pts'] > 0])
    call_net = calls['realized_r'].sum()
    put_net = puts['realized_r'].sum()

    idx_trades = df_res[df_res['is_index'] == True]
    stk_trades = df_res[df_res['is_index'] == False]
    idx_wins = len(idx_trades[idx_trades['opt_pts'] > 0])
    stk_wins = len(stk_trades[stk_trades['opt_pts'] > 0])
    idx_net = idx_trades['realized_r'].sum()
    stk_net = stk_trades['realized_r'].sum()

    print(f"Wins: {len(wins)} ({win_rate:.1f}%) | Losses/Scratches: {len(losses)} | Full SL Hits: {len(full_sl)}")
    print(f"Gross Profit: +{gross_r:.2f}R | Gross Loss: -{loss_r:.2f}R | Net Profit: +{net_r:.2f}R | Net PnL: Rs {total_net_pnl:,.2f}")
    print(f"Profit Factor: {pf} | Expectancy: +{expectancy_r:.2f}R per trade | Max Drawdown: {max_dd:.2f}R")
    print(f"Average MFE: +{avg_mfe:.2f}R | Average MAE: -{avg_mae:.2f}R")
    print(f"INDEX Trades: {len(idx_trades)} (Wins: {idx_wins}, Net: {idx_net:+.2f}R) | STOCK Trades: {len(stk_trades)} (Wins: {stk_wins}, Net: {stk_net:+.2f}R)")
    print(f"CALL Trades: {len(calls)} (Wins: {call_wins}, Net: {call_net:+.2f}R) | PUT Trades: {len(puts)} (Wins: {put_wins}, Net: {put_net:+.2f}R)")

    print("\n--- ALL EXECUTED OPTION TRADES ---")
    for _, r in df_res.iterrows():
        e_str = r['eval_time'].strftime('%Y-%m-%d %H:%M')
        print(f"{e_str} | {r['contract']:<18} | SpotEn: {r['spot_entry']:8.2f} | OptEn: {r['opt_entry']:6.2f} | OptEx: {r['opt_exit']:6.2f} | OptPts: {r['opt_pts']:+6.2f} | Lot: {r['lot_size']:4} | Net: Rs {r['net_pnl']:+9,.2f} | R: {r['realized_r']:+5.2f}R | {r['exit_reason']}")
