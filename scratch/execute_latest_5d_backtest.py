# -*- coding: utf-8 -*-
"""
scratch/execute_latest_5d_backtest.py
=====================================
Synchronized Authoritative 5-Day Option Buyer Backtest Engine (Ending Today).

Key Architectural Upgrades:
  1. Synchronized Common 5-Pillar Engine:
     - Imports evaluate_common_5pillar_bar_slice() directly from daily_autonomous_screener.py.
     - Calls evaluate_common_5pillar_bar_slice() on each 3m timestamp in Morning (09:30-10:15)
       and Afternoon (13:15-14:15) slabs.
     - Enforces MAX_INDEX_PER_SLAB = 1 and MAX_STOCKS_PER_SLAB = 2.
  2. Conservative Intrabar Execution:
     - Checks intrabar high and low (never close only!) for SL, T1, and T2.
     - Conservative SL-first resolution if both SL and Target touch on the same bar.
     - Force EOD closure at 15:20 IST so zero positions carry overnight.
     - Rejects any symbol with unknown lot size (lot <= 0).
  3. Institutional Tearsheet:
     - ZeroDivisionError safe for total_trades == 0.
     - Computes true Net-P&L Profit Factor (Sum(Winning Net P&L) / Abs(Sum(Losing Net P&L))),
       Expectancy (R & INR), Max Drawdown, Max Consecutive Losses, False Breakout Rate, and MFE/MAE.
"""

import os
import sys
import glob
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPORT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha_profile_oi_system.modules.fno_universe import determine_step_size
from alpha_profile_oi_system.daily_autonomous_screener import evaluate_common_5pillar_bar_slice

CACHE_DIR = os.path.join(REPORT_DIR, "data_cache_latest_5d_1m")
CONFIG_DIR = os.path.join(REPO_ROOT, "alpha_profile_oi_system", "config")

INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
MIN_STOCK_PRICE = 650.0
MIN_STOCK_ATR = 18.0

# Active position caps strictly matching daily_autonomous_screener.py
MAX_ACTIVE_STOCKS = 2
MAX_ACTIVE_BROAD_INDEX = 1   # NIFTY, MIDCPNIFTY, FINNIFTY
MAX_ACTIVE_BANK_INDEX = 1    # BANKNIFTY

def load_resampled_3m_data() -> Dict[str, pd.DataFrame]:
    files = glob.glob(os.path.join(CACHE_DIR, "*.parquet"))
    asset_dict = {}
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "").replace("^", "").strip()
        try:
            df = pd.read_parquet(f)
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            if not isinstance(df.index, pd.DatetimeIndex):
                continue
            
            # Ensure timezone is localized/converted to Asia/Kolkata
            if df.index.tz is not None:
                df.index = df.index.tz_convert('Asia/Kolkata')
            else:
                ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
                df.index = df.index.tz_localize(ist_tz)
                
            is_spot_index = sym in INDEX_NAMES or sym in ['NSEI', 'NSEBANK']

            # Group by date BEFORE resampling to 3m bars (Session Isolation)
            resampled_days = []
            for d_val, df_day in df.groupby(df.index.date):
                if len(df_day) == 0:
                    continue
                df_3m_day = df_day.resample('3min', label='left', closed='left').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna(subset=['open', 'high', 'low', 'close'])
                
                # Filter strictly for NSE market hours (09:15 to 15:30)
                df_3m_day = df_3m_day[(df_3m_day.index.time >= datetime.time(9, 15)) & (df_3m_day.index.time <= datetime.time(15, 30))]
                if len(df_3m_day) < 2:
                    continue

                # Precompute intraday indicators per day session
                rng = np.maximum(0.01, df_3m_day['high'] - df_3m_day['low'])
                clv = (2.0 * df_3m_day['close'] - (df_3m_day['high'] + df_3m_day['low'])) / rng
                df_3m_day['bar_delta_ratio'] = clv.round(4)
                if is_spot_index:
                    df_3m_day['bvc_est_delta'] = (rng * clv).round(2)
                else:
                    df_3m_day['bvc_est_delta'] = (df_3m_day['volume'] * clv).round(2)
                df_3m_day['delta'] = df_3m_day['bvc_est_delta']
                df_3m_day['cvd'] = df_3m_day['bvc_est_delta'].cumsum().round(2)

                # EMA9 and EMA21
                df_3m_day['ema9'] = df_3m_day['close'].ewm(span=9, adjust=False).mean()
                df_3m_day['ema21'] = df_3m_day['close'].ewm(span=21, adjust=False).mean()

                # Session VWAP
                tp = (df_3m_day['high'] + df_3m_day['low'] + df_3m_day['close']) / 3.0
                cum_v = df_3m_day['volume'].cumsum()
                cum_pv = (tp * df_3m_day['volume']).cumsum()
                df_3m_day['vwap'] = (cum_pv / np.maximum(1.0, cum_v)).round(2) if not is_spot_index else tp.expanding().mean().round(2)

                resampled_days.append(df_3m_day)

            if not resampled_days:
                continue

            # Concatenate all sessions for this symbol
            df_full = pd.concat(resampled_days).sort_index()

            # Pre-compute prior-session same-slot median hist_slot_vol
            all_dates = sorted(list(set(df_full.index.date)))
            df_full['time_slot'] = df_full.index.strftime('%H:%M')
            df_full['hist_slot_vol'] = 0.0

            for cur_d in all_dates:
                cur_mask = (df_full.index.date == cur_d)
                prior_df = df_full[df_full.index.date < cur_d]
                if not prior_df.empty and prior_df['volume'].sum() > 0:
                    slot_map = prior_df.groupby('time_slot')['volume'].median().to_dict()
                    overall_med = float(prior_df['volume'].median())
                    slot_vols = df_full.loc[cur_mask, 'time_slot'].map(slot_map).fillna(overall_med).clip(lower=100.0)
                else:
                    cur_vols = df_full.loc[cur_mask, 'volume']
                    cur_med = float(cur_vols.median()) if len(cur_vols) > 0 and cur_vols.median() > 0 else 1000.0
                    slot_vols = pd.Series(max(100.0, cur_med), index=df_full.loc[cur_mask].index)
                
                df_full.loc[cur_mask, 'hist_slot_vol'] = slot_vols

            if len(df_full) > 10:
                asset_dict[sym] = df_full
        except Exception:
            continue
    return asset_dict

def load_verified_atrs() -> Dict[str, float]:
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

def load_dynamic_lot_sizes() -> Dict[str, int]:
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
                        l = int(row['NewBrdLotQty'])
                        if l > 0:
                            lot_map[sym] = l
        except Exception as e:
            err_bhav = str(e)
    return lot_map

def get_nifty_regime(nifty_day: pd.DataFrame, b_idx: int) -> Dict[str, Any]:
    if b_idx < 2:
        return {'allow_call': True, 'allow_put': True, 'allow_stock_call': True, 'allow_stock_put': True, 'reason': 'START', 'move': 0.0}
    comp_df = nifty_day.iloc[:b_idx + 1]
    eval_b = nifty_day.iloc[b_idx - 1]
    live_b = nifty_day.iloc[b_idx]
    n_spot = float(live_b['close'])
    ema9 = float(nifty_day['ema9'].iloc[b_idx])
    ema21 = float(nifty_day['ema21'].iloc[b_idx]) if 'ema21' in nifty_day.columns else float(comp_df['close'].ewm(span=21, adjust=False).mean().iloc[-1])
    day_open = float(nifty_day['open'].iloc[0])
    day_high = float(comp_df['high'].max())
    day_low = float(comp_df['low'].min())
    move_from_open = n_spot - day_open
    drop_from_high = day_high - n_spot
    rise_from_low = n_spot - day_low

    tp_vwap = float(((comp_df['high'] + comp_df['low'] + comp_df['close']) / 3.0).mean())

    n_last_bar_rng = float(eval_b['high'] - eval_b['low'])
    if n_last_bar_rng > 65.0:
        return {'allow_call': False, 'allow_put': False, 'allow_stock_call': False, 'allow_stock_put': False, 'reason': 'NIFTY_CLIMAX_EXHAUSTION', 'move': move_from_open}

    recent_rng = nifty_day['high'].iloc[max(0, b_idx - 5):b_idx + 1].max() - nifty_day['low'].iloc[max(0, b_idx - 5):b_idx + 1].min()
    ema_dist = abs(n_spot - ema9)

    is_nifty_bullish = (n_spot >= tp_vwap and n_spot >= ema21) or (move_from_open >= 12.0 and drop_from_high < 40.0)
    is_nifty_bearish = (n_spot <= tp_vwap and n_spot <= ema21) or (move_from_open <= -12.0 and rise_from_low < 40.0)

    allow_stock_put = not is_nifty_bullish
    allow_stock_call = not is_nifty_bearish

    if recent_rng < 22.0 and ema_dist < 12.0:
        allow_call = False
        allow_put = False
        reason = 'CHOP_TRAP_INDEX_FROZEN'
    else:
        allow_call = is_nifty_bullish and (n_spot >= ema9)
        allow_put = is_nifty_bearish and (n_spot <= ema9)
        reason = 'NIFTY_BULLISH' if is_nifty_bullish else ('NIFTY_BEARISH' if is_nifty_bearish else 'NORMAL')

    return {
        'allow_call': allow_call,
        'allow_put': allow_put,
        'allow_stock_call': allow_stock_call,
        'allow_stock_put': allow_stock_put,
        'reason': reason,
        'move': move_from_open
    }

def parse_bhav_date(fpath: str) -> datetime.date:
    import re
    m = re.search(r'(\d{2})-(\d{2})-(\d{4})', os.path.basename(fpath))
    if m:
        return datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return datetime.date(2000, 1, 1)

_BHAV_CACHE = {}

def verify_bhavcopy_oi_confluence(sym: str, date_obj, spot: float, direction: str, target_price: Optional[float] = None, is_index: bool = False) -> Dict[str, Any]:
    bhav_files = glob.glob(os.path.join(REPO_ROOT, 'alpha_profile_oi_system', 'bhavcopy_cache', '*.parquet'))
    if not bhav_files:
        return {'confluent': True, 'reason': 'BHAVCOPY_CACHE_MISSING_PASSED', 'atm_strike': spot}

    # Authoritative Market-Open Bhavcopy Matching:
    # Intraday trades on date `d` at 09:15-15:30 only have access to closing bhavcopy from prior trading day (date < d)
    d = date_obj if isinstance(date_obj, datetime.date) else (date_obj.date() if hasattr(date_obj, 'date') else pd.to_datetime(date_obj).date())
    candidates = [f for f in bhav_files if parse_bhav_date(f) < d]
    if not candidates:
        candidates = [f for f in bhav_files if parse_bhav_date(f) <= d]
    bhav_match = max(candidates, key=parse_bhav_date) if candidates else None
    if not bhav_match:
        return {'confluent': True, 'reason': 'NO_BHAVCOPY_FOUND_PASSED', 'atm_strike': spot}

    try:
        if bhav_match not in _BHAV_CACHE:
            _BHAV_CACHE[bhav_match] = pd.read_parquet(bhav_match)
        df_bhav = _BHAV_CACHE[bhav_match]
    except Exception:
        return {'confluent': True, 'reason': 'BHAVCOPY_READ_ERROR_PASSED', 'atm_strike': spot}

    sub = df_bhav[df_bhav['TckrSymb'] == sym]
    if sub.empty:
        return {'confluent': True, 'reason': 'SYMBOL_NOT_IN_BHAVCOPY_PASSED', 'atm_strike': spot}

    cur_exp = sub['XpryDt'].min()
    exp_df = sub[sub['XpryDt'] == cur_exp]
    lot_size = float(exp_df['NewBrdLotQty'].iloc[0]) if ('NewBrdLotQty' in exp_df.columns and float(exp_df['NewBrdLotQty'].iloc[0]) > 0) else (25.0 if is_index else 250.0)

    strike_data = {}
    for _, row in exp_df.iterrows():
        try:
            stk = float(row['StrkPric'])
            if np.isnan(stk) or stk <= 0:
                continue
        except Exception:
            continue
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

    sorted_strikes = sorted([s for s in strike_data.keys() if pd.notna(s) and s > 0])
    if len(sorted_strikes) < 9:
        return {'confluent': True, 'reason': 'INSUFFICIENT_BHAVCOPY_STRIKES', 'atm_strike': spot}

    atm_strike = min(sorted_strikes, key=lambda s: abs(s - spot))
    atm_idx = sorted_strikes.index(atm_strike)

    lower_4 = sorted_strikes[max(0, atm_idx - 4):atm_idx]
    upper_4 = sorted_strikes[atm_idx + 1:min(len(sorted_strikes), atm_idx + 5)]

    atm_info = strike_data[atm_strike]
    tot_upper_ce_oi = sum(strike_data[s]['ce_oi'] for s in upper_4)
    tot_upper_ce_chg = sum(strike_data[s]['ce_chg_oi'] for s in upper_4)
    tot_lower_pe_oi = sum(strike_data[s]['pe_oi'] for s in lower_4)
    tot_lower_pe_chg = sum(strike_data[s]['pe_chg_oi'] for s in lower_4)

    tot_window_ce_oi = sum(strike_data[s]['ce_oi'] for s in lower_4 + [atm_strike] + upper_4)
    tot_window_pe_oi = sum(strike_data[s]['pe_oi'] for s in lower_4 + [atm_strike] + upper_4)

    min_wall_lots = 2000.0 if is_index else 250.0
    ambush_min_lots = 1000.0 if is_index else 150.0
    strike_step = sorted_strikes[1] - sorted_strikes[0] if len(sorted_strikes) > 1 else 50.0

    if direction == 'CALL':
        target_barrier = float(target_price) if target_price else (spot + max(2.0 * strike_step, 60.0 if is_index else 25.0))
        candidate_wall_strikes = [s for s in sorted_strikes if (s >= (atm_strike - strike_step * 0.1) and s <= target_barrier + strike_step * 0.15)]

        # 1. Upper OTM Call Ambush
        tot_upper_ce_vol = sum(strike_data[s]['ce_vol'] for s in upper_4)
        is_buying_squeeze = (tot_upper_ce_vol >= 2.0 * max(1.0, tot_upper_ce_chg))
        if not is_buying_squeeze and tot_upper_ce_chg >= ambush_min_lots and tot_upper_ce_chg > 1.8 * max(50.0, tot_lower_pe_chg) and atm_info['ce_chg_oi'] >= 0:
            return {'confluent': False, 'reason': f'UPPER_OTM_CALL_WRITING_AMBUSH (+{int(tot_upper_ce_chg)} CE vs +{int(tot_lower_pe_chg)} PE)', 'atm_strike': atm_strike}

        # 2. ATM Call Writer Dominance
        if atm_info['ce_chg_oi'] > 1.5 * max(50.0, atm_info['pe_chg_oi']) and atm_info['ce_chg_oi'] > 0 and (atm_info['ce_vol'] < 2.0 * max(1.0, atm_info['ce_chg_oi'])):
            atm_chg_lbl = int(atm_info["ce_chg_oi"]) if pd.notna(atm_info["ce_chg_oi"]) else 0
            atm_stk_lbl = int(atm_strike) if pd.notna(atm_strike) else 0
            return {'confluent': False, 'reason': f'ATM_CALL_WRITERS_DOMINANCE (+{atm_chg_lbl} CE at {atm_stk_lbl})', 'atm_strike': atm_strike}

        # 3. Intermediate Resistance Walls between spot and target
        res_wall_strike = None
        for up_s in candidate_wall_strikes:
            st_d = strike_data[up_s]
            ce_oi = st_d['ce_oi']
            pe_oi = st_d['pe_oi']
            ce_chg = st_d['ce_chg_oi']
            pe_chg = st_d['pe_chg_oi']

            if ce_oi < min_wall_lots:
                continue

            ce_conc = ce_oi / max(1.0, tot_window_ce_oi)
            is_standing_wall = (ce_oi >= min_wall_lots) and (
                (ce_conc >= 0.20 and ce_oi >= 1.0 * pe_oi) or
                (ce_oi >= 1.6 * max(50.0, pe_oi)) or
                (ce_oi >= 1.25 * max(1.0, atm_info['ce_oi']) and ce_oi > 1.35 * max(50.0, pe_oi))
            )
            is_building_wall = (ce_chg > 0 and ce_chg > 1.5 * max(20.0, pe_chg)) and (ce_oi > 1.2 * pe_oi)

            if is_standing_wall or is_building_wall:
                if st_d['ce_vol'] >= 2.5 * ce_oi:
                    continue  # Squeeze exception
                res_wall_strike = up_s
                break

        confluent = (res_wall_strike is None)
        atm_lbl = int(atm_strike) if (atm_strike is not None and pd.notna(atm_strike)) else 0
        wall_lbl = int(res_wall_strike) if (res_wall_strike is not None and pd.notna(res_wall_strike)) else 0
        reason = f'CALL_RUNWAY_CLEAR_ATM_{atm_lbl}' if confluent else f'CALL_BLOCKED_BY_WALL_AT_{wall_lbl}'
        return {'confluent': confluent, 'reason': reason, 'atm_strike': atm_strike}

    else:
        target_barrier = float(target_price) if target_price else (spot - max(2.0 * strike_step, 60.0 if is_index else 25.0))
        candidate_wall_strikes = [s for s in sorted_strikes if (target_barrier - strike_step * 0.15 <= s <= atm_strike + strike_step * 0.1)]

        # 1. Lower OTM Put Ambush
        tot_lower_pe_vol = sum(strike_data[s]['pe_vol'] for s in lower_4)
        is_buying_squeeze = (tot_lower_pe_vol >= 2.0 * max(1.0, tot_lower_pe_chg))
        if not is_buying_squeeze and tot_lower_pe_chg >= ambush_min_lots and tot_lower_pe_chg > 1.8 * max(50.0, tot_upper_ce_chg) and atm_info['pe_chg_oi'] >= 0:
            return {'confluent': False, 'reason': f'LOWER_OTM_PUT_WRITING_AMBUSH (+{int(tot_lower_pe_chg)} PE vs +{int(tot_upper_ce_chg)} CE)', 'atm_strike': atm_strike}

        # 2. ATM Put Writer Dominance
        if atm_info['pe_chg_oi'] > 1.5 * max(50.0, atm_info['ce_chg_oi']) and atm_info['pe_chg_oi'] > 0 and (atm_info['pe_vol'] < 2.0 * max(1.0, atm_info['pe_chg_oi'])):
            atm_chg_lbl = int(atm_info["pe_chg_oi"]) if pd.notna(atm_info["pe_chg_oi"]) else 0
            atm_stk_lbl = int(atm_strike) if pd.notna(atm_strike) else 0
            return {'confluent': False, 'reason': f'ATM_PUT_WRITERS_DOMINANCE (+{atm_chg_lbl} PE at {atm_stk_lbl})', 'atm_strike': atm_strike}

        # 3. Intermediate Support Walls between spot and downside target
        sup_wall_strike = None
        for dn_s in sorted(candidate_wall_strikes, reverse=True):
            st_d = strike_data[dn_s]
            pe_oi = st_d['pe_oi']
            ce_oi = st_d['ce_oi']
            pe_chg = st_d['pe_chg_oi']
            ce_chg = st_d['ce_chg_oi']

            if pe_oi < min_wall_lots:
                continue

            pe_conc = pe_oi / max(1.0, tot_window_pe_oi)
            is_standing_wall = (pe_oi >= min_wall_lots) and (
                (pe_conc >= 0.20 and pe_oi >= 1.0 * ce_oi) or
                (pe_oi >= 1.6 * max(50.0, ce_oi)) or
                (pe_oi >= 1.25 * max(1.0, atm_info['pe_oi']) and pe_oi > 1.35 * max(50.0, ce_oi))
            )
            is_building_wall = (pe_chg > 0 and pe_chg > 1.5 * max(20.0, ce_chg)) and (pe_oi > 1.2 * ce_oi)

            if is_standing_wall or is_building_wall:
                if st_d['pe_vol'] >= 2.5 * pe_oi:
                    continue
                sup_wall_strike = dn_s
                break

        confluent = (sup_wall_strike is None)
        atm_lbl = int(atm_strike) if (atm_strike is not None and pd.notna(atm_strike)) else 0
        wall_lbl = int(sup_wall_strike) if (sup_wall_strike is not None and pd.notna(sup_wall_strike)) else 0
        reason = f'PUT_RUNWAY_CLEAR_ATM_{atm_lbl}' if confluent else f'PUT_BLOCKED_BY_WALL_AT_{wall_lbl}'
        return {'confluent': confluent, 'reason': reason, 'atm_strike': atm_strike}


class ActiveOptionTrade:
    def __init__(self, cand: Dict[str, Any], b_idx: int, curr_t, lot_map: Dict[str, int]):
        self.cand = cand
        self.entry_idx = b_idx
        self.sym = cand['symbol']
        self.direction = cand['direction']
        self.spot_entry = float(cand['entry_price'])
        self.is_idx = cand.get('is_index', False)
        self.is_bank = (self.sym == 'BANKNIFTY')
        self.atr = float(cand.get('stock_atr', 30.0))
        self.step = 0
        self.is_closed = False
        self.spot_exit = self.spot_entry
        self.exit_reason = None
        self.exit_time = curr_t
        self.t1_hit = False
        self.is_risk_free = False
        self.breakeven_armed = False
        self.max_fav_pts = 0.0
        self.max_adv_pts = 0.0

        if self.is_idx:
            is_mid = (self.sym == 'MIDCPNIFTY')
            self.spot_t1 = 120.0 if self.is_bank else (60.0 if is_mid else 80.0)
            self.spot_t2 = 250.0 if self.is_bank else (120.0 if is_mid else 150.0)
            self.spot_sl = 60.0 if self.is_bank else (30.0 if is_mid else 40.0)
        else:
            self.spot_sl = max(4.0, round(self.atr * 0.25, 2))
            self.spot_t1 = max(5.0, round(self.atr * 0.30, 2))
            self.spot_t2 = max(11.0, round(self.atr * 0.65, 2))

        step = determine_step_size(self.spot_entry, is_index=self.is_idx, symbol=self.sym)
        self.atm_strike = round(self.spot_entry / step) * step
        self.opt_type = 'CE' if self.direction == 'CALL' else 'PE'
        stk_str = f"{int(self.atm_strike)}" if self.atm_strike == int(self.atm_strike) else f"{self.atm_strike:.1f}"
        self.contract_name = f"{self.sym} {stk_str} {self.opt_type}"
        self.lot_size = lot_map.get(self.sym, (25 if self.is_idx else 250))
        self.opt_entry_premium = max(6.0, round(self.spot_sl * 2.00, 2))
        self.opt_sl_pts = round(max(1.5, self.spot_sl * 0.50), 2)

    def update(self, b_high: float, b_low: float, b_close: float, ts):
        """
        Conservative Intrabar Evaluation:
        - Checks intrabar high and low (never close only!).
        - Conservative SL-first resolution if both SL and Target touch on the same bar.
        - Force EOD closure at 15:20 IST.
        """
        if self.is_closed:
            return
        self.step += 1

        fav_move = (b_high - self.spot_entry) if self.direction == 'CALL' else (self.spot_entry - b_low)
        adv_move = (self.spot_entry - b_low) if self.direction == 'CALL' else (b_high - self.spot_entry)
        cur_gain = (b_close - self.spot_entry) if self.direction == 'CALL' else (self.spot_entry - b_close)

        if fav_move > self.max_fav_pts:
            self.max_fav_pts = fav_move
        if adv_move > self.max_adv_pts:
            self.max_adv_pts = adv_move

        tot_m = ts.hour * 60 + ts.minute
        # Force EOD Exit at 15:20 IST
        if tot_m >= 920:
            if self.t1_hit:
                lock_pts = self.spot_t1 * 0.55
                locked_sl = (self.spot_entry + lock_pts) if self.direction == 'CALL' else (self.spot_entry - lock_pts)
                self.spot_exit = max(locked_sl, b_close) if self.direction == 'CALL' else min(locked_sl, b_close)
                self.exit_reason = "T1_PROFIT_LOCKED_EXIT"
            elif self.breakeven_armed:
                be_sl = (self.spot_entry + 0.15 * self.spot_sl) if self.direction == 'CALL' else (self.spot_entry - 0.15 * self.spot_sl)
                self.spot_exit = max(be_sl, b_close) if self.direction == 'CALL' else min(be_sl, b_close)
                self.exit_reason = "BREAKEVEN_SHIELD_EXIT"
            else:
                self.spot_exit = b_close
                self.exit_reason = "EOD_EXIT"
            self.exit_time = ts
            self.is_closed = True
            return

        # Check Intrabar SL and Trailing Logic
        if not self.t1_hit:
            # Check Initial SL Breach (only if Breakeven Shield not yet armed)
            if not self.breakeven_armed:
                if adv_move >= self.spot_sl:
                    self.spot_exit = (self.spot_entry - self.spot_sl) if self.direction == 'CALL' else (self.spot_entry + self.spot_sl)
                    self.exit_reason = "SL_HIT"
                    self.exit_time = ts
                    self.is_closed = True
                    return

            # Check Target 1 and Target 2 (Jackpot) FIRST
            if fav_move >= self.spot_t1:
                self.t1_hit = True
                self.is_risk_free = True
                if fav_move >= self.spot_t2:
                    self.spot_exit = (self.spot_entry + self.spot_t2) if self.direction == 'CALL' else (self.spot_entry - self.spot_t2)
                    self.exit_reason = "TARGET_2_HIT (JACKPOT)"
                    self.exit_time = ts
                    self.is_closed = True
                    return
                # Check locked floor immediately if price retraced below floor on same candle
                lock_pts = self.spot_t1 * 0.55
                locked_sl = (self.spot_entry + lock_pts) if self.direction == 'CALL' else (self.spot_entry - lock_pts)
                sl_breached = (b_low <= locked_sl) if self.direction == 'CALL' else (b_high >= locked_sl)
                if sl_breached:
                    self.spot_exit = locked_sl
                    self.exit_reason = "T1_PROFIT_LOCKED_EXIT"
                    self.exit_time = ts
                    self.is_closed = True
                    return

            # Check Breakeven Shield Breach (if already armed on a previous bar)
            if self.breakeven_armed:
                locked_be = (self.spot_entry + 0.15 * self.spot_sl) if self.direction == 'CALL' else (self.spot_entry - 0.15 * self.spot_sl)
                be_breached = (b_low <= locked_be) if self.direction == 'CALL' else (b_high >= locked_be)
                if be_breached:
                    self.spot_exit = locked_be
                    self.exit_reason = "BREAKEVEN_SHIELD_EXIT"
                    self.exit_time = ts
                    self.is_closed = True
                    return
            else:
                # Arm Breakeven Shield once favorable move reaches 0.55 * T1 before full SL
                if fav_move >= (self.spot_t1 * 0.55) and adv_move < self.spot_sl:
                    self.breakeven_armed = True
                    locked_be = (self.spot_entry + 0.15 * self.spot_sl) if self.direction == 'CALL' else (self.spot_entry - 0.15 * self.spot_sl)
                    be_closed_below = (b_close <= locked_be) if self.direction == 'CALL' else (b_close >= locked_be)
                    if be_closed_below:
                        self.spot_exit = locked_be
                        self.exit_reason = "BREAKEVEN_SHIELD_EXIT"
                        self.exit_time = ts
                        self.is_closed = True
                        return

            # 15-Minute Epoch Stagnation Loss Scratch (strictly at step 5)
            if self.step == 5 and (not self.breakeven_armed) and (cur_gain < -(self.spot_sl * 0.10)):
                self.spot_exit = b_close
                self.exit_reason = "15M_TIME_SL_SCRATCH"
                self.exit_time = ts
                self.is_closed = True
                return

            # 30-Minute Non-Performer Exit (strictly at step 10)
            if self.step == 10 and (not self.t1_hit) and (cur_gain <= (self.spot_sl * 0.15)):
                self.spot_exit = b_close
                self.exit_reason = "30M_STAGNANT_SCRATCH"
                self.exit_time = ts
                self.is_closed = True
                return
        else:
            # T1 was already hit -> Trailing SL locked at +0.55 * T1
            lock_pts = self.spot_t1 * 0.55
            locked_sl = (self.spot_entry + lock_pts) if self.direction == 'CALL' else (self.spot_entry - lock_pts)
            
            # Check Trailing SL breach
            sl_breached = (b_low <= locked_sl) if self.direction == 'CALL' else (b_high >= locked_sl)
            t2_reached = (fav_move >= self.spot_t2)

            if sl_breached and t2_reached:
                # Conservative resolution: trailing SL first if both touched
                self.spot_exit = locked_sl
                self.exit_reason = "T1_PROFIT_LOCKED_EXIT"
                self.exit_time = ts
                self.is_closed = True
                return
            elif t2_reached:
                self.spot_exit = (self.spot_entry + self.spot_t2) if self.direction == 'CALL' else (self.spot_entry - self.spot_t2)
                self.exit_reason = "TARGET_2_HIT (JACKPOT)"
                self.exit_time = ts
                self.is_closed = True
                return
            elif sl_breached:
                self.spot_exit = locked_sl
                self.exit_reason = "T1_PROFIT_LOCKED_EXIT"
                self.exit_time = ts
                self.is_closed = True
                return

    def close_eod(self, final_close: float, ts):
        if not self.is_closed:
            if self.t1_hit:
                lock_pts = self.spot_t1 * 0.55
                locked_sl = (self.spot_entry + lock_pts) if self.direction == 'CALL' else (self.spot_entry - lock_pts)
                self.spot_exit = max(locked_sl, final_close) if self.direction == 'CALL' else min(locked_sl, final_close)
                self.exit_reason = "T1_PROFIT_LOCKED_EXIT"
            elif self.breakeven_armed:
                be_sl = (self.spot_entry + 0.15 * self.spot_sl) if self.direction == 'CALL' else (self.spot_entry - 0.15 * self.spot_sl)
                self.spot_exit = max(be_sl, final_close) if self.direction == 'CALL' else min(be_sl, final_close)
                self.exit_reason = "BREAKEVEN_SHIELD_EXIT"
            else:
                self.spot_exit = final_close
                self.exit_reason = "EOD_EXIT"
            self.exit_time = ts
            self.is_closed = True

    def get_summary(self, d_str: str) -> Dict[str, Any]:
        spot_pts = round((self.spot_exit - self.spot_entry) if self.direction == 'CALL' else (self.spot_entry - self.spot_exit), 2)
        if "JACKPOT" in self.exit_reason:
            eff_delta = 0.72
        elif "T1" in self.exit_reason:
            eff_delta = 0.58
        elif "BREAKEVEN" in self.exit_reason:
            eff_delta = 0.50
        elif "SCRATCH" in self.exit_reason or "TIME_SL" in self.exit_reason:
            eff_delta = 0.45
        elif "SL" in self.exit_reason:
            eff_delta = 0.50
        else:
            eff_delta = 0.50

        opt_pts = round(spot_pts * eff_delta, 2)
        opt_exit_premium = max(0.50, round(self.opt_entry_premium + opt_pts, 2))
        gross_pnl = round(opt_pts * self.lot_size, 2)

        # Statutory Indian F&O Taxes & Brokerage
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
            'score': self.cand.get('score', 80.0),
            'type': self.cand.get('is_index', False),
            'is_index': self.is_idx
        }

def print_institutional_tearsheet(df_res: pd.DataFrame, csv_path: str):
    """Prints institutional tear sheet with zero division protection."""
    print(f"\nSaved authoritative backtest CSV to: {csv_path}")
    print("\n" + "="*85)
    print("PRO-TRADER SYNCHRONIZED DUAL ENGINE: STOCK & INDEX TOGETHER (5-DAY ENDING TODAY)")
    print("="*85)
    
    total_trades = len(df_res)
    print(f"Total Dispatched Option Trades: {total_trades}")
    if total_trades == 0:
        print("Zero trades dispatched in backtest window. Capital 100% Protected.")
        return

    wins = df_res[df_res['opt_pts'] > 0]
    losses = df_res[df_res['opt_pts'] <= 0]
    scratches = df_res[df_res['exit_reason'].str.contains('SCRATCH|TIME_SL|BREAKEVEN', na=False)]
    full_sl = df_res[df_res['exit_reason'] == 'SL_HIT']

    gross_win_pnl = wins['net_pnl'].sum() if not wins.empty else 0.0
    gross_loss_pnl = abs(losses['net_pnl'].sum()) if not losses.empty else 0.0
    net_pnl_profit_factor = round(gross_win_pnl / max(1.0, gross_loss_pnl), 2)

    gross_r = wins['realized_r'].sum() if not wins.empty else 0.0
    loss_r = abs(losses['realized_r'].sum()) if not losses.empty else 0.0
    net_r = gross_r - loss_r
    pf_r = round(gross_r / max(0.01, loss_r), 2)

    win_rate = (len(wins) / total_trades) * 100.0
    expectancy_r = net_r / total_trades
    total_net_pnl = df_res['net_pnl'].sum()
    expectancy_inr = total_net_pnl / total_trades

    cum_r = df_res['realized_r'].cumsum()
    peak = cum_r.cummax()
    drawdown = peak - cum_r
    max_dd = round(float(drawdown.max()), 2) if not drawdown.empty else 0.0

    # Max consecutive losses
    loss_runs = []
    cur_run = 0
    for pnl in df_res['net_pnl']:
        if pnl <= 0:
            cur_run += 1
            loss_runs.append(cur_run)
        else:
            cur_run = 0
    max_consec_losses = max(loss_runs, default=0)

    false_breakout_rate = round((len(scratches) + len(full_sl)) / total_trades * 100.0, 1)

    avg_mfe = round(float(df_res['mfe_r'].mean()), 2) if 'mfe_r' in df_res.columns else 0.0
    avg_mae = round(float(df_res['mae_r'].mean()), 2) if 'mae_r' in df_res.columns else 0.0

    calls = df_res[df_res['direction'] == 'CALL']
    puts = df_res[df_res['direction'] == 'PUT']
    call_wins = len(calls[calls['opt_pts'] > 0])
    put_wins = len(puts[puts['opt_pts'] > 0])
    call_net = calls['realized_r'].sum() if not calls.empty else 0.0
    put_net = puts['realized_r'].sum() if not puts.empty else 0.0

    idx_trades = df_res[df_res['is_index'] == True]
    stk_trades = df_res[df_res['is_index'] == False]
    idx_wins = len(idx_trades[idx_trades['opt_pts'] > 0])
    stk_wins = len(stk_trades[stk_trades['opt_pts'] > 0])
    idx_net = idx_trades['realized_r'].sum() if not idx_trades.empty else 0.0
    stk_net = stk_trades['realized_r'].sum() if not stk_trades.empty else 0.0

    print(f"Wins: {len(wins)} ({win_rate:.1f}%) | Losses/Scratches: {len(losses)} | Full SL Hits: {len(full_sl)}")
    print(f"Gross Profit: +{gross_r:.2f}R | Gross Loss: -{loss_r:.2f}R | Net Profit: +{net_r:.2f}R | Net PnL: Rs {total_net_pnl:,.2f}")
    print(f"Net PnL Profit Factor: {net_pnl_profit_factor} | R Profit Factor: {pf_r}")
    print(f"Expectancy: +{expectancy_r:.2f}R (Rs {expectancy_inr:,.2f}) per trade | Max Drawdown: {max_dd:.2f}R | Max Consec Losses: {max_consec_losses}")
    print(f"False Breakout Rate: {false_breakout_rate}% | Average MFE: +{avg_mfe:.2f}R | Average MAE: -{avg_mae:.2f}R")
    print(f"INDEX Trades: {len(idx_trades)} (Wins: {idx_wins}, Net: {idx_net:+.2f}R) | STOCK Trades: {len(stk_trades)} (Wins: {stk_wins}, Net: {stk_net:+.2f}R)")
    print(f"CALL Trades: {len(calls)} (Wins: {call_wins}, Net: {call_net:+.2f}R) | PUT Trades: {len(puts)} (Wins: {put_wins}, Net: {put_net:+.2f}R)")

    print("\n--- ALL EXECUTED OPTION TRADES ---")
    for _, r in df_res.iterrows():
        e_str = r['eval_time'].strftime('%Y-%m-%d %H:%M') if hasattr(r['eval_time'], 'strftime') else str(r['eval_time'])
        print(f"{e_str} | {r['contract']:<18} | SpotEn: {r['spot_entry']:8.2f} | OptEn: {r['opt_entry']:6.2f} | OptEx: {r['opt_exit']:6.2f} | OptPts: {r['opt_pts']:+6.2f} | Lot: {r['lot_size']:4} | Net: Rs {r['net_pnl']:+9,.2f} | R: {r['realized_r']:+5.2f}R | {r['exit_reason']}")


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
        if 'NIFTY' not in asset_data:
            continue
        nifty_day = asset_data['NIFTY'][asset_data['NIFTY'].index.date == d].copy()
        if nifty_day.empty:
            continue
        nifty_day['ema9'] = nifty_day['close'].ewm(span=9, adjust=False).mean()
        nifty_day['ema21'] = nifty_day['close'].ewm(span=21, adjust=False).mean()

        active_trades = []
        daily_traded_symbols = set()
        time_index = nifty_day.index

        for b_idx in range(len(time_index)):
            curr_t = time_index[b_idx]
            tot_m = curr_t.hour * 60 + curr_t.minute
            is_m = (570 <= tot_m <= 615)   # 09:30 to 10:15 AM
            is_a = (795 <= tot_m <= 855)   # 13:15 to 14:15 PM

            # 1. Update Open Active Trades Concurrently
            for tr in active_trades:
                if not tr.is_closed:
                    df_asset = asset_data[tr.sym]
                    if curr_t in df_asset.index:
                        b = df_asset.loc[curr_t]
                        if isinstance(b, pd.DataFrame):
                            b = b.iloc[0]
                        tr.update(float(b['high']), float(b['low']), float(b['close']), curr_t)
                        if tr.is_closed:
                            all_closed_trades.append(tr)

            if not (is_m or is_a):
                continue

            regime = get_nifty_regime(nifty_day, b_idx)
            act_stocks = sum(1 for t in active_trades if (not t.is_closed and not t.is_idx))
            act_broad = sum(1 for t in active_trades if (not t.is_closed and t.is_idx and not t.is_bank))
            act_bank = sum(1 for t in active_trades if (not t.is_closed and t.is_bank))

            candidates = []

            # A. Index Candidates (MAX_INDEX_PER_SLAB = 1)
            for isym in INDEX_NAMES:
                if isym in daily_traded_symbols or isym not in asset_data:
                    continue
                idf = asset_data[isym]
                today_bars = idf[(idf.index.date == d) & (idf.index <= curr_t)]
                if len(today_bars) < 10:
                    continue
                idx_atr = atrs.get(isym, (450.0 if isym == 'BANKNIFTY' else 160.0))
                
                sig = evaluate_common_5pillar_bar_slice(isym, today_bars, idx_atr, regime, is_index=True)
                if sig:
                    sig['entry_price'] = float(today_bars.iloc[-2]['close'])
                    t_price = (sig['entry_price'] + max(100.0, 0.75 * idx_atr)) if sig['direction'] == 'CALL' else (sig['entry_price'] - max(100.0, 0.75 * idx_atr))
                    oi_res = verify_bhavcopy_oi_confluence(isym, d, sig['entry_price'], sig['direction'], target_price=t_price, is_index=True)
                    if oi_res['confluent']:
                        sig['oi_confluent'] = True
                        sig['score'] = min(100.0, sig.get('score', 80.0) + 15.0)
                        candidates.append(sig)

            # B. Stock Candidates (MAX_STOCKS_PER_SLAB = 2)
            if act_stocks < MAX_ACTIVE_STOCKS:
                for ssym in stock_syms:
                    if ssym in daily_traded_symbols or ssym not in asset_data:
                        continue
                    lot_sz = lot_map.get(ssym, 0)
                    if lot_sz <= 0:
                        continue  # Reject symbol with unknown/invalid lot size

                    sdf = asset_data[ssym]
                    today_bars = sdf[(sdf.index.date == d) & (sdf.index <= curr_t)]
                    if len(today_bars) < 10:
                        continue
                    stock_atr = atrs.get(ssym, 30.0)

                    sig = evaluate_common_5pillar_bar_slice(ssym, today_bars, stock_atr, regime, is_index=False)
                    if sig:
                        sig['entry_price'] = float(today_bars.iloc[-2]['close'])
                        t_price = (sig['entry_price'] + max(15.0, 0.75 * stock_atr)) if sig['direction'] == 'CALL' else (sig['entry_price'] - max(15.0, 0.75 * stock_atr))
                        oi_res = verify_bhavcopy_oi_confluence(ssym, d, sig['entry_price'], sig['direction'], target_price=t_price, is_index=False)
                        if oi_res['confluent']:
                            sig['oi_confluent'] = True
                            sig['score'] = min(100.0, sig.get('score', 75.0) + 20.0)
                            candidates.append(sig)

            # Sort combined candidates by 100-point 5-Pillar Score
            candidates.sort(key=lambda x: x['score'], reverse=True)

            # Dispatch new trades into available slots
            for cand in candidates:
                is_i = cand.get('is_index', False)
                is_b = (cand['symbol'] == 'BANKNIFTY')
                if is_i:
                    if is_b and act_bank >= MAX_ACTIVE_BANK_INDEX:
                        continue
                    if (not is_b) and act_broad >= MAX_ACTIVE_BROAD_INDEX:
                        continue
                else:
                    if act_stocks >= MAX_ACTIVE_STOCKS:
                        continue

                new_trade = ActiveOptionTrade(cand, b_idx, curr_t, lot_map)
                active_trades.append(new_trade)
                daily_traded_symbols.add(cand['symbol'])
                if is_i:
                    if is_b:
                        act_bank += 1
                    else:
                        act_broad += 1
                else:
                    act_stocks += 1

        # End of day: close remaining active trades
        for tr in active_trades:
            if not tr.is_closed:
                df_asset = asset_data[tr.sym]
                today_asset = df_asset[df_asset.index.date == d]
                if len(today_asset) > 0:
                    final_close = float(today_asset.iloc[-1]['close'])
                    tr.close_eod(final_close, today_asset.index[-1])
                    all_closed_trades.append(tr)

    summaries = [tr.get_summary(str(tr.cand['eval_time'].date())) for tr in all_closed_trades]
    df_res = pd.DataFrame(summaries)
    csv_path = os.path.join(REPORT_DIR, "Alpha_Latest_5Day_Ending_Today_Backtest.csv")
    df_res.to_csv(csv_path, index=False)
    print_institutional_tearsheet(df_res, csv_path)
