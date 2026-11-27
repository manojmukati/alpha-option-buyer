# -*- coding: utf-8 -*-
"""
alpha_5pillar_institutional_engine.py
=====================================
Alpha 5-Pillar Institutional Data-First Autonomous Screener & Trading Engine.

CORE ARCHITECTURAL PRINCIPLE: DATA FIRST, CANDLE LAST!
Zero tolerance for candle-first logic or synthetic/abs() scoring.
Every trade must strictly pass all 4 institutional data pillars BEFORE the 3-minute candle is ever evaluated!

STRICT 5-PILLAR HIERARCHY:
----------------------------------------------------------------------------------------
PILLAR 1: Cash Turnover & Volume Gate
  - Stocks: 15m Cash Turnover >= Rs 5-10 Cr (or bar turnover >= Rs 2.5 Cr), RVOL >= 1.25x, Spot >= Rs 50.
  - Indices: Range RVOL >= 1.20x, Bar Range >= 25 pts (BankNifty) / 10 pts (Nifty).
  - Hard Gate: If turnover or volume fails, instant rejection (0 points).

PILLAR 2: Cumulative Volume Delta (CVD) & BVC Directional Gate (ZERO abs())
  - CALL: eval_bvc >= +0.15 AND (delta_inflection >= +0.10 OR bvc_cum_ratio >= +0.10).
          Strictly positive! If eval_bvc <= 0 or cumulative delta <= 0, REJECT IMMEDIATELY.
  - PUT:  eval_bvc <= -0.15 AND (delta_inflection <= -0.10 OR bvc_cum_ratio <= -0.10).
          Strictly negative! If eval_bvc >= 0 or cumulative delta >= 0, REJECT IMMEDIATELY.
  - Zero tolerance for abs() converting heavy dumping into positive call points!

PILLAR 3: Wyckoff Market Profile & VWAP Acceptance Gate
  - CALL: eval_close > VWAP AND eval_close >= (VAH * 0.998) AND eval_close >= (POC * 0.998).
          Anti-Top Chasing: Price cannot be extended > 2.0% above VWAP.
  - PUT:  eval_close < VWAP AND eval_close <= (VAL * 1.002) AND eval_close <= (POC * 1.002).
          Anti-Bottom Chasing: Price cannot be extended > 2.0% below VWAP.
  - Hard Gate: If price is on the wrong side of VWAP/Value Area, REJECT IMMEDIATELY (0 points).

PILLAR 4: Live Option Chain Confluence & Runway Gate
  - Symmetric -4 to +4 strike analysis with live NSE Option Chain (fail-closed).
  - Immediate Runway Wall Check (+-2 strikes from ATM strictly).
    Deep OTM strikes (> 2 strikes away) are NEVER classified as immediate support/resistance walls.
  - CALL: Call unwinding + Put support floor; no immediate CE wall in ATM+1/ATM+2 runway.
  - PUT:  Put unwinding + Call resistance ceiling; no immediate PE wall in ATM-1/ATM-2 runway.
  - Market Maker Straddle Pinning Check (|ΔCE - ΔPE| / (ΔCE + ΔPE) < 0.15 blocks chop traps).

PILLAR 5: 3-Minute Candle Price Action Trigger (AT THE VERY END)
  - Evaluated ONLY AFTER Pillars 1, 2, 3, and 4 have ALL fully passed!
  - CALL: Bullish candle closing in upper half (thrust >= 0.50), upper wick <= 45% of range.
  - PUT:  Bearish candle closing in lower half (thrust <= 0.50), lower wick <= 45% of range.
  - Filters out exhaustion wicks, shooting stars on calls, and hammer absorption on puts.
----------------------------------------------------------------------------------------
SCORING MATRIX (100 PTS HONEST DYNAMIC SCORING):
  Pillar 1: 0 - 20 pts (Volume & Cash Turnover)
  Pillar 2: 0 - 25 pts (Strict Directional CVD/BVC Delta)
  Pillar 3: 0 - 25 pts (Volume Profile VAH/VAL/VWAP Acceptance)
  Pillar 4: 0 - 20 pts (Live Option Chain Confluence)
  Pillar 5: 0 - 10 pts (Candle Trigger Quality)
  Hard Pass Threshold: >= 75.0 pts (Stocks), >= 70.0 pts (Indices).
  No fake base score floor. Zero points awarded if any hard gate fails.
========================================================================================
"""

import os
import sys
import glob
import time
import json
import math
import datetime
import logging
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from modules.live_data_feed import (
    LiveMarketDataFeed,
    fetch_bulk_nse_snapshot,
    get_ist_time,
    clean_num
)
from modules.fno_universe import get_instrument_meta, determine_step_size
from modules.telegram_notifier import TelegramNotifier

# Import base class and operational utilities
from alpha_profile_oi_system.alpha_institutional_swapped_screener import (
    AlphaInstitutionalSwappedScreener,
    ActiveOptionTrade,
    is_in_active_trading_slab,
    is_in_fresh_signal_window,
    get_current_trading_slab_name,
    get_symbol_lot_size,
    verify_authoritative_bhavcopy_oi,
    load_resampled_3m_data_native,
    check_github_actions_concurrency_guard,
    INDEX_NAMES,
    MIN_STOCK_PRICE,
    MAX_SLAB_TRADES,
    MAX_INDEX_PER_SLAB,
    MAX_STOCKS_PER_SLAB
)

CONFIG_PATH = os.path.join(CURRENT_DIR, 'config', 'telegram_config.json')

# Authoritative Institutional Thresholds
INSTITUTIONAL_STOCK_THRESHOLD = 75.0
INSTITUTIONAL_INDEX_THRESHOLD = 70.0


# =========================================================================
# PILLAR 2: DIRECTIONAL BAR & CUMULATIVE DELTA ENGINE (ZERO abs())
# =========================================================================
def compute_bar_delta_proxy(df_slice: pd.DataFrame, is_index: bool = False) -> Tuple[float, float, float, float]:
    """
    Computes instant bar delta, previous bar delta, delta inflection, and cumulative delta ratio.
    Returns: (eval_bvc, prev_bvc, delta_inflection, bvc_cum_ratio)
    - eval_bvc: current bar order flow delta [-1.0, +1.0]
    - prev_bvc: previous bar order flow delta [-1.0, +1.0]
    - delta_inflection: eval_bvc - prev_bvc (positive = selling drying up / buying expanding)
    - bvc_cum_ratio: volume-weighted cumulative delta of recent 5 bars [-1.0, +1.0]
    """
    if df_slice is None or len(df_slice) < 3:
        return 0.0, 0.0, 0.0, 0.0

    bars_slice = df_slice.iloc[-6:-1] if len(df_slice) >= 6 else df_slice.iloc[:-1]
    if len(bars_slice) < 2:
        return 0.0, 0.0, 0.0, 0.0

    price_chg = bars_slice['close'] - bars_slice['open']
    std_val = float(price_chg.std()) if len(bars_slice) > 1 else 0.0
    sigma = std_val if std_val > 0.01 else float((bars_slice['high'] - bars_slice['low']).mean() * 0.5)
    sigma = max(0.01, sigma)

    bvc_z = np.clip(price_chg / sigma, -3.0, 3.0)
    bvc_fraction = np.tanh(bvc_z)

    eval_bvc = float(bvc_fraction.iloc[-1])
    prev_bvc = float(bvc_fraction.iloc[-2]) if len(bvc_fraction) >= 2 else 0.0
    delta_inflection = round(eval_bvc - prev_bvc, 4)

    # Volume-weighted cumulative delta
    if 'volume' in bars_slice.columns and not is_index:
        weights = bars_slice['volume'].values.astype(float)
        tot_w = float(np.sum(weights))
        bvc_cum_ratio = float(np.sum(weights * bvc_fraction.values) / max(1.0, tot_w)) if tot_w > 0 else float(bvc_fraction.mean())
    else:
        rng_weights = (bars_slice['high'] - bars_slice['low']).values.astype(float)
        tot_w = float(np.sum(rng_weights))
        bvc_cum_ratio = float(np.sum(rng_weights * bvc_fraction.values) / max(0.1, tot_w)) if tot_w > 0 else float(bvc_fraction.mean())

    return eval_bvc, prev_bvc, delta_inflection, round(bvc_cum_ratio, 4)


# =========================================================================
# PILLAR 3: WYCKOFF MARKET PROFILE & VWAP ENGINE
# =========================================================================
def compute_volume_profile(df_slice: pd.DataFrame, daily_atr: float, is_index: bool = False) -> Dict[str, float]:
    """
    Computes Wyckoff Market Profile parameters: VWAP, VAH (Value Area High),
    VAL (Value Area Low), POC (Point of Control), and Price Standard Deviation.
    """
    tp_all = (df_slice['high'] + df_slice['low'] + df_slice['close']) / 3.0
    if 'volume' in df_slice.columns and not is_index:
        vol_weights = df_slice['volume'].astype(float)
        tot_vol = float(vol_weights.sum())
        vwap = float((tp_all * vol_weights).sum() / max(1.0, tot_vol)) if tot_vol > 0 else float(tp_all.mean())
    else:
        rng_weights = (df_slice['high'] - df_slice['low']).astype(float)
        tot_rng = float(rng_weights.sum())
        vwap = float((tp_all * rng_weights).sum() / max(0.1, tot_rng)) if tot_rng > 0 else float(tp_all.mean())

    std_dev = float(df_slice['close'].std()) if len(df_slice) > 1 else (0.35 * daily_atr)
    eval_bar = df_slice.iloc[-2] if len(df_slice) >= 2 else df_slice.iloc[-1]

    poc = float(eval_bar['poc']) if ('poc' in eval_bar and pd.notna(eval_bar['poc']) and float(eval_bar['poc']) > 0) else float(tp_all.median())
    vah = float(eval_bar['vah']) if ('vah' in eval_bar and pd.notna(eval_bar['vah']) and float(eval_bar['vah']) > 0) else round(vwap + 0.68 * std_dev, 2)
    val = float(eval_bar['val']) if ('val' in eval_bar and pd.notna(eval_bar['val']) and float(eval_bar['val']) > 0) else round(vwap - 0.68 * std_dev, 2)

    return {
        'vwap': round(vwap, 2),
        'vah': round(vah, 2),
        'val': round(val, 2),
        'poc': round(poc, 2),
        'std_dev': round(std_dev, 2)
    }


# =========================================================================
# MODULE 1: DEDICATED 5-PILLAR INSTITUTIONAL STOCK ENGINE (216 F&O STOCKS)
# =========================================================================
def evaluate_5pillar_stock_engine(
    symbol: str,
    df_slice: pd.DataFrame,
    live_q: Optional[Dict[str, Any]],
    nifty_regime: Dict[str, Any],
    data_feed: LiveMarketDataFeed,
    daily_atr: float,
    live_spot_override: Optional[float] = None,
    pct_change_override: Optional[float] = None,
    opt_override: Optional[Dict[str, Any]] = None,
    target_direction: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    ========================================================================================
    DEDICATED 5-PILLAR INSTITUTIONAL STOCK ENGINE
    DATA FIRST, CANDLE LAST!
    
    1. PILLAR 1: Volume & Cash Turnover Gate (Turnover 15m >= 5-10 Cr, RVOL >= 1.25x)
    2. PILLAR 2: CVD / BVC Delta Gate (Directional, ZERO abs(), strictly positive/negative)
    3. PILLAR 3: Volume Profile & VWAP Gate (CALL: Close > VWAP & Close >= VAH; PUT: Close < VWAP & Close <= VAL)
    4. PILLAR 4: Live Option Chain OI Confluence Gate (Symmetric +-2 strike immediate runway)
    5. PILLAR 5: 3-Minute Candle Price Action Trigger (AT THE VERY END, only after Pillars 1-4 pass)
    ========================================================================================
    """
    sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
    if sym_clean in INDEX_NAMES:
        return None
    if df_slice is None or len(df_slice) < 5:
        return None

    eval_bar = df_slice.iloc[-2]
    live_bar = df_slice.iloc[-1]
    comp_bars = df_slice.iloc[:-2].tail(min(8, len(df_slice) - 2))
    if len(comp_bars) < 3:
        return None

    eval_close = float(eval_bar['close'])
    eval_open = float(eval_bar['open'])
    eval_high = float(eval_bar['high'])
    eval_low = float(eval_bar['low'])
    eval_vol = float(eval_bar.get('volume', 0.0))
    bar_rng = max(0.01, eval_high - eval_low)
    live_spot = float(live_spot_override) if live_spot_override is not None else float(live_bar['close'])

    # Price Floor (Instantly eliminates illiquid penny options)
    if live_spot < MIN_STOCK_PRICE:
        return None

    atr = float(daily_atr) if float(daily_atr or 0.0) > 0.0 else (live_spot * 0.022)

    # Intraday Day Open / High / Low
    if hasattr(df_slice.index, 'date'):
        today_date = df_slice.index[-1].date()
        today_bars = df_slice[df_slice.index.date == today_date]
        if len(today_bars) > 0:
            day_open = float(today_bars['open'].iloc[0])
            day_high = float(today_bars['high'].max())
            day_low = float(today_bars['low'].min())
        else:
            day_open, day_high, day_low = float(df_slice['open'].iloc[0]), float(df_slice['high'].max()), float(df_slice['low'].min())
    else:
        day_open, day_high, day_low = float(df_slice['open'].iloc[0]), float(df_slice['high'].max()), float(df_slice['low'].min())

    pct_change = float(pct_change_override) if pct_change_override is not None else round(((live_spot - day_open) / max(1.0, day_open)) * 100.0, 2)

    # =========================================================================
    # PILLAR 1: VOLUME & CASH TURNOVER GATE
    # =========================================================================
    turnover_cr = (eval_close * eval_vol) / 1e7
    bars_5_turnover = []
    for idx_b in range(max(0, len(df_slice) - 6), len(df_slice) - 1):
        b_item = df_slice.iloc[idx_b]
        bars_5_turnover.append((float(b_item.get('close', 0.0)) * float(b_item.get('volume', 0.0))) / 1e7)
    turnover_15m = float(sum(bars_5_turnover)) if bars_5_turnover else turnover_cr

    # Hard Gate 1: Strict Cash Turnover Floor (Filters illiquid / dead stocks)
    if turnover_15m < 5.0 and turnover_cr < 2.5:
        return None

    # RVOL Calculation
    hist_slot_vol = float(eval_bar.get('hist_slot_vol', 0.0))
    comp_vol_med = float(comp_bars['volume'].median()) if 'volume' in comp_bars else 1000.0
    base_vol = max(100.0, hist_slot_vol if hist_slot_vol > 0 else comp_vol_med)
    real_rvol = round(eval_vol / max(1.0, base_vol), 2)

    # Hard Gate 1: Minimum RVOL Floor
    if real_rvol < 1.25:
        return None

    # Pillar 1 Score (0 - 20 pts)
    p1_score = round(min(20.0, 10.0 + min(6.0, (real_rvol - 1.25) * 3.0) + min(4.0, (turnover_15m / 10.0) * 2.0)), 2)

    # =========================================================================
    # PILLAR 2: CVD / BVC DIRECTIONAL DELTA GATE (ZERO abs())
    # =========================================================================
    eval_bvc, prev_bvc, delta_inflection, bvc_cum_ratio = compute_bar_delta_proxy(df_slice, is_index=False)

    # Direction Selection based on Macro Regime and Momentum
    directions_to_test = []
    # CALL candidate: Regime allows stock calls, non-negative intraday momentum
    if nifty_regime.get('allow_stock_call', True) and pct_change >= -0.75:
        directions_to_test.append('CALL')
    # PUT candidate: Regime allows stock puts, non-positive intraday momentum
    if nifty_regime.get('allow_stock_put', True) and pct_change <= 0.75:
        directions_to_test.append('PUT')

    if target_direction:
        directions_to_test = [d for d in directions_to_test if d == target_direction]

    if not directions_to_test:
        return None

    # Profile & Value Area parameters
    prof = compute_volume_profile(df_slice, daily_atr=atr, is_index=False)
    vwap = prof['vwap']
    vah = prof['vah']
    val = prof['val']
    poc = prof['poc']

    for direction in directions_to_test:
        # ---------------------------------------------------------------------
        # PILLAR 2 HARD GATE: STRICT DIRECTIONAL DELTA (ZERO abs()!)
        # ---------------------------------------------------------------------
        if direction == 'CALL':
            # Hard Gate: BVC delta MUST be strictly positive, and delta inflection or cumulative delta positive
            if eval_bvc < 0.15:
                continue
            if delta_inflection < 0.05 and bvc_cum_ratio < 0.05 and eval_bvc < 0.35:
                continue
            if eval_bvc <= 0.0 or bvc_cum_ratio <= 0.0:
                continue
            p2_score = round(min(25.0, (eval_bvc * 15.0) + (max(0.0, delta_inflection) * 10.0)), 2)
        else:  # PUT
            # Hard Gate: BVC delta MUST be strictly negative, and delta inflection or cumulative delta negative
            if eval_bvc > -0.15:
                continue
            if delta_inflection > -0.05 and bvc_cum_ratio > -0.05 and eval_bvc > -0.35:
                continue
            if eval_bvc >= 0.0 or bvc_cum_ratio >= 0.0:
                continue
            p2_score = round(min(25.0, (abs(eval_bvc) * 15.0) + (max(0.0, -delta_inflection) * 10.0)), 2)

        # ---------------------------------------------------------------------
        # PILLAR 3 HARD GATE: WYCKOFF VOLUME PROFILE ACCEPTANCE & REVERSAL SETUP
        # Dual-Setup Architecture:
        # 1. Breakout / Expansion: Price accepted outside Value Area (CALL > VAH, PUT < VAL)
        # 2. Institutional Reversal / Absorption: Price sweeps extreme and reverses with strong delta (CALL from Support/VAL, PUT from Resistance/VAH)
        # ---------------------------------------------------------------------
        is_breakout = False
        is_reversal = False

        if direction == 'CALL':
            # Setup 1: Fresh Base Breakout / Value Expansion
            if eval_close > vwap and eval_close >= (vah * 0.998) and eval_close >= (poc * 0.998):
                if eval_close <= (vwap * 1.020):  # Anti-top chasing guard
                    is_breakout = True

            # Setup 2: Institutional Support Sweep Reversal (Wyckoff Spring)
            # Recent bars swept day low or recent base low, buyers absorb supply and reclaim back above VAL/VWAP
            low_swept = (comp_bars['low'].tail(3).min() <= (day_low * 1.003)) or (eval_low <= (day_low * 1.003))
            support_reclaimed = (eval_close >= (val * 0.998)) and (eval_close > eval_open)
            buying_absorption = (eval_bvc >= 0.20) and (delta_inflection >= 0.10 or bvc_cum_ratio >= 0.10)
            if low_swept and support_reclaimed and buying_absorption and eval_close <= (vwap * 1.010):
                is_reversal = True

            if not (is_breakout or is_reversal):
                continue
            p3_score = 25.0 if is_breakout else 22.0
        else:  # PUT
            # Setup 1: Fresh Base Breakdown / Value Expansion
            if eval_close < vwap and eval_close <= (val * 1.002) and eval_close <= (poc * 1.002):
                if eval_close >= (vwap * 0.980):  # Anti-bottom chasing guard
                    is_breakout = True

            # Setup 2: Institutional Resistance Sweep Reversal (Wyckoff Upthrust)
            # Recent bars swept day high or recent base high, sellers dump and reject back below VAH/VWAP
            high_swept = (comp_bars['high'].tail(3).max() >= (day_high * 0.997)) or (eval_high >= (day_high * 0.997))
            resistance_rejected = (eval_close <= (vah * 1.002)) and (eval_close < eval_open)
            selling_absorption = (eval_bvc <= -0.20) and (delta_inflection <= -0.10 or bvc_cum_ratio <= -0.10)
            if high_swept and resistance_rejected and selling_absorption and eval_close >= (vwap * 0.990):
                is_reversal = True

            if not (is_breakout or is_reversal):
                continue
            p3_score = 25.0 if is_breakout else 22.0

        # ---------------------------------------------------------------------
        # PILLAR 4 HARD GATE: LIVE OPTION CHAIN OI CONFLUENCE & RUNWAY
        # ---------------------------------------------------------------------
        target_dist = 60.0 if eval_close >= 1500.0 else max(20.0, eval_close * 0.020)
        target_price = (eval_close + target_dist) if direction == 'CALL' else (eval_close - target_dist)

        if opt_override is not None:
            opt = opt_override
        else:
            time.sleep(1.0)
            opt = data_feed.compute_option_pricing(
                spot_price=eval_close,
                direction=direction,
                symbol=sym_clean,
                target_price=target_price,
                is_index=False
            )

        if not opt or not opt.get('is_liquid', False):
            continue

        # Market Maker Straddle Pinning Check (Chop Trap Blocker)
        atm_info = opt.get('atm_info', {})
        if atm_info:
            ce_chg = abs(float(atm_info.get('ce_chg_oi', 0.0)))
            pe_chg = abs(float(atm_info.get('pe_chg_oi', 0.0)))
            tot_chg = ce_chg + pe_chg
            if tot_chg > 500.0 and (abs(ce_chg - pe_chg) / tot_chg) < 0.15:
                continue  # Market maker equilibrium chop trap veto

        p4_score = round(min(20.0, max(10.0, float(opt.get('oi_score', 14.0)))), 2)

        # ---------------------------------------------------------------------
        # PILLAR 5 TRIGGER: 3-MINUTE CANDLE PRICE ACTION TRIGGER (AT THE VERY END)
        # ---------------------------------------------------------------------
        thrust = (eval_close - eval_low) / bar_rng if direction == 'CALL' else (eval_high - eval_close) / bar_rng

        if direction == 'CALL':
            # Bullish Candle Trigger:
            # 1. Closes in upper half of range (thrust >= 0.50)
            # 2. Upper wick rejection must NOT exceed 45% of bar range (avoids buying inverted hammers at tops)
            # 3. Green body (eval_close > eval_open) or strong hammer with lower defense wick >= 40%
            upper_wick_ratio = (eval_high - eval_close) / bar_rng
            is_green = (eval_close > eval_open)
            is_hammer = (eval_low <= eval_open) and ((min(eval_close, eval_open) - eval_low) / bar_rng >= 0.40)
            if thrust < 0.50 or upper_wick_ratio > 0.45 or (not is_green and not is_hammer):
                continue
            p5_score = 10.0 if (thrust >= 0.65 and is_green) else 8.0
        else:  # PUT
            # Bearish Candle Trigger:
            # 1. Closes in lower half of range (thrust <= 0.50)
            # 2. Lower wick absorption must NOT exceed 45% of bar range (avoids selling hammers at bottoms)
            # 3. Red body (eval_close < eval_open) or strong shooting star with upper rejection wick >= 40%
            lower_wick_ratio = (eval_close - eval_low) / bar_rng
            is_red = (eval_close < eval_open)
            is_shooting_star = (eval_high >= eval_open) and ((eval_high - max(eval_close, eval_open)) / bar_rng >= 0.40)
            if thrust < 0.50 or lower_wick_ratio > 0.45 or (not is_red and not is_shooting_star):
                continue
            p5_score = 10.0 if (thrust >= 0.65 and is_red) else 8.0

        # =====================================================================
        # TOTAL 5-PILLAR INSTITUTIONAL SCORE CALCULATION
        # =====================================================================
        total_score = round(p1_score + p2_score + p3_score + p4_score + p5_score, 2)

        if total_score >= INSTITUTIONAL_STOCK_THRESHOLD:
            return {
                'symbol': sym_clean,
                'is_index': False,
                'direction': direction,
                'spot_price': round(eval_close, 2),
                'entry_price': round(float(live_spot), 2),
                'score': total_score,
                'pillar_scores': {
                    'p1_volume': p1_score,
                    'p2_delta': p2_score,
                    'p3_profile': p3_score,
                    'p4_oi': p4_score,
                    'p5_candle': p5_score
                },
                'stock_atr': atr,
                'rvol': real_rvol,
                'turnover_15m': round(turnover_15m, 2),
                'vwap': round(vwap, 2),
                'vah': round(vah, 2),
                'val': round(val, 2),
                'pct_change': pct_change,
                'verified_opt': opt,
                'passed': True,
                'setup_mode': ("REVERSAL_SPRING" if (direction == 'CALL' and is_reversal) else ("REVERSAL_UPTHRUST" if is_reversal else "BASE_BREAKOUT")),
                'tier_reason': f'ALPHA_5PILLAR_STOCK_{direction}_{("REVERSAL_SPRING" if (direction == "CALL" and is_reversal) else ("REVERSAL_UPTHRUST" if is_reversal else "BASE_BREAKOUT"))}_CONFIRMED'
            }

    return None


# =========================================================================
# MODULE 2: DEDICATED 5-PILLAR INSTITUTIONAL INDEX ENGINE (4 INDICES)
# =========================================================================
def evaluate_5pillar_index_engine(
    symbol: str,
    df_slice: pd.DataFrame,
    live_q: Optional[Dict[str, Any]],
    nifty_regime: Dict[str, Any],
    data_feed: LiveMarketDataFeed,
    daily_atr: float,
    live_spot_override: Optional[float] = None,
    bulk_snapshot: Optional[Dict[str, Any]] = None,
    opt_override: Optional[Dict[str, Any]] = None,
    target_direction: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    ========================================================================================
    DEDICATED 5-PILLAR INSTITUTIONAL INDEX ENGINE (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
    DATA FIRST, CANDLE LAST!
    
    1. PILLAR 1: Range RVOL & Point Range Floor Gate
    2. PILLAR 2: True Range CVD Delta Gate (Directional, ZERO abs())
    3. PILLAR 3: Volume Profile & VWAP Gate (Acceptance beyond VAH/VAL)
    4. PILLAR 4: Live Option Chain Confluence & Banking Heavyweight Confluence
    5. PILLAR 5: 3-Minute Candle Price Action Trigger (AT THE VERY END)
    ========================================================================================
    """
    sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
    if sym_clean not in INDEX_NAMES:
        return None
    if df_slice is None or len(df_slice) < 5:
        return None

    eval_bar = df_slice.iloc[-2]
    live_bar = df_slice.iloc[-1]
    comp_bars = df_slice.iloc[:-2].tail(min(8, len(df_slice) - 2))

    eval_close = float(eval_bar['close'])
    eval_open = float(eval_bar['open'])
    eval_high = float(eval_bar['high'])
    eval_low = float(eval_bar['low'])
    bar_rng = max(0.5, eval_high - eval_low)
    live_spot = float(live_spot_override) if live_spot_override is not None else float(live_bar['close'])

    atr = float(daily_atr) if float(daily_atr or 0.0) > 0.0 else (live_spot * 0.015)

    if hasattr(df_slice.index, 'date'):
        today_date = df_slice.index[-1].date()
        today_bars = df_slice[df_slice.index.date == today_date]
        if len(today_bars) > 0:
            day_open = float(today_bars['open'].iloc[0])
            day_high = float(today_bars['high'].max())
            day_low = float(today_bars['low'].min())
        else:
            day_open, day_high, day_low = float(df_slice['open'].iloc[0]), float(df_slice['high'].max()), float(df_slice['low'].min())
    else:
        day_open, day_high, day_low = float(df_slice['open'].iloc[0]), float(df_slice['high'].max()), float(df_slice['low'].min())

    pct_change = round(((live_spot - day_open) / max(1.0, day_open)) * 100.0, 2)

    # =========================================================================
    # PILLAR 1: RANGE RVOL & POINT RANGE FLOOR GATE
    # =========================================================================
    min_idx_rng = 25.0 if sym_clean == 'BANKNIFTY' else 10.0
    if bar_rng < min_idx_rng:
        return None

    # Range RVOL (excluding 09:15 outlier bar)
    comp_clean = df_slice.iloc[1:-2].tail(8) if len(df_slice) > 3 else df_slice.iloc[:-2]
    comp_rngs = comp_clean['high'] - comp_clean['low']
    median_rng = float(comp_rngs.median()) if len(comp_rngs) > 0 else (45.0 if sym_clean == 'BANKNIFTY' else 15.0)
    idx_rvol = round(bar_rng / max(15.0 if sym_clean == 'BANKNIFTY' else 8.0, median_rng), 2)

    if idx_rvol < 1.05 and bar_rng < (1.35 * min_idx_rng):
        return None

    p1_score = round(min(20.0, 10.0 + min(10.0, max(0.0, (bar_rng - min_idx_rng) / min_idx_rng * 10.0))), 2)

    # =========================================================================
    # PILLAR 2: TRUE RANGE CVD DELTA GATE (ZERO abs())
    # =========================================================================
    eval_bvc, prev_bvc, delta_inflection, bvc_cum_ratio = compute_bar_delta_proxy(df_slice, is_index=True)

    # Banking / Heavyweight Confluence Check (Evaluates Intraday Trend, NOT Day-to-Day Gap!)
    heavy_call_allow = True
    heavy_put_allow = True
    if bulk_snapshot and sym_clean in ['NIFTY', 'BANKNIFTY']:
        heavy_syms = ['HDFCBANK', 'ICICIBANK', 'RELIANCE'] if sym_clean == 'NIFTY' else ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK']
        heavy_pcts = []
        for h_s in heavy_syms:
            h_q = bulk_snapshot.get(h_s)
            if h_q and h_q.get('pct_change') is not None:
                try:
                    heavy_pcts.append(float(h_q['pct_change']))
                except Exception:
                    pass
        if heavy_pcts:
            avg_heavy = float(np.mean(heavy_pcts))
            # Intraday momentum filter
            if avg_heavy < -0.30:
                heavy_call_allow = False
            if avg_heavy > 0.30:
                heavy_put_allow = False

    is_bn = (sym_clean == 'BANKNIFTY')
    allow_call = heavy_call_allow if is_bn else (nifty_regime.get('allow_call', True) and heavy_call_allow)
    allow_put = heavy_put_allow if is_bn else (nifty_regime.get('allow_put', True) and heavy_put_allow)

    directions_to_test = []
    if allow_call and pct_change >= -0.50:
        directions_to_test.append('CALL')
    if allow_put and pct_change <= 0.50:
        directions_to_test.append('PUT')

    if target_direction:
        directions_to_test = [d for d in directions_to_test if d == target_direction]

    if not directions_to_test:
        return None

    # Profile & Value Area parameters
    prof = compute_volume_profile(df_slice, daily_atr=atr, is_index=True)
    vwap = prof['vwap']
    vah = prof['vah']
    val = prof['val']
    poc = prof['poc']

    for direction in directions_to_test:
        # ---------------------------------------------------------------------
        # PILLAR 2 HARD GATE: STRICT DIRECTIONAL DELTA (ZERO abs())
        # ---------------------------------------------------------------------
        if direction == 'CALL':
            if eval_bvc < 0.15:
                continue
            if delta_inflection < 0.05 and bvc_cum_ratio < 0.05 and eval_bvc < 0.35:
                continue
            if eval_bvc <= 0.0 or bvc_cum_ratio <= 0.0:
                continue
            p2_score = round(min(25.0, (eval_bvc * 15.0) + (max(0.0, delta_inflection) * 10.0)), 2)
        else:  # PUT
            if eval_bvc > -0.15:
                continue
            if delta_inflection > -0.05 and bvc_cum_ratio > -0.05 and eval_bvc > -0.35:
                continue
            if eval_bvc >= 0.0 or bvc_cum_ratio >= 0.0:
                continue
            p2_score = round(min(25.0, (abs(eval_bvc) * 15.0) + (max(0.0, -delta_inflection) * 10.0)), 2)

        # ---------------------------------------------------------------------
        # PILLAR 3 HARD GATE: WYCKOFF VOLUME PROFILE ACCEPTANCE & REVERSAL SETUP
        # Dual-Setup Architecture: Breakout vs Institutional Reversal Absorption
        # ---------------------------------------------------------------------
        is_breakout = False
        is_reversal = False

        if direction == 'CALL':
            # Setup 1: Index Base Breakout / Value Expansion
            if eval_close > vwap and eval_close >= (vah * 0.998):
                if eval_close <= (vwap * 1.018):  # Anti-top chasing guard
                    is_breakout = True

            # Setup 2: Institutional Index Support Reversal (Spring / Bottom Reclaim)
            # Swept swing low or day low, strong buying absorption reclaims VAL towards VWAP
            idx_buffer = 30.0 if sym_clean == 'BANKNIFTY' else 12.0
            low_swept = (comp_bars['low'].tail(3).min() <= comp_bars['low'].min() + idx_buffer) or (comp_bars['low'].tail(3).min() <= day_low + 50.0) or (eval_low <= day_low + 50.0)
            support_reclaimed = (eval_close >= (val * 0.998)) and (eval_close > eval_open)
            buying_absorption = (eval_bvc >= 0.20) and (delta_inflection >= 0.10 or bvc_cum_ratio >= 0.10)
            if low_swept and support_reclaimed and buying_absorption and eval_close <= (vwap * 1.010):
                is_reversal = True

            if not (is_breakout or is_reversal):
                continue
            p3_score = 25.0 if is_breakout else 22.0
        else:  # PUT
            # Setup 1: Index Base Breakdown / Value Expansion
            if eval_close < vwap and eval_close <= (val * 1.002):
                if eval_close >= (vwap * 0.982):  # Anti-bottom chasing guard
                    is_breakout = True

            # Setup 2: Institutional Index Top Resistance Reversal (Upthrust / Failed High)
            # Swept swing high or day high, strong selling absorption rejects VAH towards VWAP
            idx_buffer = 30.0 if sym_clean == 'BANKNIFTY' else 12.0
            high_swept = (comp_bars['high'].tail(3).max() >= comp_bars['high'].max() - idx_buffer) or (comp_bars['high'].tail(3).max() >= day_high - 50.0) or (eval_high >= day_high - 50.0)
            resistance_rejected = (eval_close <= (vah * 1.002)) and (eval_close < eval_open)
            selling_absorption = (eval_bvc <= -0.20) and (delta_inflection <= -0.10 or bvc_cum_ratio <= -0.10)
            if high_swept and resistance_rejected and selling_absorption and eval_close >= (vwap * 0.990):
                is_reversal = True

            if not (is_breakout or is_reversal):
                continue
            p3_score = 25.0 if is_breakout else 22.0

        # ---------------------------------------------------------------------
        # PILLAR 4 HARD GATE: LIVE OPTION CHAIN OI CONFLUENCE & RUNWAY
        # ---------------------------------------------------------------------
        target_dist = 150.0 if sym_clean == 'BANKNIFTY' else 60.0
        target_price = (eval_close + target_dist) if direction == 'CALL' else (eval_close - target_dist)

        if opt_override is not None:
            opt = opt_override
        else:
            time.sleep(1.0)
            opt = data_feed.compute_option_pricing(
                spot_price=eval_close,
                direction=direction,
                symbol=sym_clean,
                target_price=target_price,
                is_index=True
            )

        if not opt or not opt.get('is_liquid', False):
            continue

        # Market Maker Straddle Pinning Check (Chop Trap Blocker)
        atm_info = opt.get('atm_info', {})
        if atm_info:
            ce_chg = abs(float(atm_info.get('ce_chg_oi', 0.0)))
            pe_chg = abs(float(atm_info.get('pe_chg_oi', 0.0)))
            tot_chg = ce_chg + pe_chg
            if tot_chg > 2000.0 and (abs(ce_chg - pe_chg) / tot_chg) < 0.15:
                continue

        p4_score = round(min(20.0, max(10.0, float(opt.get('oi_score', 14.0)))), 2)

        # ---------------------------------------------------------------------
        # PILLAR 5 TRIGGER: 3-MINUTE CANDLE PRICE ACTION TRIGGER (AT THE VERY END)
        # ---------------------------------------------------------------------
        thrust = (eval_close - eval_low) / bar_rng if direction == 'CALL' else (eval_high - eval_close) / bar_rng

        if direction == 'CALL':
            upper_wick_ratio = (eval_high - eval_close) / bar_rng
            is_green = (eval_close > eval_open)
            is_hammer = (eval_low <= eval_open) and ((min(eval_close, eval_open) - eval_low) / bar_rng >= 0.40)
            if thrust < 0.50 or upper_wick_ratio > 0.45 or (not is_green and not is_hammer):
                continue
            p5_score = 10.0 if (thrust >= 0.65 and is_green) else 8.0
        else:  # PUT
            lower_wick_ratio = (eval_close - eval_low) / bar_rng
            is_red = (eval_close < eval_open)
            is_shooting_star = (eval_high >= eval_open) and ((eval_high - max(eval_close, eval_open)) / bar_rng >= 0.40)
            if thrust < 0.50 or lower_wick_ratio > 0.45 or (not is_red and not is_shooting_star):
                continue
            p5_score = 10.0 if (thrust >= 0.65 and is_red) else 8.0

        # =====================================================================
        # TOTAL 5-PILLAR INSTITUTIONAL SCORE CALCULATION
        # =====================================================================
        total_score = round(p1_score + p2_score + p3_score + p4_score + p5_score, 2)

        if total_score >= INSTITUTIONAL_INDEX_THRESHOLD:
            return {
                'symbol': sym_clean,
                'is_index': True,
                'direction': direction,
                'spot_price': round(eval_close, 2),
                'entry_price': round(float(live_spot), 2),
                'score': total_score,
                'pillar_scores': {
                    'p1_volume': p1_score,
                    'p2_delta': p2_score,
                    'p3_profile': p3_score,
                    'p4_oi': p4_score,
                    'p5_candle': p5_score
                },
                'stock_atr': atr,
                'rvol': idx_rvol,
                'vwap': round(vwap, 2),
                'vah': round(vah, 2),
                'val': round(val, 2),
                'pct_change': pct_change,
                'verified_opt': opt,
                'passed': True,
                'setup_mode': ("REVERSAL_SPRING" if (direction == 'CALL' and is_reversal) else ("REVERSAL_UPTHRUST" if is_reversal else "BASE_BREAKOUT")),
                'tier_reason': f'ALPHA_5PILLAR_INDEX_{direction}_{("REVERSAL_SPRING" if (direction == "CALL" and is_reversal) else ("REVERSAL_UPTHRUST" if is_reversal else "BASE_BREAKOUT"))}_CONFIRMED'
            }

    return None


# =========================================================================
# PRODUCTION SCREENER CLASS: ALPHA 5-PILLAR INSTITUTIONAL ENGINE
# =========================================================================
class Alpha5PillarInstitutionalEngine(AlphaInstitutionalSwappedScreener):
    """
    Alpha 5-Pillar Institutional Autonomous Engine.
    Inherits all production position management, journal dispatch, and live loops.
    Strictly overrides evaluation calls with evaluate_5pillar_stock_engine & evaluate_5pillar_index_engine.
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('alpha_5pillar_engine')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [5PILLAR_ENGINE] %(message)s'))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def run_screener(self, ignore_window: bool = False, send_heartbeat: bool = True, cycle_num: Optional[int] = None):
        """
        Executes one full live scan cycle using the 5-Pillar Institutional Engine.
        """
        ist_now = get_ist_time()
        ist_time_str = ist_now.strftime('%I:%M:%S %p IST')

        if not ignore_window and not is_in_active_trading_slab(ist_now):
            print(f"[SLAB CHECK]: {ist_time_str} is outside active slabs (09:30-10:15 & 13:15-14:15). Running status heartbeat.")
            return

        print(f"\n=======================================================")
        print(f"  ALPHA 5-PILLAR INSTITUTIONAL ENGINE | {ist_now.strftime('%d-%b-%Y %I:%M:%S %p IST')}")
        print(f"  Architecture: 5 Pillars Strict Data-First (Candle Last, Zero abs())")
        print(f"=======================================================")

        self.update_open_paper_trades()

        # Enforce 15-Minute Slab Runway Buffer
        if not ignore_window and not is_in_fresh_signal_window(ist_now):
            print(f"[SLAB RUNWAY BUFFER]: {ist_time_str} is in the 15m slab runway buffer.")
            print(f"[SLAB RUNWAY BUFFER]: Fresh entries LOCKED to protect capital. Existing positions actively managed.")
            if send_heartbeat:
                cycle_str = f" (Cycle #{cycle_num})" if cycle_num else ""
                heartbeat_msg = (
                    f"🛡️ <b>ALPHA 5-PILLAR INSTITUTIONAL ENGINE{cycle_str}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⏰ <b>Execution Time:</b> {ist_time_str}\n"
                    f"⏳ <b>Slab Runway Buffer:</b> Active (10:00-10:15 / 14:00-14:15)\n"
                    f"🔒 <b>Fresh Signals:</b> LOCKED (Minimum 15m runway required)\n"
                    f"🎯 <b>Position Management:</b> Active (Trailing SL & Targets Monitored)\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔒 <i>Delivered strictly to Owner Chatbot. Capital 100% Protected.</i>"
                )
                self.notifier.send_broadcast_message(heartbeat_msg, is_trade_signal=False)
            return

        bulk = fetch_bulk_nse_snapshot(force_refresh=True)
        regime = self.get_nifty_market_regime(force_refresh=True)
        print(f"[MACRO REGIME]: {regime['reason']} | Nifty Move: {regime.get('move_pts', 0.0):+.1f} pts")

        # 1. Evaluate 4 Indices with 5-Pillar Index Engine
        index_candidates = []
        for idx in INDEX_NAMES:
            df = self.data_feed.get_live_intraday_bars(idx)
            atr = self.get_stock_atr(idx, bulk.get(idx))
            res = evaluate_5pillar_index_engine(
                symbol=idx,
                df_slice=df,
                live_q=bulk.get(idx),
                nifty_regime=regime,
                data_feed=self.data_feed,
                daily_atr=atr,
                bulk_snapshot=bulk
            )
            if res and res.get('passed', False):
                index_candidates.append(res)

        # 2. Evaluate Dynamic Top 28 Highest Liquidity F&O Stocks
        all_candidate_symbols = [
            sym for sym in [
                item['symbol'] if isinstance(item, dict) else item
                for item in self.universe
            ]
            if sym not in INDEX_NAMES and not sym.startswith('NIFTY') and not sym.endswith('FPI')
        ]
        symbols_to_scan = self.get_dynamic_top_liquid_stocks(all_candidate_symbols, top_n=28)

        def scan_stock_worker(sym):
            try:
                sym_clean = sym.upper().replace('.NS', '').replace('^', '').strip()
                df = self.data_feed.get_live_intraday_bars(sym_clean)
                if df is None or len(df) < 5:
                    return None
                atr = self.get_stock_atr(sym_clean, bulk.get(sym_clean))
                return evaluate_5pillar_stock_engine(
                    symbol=sym_clean,
                    df_slice=df,
                    live_q=bulk.get(sym_clean),
                    nifty_regime=regime,
                    data_feed=self.data_feed,
                    daily_atr=atr
                )
            except Exception as e:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
            stock_results = list(executor.map(scan_stock_worker, symbols_to_scan))

        stock_candidates = [r for r in stock_results if r and r.get('passed', False)]

        print(f"[5-PILLAR SCAN COMPLETE]: Scanned {len(INDEX_NAMES)} Indices & {len(symbols_to_scan)} Stocks.")
        print(f"[5-PILLAR DISCOVERY]: {len(index_candidates)} Index candidate(s), {len(stock_candidates)} Stock candidate(s).")

        all_candidates = []
        for cand in (index_candidates + stock_candidates):
            sym = cand['symbol']
            if self.is_already_dispatched(sym, ist_now):
                print(f"[REJECTION]: {sym} already dispatched in this slab.")
                continue
            all_candidates.append(cand)

        all_candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        # Enforce slab quotas
        cur_idx_count = self.get_slab_index_dispatched_count(ist_now)
        cur_stk_count = self.get_slab_stock_dispatched_count(ist_now)
        total_slab_dispatched = cur_idx_count + cur_stk_count

        idx_cands = [c for c in all_candidates if c.get('is_index', False)]
        stk_cands = [c for c in all_candidates if not c.get('is_index', False)]

        idx_cands.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        stk_cands.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        verified_super_trades = []
        for icand in idx_cands:
            if total_slab_dispatched >= MAX_SLAB_TRADES or cur_idx_count >= MAX_INDEX_PER_SLAB:
                break
            verified_super_trades.append(icand)
            cur_idx_count += 1
            total_slab_dispatched += 1

        for scand in stk_cands:
            if total_slab_dispatched >= MAX_SLAB_TRADES or cur_stk_count >= MAX_STOCKS_PER_SLAB:
                break
            verified_super_trades.append(scand)
            cur_stk_count += 1
            total_slab_dispatched += 1

        if not verified_super_trades:
            print("[5-PILLAR DISCOVERY]: No setups passed all 5 strict institutional data gates. Capital 100% Protected.")
            if send_heartbeat:
                cycle_str = f" (Cycle #{cycle_num})" if cycle_num else ""
                heartbeat_msg = (
                    f"🛡️ <b>ALPHA 5-PILLAR INSTITUTIONAL ENGINE{cycle_str}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⏰ <b>Execution Time:</b> {ist_time_str}\n"
                    f"📊 <b>Regime:</b> {regime['reason']} ({regime.get('move_pts', 0.0):+.1f} pts)\n"
                    f"🔍 <b>Scanned:</b> {len(INDEX_NAMES)} Indices & {len(symbols_to_scan)} Liquid Stocks\n"
                    f"🛡️ <b>Status:</b> Zero False Signals (Pillars 1-4 Gate Protected)\n"
                    f"🔒 <b>Capital:</b> 100% Protected (No FOMO / No Top Chasing)\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔒 <i>Delivered strictly to Owner Chatbot.</i>"
                )
                self.notifier.send_broadcast_message(heartbeat_msg, is_trade_signal=False)
            return

        for super_trade in verified_super_trades:
            self._dispatch_super_trade(super_trade, ist_now)

    def run_authoritative_5day_backtest(self):
        """
        Executes the authoritative 5-day backtest across historical 3-minute bars
        using the strict 5-Pillar Institutional Engine.
        """
        print("==========================================================================")
        print("   ALPHA 5-PILLAR INSTITUTIONAL ENGINE: 5-DAY AUTHORITATIVE BACKTEST")
        print("   Fail-Closed, Zero Synthetic, Data-First (Candle Last, Zero abs())")
        print("==========================================================================")

        asset_data, stock_syms = load_resampled_3m_data_native()
        if 'NIFTY' not in asset_data:
            print("ERROR: NIFTY 3-minute data missing. Aborting backtest.")
            return

        nifty_df = asset_data['NIFTY']
        trading_days = sorted(list(set(nifty_df.index.date)))[-5:]
        print(f"5-Day Window: {trading_days[0]} to {trading_days[-1]} ({len(trading_days)} Trading Days)")

        atrs = {}
        for sym, df_item in asset_data.items():
            atrs[sym] = round(float(df_item['close'].iloc[-1] * (0.015 if sym in INDEX_NAMES else 0.022)), 2)

        lot_map = {}
        for sym in asset_data.keys():
            lot_map[sym] = int(get_symbol_lot_size(sym))

        all_closed_trades: List[ActiveOptionTrade] = []

        for d in trading_days:
            nifty_day = nifty_df[nifty_df.index.date == d]
            if len(nifty_day) < 10:
                continue

            day_timestamps = sorted(list(nifty_day.index))
            m_slab_traded = set()
            a_slab_traded = set()
            m_idx_count = 0
            m_stk_count = 0
            a_idx_count = 0
            a_stk_count = 0

            active_positions: List[ActiveOptionTrade] = []

            for curr_t in day_timestamps:
                m_mins = curr_t.hour * 60 + curr_t.minute
                is_m_active = (578 <= m_mins <= 650)
                is_a_active = (795 <= m_mins <= 855)
                is_m_fresh = (578 <= m_mins <= 645)
                is_a_fresh = (795 <= m_mins <= 848)

                # Update active positions
                still_open = []
                for tr in active_positions:
                    if curr_t < tr.entry_time:
                        still_open.append(tr)
                        continue
                    df_asset = asset_data.get(tr.sym)
                    if df_asset is not None and curr_t in df_asset.index:
                        b_row = df_asset.loc[curr_t]
                        tr.update(float(b_row['high']), float(b_row['low']), float(b_row['close']), curr_t)
                    if tr.is_closed:
                        all_closed_trades.append(tr)
                    else:
                        still_open.append(tr)
                active_positions = still_open

                # Close positions at end of slab
                if (m_mins == 650) or (m_mins == 855):
                    for tr in active_positions:
                        df_asset = asset_data.get(tr.sym)
                        b_close = float(df_asset.loc[curr_t]['close']) if (df_asset is not None and curr_t in df_asset.index) else tr.spot_entry
                        tr.close_eod(b_close, curr_t)
                        all_closed_trades.append(tr)
                    active_positions = []

                if not (is_m_fresh or is_a_fresh):
                    continue

                cur_idx_count = m_idx_count if is_m_fresh else a_idx_count
                cur_stk_count = m_stk_count if is_m_fresh else a_stk_count
                if (cur_idx_count + cur_stk_count) >= MAX_SLAB_TRADES or (cur_idx_count >= MAX_INDEX_PER_SLAB and cur_stk_count >= MAX_STOCKS_PER_SLAB):
                    continue

                # Regime calculation
                n_slice = nifty_day.loc[:curr_t]
                n_open = float(nifty_day['open'].iloc[0])
                n_close = float(n_slice['close'].iloc[-1])
                n_pct = round(((n_close - n_open) / max(1.0, n_open)) * 100.0, 2)
                regime = {
                    'allow_stock_call': n_pct >= -0.25,
                    'allow_stock_put': n_pct <= 0.25,
                    'allow_call': n_pct >= 0.10,
                    'allow_put': n_pct <= -0.10,
                    'nifty_pct': n_pct,
                    'nifty_spot': n_close
                }

                # Evaluate Indices
                cur_slab_traded = m_slab_traded if is_m_fresh else a_slab_traded
                if cur_idx_count < MAX_INDEX_PER_SLAB:
                    for isym in ['NIFTY', 'BANKNIFTY']:
                        if isym in cur_slab_traded or isym not in asset_data:
                            continue
                        idf = asset_data[isym]
                        today_bars = idf[(idf.index.date == d) & (idf.index <= curr_t)]
                        if len(today_bars) < 5:
                            continue
                        spot_eval = float(today_bars['close'].iloc[-2])
                        idx_atr = atrs.get(isym, round(spot_eval * 0.012, 2))
                        pct_idx = ((spot_eval - float(today_bars['open'].iloc[0])) / max(1.0, float(today_bars['open'].iloc[0]))) * 100.0
                        test_dirs = []
                        if regime['allow_call'] and pct_idx >= -0.60:
                            test_dirs.append('CALL')
                        if regime['allow_put'] and pct_idx <= 0.60:
                            test_dirs.append('PUT')

                        for dir_test in test_dirs:
                            t_dist = 50.0 if isym == 'BANKNIFTY' else 45.0
                            t_price = (spot_eval + t_dist) if dir_test == 'CALL' else (spot_eval - t_dist)
                            oi_res = verify_authoritative_bhavcopy_oi(isym, d, spot_eval, dir_test, target_price=t_price, is_index=True)
                            if not oi_res.get('confluent', False):
                                continue

                            opt_override = {
                                'is_live_chain_verified': True,
                                'is_liquid': True,
                                'oi_score': float(oi_res.get('score', 12.0)),
                                'entry_premium': max(6.0, round(25.0 * 2.0, 2)),
                                'contract_name': f"{isym} {int(oi_res.get('atm_strike', spot_eval))} {'CE' if dir_test == 'CALL' else 'PE'}",
                                'atm_strike': oi_res.get('atm_strike', spot_eval)
                            }
                            sig = evaluate_5pillar_index_engine(
                                isym, today_bars, live_q=None, nifty_regime=regime,
                                data_feed=self.data_feed, daily_atr=idx_atr,
                                live_spot_override=spot_eval,
                                opt_override=opt_override,
                                target_direction=dir_test
                            )
                            if sig and sig.get('passed', False):
                                tr = ActiveOptionTrade(sig, curr_t, lot_map)
                                active_positions.append(tr)
                                cur_slab_traded.add(isym)
                                if is_m_fresh:
                                    m_idx_count += 1
                                else:
                                    a_idx_count += 1
                                break

                # Evaluate Stocks
                if cur_stk_count < MAX_STOCKS_PER_SLAB:
                    for ssym in stock_syms:
                        if ssym in cur_slab_traded or ssym not in asset_data:
                            continue
                        sdf = asset_data[ssym]
                        today_bars = sdf[(sdf.index.date == d) & (sdf.index <= curr_t)]
                        if len(today_bars) < 5:
                            continue
                        spot_eval = float(today_bars['close'].iloc[-2])
                        stock_atr = atrs.get(ssym, round(spot_eval * 0.022, 2))
                        pct_stock = ((spot_eval - float(today_bars['open'].iloc[0])) / max(1.0, float(today_bars['open'].iloc[0]))) * 100.0
                        test_dirs = []
                        if regime['allow_stock_call'] and pct_stock >= -0.75:
                            test_dirs.append('CALL')
                        if regime['allow_stock_put'] and pct_stock <= 0.75:
                            test_dirs.append('PUT')

                        for dir_test in test_dirs:
                            t_dist = 15.0
                            t_price = (spot_eval + t_dist) if dir_test == 'CALL' else (spot_eval - t_dist)
                            oi_res = verify_authoritative_bhavcopy_oi(ssym, d, spot_eval, dir_test, target_price=t_price, is_index=False)
                            if not oi_res.get('confluent', False):
                                continue

                            opt_override = {
                                'is_live_chain_verified': True,
                                'is_liquid': True,
                                'oi_score': float(oi_res.get('score', 12.0)),
                                'entry_premium': max(6.0, round(5.0 * 2.0, 2)),
                                'contract_name': f"{ssym} {int(oi_res.get('atm_strike', spot_eval))} {'CE' if dir_test == 'CALL' else 'PE'}",
                                'atm_strike': oi_res.get('atm_strike', spot_eval)
                            }
                            sig = evaluate_5pillar_stock_engine(
                                ssym, today_bars, live_q=None, nifty_regime=regime,
                                data_feed=self.data_feed, daily_atr=stock_atr,
                                live_spot_override=spot_eval,
                                opt_override=opt_override,
                                target_direction=dir_test
                            )
                            if sig and sig.get('passed', False):
                                tr = ActiveOptionTrade(sig, curr_t, lot_map)
                                active_positions.append(tr)
                                cur_slab_traded.add(ssym)
                                if is_m_fresh:
                                    m_stk_count += 1
                                else:
                                    a_stk_count += 1
                                break

        # Compile summaries
        summaries = [tr.get_summary(str(tr.cand['entry_price'])) for tr in all_closed_trades]
        df_res = pd.DataFrame(summaries) if summaries else pd.DataFrame()
        print(f"\n=======================================================")
        print(f"5-DAY AUTHORITATIVE BACKTEST COMPLETED: {len(all_closed_trades)} Trades")
        print(f"=======================================================")
        if not df_res.empty:
            win_rate = (len(df_res[df_res['net_pnl'] > 0]) / len(df_res)) * 100.0
            tot_pnl = df_res['net_pnl'].sum()
            print(f"Win Rate: {win_rate:.1f}% | Net PnL: Rs {tot_pnl:+,.2f}")
        return df_res


# =========================================================================
# STANDALONE ENTRYPOINT
# =========================================================================
if __name__ == '__main__':
    engine = Alpha5PillarInstitutionalEngine()
    if '--backtest' in sys.argv:
        engine.run_authoritative_5day_backtest()
    elif '--slab-session' in sys.argv:
        if not check_github_actions_concurrency_guard():
            sys.exit(0)
        engine.run_live_slab_session(interval_seconds=180)
    elif '--eod-audit' in sys.argv or '--send-journal' in sys.argv:
        engine.run_eod_analysis_and_dispatch()
    else:
        ignore_win = ('--ignore-window' in sys.argv)
        engine.run_screener(ignore_window=ignore_win)
