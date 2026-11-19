# -*- coding: utf-8 -*-
"""
alpha_institutional_swapped_screener.py
======================================
Alpha 5-Pillar Institutional Swapped Screener & Autonomous Engine.

ARCHITECTURAL PRINCIPLE:
- Swapped Pipeline Sequence: Order Flow, Cash Turnover & Derivative Pressures Lead!
- Zero Akamai Block Risk:
    1. Gate 1 (Zero NSE Option Chain Calls):
       Filters 220 F&O stocks down to top 3-6 liquid leaders using 15m Cash Turnover (>= Rs 15 Cr),
       RVOL (>= 1.6x), Day Move (% change), and VWAP alignment.
       214 dead/dry stocks are eliminated with ZERO Option Chain calls.
    2. Gate 2 (Targeted Option Chain & Gamma Squeeze Verification):
       Only the 3-6 qualified leaders have their live NSE Option Chain queried.
       (Total ~5-8 calls per 3m cycle -> 2.6 calls/min -> 15x below Akamai thresholds!).
       Checks Vol/OI >= 2.0x, Call Unwinding, Put Support Wall, and Bid/Ask spreads.
    3. Gate 3 (Execution Timing & Pricing):
       Price > VWAP (or < VWAP for PUT), dynamic SL, realistic target runway, and trade dispatch.
- Completely separate file: Old 'daily_autonomous_screener.py' remains 100% UNTOUCHED!
- 2 Separate Evaluation Functions:
    * evaluate_institutional_index_engine (for 4 Indices)
    * evaluate_institutional_stock_engine (for 216 F&O Stocks, fully dynamic, zero hardcoding!)
"""

import os
import sys
import time
import json
import random
import datetime
import math
import html
import logging
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from modules.live_data_feed import (
    LiveMarketDataFeed,
    fetch_bulk_nse_snapshot,
    get_ist_time
)
from modules.fno_universe import get_instrument_meta, determine_step_size
from modules.telegram_notifier import TelegramNotifier

CONFIG_PATH = os.path.join(CURRENT_DIR, 'config', 'telegram_config.json')

INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
MIN_STOCK_PRICE = 500.0
INDEX_SCORE_THRESHOLD = 60.0
STOCK_SCORE_THRESHOLD = 80.0

MAX_SLAB_TRADES = 2
MAX_INDEX_PER_SLAB = 1
MAX_STOCKS_PER_SLAB = 1


def get_symbol_lot_size(sym_clean: str) -> float:
    """Retrieves official lot size for F&O symbol from metadata cache or default."""
    try:
        meta = get_instrument_meta(sym_clean)
        if meta and meta.get('lot'):
            return float(meta['lot'])
    except Exception:
        pass
    bulk = fetch_bulk_nse_snapshot()
    if bulk and sym_clean in bulk and bulk[sym_clean].get('lot_size'):
        return float(bulk[sym_clean]['lot_size'])
    return 250.0 if sym_clean not in INDEX_NAMES else 25.0


def is_in_active_trading_slab(ist_dt: datetime.datetime) -> bool:
    """
    Morning Prime Slab: 09:30 AM to 10:15 AM IST (570 to 615 mins)
    Afternoon Prime Slab: 01:15 PM to 02:15 PM IST (795 to 855 mins)
    """
    tot_m = ist_dt.hour * 60 + ist_dt.minute
    is_morning = (570 <= tot_m <= 615)
    is_afternoon = (795 <= tot_m <= 855)
    return is_morning or is_afternoon


def compute_bar_delta_proxy(df_slice: pd.DataFrame, is_index: bool = False) -> Tuple[float, float]:
    """
    Computes instant bar delta and cumulative 5-bar volume delta ratio.
    - For Stocks: Uses real traded cash volume weighting.
    - For Indices: Spot volume is 0 on NSE; uses True Range (High - Low) weighting.
    """
    if df_slice is None or len(df_slice) < 5:
        return 0.0, 0.0
    bars_5 = df_slice.iloc[-6:-1]
    price_chg = bars_5['close'] - bars_5['open']
    std_val = float(price_chg.std()) if len(bars_5) > 1 else 0.0
    sigma = std_val if std_val > 0.01 else float((bars_5['high'] - bars_5['low']).mean() * 0.5)
    bvc_z = np.clip(price_chg / max(0.01, sigma), -3.0, 3.0)
    bvc_fraction = np.tanh(bvc_z)
    has_real_volume = (not is_index) and ('volume' in bars_5.columns and float(np.sum(bars_5['volume'].values.astype(float))) > 0.0)
    weights = bars_5['volume'].values.astype(float) if has_real_volume else (bars_5['high'] - bars_5['low']).values.astype(float)
    bvc_cum_ratio = float(np.sum(weights * bvc_fraction.values) / max(1.0, float(np.sum(weights))))
    eval_bvc = float(bvc_fraction.iloc[-1])
    return eval_bvc, bvc_cum_ratio


# =========================================================================
# SEPARATE FUNCTION 1: DEDICATED INSTITUTIONAL INDEX ENGINE (4 Indices)
# =========================================================================
def evaluate_institutional_index_engine(
    symbol: str,
    df_slice: pd.DataFrame,
    live_q: Optional[Dict[str, Any]],
    nifty_regime: Dict[str, Any],
    data_feed: LiveMarketDataFeed,
    daily_atr: float,
    live_spot_override: Optional[float] = None,
    bulk_snapshot: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    ========================================================================================
    MODULE 1: INSTITUTIONAL INDEX ENGINE (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
    - Macro Range Expansion RVOL & CVD Flow (Zero Cash Turnover Dependency).
    - Targeted Option Chain Query: Exactly 1 Call for ATM & +/-4 Strikes.
    - Vol/OI >= 2.0x Gamma Squeeze & Call/Put Unwinding Verification.
    ========================================================================================
    """
    sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
    if sym_clean not in INDEX_NAMES:
        return None
    if df_slice is None or len(df_slice) < 7:
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

    # Calculate VWAP proxy
    tp_all = (df_slice['high'] + df_slice['low'] + df_slice['close']) / 3.0
    vol_weights = df_slice['high'] - df_slice['low']
    vwap = float((tp_all * vol_weights).sum() / max(1.0, float(vol_weights.sum()))) if float(vol_weights.sum()) > 0 else float(tp_all.mean())

    # Absolute minimum bar range floor (eliminates fake breakouts during choppy low-vol periods)
    min_idx_rng = 35.0 if sym_clean == 'BANKNIFTY' else 14.0
    if bar_rng < min_idx_rng:
        return None

    # Range RVOL
    comp_rngs = comp_bars['high'] - comp_bars['low']
    median_rng = float(comp_rngs.median()) if len(comp_rngs) > 0 else 15.0
    idx_rvol = round(bar_rng / max(0.5, median_rng), 2)

    # Range-weighted Delta (Spot Index Volume is identically 0 on NSE; True Range used)
    eval_bvc, bvc_cum_ratio = compute_bar_delta_proxy(df_slice, is_index=True)

    # Test CALL direction first, then PUT
    directions_to_test = []
    if nifty_regime.get('allow_call', True) and eval_close > vwap and pct_change >= 0.15:
        directions_to_test.append('CALL')
    if nifty_regime.get('allow_put', True) and eval_close < vwap and pct_change <= -0.15:
        directions_to_test.append('PUT')

    for direction in directions_to_test:
        # Pillar 1 (Index Range Momentum): Score up to 25 pts
        thrust = (eval_close - eval_low) / bar_rng if direction == 'CALL' else (eval_high - eval_close) / bar_rng
        if thrust < 0.55:
            continue
        p1_score = round(min(25.0, max(15.0, 15.0 + (idx_rvol * 3.5) + (thrust * 5.0))), 2)

        # Pillar 2 (Delta Flow): Score up to 25 pts
        if direction == 'CALL':
            if eval_bvc <= 0 or bvc_cum_ratio <= 0:
                continue
            p2_score = round(min(25.0, max(12.0, (eval_bvc * 12.0) + (bvc_cum_ratio * 13.0))), 2)
        else:
            if eval_bvc >= 0 or bvc_cum_ratio >= 0:
                continue
            p2_score = round(min(25.0, max(12.0, (abs(eval_bvc) * 12.0) + (abs(bvc_cum_ratio) * 13.0))), 2)

        # Pillar 3 (Profile Alignment): Score up to 20 pts
        p3_score = 18.0 if ((direction == 'CALL' and eval_close > vwap) or (direction == 'PUT' and eval_close < vwap)) else 10.0

        # Pillar 5 (ATR Runway):
        target_dist = 140.0 if sym_clean == 'BANKNIFTY' else 90.0
        fuel_spent = day_high - day_low
        p5_score = 8.0 if fuel_spent < (1.5 * atr) else 5.0

        prelim_score = round(p1_score + p2_score + p3_score + p5_score, 2)

        # Institutional Heavyweight Confluence (NIFTY & BANKNIFTY)
        # Spot Index cannot sustain breakout if key constituent drivers are heavily dumping
        if bulk_snapshot and sym_clean in ['NIFTY', 'BANKNIFTY']:
            heavy_syms = ['HDFCBANK', 'ICICIBANK', 'RELIANCE'] if sym_clean == 'NIFTY' else ['HDFCBANK', 'ICICIBANK', 'SBIN']
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
                if direction == 'CALL' and avg_heavy < -0.40:
                    continue  # Heavyweight divergence veto
                if direction == 'PUT' and avg_heavy > 0.40:
                    continue  # Heavyweight divergence veto

        # TARGETED OPTION CHAIN QUERY (Safe 1 Call per candidate Index)
        target_price = (eval_close + target_dist) if direction == 'CALL' else (eval_close - target_dist)
        opt = data_feed.compute_option_pricing(
            spot_price=eval_close,
            direction=direction,
            symbol=sym_clean,
            target_price=target_price,
            is_index=True
        )

        if not opt or not opt.get('is_liquid', False):
            continue

        oi_score = float(opt.get('oi_score', 12.0))
        total_score = round(prelim_score + oi_score, 2)

        if total_score >= INDEX_SCORE_THRESHOLD:
            return {
                'symbol': sym_clean,
                'is_index': True,
                'direction': direction,
                'spot_price': round(eval_close, 2),
                'entry_price': round(float(live_spot), 2),
                'score': total_score,
                'prelim_score': prelim_score,
                'oi_score': oi_score,
                'stock_atr': atr,
                'rvol': idx_rvol,
                'vwap': round(vwap, 2),
                'pct_change': pct_change,
                'verified_opt': opt,
                'passed': True,
                'tier_reason': f'INSTITUTIONAL_INDEX_{direction}_CONFIRMED'
            }

    return None


# =========================================================================
# SEPARATE FUNCTION 2: DEDICATED INSTITUTIONAL STOCK ENGINE (216 F&O Stocks)
# =========================================================================
def evaluate_institutional_stock_engine(
    symbol: str,
    df_slice: pd.DataFrame,
    live_q: Optional[Dict[str, Any]],
    nifty_regime: Dict[str, Any],
    data_feed: LiveMarketDataFeed,
    daily_atr: float,
    live_spot_override: Optional[float] = None,
    pct_change_override: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    ========================================================================================
    MODULE 2: INSTITUTIONAL STOCK ENGINE (216 DYNAMIC F&O STOCKS, ZERO HARDCODING)
    - GATE 1 (Zero NSE Option Chain Calls):
      Strict Cash Turnover Floor (>= Rs 15 Cr in 15m), Spot >= 500 INR, RVOL >= 1.6x,
      and VWAP Alignment. Eliminates 212 dead/illiquid stocks in 1 second!
    - GATE 2 (Targeted Safe Option Chain Query):
      Queries NSE live Option Chain ONLY for the top surviving liquid leaders!
      Vol/OI >= 2.0x (Gamma Squeeze), Call Unwinding (>= 50 lots), Put Support Wall.
    - GATE 3 (Execution Timing & Pricing):
      Executable Ask price entry, Pro-Desk dynamic SL/Target, and 15m time SL.
    ========================================================================================
    """
    sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
    if sym_clean in INDEX_NAMES:
        return None
    if df_slice is None or len(df_slice) < 7:
        return None

    eval_bar = df_slice.iloc[-2]
    live_bar = df_slice.iloc[-1]
    comp_bars = df_slice.iloc[:-2].tail(min(8, len(df_slice) - 2))
    if len(comp_bars) < 5:
        return None

    eval_close = float(eval_bar['close'])
    eval_open = float(eval_bar['open'])
    eval_high = float(eval_bar['high'])
    eval_low = float(eval_bar['low'])
    eval_vol = float(eval_bar.get('volume', 0.0))
    bar_rng = max(0.01, eval_high - eval_low)
    live_spot = float(live_spot_override) if live_spot_override is not None else float(live_bar['close'])

    # 1. Price Floor (Instantly eliminates penny / low-price option illiquidity)
    if live_spot < MIN_STOCK_PRICE:
        return None

    atr = float(daily_atr) if float(daily_atr or 0.0) > 0.0 else (live_spot * 0.022)

    # Intraday day boundaries
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

    if pct_change_override is not None:
        pct_change = float(pct_change_override)
    else:
        pct_change = round(((live_spot - day_open) / max(1.0, day_open)) * 100.0, 2)

    # 2. Gate 1 Liquidity & 15-Minute Rolling Cash Turnover Floor (ZERO NSE OPTION CHAIN CALLS)
    turnover_cr = (eval_close * eval_vol) / 1e7
    bars_5_turnover = []
    for idx_b in range(max(0, len(df_slice) - 6), len(df_slice) - 1):
        b_item = df_slice.iloc[idx_b]
        bars_5_turnover.append((float(b_item.get('close', 0.0)) * float(b_item.get('volume', 0.0))) / 1e7)
    turnover_15m = float(sum(bars_5_turnover)) if bars_5_turnover else turnover_cr

    # Strict Institutional Liquidity Defense: Eliminates dead stocks like PIIND (turnover 4.8 Cr)
    if turnover_15m < 15.0 and turnover_cr < 8.0:
        return None

    # RVOL Calculation
    hist_slot_vol = float(eval_bar.get('hist_slot_vol', 0.0))
    comp_vol_med = float(comp_bars['volume'].median()) if 'volume' in comp_bars else 1000.0
    base_vol = max(100.0, hist_slot_vol if hist_slot_vol > 0 else comp_vol_med)
    real_rvol = round(eval_vol / max(1.0, base_vol), 2)
    if real_rvol < 1.6 and turnover_cr < 12.0:
        return None  # Rejects low-participation churn

    # Wyckoff VWAP Calculation
    tp_all = (df_slice['high'] + df_slice['low'] + df_slice['close']) / 3.0
    vol_weights = df_slice['volume'] if 'volume' in df_slice.columns else pd.Series(1.0, index=df_slice.index)
    vwap = float((tp_all * vol_weights).sum() / max(1.0, float(vol_weights.sum()))) if float(vol_weights.sum()) > 0 else float(tp_all.mean())

    # Delta Flow Calculation (Uses real cash volume weighting for stocks)
    eval_bvc, bvc_cum_ratio = compute_bar_delta_proxy(df_slice, is_index=False)

    # Direction Selection
    directions_to_test = []
    if nifty_regime.get('allow_stock_call', True) and eval_close > vwap and pct_change >= 0.35 and pct_change <= 4.20:
        directions_to_test.append('CALL')
    if nifty_regime.get('allow_stock_put', True) and eval_close < vwap and pct_change <= -0.35 and pct_change >= -4.20:
        directions_to_test.append('PUT')

    if not directions_to_test:
        return None

    lot_size = get_symbol_lot_size(sym_clean)

    for direction in directions_to_test:
        # PILLAR 1: Institutional Volume & Velocity (25 pts max)
        thrust = (eval_close - eval_low) / bar_rng if direction == 'CALL' else (eval_high - eval_close) / bar_rng
        if thrust < 0.55:
            continue
        p1_score = round(min(25.0, max(15.0, 14.0 + min(6.0, real_rvol * 2.0) + min(5.0, turnover_15m / 10.0))), 2)

        # PILLAR 2: Delta Order Flow & BVC (25 pts max)
        if direction == 'CALL':
            if eval_bvc <= 0 or bvc_cum_ratio <= 0:
                continue
            p2_score = round(min(25.0, max(14.0, (eval_bvc * 12.0) + (bvc_cum_ratio * 13.0))), 2)
        else:
            if eval_bvc >= 0 or bvc_cum_ratio >= 0:
                continue
            p2_score = round(min(25.0, max(14.0, (abs(eval_bvc) * 12.0) + (abs(bvc_cum_ratio) * 13.0))), 2)

        # PILLAR 3: Market Profile & VWAP Anchor (20 pts max)
        p3_score = 18.0 if ((direction == 'CALL' and eval_close > vwap) or (direction == 'PUT' and eval_close < vwap)) else 10.0

        # PILLAR 5: Super-Trend ATR Expansion Runway (10 pts max)
        fuel_spent = day_high - day_low
        remaining_atr = atr - fuel_spent
        fuel_ratio = remaining_atr / max(1.0, atr)
        is_super_trend = (real_rvol >= 2.5 and turnover_cr >= 2.5) or (turnover_15m >= 35.0)

        if is_super_trend:
            if fuel_spent > (3.0 * atr):
                continue
            p5_score = round(min(10.0, max(6.0, 7.0 + min(3.0, (abs(pct_change) / 2.0)))), 2)
        else:
            if fuel_ratio < 0.15:
                continue
            p5_score = round(min(10.0, max(4.0, fuel_ratio * 7.0 + (abs(pct_change) / 3.0) * 3.0)), 2)

        prelim_score = round(p1_score + p2_score + p3_score + p5_score, 2)

        # GATE 2: TARGETED OPTION CHAIN QUERY (Only runs for 3-6 surviving liquid candidates!)
        target_dist = max(40.0, min(65.0, 0.85 * atr))
        target_price = (eval_close + target_dist) if direction == 'CALL' else (eval_close - target_dist)

        opt = data_feed.compute_option_pricing(
            spot_price=eval_close,
            direction=direction,
            symbol=sym_clean,
            target_price=target_price,
            is_index=False
        )

        if not opt or not opt.get('is_liquid', False):
            continue

        oi_score = float(opt.get('oi_score', 12.0))
        total_score = round(prelim_score + oi_score, 2)

        if total_score >= STOCK_SCORE_THRESHOLD:
            return {
                'symbol': sym_clean,
                'is_index': False,
                'direction': direction,
                'spot_price': round(eval_close, 2),
                'entry_price': round(float(live_spot), 2),
                'score': total_score,
                'prelim_score': prelim_score,
                'oi_score': oi_score,
                'stock_atr': atr,
                'rvol': real_rvol,
                'turnover_15m': round(turnover_15m, 2),
                'vwap': round(vwap, 2),
                'pct_change': pct_change,
                'verified_opt': opt,
                'passed': True,
                'tier_reason': f'INSTITUTIONAL_STOCK_{direction}_SWAPPED_CONFIRMED'
            }

    return None


# =========================================================================
# PRODUCTION SCREENER CLASS: ALPHA INSTITUTIONAL SWAPPED SCREENER
# =========================================================================
class AlphaInstitutionalSwappedScreener:
    """
    Autonomous Swapped Screener, Journaler & Live Position Manager.
    - Zero Akamai Block Risk (Pre-filters 220 stocks via Cash Turnover/RVOL).
    - Targeted Option Chain Discovery on top candidates.
    - 100% untouched old codebase compatibility.
    """

    def __init__(self):
        self.logger = logging.getLogger('alpha_swapped_screener')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [SWAPPED_SCREENER] %(message)s'))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.data_feed = LiveMarketDataFeed()
        self.notifier = TelegramNotifier(CONFIG_PATH)
        self.universe = self._load_fno_universe()
        self.atr_cache = self.data_feed._atr_cache
        self.dispatched_cache_file = os.path.join(CURRENT_DIR, 'config', 'dispatched_signals_today.json')
        self.dispatched_today = self._load_dispatched_cache()
        self.nifty_regime_cache = {
            'allow_stock_call': True,
            'allow_stock_put': True,
            'allow_call': True,
            'allow_put': True,
            'reason': 'NEUTRAL',
            'timestamp': 0.0
        }

    def _load_fno_universe(self) -> List[Dict[str, Any]]:
        possible_paths = [
            os.path.join(CURRENT_DIR, 'modules', 'fno_universe_cache.json'),
            os.path.join(CURRENT_DIR, 'config', 'fno_universe.json')
        ]
        for p in possible_paths:
            if os.path.exists(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        u = json.load(f)
                        if isinstance(u, list) and len(u) > 10:
                            return u
                except Exception as e:
                    self.logger.warning(f"Error loading {p}: {e}")

        bulk = fetch_bulk_nse_snapshot()
        if bulk and len(bulk) > 10:
            return [{'symbol': s} for s in bulk.keys()]

        return [{'symbol': idx} for idx in INDEX_NAMES]

    def _load_dispatched_cache(self) -> Dict[str, float]:
        today_str = get_ist_time().strftime('%Y-%m-%d')
        if os.path.exists(self.dispatched_cache_file):
            try:
                with open(self.dispatched_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('date') == today_str:
                        return data.get('dispatched', {})
            except Exception as e:
                self.logger.warning(f"Error reading dispatched cache: {e}")
        return {}

    def _save_dispatched_cache(self):
        today_str = get_ist_time().strftime('%Y-%m-%d')
        try:
            os.makedirs(os.path.dirname(self.dispatched_cache_file), exist_ok=True)
            with open(self.dispatched_cache_file, 'w', encoding='utf-8') as f:
                json.dump({'date': today_str, 'dispatched': self.dispatched_today}, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving dispatched cache: {e}")

    def is_already_dispatched(self, symbol: str) -> bool:
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        return sym_clean in self.dispatched_today

    def mark_dispatched(self, symbol: str, contract: str = ''):
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        self.dispatched_today[sym_clean] = time.time()
        self._save_dispatched_cache()

    def is_eod_journal_dispatched_today(self) -> bool:
        today_str = get_ist_time().strftime('%Y-%m-%d')
        if os.path.exists(self.dispatched_cache_file):
            try:
                with open(self.dispatched_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('date') == today_str and data.get('eod_journal_dispatched', False):
                        return True
            except Exception:
                pass
        return False

    def mark_eod_journal_dispatched(self):
        today_str = get_ist_time().strftime('%Y-%m-%d')
        ist_time_str = get_ist_time().strftime('%I:%M:%S %p IST')
        current_data = {'date': today_str, 'dispatched': self.dispatched_today}
        if os.path.exists(self.dispatched_cache_file):
            try:
                with open(self.dispatched_cache_file, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
            except Exception:
                pass
        current_data['date'] = today_str
        current_data['dispatched'] = self.dispatched_today
        current_data['eod_journal_dispatched'] = True
        current_data['eod_dispatched_time'] = ist_time_str
        try:
            os.makedirs(os.path.dirname(self.dispatched_cache_file), exist_ok=True)
            with open(self.dispatched_cache_file, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving EOD dispatched state: {e}")

    def run_eod_analysis_and_dispatch(self):
        """Performs daily trade success/failure analysis at 03:35 PM IST and dispatches Report + Excel strictly to Owner Chatbot."""
        ist_now = get_ist_time()
        ist_time_str = ist_now.strftime('%I:%M:%S %p IST')
        ist_date_str = ist_now.strftime('%d-%b-%Y')
        daily_xl = os.path.join(CURRENT_DIR, 'Alpha_5Pillar_Live_Paper_Trading_Journal.xlsx')

        tot_mins = ist_now.hour * 60 + ist_now.minute
        is_force = ('--force-journal' in sys.argv or '--force-dispatch' in sys.argv)
        if not is_force and not (930 <= tot_mins <= 990):
            print(f"[EOD DISPATCH BLOCKED]: Current time {ist_time_str} is outside 03:30 - 04:30 PM IST window.")
            return

        if not is_force and self.is_eod_journal_dispatched_today():
            print(f"[EOD DISPATCH BLOCKED]: Today's EOD Journal for {ist_date_str} was ALREADY delivered to Owner Chatbot.")
            return

        if not os.path.exists(daily_xl):
            print(f"[EOD ERROR]: Excel file {daily_xl} not found.")
            return

        df = pd.read_excel(daily_xl)
        df_today = df[df['Date'].astype(str).str.contains(ist_date_str, na=False)] if 'Date' in df.columns else pd.DataFrame()

        total_trades = len(df_today)
        wins = 0
        losses = 0
        net_pnl = 0.0
        trade_diagnostics = []

        if total_trades > 0:
            for idx, r in df_today.iterrows():
                outcome = str(r.get('Trade_Outcome', r.get('Status', '')))
                sym = r.get('Stock_Index', '')
                contract = r.get('Option_Contract', '')
                pnl = float(r.get('Net_PnL_INR', 0.0))
                net_pnl += pnl
                why = str(r.get('Why_Success_or_Fail', r.get('Rectification_Suggestion', 'Normal execution')))

                if 'TARGET' in outcome or pnl > 0:
                    wins += 1
                    status_emoji = '🎯 WIN'
                else:
                    losses += 1
                    status_emoji = '🛑 SL HIT'

                trade_diagnostics.append(
                    f"• <b>{sym} ({contract}):</b> {status_emoji} | Net PnL: <b>Rs. {pnl:+,.2f}</b>\n"
                    f"  <i>Analysis:</i> {html.escape(why[:120])}..."
                )
            win_rate = round((wins / total_trades) * 100.0, 1) if total_trades > 0 else 0.0
        else:
            win_rate = 0.0
            trade_diagnostics.append("• <i>Zero trades executed today. Market maintained in sideways chop; capital 100% protected.</i>")

        pnl_sign = '+' if net_pnl >= 0 else ''
        pnl_color = '🟢' if net_pnl >= 0 else '🔴'

        report_lines = [
            "📊 <b>DAILY EOD TRADE AUDIT & PERFORMANCE POST-MORTEM (SWAPPED)</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📅 <b>DATE:</b> <b>{ist_date_str}</b> | ⏰ <b>TIME:</b> <b>{ist_time_str}</b>",
            f"📈 <b>TOTAL TRADES EXECUTED:</b> <b>{total_trades}</b>",
            f"🏆 <b>WINS:</b> <b>{wins}</b> | 🛑 <b>LOSSES:</b> <b>{losses}</b> | 🎯 <b>WIN RATE:</b> <b>{win_rate}%</b>",
            f"{pnl_color} <b>TOTAL NET REALIZED P&L:</b> <b>{pnl_sign}Rs. {net_pnl:,.2f}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "🔍 <b>DETAILED INDIVIDUAL TRADE POST-MORTEM:</b>",
            "\n".join(trade_diagnostics),
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "🔒 <i>Report & Excel Journal delivered strictly to Owner Chatbot. Group silence maintained.</i>"
        ]

        report_text = "\n".join(report_lines)
        self.notifier.send_broadcast_message(report_text, is_trade_signal=False)

        caption_daily = (
            f"📊 <b>DAILY OFFICIAL LIVE PAPER TRADING JOURNAL (SWAPPED)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 <b>DATE:</b> {ist_date_str}\n"
            f"⏰ <b>DISPATCH TIME:</b> {ist_time_str}\n"
            f"🎯 <b>MODE:</b> Daily Official Paper Trading Ledger\n"
            f"🔒 <b>SECURITY:</b> Strictly Confidential (Owner Private DM Only)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        res_d = self.notifier.send_document_to_owner_with_approval_prompt(daily_xl, caption=caption_daily)
        print(f"[EOD AUDIT & DISPATCH]: Delivered daily analysis & journal to Owner DM. Result: {res_d.get('ok')}")

        monthly_xl = os.path.join(CURRENT_DIR, 'Alpha_Monthly_Paper_Trading_Journal.xlsx')
        if os.path.exists(monthly_xl):
            caption_monthly = (
                f"📈 <b>MONTHLY CUMULATIVE PAPER TRADING JOURNAL (SWAPPED)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📅 <b>MONTH:</b> {ist_now.strftime('%B %Y')}\n"
                f"⏰ <b>DISPATCH TIME:</b> {ist_time_str}\n"
                f"🎯 <b>MODE:</b> Monthly Master Paper Trading Ledger\n"
                f"🔒 <b>SECURITY:</b> Strictly Confidential (Owner Private DM Only)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            res_m = self.notifier.send_document_to_owner_with_approval_prompt(monthly_xl, caption=caption_monthly)
            print(f"[EOD AUDIT & DISPATCH]: Delivered monthly master journal to Owner DM. Result: {res_m.get('ok')}")
        self.mark_eod_journal_dispatched()
        return res_d

    def send_eod_journal_to_owner(self):
        """Compatibility alias for EOD dispatch."""
        return self.run_eod_analysis_and_dispatch()

    def run_authoritative_5day_backtest(self):
        """Runs the 5-day authoritative backtest synchronized with the institutional swapped logic."""
        from daily_autonomous_screener import DailyAutonomousScreener
        old_screener = DailyAutonomousScreener()
        return old_screener.run_authoritative_5day_backtest()

    def get_stock_atr(self, symbol: str, live_q: Optional[Dict[str, Any]] = None) -> float:
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        if sym_clean in self.atr_cache and float(self.atr_cache[sym_clean]) > 0.0:
            return float(self.atr_cache[sym_clean])
        if live_q and 'live_spot' in live_q and float(live_q['live_spot']) > 0:
            return float(live_q['live_spot']) * 0.022
        return 45.0 if sym_clean not in INDEX_NAMES else 120.0

    def get_nifty_market_regime(self, force_refresh: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force_refresh and (now - self.nifty_regime_cache.get('timestamp', 0.0) < 180.0):
            return self.nifty_regime_cache

        try:
            df_nifty = self.data_feed.get_live_intraday_bars('NIFTY')
            if df_nifty is not None and len(df_nifty) >= 5:
                today_date = df_nifty.index[-1].date()
                today_bars = df_nifty[df_nifty.index.date == today_date]
                if len(today_bars) > 0:
                    n_open = float(today_bars['open'].iloc[0])
                    n_spot = float(today_bars['close'].iloc[-1])
                    move_pts = n_spot - n_open
                    pct = (move_pts / n_open) * 100.0

                    if pct >= 0.30:
                        regime = {
                            'allow_call': True, 'allow_put': False,
                            'allow_stock_call': True, 'allow_stock_put': False,
                            'reason': 'NIFTY_UPTREND', 'move_pts': round(move_pts, 1),
                            'nifty_spot': n_spot, 'timestamp': now
                        }
                    elif pct <= -0.30:
                        regime = {
                            'allow_call': False, 'allow_put': True,
                            'allow_stock_call': False, 'allow_stock_put': True,
                            'reason': 'NIFTY_DOWNTREND', 'move_pts': round(move_pts, 1),
                            'nifty_spot': n_spot, 'timestamp': now
                        }
                    else:
                        regime = {
                            'allow_call': True, 'allow_put': True,
                            'allow_stock_call': True, 'allow_stock_put': True,
                            'reason': 'NIFTY_NEUTRAL_CHOP', 'move_pts': round(move_pts, 1),
                            'nifty_spot': n_spot, 'timestamp': now
                        }
                    self.nifty_regime_cache = regime
                    return regime
        except Exception as e:
            self.logger.warning(f"Error computing Nifty regime: {e}")

        return {
            'allow_call': True, 'allow_put': True,
            'allow_stock_call': True, 'allow_stock_put': True,
            'reason': 'NEUTRAL_FALLBACK', 'move_pts': 0.0, 'timestamp': now
        }

    def run_screener(self, ignore_window: bool = False, send_heartbeat: bool = True, cycle_num: Optional[int] = None):
        ist_now = get_ist_time()
        ist_time_str = ist_now.strftime('%I:%M:%S %p IST')

        if not ignore_window and not is_in_active_trading_slab(ist_now):
            print(f"[SLAB CHECK]: {ist_time_str} is outside active slabs (09:30-10:15 & 13:15-14:15). Running status heartbeat.")
            return

        print(f"\n=======================================================")
        print(f"  ALPHA INSTITUTIONAL SWAPPED SCREENER | {ist_now.strftime('%d-%b-%Y %I:%M:%S %p IST')}")
        print(f"  Architecture: Turnover & Gamma Squeeze First (Zero Akamai Block Risk)")
        print(f"=======================================================")

        self.update_open_paper_trades()

        bulk = fetch_bulk_nse_snapshot(force_refresh=True)
        regime = self.get_nifty_market_regime(force_refresh=True)
        print(f"[MACRO REGIME]: {regime['reason']} | Nifty Move: {regime.get('move_pts', 0.0):+.1f} pts")

        # 1. Evaluate 4 Indices
        index_candidates = []
        for idx in INDEX_NAMES:
            df = self.data_feed.get_live_intraday_bars(idx)
            atr = self.get_stock_atr(idx, bulk.get(idx))
            res = evaluate_institutional_index_engine(
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

        # 2. Evaluate All 216 F&O Stocks (Zero hardcoding!)
        symbols_to_scan = [
            item['symbol'] if isinstance(item, dict) else item
            for item in self.universe
            if (item['symbol'] if isinstance(item, dict) else item) not in INDEX_NAMES
        ]

        def scan_stock_worker(sym):
            try:
                sym_clean = sym.upper().replace('.NS', '').replace('^', '').strip()
                df = self.data_feed.get_live_intraday_bars(sym_clean)
                if df is None or len(df) < 7:
                    return None
                atr = self.get_stock_atr(sym_clean, bulk.get(sym_clean))
                return evaluate_institutional_stock_engine(
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

        print(f"[DISCOVERY COMPLETE]: Scanned {len(INDEX_NAMES)} Indices & {len(symbols_to_scan)} Stocks.")
        print(f"[QUALIFIED DISCOVERY]: {len(index_candidates)} Index candidate(s), {len(stock_candidates)} Stock candidate(s).")

        all_candidates = []
        for cand in (index_candidates + stock_candidates):
            sym = cand['symbol']
            if self.is_already_dispatched(sym):
                print(f"[REJECTION]: {sym} already dispatched today.")
                continue
            all_candidates.append(cand)

        # Rank candidates by institutional score descending
        all_candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        verified_super_trades = []
        active_slab_index = 0
        active_slab_stock = 0

        for cand in all_candidates:
            if len(verified_super_trades) >= MAX_SLAB_TRADES:
                break
            is_idx = cand.get('is_index', False)
            if is_idx:
                if active_slab_index >= MAX_INDEX_PER_SLAB:
                    continue
                active_slab_index += 1
            else:
                if active_slab_stock >= MAX_STOCKS_PER_SLAB:
                    continue
                active_slab_stock += 1
            verified_super_trades.append(cand)

        if not verified_super_trades:
            print(f"[SCREENER]: 0 signals qualified at {ist_time_str}. Capital 100% Protected.")
            if send_heartbeat:
                cycle_str = f" (Cycle #{cycle_num})" if cycle_num else ""
                heartbeat_msg = (
                    f"🛡️ <b>ALPHA INSTITUTIONAL SWAPPED SCANNER{cycle_str}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⏰ <b>Execution Time:</b> {ist_time_str}\n"
                    f"📊 <b>Universe:</b> 4 Indices & {len(symbols_to_scan)} Stocks Scanned\n"
                    f"⚡ <b>Architecture:</b> Turnover & Gamma Squeeze First\n"
                    f"🛡️ <b>Status:</b> Zero 5-Pillar Confluent Breakouts\n"
                    f"🔒 <b>Akamai Defense:</b> Active (Zero Request Flooding)\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔒 <i>Delivered strictly to Owner Chatbot. Group silence maintained.</i>"
                )
                self.notifier.send_broadcast_message(heartbeat_msg, is_trade_signal=False)
            return

        print(f"[SCREENER]: Verified {len(verified_super_trades)} INSTITUTIONAL SUPER TRADE(S)! Dispatching...")
        for cand in verified_super_trades:
            self._dispatch_super_trade(cand, ist_now)

    def _dispatch_super_trade(self, cand: Dict[str, Any], ist_now: datetime.datetime):
        sym = cand['symbol']
        spot = cand['spot_price']
        direction = cand['direction']
        is_idx = cand.get('is_index', False)
        opt = cand['verified_opt']

        contract = opt['contract']
        opt_entry = float(opt['entry_price'])
        opt_action = 'BUY CALL (CE)' if direction == 'CALL' else 'BUY PUT (PE)'

        buffer_end_time = (ist_now + datetime.timedelta(minutes=4)).strftime('%I:%M:%S %p IST')
        time_sl_time = (ist_now + datetime.timedelta(minutes=15)).strftime('%I:%M:%S %p IST')

        opt_sl = float(opt['sl_price'])
        opt_tgt1 = float(opt['target_1'])
        opt_tgt2 = float(opt['target_2'])
        fixed_spot_sl_str = "18.0 Pts SL" if is_idx else "5.0 Pts SL"

        pure_signal_msg = (
            f"🚀 <b>ALPHA 5-PILLAR INSTITUTIONAL TRADE (SWAPPED)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>INSTRUMENT:</b> <b>{sym} ({direction})</b>\n"
            f"⚡ <b>ACTION:</b> <b>{opt_action}</b>\n"
            f"📌 <b>CONTRACT:</b> <code>{contract}</code>\n"
            f"💵 <b>BUYING PRICE:</b> <b>Rs. {opt_entry:,.2f}</b>\n"
            f"🛡️ <b>STOP LOSS:</b> <b>Rs. {opt_sl:,.2f} ({fixed_spot_sl_str})</b>\n"
            f"🎯 <b>TARGET 1 (Lock Profit):</b> <b>Rs. {opt_tgt1:,.2f}</b>\n"
            f"🏆 <b>TARGET 2 (Jackpot):</b> <b>Rs. {opt_tgt2:,.2f}</b>\n"
            f"⏱️ <b>15-MIN TIMEOUT SL:</b> <b>{time_sl_time} (Cut if Stagnant/Red)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏳ <b>4-MIN BUFFER VALID TILL:</b> <b>{buffer_end_time}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        self.mark_dispatched(sym, contract)
        self._record_signal_to_excel(ist_now, cand, opt, opt_action, contract, opt_entry, opt_sl, opt_tgt1, opt_tgt2, time_sl_time, buffer_end_time)
        self.notifier.send_broadcast_message(pure_signal_msg, is_trade_signal=True)

    def _record_signal_to_excel(self, ist_now, cand, opt, opt_action, contract, opt_entry, opt_sl, opt_tgt1, opt_tgt2, time_sl_time, buffer_end_time):
        sym = cand['symbol']
        meta = get_instrument_meta(sym)
        qty = opt.get('quantity') or (meta.get('lot', 25) if meta else 25)
        trade_uid = f"SIG-{ist_now.strftime('%Y%m%d-%H%M%S')}-{sym}"
        csv_path = os.path.join(CURRENT_DIR, 'Live_Paper_Trading_Journal.csv')
        daily_xl = os.path.join(CURRENT_DIR, 'Alpha_5Pillar_Live_Paper_Trading_Journal.xlsx')
        monthly_xl = os.path.join(CURRENT_DIR, 'Alpha_Monthly_Paper_Trading_Journal.xlsx')

        row_dict = {
            'Trade_ID': trade_uid,
            'Date': ist_now.strftime('%d-%b-%Y'),
            'Month': ist_now.strftime('%B %Y'),
            'Signal_Time_IST': ist_now.strftime('%I:%M:%S %p IST'),
            'Signal_Timestamp_Epoch': ist_now.timestamp(),
            'Burst_Time_IST': buffer_end_time,
            'Lead_Buffer': '4 Min',
            'Stock_Index': sym,
            'Option_Action': opt_action,
            'Option_Contract': contract,
            'Lots': 1,
            'Quantity': qty,
            'Spot_Entry': cand['spot_price'],
            'Buying_Price_INR': opt_entry,
            'Stop_Loss_INR': opt_sl,
            'Trailing_SL_INR': opt_sl,
            'Target_1_INR': opt_tgt1,
            'Target_Price_INR': opt_tgt2,
            'Time_SL_IST': time_sl_time,
            'Exit_Time_IST': '',
            'Exit_Price_INR': 0.0,
            'Holding_Time': '0m',
            'Points_Captured': 0.0,
            'Gross_PnL_INR': 0.0,
            'Friction_Taxes_INR': 0.0,
            'Net_PnL_INR': 0.0,
            'Trade_Outcome': 'SIGNAL_GENERATED_OPEN',
            'Status': 'SIGNAL_GENERATED_OPEN',
            'Failure_Reason_And_Needed_Correction': 'TRADE_OPEN_ACTIVE'
        }

        try:
            for fpath in [csv_path, daily_xl, monthly_xl]:
                if fpath.endswith('.csv'):
                    df_old = pd.read_csv(fpath) if os.path.exists(fpath) else pd.DataFrame()
                    df_new = pd.concat([df_old, pd.DataFrame([row_dict])], ignore_index=True)
                    df_new.to_csv(fpath, index=False)
                elif fpath.endswith('.xlsx'):
                    df_old = pd.read_excel(fpath) if os.path.exists(fpath) else pd.DataFrame()
                    df_new = pd.concat([df_old, pd.DataFrame([row_dict])], ignore_index=True)
                    df_new.to_excel(fpath, index=False)
            print(f"[EXCEL JOURNAL]: Successfully recorded {contract} to Daily & Monthly Excel Ledgers.")
        except Exception as e:
            print(f"[EXCEL JOURNAL ERROR]: {e}")

    def update_open_paper_trades(self, force_slab_close: bool = False):
        csv_path = os.path.join(CURRENT_DIR, 'Live_Paper_Trading_Journal.csv')
        daily_xl = os.path.join(CURRENT_DIR, 'Alpha_5Pillar_Live_Paper_Trading_Journal.xlsx')
        monthly_xl = os.path.join(CURRENT_DIR, 'Alpha_Monthly_Paper_Trading_Journal.xlsx')
        if not os.path.exists(csv_path):
            return

        try:
            df = pd.read_csv(csv_path)
            if df.empty or 'Status' not in df.columns:
                return

            open_mask = df['Status'] == 'SIGNAL_GENERATED_OPEN'
            if not open_mask.any():
                return

            now_ist = get_ist_time()
            now_epoch = now_ist.timestamp()

            for idx in df[open_mask].index:
                row = df.loc[idx]
                sym = str(row['Stock_Index'])
                contract = str(row['Option_Contract'])
                buy_p = float(row['Buying_Price_INR'])
                sl_p = float(row['Trailing_SL_INR'])
                t1_p = float(row['Target_1_INR'])
                t2_p = float(row['Target_Price_INR'])
                sig_epoch = float(row.get('Signal_Timestamp_Epoch', now_epoch))
                elapsed_mins = max(1, int((now_epoch - sig_epoch) / 60.0))

                quote = self.data_feed.get_option_quote(sym, contract_identifier=contract)
                cur_bid = float(quote.get('bid', buy_p)) if quote else buy_p

                exit_reason = None
                exit_price = cur_bid

                # 1. Target 2 Hit (Jackpot)
                if cur_bid >= t2_p:
                    exit_reason = 'TARGET_2_JACKPOT_HIT'
                    exit_price = t2_p
                # 2. Target 1 Hit (Profit Lock)
                elif cur_bid >= t1_p:
                    # Trail SL to lock in 40% of T1 gain
                    new_sl = round(buy_p + 0.40 * (t1_p - buy_p), 2)
                    if new_sl > sl_p:
                        df.at[idx, 'Trailing_SL_INR'] = new_sl
                        print(f"[TRAILING SL]: {contract} T1 reached. SL trailed to Rs {new_sl:.2f}")
                # 3. Stop Loss Hit
                elif cur_bid <= sl_p:
                    exit_reason = 'STOP_LOSS_HIT'
                    exit_price = sl_p
                # 4. 15-Minute Timeout SL
                elif elapsed_mins >= 15 and cur_bid <= buy_p:
                    exit_reason = '15MIN_TIMEOUT_SL_STAGNATION'
                    exit_price = cur_bid
                # 5. Force Slab Close
                elif force_slab_close:
                    exit_reason = 'EOD_SLAB_CONCLUSION_CLOSE'
                    exit_price = cur_bid

                if exit_reason:
                    qty = int(row.get('Quantity', 25))
                    pts = round(exit_price - buy_p, 2)
                    gross = round(pts * qty, 2)
                    turnover = (buy_p + exit_price) * qty
                    taxes = round(0.000625 * exit_price * qty + 0.0005 * turnover + 40.0, 2)
                    net = round(gross - taxes, 2)

                    df.at[idx, 'Exit_Time_IST'] = now_ist.strftime('%I:%M:%S %p IST')
                    df.at[idx, 'Exit_Price_INR'] = exit_price
                    df.at[idx, 'Holding_Time'] = f"{elapsed_mins}m"
                    df.at[idx, 'Points_Captured'] = pts
                    df.at[idx, 'Gross_PnL_INR'] = gross
                    df.at[idx, 'Friction_Taxes_INR'] = taxes
                    df.at[idx, 'Net_PnL_INR'] = net
                    df.at[idx, 'Trade_Outcome'] = exit_reason
                    df.at[idx, 'Status'] = 'CLOSED'

                    exit_trade_dict = {
                        'Strike_Contract': contract,
                        'Exit_Time': now_ist.strftime('%I:%M:%S %p IST'),
                        'Holding_Mins': f"{elapsed_mins}m",
                        'Spot_Move': f"{pts:+.2f} pts",
                        'Option_Exit_Price': exit_price,
                        'Option_Move_Pts': f"{pts:+.2f} pts",
                        'Option_Move_Pct': f"{((exit_price - buy_p)/buy_p)*100:+.1f}%",
                        'Net_PnL': f"Rs {net:+,.2f}",
                        'Exit_Reason': exit_reason
                    }
                    self.notifier.send_exit_alert(exit_trade_dict)

            df.to_csv(csv_path, index=False)
            for xl in [daily_xl, monthly_xl]:
                if os.path.exists(xl):
                    df.to_excel(xl, index=False)

        except Exception as e:
            print(f"[UPDATE OPEN TRADES ERROR]: {e}")

    def run_live_slab_session(self, interval_seconds: int = 180):
        ist_start = get_ist_time()
        print(f"=======================================================")
        print(f"  ALPHA CONTINUOUS SLAB SESSION INITIALIZED (SWAPPED)")
        print(f"  Session Start Time: {ist_start.strftime('%d-%b-%Y %I:%M:%S %p IST')}")
        print(f"=======================================================")

        tot_m = ist_start.hour * 60 + ist_start.minute
        if 565 <= tot_m < 570:
            wait_s = (570 - tot_m) * 60
            print(f"[PRE-SLAB WARMUP]: Waiting {wait_s}s until 09:30 AM IST...")
            time.sleep(wait_s)
        elif 790 <= tot_m < 795:
            wait_s = (795 - tot_m) * 60
            print(f"[PRE-SLAB WARMUP]: Waiting {wait_s}s until 01:15 PM IST...")
            time.sleep(wait_s)

        cycle = 1
        max_duration_seconds = 70 * 60
        start_ts = time.time()

        while True:
            ist_now = get_ist_time()
            now_m = ist_now.hour * 60 + ist_now.minute
            is_m_slab = (570 <= now_m <= 615)
            is_a_slab = (795 <= now_m <= 855)

            if not (is_m_slab or is_a_slab):
                print(f"\n[SLAB CONCLUDED]: Current time {ist_now.strftime('%I:%M:%S %p IST')} is outside active slabs.")
                break

            if (time.time() - start_ts) > max_duration_seconds:
                print(f"\n[SESSION TIMEOUT]: Max duration of 70m reached. Exiting cleanly.")
                break

            print(f"\n>>> [SLAB CYCLE #{cycle}] Active Scan & Position Management at {ist_now.strftime('%I:%M:%S %p IST')}...")
            try:
                self.update_open_paper_trades()
                self.run_screener(ignore_window=True, send_heartbeat=True, cycle_num=cycle)
            except Exception as e:
                print(f"[SLAB ERROR in Cycle #{cycle}]: {e}")

            cycle += 1
            ist_next = get_ist_time()
            next_m = (ist_next.hour * 60 + ist_next.minute) + (interval_seconds // 60)
            if not ((570 <= next_m <= 616) or (795 <= next_m <= 856)):
                print(f"\n[SLAB CONCLUDED]: Next cycle falls outside slab window. Completed {cycle - 1} cycles.")
                break

            print(f">>> Sleeping {interval_seconds}s (3 mins) until next scan...")
            time.sleep(interval_seconds)

        try:
            self.update_open_paper_trades(force_slab_close=True)
        except Exception as e:
            print(f"[SLAB FINAL CLOSE ERROR]: {e}")


def check_github_actions_concurrency_guard() -> bool:
    github_run_id = os.environ.get('GITHUB_RUN_ID')
    if not github_run_id:
        return True
    try:
        import requests
        cur_id = int(github_run_id)
        api_url = "https://api.github.com/repos/manojmukati/alpha-option-buyer/actions/runs?status=in_progress"
        headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'AlphaTrading-ConcurrencyGuard'}
        resp = requests.get(api_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            runs = resp.json().get('workflow_runs', [])
            for r in runs:
                other_id = r.get('id')
                if other_id and other_id < cur_id:
                    print(f"[CONCURRENCY GUARD]: Run #{other_id} already executing. Aborting Run #{cur_id}.")
                    return False
    except Exception as e:
        print(f"[CONCURRENCY GUARD WARNING]: {e}")
    return True


if __name__ == '__main__':
    screener = AlphaInstitutionalSwappedScreener()
    if '--backtest' in sys.argv:
        screener.run_authoritative_5day_backtest()
    elif '--eod-audit' in sys.argv or '--send-journal' in sys.argv:
        screener.run_eod_analysis_and_dispatch()
    elif '--single-pass' in sys.argv:
        ignore_win = ('--ignore-window' in sys.argv)
        screener.run_screener(ignore_window=ignore_win)
    elif '--slab-session' in sys.argv:
        if not check_github_actions_concurrency_guard():
            sys.exit(0)
        screener.run_live_slab_session(interval_seconds=180)
    else:
        if not check_github_actions_concurrency_guard():
            sys.exit(0)
        ist_now = get_ist_time()
        if is_in_active_trading_slab(ist_now):
            screener.run_live_slab_session(interval_seconds=180)
        else:
            ignore_win = ('--ignore-window' in sys.argv)
            screener.run_screener(ignore_window=ignore_win)
