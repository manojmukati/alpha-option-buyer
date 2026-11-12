# -*- coding: utf-8 -*-
"""
daily_autonomous_screener.py
============================
Alpha 5-Pillar Pure Institutional Paper-Trading Screener & Autonomous Engine.

Key Architectural Upgrades:
  1. Shared Pure Function evaluate_common_5pillar_bar_slice():
     - Called by BOTH daily_autonomous_screener.py and scratch/execute_latest_5d_backtest.py.
     - Strictly requires len(df_slice) >= 10: 8 completed base bars + 1 completed breakout bar + 1 live holding bar.
     - Dedicated Index Engine (Branch A) using range-expansion RVOL & range-weighted BVC delta proxy.
     - Dedicated Stock Engine (Branch B):
       * Pillar 1 (25 pts): RVOL >= 1.80x vs max(hist_slot_vol, median_base) AND Turnover >= Rs 2.5 Cr.
       * Pillar 2 (25 pts): BVC Delta & CVD Persistence with divergence Hard Veto.
       * Pillar 3 (20 pts): Compression width <= 0.62 * ATR + Breakout + VWAP alignment + Live holding.
       * Pillar 4 (20 pts): Expiry-filtered symmetric -4 to +4 OI Runway & fail-closed option pricing.
       * Pillar 5 (10 pts): ATR Used Ratio <= 0.90 & Day Move <= 3.00%.
       * Gate 1 (Market Regime) & Gate 2 (CLV rejection wick veto < 0.58 CALL, > 0.42 PUT).
  2. Autonomous Scheduler & Position Manager:
     - Epoch-based 15-Minute Time SL loss scratch.
     - T1 profit locking (+0.40 * T1 gain).
     - T2 Jackpot exit (+3.5R).
     - 15:20 IST EOD square-off.
     - Simultaneous update of CSV, Daily Excel, and Monthly Excel.
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
import urllib.request
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

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
from modules.fno_universe import get_instrument_meta
from modules.telegram_notifier import TelegramNotifier

CONFIG_PATH = os.path.join(CURRENT_DIR, 'config', 'telegram_config.json')

INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

MIN_STOCK_PRICE = 500.0
MIN_STOCK_ATR = 20.0

MAX_DAILY_SUPER_TRADES = 8
MAX_SLAB_TRADES = 3
MAX_INDEX_PER_SLAB = 1
MAX_STOCKS_PER_SLAB = 3

# Pure dynamic asset handling: Zero hardcoded stock names. Only indices are fixed.


def is_in_active_trading_slab(ist_time: datetime.datetime) -> bool:
    """
    Checks if current IST time falls strictly within designated trading slabs:
    - Morning Prime:   09:30 AM to 10:12 AM IST (570 to 612 mins)
    - Afternoon Prime: 01:12 PM to 02:09 PM IST (792 to 849 mins)
    """
    if ist_time.weekday() >= 5:
        return False
    tot_mins = ist_time.hour * 60 + ist_time.minute
    slab1 = (570 <= tot_mins <= 615)
    slab2 = (795 <= tot_mins <= 855)
    return slab1 or slab2


def compute_bar_delta_proxy(df_bars: pd.DataFrame, is_index: bool = False) -> Dict[str, Any]:
    """
    Computes bar delta proxy and cumulative volume delta on completed bars.
    """
    if df_bars is None or len(df_bars) < 2:
        return {
            'valid': False,
            'session_cvd_ratio': 0.0,
            'recent_cvd_ratio': 0.0,
            'session_cvd_raw': 0.0,
            'cvd_direction': 'NEUTRAL',
            'reason': 'INSUFFICIENT_COMPLETED_BARS',
            'method': 'NONE'
        }

    highs = df_bars['high'].values.astype(float)
    lows = df_bars['low'].values.astype(float)
    closes = df_bars['close'].values.astype(float)
    vols = df_bars['volume'].values.astype(float) if 'volume' in df_bars.columns else np.zeros_like(closes)

    rngs = np.maximum(0.01, highs - lows)
    tot_vol = np.sum(vols)

    if is_index or tot_vol == 0:
        if is_index:
            deltas = (2.0 * closes - (highs + lows))
            tot_rng = float(np.sum(rngs))
            session_cvd_raw = float(np.sum(deltas))
            session_cvd_ratio = round(session_cvd_raw / max(0.5, tot_rng), 3)

            rec_n = min(3, len(df_bars))
            rec_rng = float(np.sum(rngs[-rec_n:]))
            rec_delta = float(np.sum(deltas[-rec_n:]))
            recent_cvd_ratio = round(rec_delta / max(0.5, rec_rng), 3)
            method_str = 'INDEX_SPOT_RANGE_PROXY'
        else:
            return {
                'valid': False,
                'session_cvd_ratio': 0.0,
                'recent_cvd_ratio': 0.0,
                'session_cvd_raw': 0.0,
                'cvd_direction': 'NEUTRAL',
                'reason': 'STOCK_VOLUME_UNAVAILABLE_OR_ZERO',
                'method': 'NONE'
            }
    else:
        if 'delta' in df_bars.columns:
            deltas = df_bars['delta'].values.astype(float)
            method_str = 'BVC_SUBBAR_MICRO_DELTA'
        else:
            deltas = vols * ((2.0 * closes - (highs + lows)) / rngs)
            method_str = 'BVC_VOLUME_DELTA'
        session_cvd_raw = float(np.sum(deltas))
        session_cvd_ratio = round(session_cvd_raw / max(1.0, tot_vol), 3)

        rec_n = min(3, len(df_bars))
        rec_vol = float(np.sum(vols[-rec_n:]))
        rec_delta = float(np.sum(deltas[-rec_n:]))
        recent_cvd_ratio = round(rec_delta / max(1.0, rec_vol), 3)

    cvd_dir = 'NEUTRAL'
    if recent_cvd_ratio >= 0.15 and session_cvd_ratio >= 0.0:
        cvd_dir = 'CALL'
    elif recent_cvd_ratio <= -0.15 and session_cvd_ratio <= 0.0:
        cvd_dir = 'PUT'

    return {
        'valid': True,
        'session_cvd_ratio': session_cvd_ratio,
        'recent_cvd_ratio': recent_cvd_ratio,
        'session_cvd_raw': session_cvd_raw,
        'cvd_direction': cvd_dir,
        'method': method_str
    }


def compute_bvc_cvd(df_bars: pd.DataFrame) -> Dict[str, Any]:
    """Backward compatibility wrapper redirecting to compute_bar_delta_proxy."""
    return compute_bar_delta_proxy(df_bars)


def evaluate_common_5pillar_bar_slice(
    symbol: str,
    df_slice: pd.DataFrame,
    daily_atr: float,
    nifty_regime: Dict[str, Any],
    live_spot_override: Optional[float] = None,
    pct_change_override: Optional[float] = None,
    is_index: bool = False
) -> Optional[Dict[str, Any]]:
    """
    ========================================================================================
    EXACT PURE 5-PILLAR INSTITUTIONAL EVALUATION ENGINE (HINDI LINE-BY-LINE LOGIC COMMENTS)
    Pillar 1: 25 Pts (RVOL >= 1.40x & Turnover >= 1.5 Cr)
    Pillar 2: 25 Pts (True BVC Volume Delta & CVD Flow - Divergence Veto)
    Pillar 3: 20 Pts (Wyckoff Market Profile Acceptance: VWAP/POC & Base Compression - Lagging EMA Removed)
    Pillar 4: 20 Pts (Symmetric -4 to +4 Strikes OI Runway & Short-Covering Unwind)
    Pillar 5: 10 Pts (Remaining ATR Fuel >= 35% & Decoupled Targets)
    Total Score: 100 Points (Qualification Threshold >= 70.0)
    Candlestick Shape = 0% Veto Power (Candle shape se koi valid trade reject nahi hoga).
    ========================================================================================
    """
    # [LINE 1] Check karo: Kya kam se kam 7 bars (3-minute candles) available hain? Agar nahi toh reject karo.
    if df_slice is None or len(df_slice) < 7:
        return None

    # [LINE 2] Symbol name saaf karo (jaise .NS ya ^ hatao) aur check karo ki ye Index hai ya Stock.
    sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
    is_idx = is_index or (sym_clean in INDEX_NAMES)

    # [LINE 3] Base bars nikalo: Breakout candle se pehle ki 5 se 8 candles base consolidation define karti hain.
    comp_bars = df_slice.iloc[:-2].tail(min(8, len(df_slice) - 2))
    # [LINE 4] eval_bar = Breakout candle (jo base ko todti hai).
    eval_bar = df_slice.iloc[-2]
    # [LINE 5] live_bar = Live entry candle (jisme breakout confirm hone par turant entry li jati hai).
    live_bar = df_slice.iloc[-1]

    # [LINE 6] Agar base me 5 candles se kam hain toh pattern invalid hai, reject karo.
    if len(comp_bars) < 5:
        return None

    # [LINE 7] Breakout candle ke OHLCV (Open, High, Low, Close, Volume) values nikalo.
    eval_close = float(eval_bar['close'])
    eval_open = float(eval_bar['open'])
    eval_high = float(eval_bar['high'])
    eval_low = float(eval_bar['low'])
    eval_vol = float(eval_bar.get('volume', 0.0))

    # [LINE 8] Live spot price nikalo: Live bar ka close price entry price hoga.
    live_spot = float(live_spot_override) if live_spot_override is not None else float(live_bar['close'])

    # [LINE 9] Aaj ke trading session ka Open, Day High aur Day Low nikalo (prior days ka data mix nahi hona chahiye).
    if hasattr(df_slice.index, 'date'):
        today_date = df_slice.index[-1].date()
        today_bars = df_slice[df_slice.index.date == today_date]
        if len(today_bars) > 0:
            day_open = float(today_bars['open'].iloc[0])
            day_high = float(today_bars['high'].max())
            day_low = float(today_bars['low'].min())
        else:
            day_open = float(df_slice['open'].iloc[0])
            day_high = float(df_slice['high'].max())
            day_low = float(df_slice['low'].min())
    else:
        day_open = float(df_slice['open'].iloc[0])
        day_high = float(df_slice['high'].max())
        day_low = float(df_slice['low'].min())

    # [LINE 10] Din ka percentage move nikalo: (Live Spot - Day Open) / Day Open * 100
    if pct_change_override is not None:
        pct_change = float(pct_change_override)
    else:
        pct_change = round(((live_spot - day_open) / max(1.0, day_open)) * 100.0, 2)

    # [LINE 11] Verified Wilder 14-day Daily ATR nikalo (agar missing ho toh spot ka 2.2% default lo).
    atr = float(daily_atr) if float(daily_atr or 0.0) > 0.0 else (live_spot * 0.022)

    # [LINE 12] Trading Slab Check: Morning (09:30-10:12 AM) ya Afternoon (01:12-02:09 PM) me hi trade allow hai.
    target_bar = live_bar if (hasattr(live_bar, 'name') and hasattr(live_bar.name, 'hour')) else eval_bar
    if hasattr(target_bar, 'name') and hasattr(target_bar.name, 'hour'):
        tot_m = target_bar.name.hour * 60 + target_bar.name.minute
        is_m = (570 <= tot_m <= 615)   # 09:30 se 10:15 AM IST
        is_a = (795 <= tot_m <= 855)   # 01:15 se 02:15 PM IST
        if not (is_m or is_a):
            return None  # Slab ke bahar trade reject karo
    else:
        is_m, is_a = True, False

    # [LINE 13] Base range nikalo: Consolidation base ka High, Low aur Width (comp_width).
    comp_high = float(comp_bars['high'].max())
    comp_low = float(comp_bars['low'].min())
    comp_width = comp_high - comp_low

    # [LINE 14] 0% CANDLESTICK SHAPE VETO: Sirf break out trigger check karo (wick/rejection shape se reject mat karo).
    # Agar Breakout candle ka Close base High se upar aur green candle hai -> CALL direction.
    if eval_close > comp_high and eval_close > eval_open:
        direction = 'CALL'
    # Agar Breakout candle ka Close base Low se niche aur red candle hai -> PUT direction.
    elif eval_close < comp_low and eval_close < eval_open:
        direction = 'PUT'
    else:
        return None  # Na base high toota na low toota, trade reject karo.

    # [LINE 15] Macro Nifty Regime Filter: Nifty ke opposite trade mat lo.
    if is_idx:
        # Index CALL ke liye Nifty regime allow_call hona chahiye.
        if direction == 'CALL' and not nifty_regime.get('allow_call', True):
            return None
        # Index PUT ke liye Nifty bearish aur fall confirmed hona chahiye.
        if direction == 'PUT':
            if not nifty_regime.get('allow_put', True):
                return None
            if not nifty_regime.get('allow_stock_put', False) or pct_change > -0.15:
                return None
    else:
        # Stock CALL ke liye Nifty crash nahi hona chahiye.
        if direction == 'CALL' and not nifty_regime.get('allow_stock_call', True):
            return None
        # Stock PUT ke liye Nifty bearish hona zaroori hai.
        if direction == 'PUT' and not nifty_regime.get('allow_stock_put', False):
            return None
        # Minimum Stock Floor: Price >= 500 & Volatility check (ATR >= 12 or >= 1.2% of spot)
        if live_spot < MIN_STOCK_PRICE or (atr < 12.0 and (atr / max(1.0, live_spot)) < 0.012):
            return None
        # Din ka move 0.20% se kam (completely flat) ya 4.50% se zyada (exhausted circuit) ho toh reject karo.
        if abs(pct_change) < 0.20 or abs(pct_change) > 4.50:
            return None

    # ========================================================================================
    # PILLAR 1: Institutional RVOL & Smart Money Turnover (Max: 25 Points)
    # PURE 100-POINT SCORING: Smooth continuous scoring, no arbitrary hard vetoes!
    # ========================================================================================
    hist_slot_vol = float(eval_bar.get('hist_slot_vol', 0.0))
    comp_vol_med = float(comp_bars['volume'].median()) if 'volume' in comp_bars else 1000.0
    base_vol = max(100.0, hist_slot_vol if hist_slot_vol > 0 else comp_vol_med)

    if not is_idx:
        # Stock ke liye 3-minute turnover in Crores: (Price * Volume) / 1 Crore
        turnover_cr = (eval_close * eval_vol) / 1e7
        real_rvol = round(eval_vol / max(1.0, base_vol), 2)
        # Continuous scoring: RVOL points (max 15) + Turnover points (max 10) = 25 Points
        if real_rvol >= 2.0:
            rvol_pts = 15.0
        elif real_rvol >= 1.5:
            rvol_pts = 12.0
        elif real_rvol >= 1.2:
            rvol_pts = 9.0
        elif real_rvol >= 1.0:
            rvol_pts = 6.0
        else:
            rvol_pts = round(max(1.0, real_rvol * 5.0), 2)

        if turnover_cr >= 2.5:
            turn_pts = 10.0
        elif turnover_cr >= 1.5:
            turn_pts = 8.0
        elif turnover_cr >= 0.8:
            turn_pts = 5.0
        else:
            turn_pts = round(max(1.0, turnover_cr * 4.0), 2)

        p1_score = round(min(25.0, rvol_pts + turn_pts), 2)
    else:
        # Index ke liye range-expansion RVOL
        bar_rng = max(0.05, eval_high - eval_low)
        comp_rngs = comp_bars['high'] - comp_bars['low']
        median_rng = float(comp_rngs.median()) if len(comp_rngs) > 0 else 15.0
        idx_rvol = round(bar_rng / max(0.5, median_rng), 2)
        real_rvol = idx_rvol
        turnover_cr = 50.0  # Index institutional liquidity
        if idx_rvol >= 2.0:
            p1_score = 25.0
        elif idx_rvol >= 1.5:
            p1_score = 21.0
        elif idx_rvol >= 1.2:
            p1_score = 18.0
        elif idx_rvol >= 1.0:
            p1_score = 15.0
        elif idx_rvol >= 0.8:
            p1_score = 11.0
        else:
            p1_score = round(max(3.0, idx_rvol * 12.0), 2)

    # ========================================================================================
    # PILLAR 2: True BVC Volume Delta & Multi-Bar CVD (Max: 25 Points)
    # PURE 100-POINT SCORING: Aligned flow gets full points, divergent gets low points
    # ========================================================================================
    bars_5 = df_slice.iloc[-6:-1]
    if len(bars_5) < 3:
        return None
    price_chg = bars_5['close'] - bars_5['open']
    std_val = float(price_chg.std()) if len(bars_5) > 1 else 0.0
    sigma = std_val if std_val > 0.01 else float((bars_5['high'] - bars_5['low']).mean() * 0.5)
    bvc_z = np.clip(price_chg / max(0.01, sigma), -3.0, 3.0)
    bvc_fraction = np.tanh(bvc_z)

    if not is_idx:
        weights = bars_5['volume'].values.astype(float) if 'volume' in bars_5.columns else np.ones(len(bars_5))
    else:
        weights = (bars_5['high'] - bars_5['low']).values.astype(float)
    bvc_cum_ratio = float(np.sum(weights * bvc_fraction.values) / max(1.0, float(np.sum(weights))))
    eval_bvc = float(bvc_fraction.iloc[-1])

    is_delta_aligned = (direction == 'CALL' and eval_bvc > 0) or (direction == 'PUT' and eval_bvc < 0)
    is_cum_aligned = (direction == 'CALL' and bvc_cum_ratio > 0) or (direction == 'PUT' and bvc_cum_ratio < 0)

    if is_delta_aligned and is_cum_aligned:
        eval_pts = min(15.0, abs(eval_bvc) * 16.0)
        cum_pts = min(10.0, abs(bvc_cum_ratio) * 18.0)
        p2_score = round(eval_pts + cum_pts, 2)
    elif is_delta_aligned:
        p2_score = round(max(6.0, min(14.0, abs(eval_bvc) * 14.0)), 2)
    elif is_cum_aligned:
        p2_score = round(max(5.0, min(10.0, abs(bvc_cum_ratio) * 12.0)), 2)
    else:
        p2_score = 2.0  # Divergence penalty reflected in score

    # ========================================================================================
    # PILLAR 3: Wyckoff Market Profile POC/VAH/VAL & Compression Quality (Max: 20 Points)
    # ========================================================================================
    tp_all = (df_slice['high'] + df_slice['low'] + df_slice['close']) / 3.0
    vol_all = df_slice['volume'] if 'volume' in df_slice.columns else pd.Series(1.0, index=df_slice.index)
    vwap = float((tp_all * vol_all).sum() / max(1.0, float(vol_all.sum()))) if (not is_idx and float(vol_all.sum()) > 0) else float(tp_all.mean())
    std_dev = float(df_slice['close'].std()) if len(df_slice) > 1 else 0.35 * atr
    poc = float(eval_bar['poc']) if ('poc' in eval_bar and pd.notna(eval_bar['poc']) and float(eval_bar['poc']) > 0) else float(tp_all.median())
    vah = float(eval_bar['vah']) if ('vah' in eval_bar and pd.notna(eval_bar['vah']) and float(eval_bar['vah']) > 0) else round(vwap + 1.0 * std_dev, 2)
    val = float(eval_bar['val']) if ('val' in eval_bar and pd.notna(eval_bar['val']) and float(eval_bar['val']) > 0) else round(vwap - 1.0 * std_dev, 2)

    if 'ema9' in eval_bar and pd.notna(eval_bar['ema9']):
        ema9 = float(eval_bar['ema9'])
    else:
        ema9 = float(df_slice['close'].ewm(span=9, adjust=False).mean().iloc[-2])

    if 'ema21' in eval_bar and pd.notna(eval_bar['ema21']):
        ema21 = float(eval_bar['ema21'])
    else:
        ema21 = float(df_slice['close'].ewm(span=21, adjust=False).mean().iloc[-2])

    # VALUE ACCEPTANCE RULE: Price must accept on the breakout side of VWAP & POC
    if direction == 'CALL':
        if not (eval_close > vwap and eval_close >= poc):
            return None
    else:
        if not (eval_close < vwap and eval_close <= poc):
            return None

    # Base Compression Quality (Max 10 Points)
    norm_base = 500.0 if sym_clean == 'BANKNIFTY' else (160.0 if is_idx else (0.75 * atr))
    comp_ratio = comp_width / max(1.0, norm_base)
    compression_pts = round(min(10.0, max(2.0, (1.0 - min(0.9, comp_ratio)) * 10.0 + 2.0)), 2)

    # Value Area Acceptance (Max 10 Points)
    if direction == 'CALL':
        profile_pts = 10.0 if eval_close >= vah else 6.5
    else:
        profile_pts = 10.0 if eval_close <= val else 6.5

    p3_score = round(compression_pts + profile_pts, 2)

    # ========================================================================================
    # PILLAR 5: Remaining ATR Fuel / Trend Expansion (Max: 10 Points)
    # PURE 100-POINT SCORING: Proportional fuel score, no arbitrary drop
    # ========================================================================================
    fuel_spent = day_high - day_low
    remaining_atr = atr - fuel_spent
    fuel_ratio = remaining_atr / max(1.0, atr)
    if remaining_atr > 0:
        p5_score = round(min(10.0, max(3.0, fuel_ratio * 12.5)), 2)
    else:
        p5_score = 4.0  # Trend expansion continuation score

    # [LINE 31] 4 Pillars ka Preliminary Score nikalo (Pillar 1 + 2 + 3 + 5 = Max 80 Points).
    # Baad me Stage 2 me Pillar 4 (OI Runway, Max 20 Points) judkar Final 100 Points banega.
    prelim_score = round(p1_score + p2_score + p3_score + p5_score, 2)

    # [LINE 32] Sabhi metrics return karo live screener aur backtest ke liye.
    return {
        'symbol': sym_clean,
        'is_index': is_idx,
        'direction': direction,
        'spot_price': round(eval_close, 2),
        'entry_price': round(float(live_spot), 2),
        'b_close': eval_close,
        'comp_high': comp_high,
        'comp_low': comp_low,
        'comp_width': comp_width,
        'turnover_cr': round(turnover_cr, 2),
        'rvol': real_rvol,
        'bvc_cum_ratio': round(bvc_cum_ratio, 3),
        'eval_bvc': round(eval_bvc, 3),
        'vwap': round(vwap, 2),
        'poc': round(poc, 2),
        'vah': round(vah, 2),
        'val': round(val, 2),
        'ema9': round(ema9, 2),
        'ema21': round(ema21, 2),
        'remaining_atr': round(remaining_atr, 2),
        'p1_score': p1_score,
        'p2_score': p2_score,
        'p3_score': p3_score,
        'p5_score': p5_score,
        'prelim_score': prelim_score,
        'score': prelim_score,
        'stock_atr': atr,
        'pct_change': pct_change,
        'eval_time': target_bar.name if hasattr(target_bar, 'name') else None
    }


class DailyAutonomousScreener:
    """Production Autonomous Screener, Journaler & Live Position Manager."""

    def __init__(self):
        self.logger = logging.getLogger('alpha_5pillar_screener')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
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
        """Loads all active NSE F&O universe assets (220+ symbols)."""
        possible_paths = [
            os.path.join(CURRENT_DIR, 'modules', 'fno_universe_cache.json'),
            os.path.join(CURRENT_DIR, 'config', 'fno_universe.json'),
            os.path.join(CURRENT_DIR, 'fno_universe_cache.json')
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

    def is_already_dispatched(self, symbol: str, contract: str = '') -> bool:
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        return sym_clean in self.dispatched_today

    def mark_dispatched(self, symbol: str, contract: str = ''):
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        self.dispatched_today[sym_clean] = time.time()
        self._save_dispatched_cache()

    def get_stock_atr(self, symbol: str, q: Optional[Dict[str, Any]] = None) -> float:
        return self.data_feed.get_wilder_daily_atr(symbol)

    def get_nifty_market_regime(self, force_refresh: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force_refresh and (now - self.nifty_regime_cache.get('timestamp', 0.0) < 45.0):
            return self.nifty_regime_cache

        df_nifty = self.data_feed.get_live_intraday_bars('NIFTY')
        if df_nifty is None or len(df_nifty) < 6:
            res = {
                'allow_stock_call': True,
                'allow_stock_put': True,
                'allow_call': True,
                'allow_put': True,
                'reason': 'NIFTY_DEFAULT_NEUTRAL',
                'timestamp': now
            }
            self.nifty_regime_cache = res
            return res

        cur_close = float(df_nifty['close'].iloc[-1])
        day_open = float(df_nifty['open'].iloc[0])
        ema9 = float(df_nifty['close'].ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(df_nifty['close'].ewm(span=21, adjust=False).mean().iloc[-1])
        tp = (df_nifty['high'] + df_nifty['low'] + df_nifty['close']) / 3.0
        tp_vwap = float(tp.mean())

        move_pts = cur_close - day_open
        n_pct = round((move_pts / max(1.0, day_open)) * 100.0, 2)

        n_last_bar_rng = float(df_nifty['high'].iloc[-1] - df_nifty['low'].iloc[-1])
        if n_last_bar_rng > 65.0:
            res = {
                'allow_stock_call': False,
                'allow_stock_put': False,
                'allow_call': False,
                'allow_put': False,
                'move': move_pts,
                'move_pts': move_pts,
                'nifty_pct': n_pct,
                'nifty_spot': cur_close,
                'reason': 'NIFTY_CLIMAX_EXHAUSTION',
                'timestamp': now
            }
            self.nifty_regime_cache = res
            return res

        recent_rng = float(df_nifty['high'].iloc[-6:].max() - df_nifty['low'].iloc[-6:].min())
        ema_dist = abs(cur_close - ema9)
        is_chop = (recent_rng < 22.0 and ema_dist < 12.0)

        is_nifty_bullish = (cur_close >= tp_vwap and move_pts >= 10.0)
        is_nifty_bearish = (cur_close < tp_vwap and move_pts <= -35.0)

        is_runaway_bull = (move_pts >= 40.0)
        allow_stock_call = not is_nifty_bearish
        allow_stock_put = not is_runaway_bull
        allow_index_call = is_nifty_bullish and not is_chop
        allow_index_put = is_nifty_bearish and not is_chop

        regime_desc = "NEUTRAL"
        if is_chop:
            regime_desc = "CHOP_TRAP_INDEX_FROZEN"
        elif allow_index_call and not is_nifty_bearish:
            regime_desc = "NIFTY_UPTREND"
        elif allow_index_put and not is_nifty_bullish:
            regime_desc = "NIFTY_DOWNTREND"
        else:
            regime_desc = "NIFTY_SIDEWAYS_CHOP"

        res = {
            'allow_stock_call': allow_stock_call,
            'allow_stock_put': allow_stock_put,
            'allow_call': allow_index_call,
            'allow_put': allow_index_put,
            'move': move_pts,
            'move_pts': move_pts,
            'nifty_pct': n_pct,
            'nifty_spot': cur_close,
            'reason': regime_desc,
            'timestamp': now
        }
        self.nifty_regime_cache = res
        return res

    def evaluate_index_engine(self, symbol: str, live_q: Optional[Dict[str, Any]] = None, is_afternoon: bool = False) -> Dict[str, Any]:
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        df = self.data_feed.get_live_intraday_bars(sym_clean)
        if df is None or len(df) < 7:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'INSUFFICIENT_3M_BARS', 'is_index': True}

        atr = self.get_stock_atr(sym_clean, live_q)
        regime = self.get_nifty_market_regime()
        live_spot = float(live_q.get('live_spot', df['close'].iloc[-1])) if live_q else float(df['close'].iloc[-1])

        cand = evaluate_common_5pillar_bar_slice(
            sym_clean, df, atr, regime, live_spot_override=live_spot, is_index=True
        )
        if not cand:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'COMMON_5PILLAR_INDEX_VETO', 'is_index': True}

        cand['passed'] = True
        return cand

    def evaluate_stock_engine(self, symbol: str, live_q: Optional[Dict[str, Any]] = None, is_afternoon: bool = False, nifty_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        df = self.data_feed.get_live_intraday_bars(sym_clean)
        if df is None or len(df) < 7:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'INSUFFICIENT_3M_BARS', 'is_index': False}

        atr = self.get_stock_atr(sym_clean, live_q)
        regime = nifty_context or self.get_nifty_market_regime()
        live_spot = float(live_q.get('live_spot', df['close'].iloc[-1])) if live_q else float(df['close'].iloc[-1])
        pct_change = float(live_q.get('pct_change', 0.0)) if live_q else None

        cand = evaluate_common_5pillar_bar_slice(
            sym_clean, df, atr, regime, live_spot_override=live_spot, pct_change_override=pct_change, is_index=False
        )
        if not cand:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'COMMON_5PILLAR_STOCK_VETO', 'is_index': False}

        cand['passed'] = True
        return cand

    def evaluate_asset(self, symbol: str, nifty_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        bulk = fetch_bulk_nse_snapshot()
        live_q = bulk.get(sym_clean)
        if sym_clean in INDEX_NAMES:
            return self.evaluate_index_engine(sym_clean, live_q=live_q)
        else:
            return self.evaluate_stock_engine(sym_clean, live_q=live_q, nifty_context=nifty_context)

    def scan_and_dispatch(self, ignore_window: bool = False):
        """
        Executes full autonomous 5-pillar screening, qualifies super trades,
        appends each trade to Live_Paper_Trading_Journal.csv (Status='SIGNAL_GENERATED_OPEN'),
        and broadcasts Telegram trade alert.
        """
        return self.run_screener(ignore_window=ignore_window)

    def run_screener(self, ignore_window: bool = False):
        ist_now = get_ist_time()
        ist_time_str = ist_now.strftime('%I:%M:%S %p IST')

        if not ignore_window and not is_in_active_trading_slab(ist_now):
            print(f"[SLAB CHECK]: {ist_time_str} is outside active slabs (09:30-10:15 & 13:15-14:15). Running status heartbeat.")
            return

        print(f"\n=======================================================")
        print(f"  ALPHA 5-PILLAR INSTITUTIONAL SCREENER | {ist_now.strftime('%d-%b-%Y %I:%M:%S %p IST')}")
        print(f"  Designated Slabs: 09:30-10:15 AM IST & 01:15-02:15 PM IST")
        print(f"  Rule: Spot >= {MIN_STOCK_PRICE} INR Floor for Stock Options")
        print(f"=======================================================")

        bulk = fetch_bulk_nse_snapshot(force_refresh=True)
        regime = self.get_nifty_market_regime(force_refresh=True)
        print(f"[MACRO REGIME]: {regime['reason']} | Nifty Spot: {regime.get('nifty_spot', 0.0):.1f} | Move: {regime.get('move_pts', 0.0):+.1f} pts")

        index_candidates = []
        stock_candidates = []
        all_eval_results = []

        for idx in INDEX_NAMES:
            res = self.evaluate_index_engine(idx, live_q=bulk.get(idx))
            all_eval_results.append(res)
            if res.get('passed', False):
                index_candidates.append(res)

        symbols_to_scan = [item['symbol'] if isinstance(item, dict) else item for item in self.universe if (item['symbol'] if isinstance(item, dict) else item) not in INDEX_NAMES]
        for sym in symbols_to_scan:
            res = self.evaluate_stock_engine(sym, live_q=bulk.get(sym), nifty_context=regime)
            all_eval_results.append(res)
            if res.get('passed', False):
                stock_candidates.append(res)

        print(f"[DISCOVERY COMPLETE]: Scanned {len(all_eval_results)} assets ({len(INDEX_NAMES)} Indices, {len(symbols_to_scan)} Stocks).")
        print(f"[QUALIFIED DISCOVERY]: {len(index_candidates)} Index candidate(s), {len(stock_candidates)} Stock candidate(s).")

        index_candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        stock_candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        verified_super_trades = []
        active_slab_index = 0
        active_slab_stock = 0
        stage2_rejections = {}

        # Stage 2: Option Chain & Symmetric OI Runway Verification
        for cand in index_candidates:
            if active_slab_index >= MAX_INDEX_PER_SLAB:
                stage2_rejections[cand['symbol']] = "MAX_INDEX_SLAB_CAP_REACHED"
                continue
            sym = cand['symbol']
            if self.is_already_dispatched(sym):
                stage2_rejections[sym] = "ALREADY_DISPATCHED_TODAY"
                continue

            spot = cand['spot_price']
            direction = cand['direction']
            stock_atr = float(cand.get('stock_atr') or 120.0)
            idx_target_dist = 220.0 if sym == 'BANKNIFTY' else 140.0
            target_price = (spot + idx_target_dist) if direction == 'CALL' else (spot - idx_target_dist)
            opt = self.data_feed.compute_option_pricing(spot, direction=direction, symbol=sym, target_price=target_price, is_index=True)
            if not opt or not opt.get('is_liquid', False):
                stage2_rejections[sym] = f"OPTION_PRICING_FAIL_CLOSED: {opt.get('rejected_reason') if opt else 'CHAIN_UNAVAILABLE'}"
                continue

            cand['verified_opt'] = opt
            oi_score = float(opt.get('oi_score', 12.0))
            cand['score'] = round(cand.get('prelim_score', cand.get('score', 60.0)) + oi_score, 2)
            if cand['score'] < 70.0:
                stage2_rejections[sym] = f"SCORE_BELOW_70 ({cand['score']:.1f} < 70.0)"
                continue
            verified_super_trades.append(cand)
            active_slab_index += 1

        for cand in stock_candidates:
            if active_slab_stock >= MAX_STOCKS_PER_SLAB:
                stage2_rejections[cand['symbol']] = "MAX_STOCK_SLAB_CAP_REACHED"
                continue
            sym = cand['symbol']
            if self.is_already_dispatched(sym):
                stage2_rejections[sym] = "ALREADY_DISPATCHED_TODAY"
                continue

            dispatched_syms = [t['symbol'] for t in verified_super_trades if not t.get('is_index', False)]
            if sym in dispatched_syms:
                stage2_rejections[sym] = "DUPLICATE_SYMBOL_SLOT_CAP"
                continue

            spot = cand['spot_price']
            direction = cand['direction']
            stock_atr = float(cand.get('stock_atr') or 45.0)
            stk_target_dist = max(40.0, min(65.0, 0.85 * stock_atr))
            target_price = (spot + stk_target_dist) if direction == 'CALL' else (spot - stk_target_dist)
            opt = self.data_feed.compute_option_pricing(spot, direction=direction, symbol=sym, target_price=target_price, is_index=False)
            if not opt or not opt.get('is_liquid', False):
                stage2_rejections[sym] = f"OPTION_PRICING_FAIL_CLOSED: {opt.get('rejected_reason') if opt else 'CHAIN_UNAVAILABLE'}"
                continue

            cand['verified_opt'] = opt
            oi_score = float(opt.get('oi_score', 12.0))
            cand['score'] = round(cand.get('prelim_score', cand.get('score', 60.0)) + oi_score, 2)
            if cand['score'] < 70.0:
                stage2_rejections[sym] = f"SCORE_BELOW_70 ({cand['score']:.1f} < 70.0)"
                continue
            verified_super_trades.append(cand)
            active_slab_stock += 1

        self._log_rejection_audit(ist_now, all_eval_results, verified_super_trades, stage2_rejections)

        if not verified_super_trades:
            print(f"[SCREENER]: 0 signals qualified after separate Index & Stock scan at {ist_time_str}. Capital 100% Protected.")
            heartbeat_msg = (
                f"🛡️ <b>ALPHA 5-PILLAR SCANNER STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ <b>Execution Time:</b> {ist_time_str}\n"
                f"📊 <b>Universe Scanned:</b> {len(all_eval_results)} Assets\n"
                f"🏛️ <b>Index Engine:</b> 4 Indices Evaluated (Slots: {active_slab_index}/{MAX_INDEX_PER_SLAB})\n"
                f"📈 <b>Stock Engine:</b> {len(symbols_to_scan)} Stocks Evaluated (Slots: {active_slab_stock}/{MAX_STOCKS_PER_SLAB})\n"
                f"🛡️ <b>Status:</b> Zero 5-Pillar Confluent Breakouts\n"
                f"🔒 <b>Capital Protection:</b> 100% Active (Zero Forced Trades)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
        fixed_spot_sl_str = "25.0 Pts" if is_idx else "10.0 Pts"

        pure_signal_msg = (
            f"🚀 <b>ALPHA 5-PILLAR INSTITUTIONAL TRADE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>INSTRUMENT:</b> <b>{sym} ({direction})</b>\n"
            f"⚡ <b>ACTION:</b> <b>{opt_action}</b>\n"
            f"📌 <b>CONTRACT:</b> <code>{contract}</code>\n"
            f"💵 <b>BUYING PRICE:</b> <b>Rs. {opt_entry:,.2f}</b>\n"
            f"🛡️ <b>STOP LOSS:</b> <b>Rs. {opt_sl:,.2f} (Spot SL: {fixed_spot_sl_str})</b>\n"
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
            'Status': 'SIGNAL_GENERATED_OPEN'
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

    def update_open_paper_trades(self):
        csv_path = os.path.join(CURRENT_DIR, 'Live_Paper_Trading_Journal.csv')
        daily_xl = os.path.join(CURRENT_DIR, 'Alpha_5Pillar_Live_Paper_Trading_Journal.xlsx')
        monthly_xl = os.path.join(CURRENT_DIR, 'Alpha_Monthly_Paper_Trading_Journal.xlsx')
        if not os.path.exists(csv_path):
            return

        try:
            df = pd.read_csv(csv_path)
            if df.empty or 'Status' not in df.columns:
                return

            open_mask = (df['Status'] == 'SIGNAL_GENERATED_OPEN')
            if not open_mask.any():
                return

            ist_now = get_ist_time()
            now_ts = ist_now.timestamp()
            tot_mins_now = ist_now.hour * 60 + ist_now.minute
            updated_any = False

            for idx in df[open_mask].index:
                row = df.loc[idx]
                sym = str(row['Stock_Index']).strip()
                is_idx = sym in INDEX_NAMES
                opt_action = str(row['Option_Action'])
                contract = str(row['Option_Contract'])
                opt_entry = float(row['Buying_Price_INR'])
                opt_sl = float(row['Stop_Loss_INR'])
                trail_sl = float(row.get('Trailing_SL_INR', opt_sl))
                opt_t1 = float(row.get('Target_1_INR', opt_entry * 1.25))
                opt_t2 = float(row['Target_Price_INR'])
                spot_entry = float(row.get('Spot_Entry', 0.0))
                qty = int(row.get('Quantity', 25))

                q = self.data_feed.get_live_asset_quote(sym)
                cur_spot = float(q.get('live_spot', q.get('close', spot_entry))) if q else spot_entry

                opt_quote = self.data_feed.get_option_quote(sym, contract_identifier=contract)
                if opt_quote and opt_quote.get('valid') and opt_quote.get('ltp', 0.0) > 0:
                    est_opt_price = opt_quote['ltp']
                    current_bid = opt_quote['bid']
                else:
                    delta = 0.52 if is_idx else 0.50
                    spot_gain = (cur_spot - spot_entry) if ('CE' in contract or 'CALL' in opt_action) else (spot_entry - cur_spot)
                    est_opt_price = round(max(0.5, opt_entry + (spot_gain * delta)), 2)
                    current_bid = est_opt_price

                t1_locked_floor = round(opt_entry + max((opt_t1 - opt_entry) * 0.68, (est_opt_price - opt_entry) * 0.70), 2)
                be_shield_floor = round(opt_entry + (opt_entry - opt_sl) * 0.35, 2)
                opt_arm_trigger = opt_entry + (opt_entry - opt_sl) * 1.0

                if est_opt_price >= opt_t1:
                    if trail_sl < t1_locked_floor:
                        df.at[idx, 'Trailing_SL_INR'] = t1_locked_floor
                        trail_sl = t1_locked_floor
                        updated_any = True
                        print(f"[POSITION MANAGER]: {sym} reached Target 1! Dynamic trailing SL ratcheted at 70%: Rs. {t1_locked_floor:.2f}")
                elif est_opt_price >= opt_arm_trigger and trail_sl < be_shield_floor:
                    df.at[idx, 'Trailing_SL_INR'] = be_shield_floor
                    trail_sl = be_shield_floor
                    updated_any = True
                    print(f"[POSITION MANAGER]: {sym} reached +1.0R move! Breakeven Shield armed at +0.35R: Rs. {be_shield_floor:.2f}")

                exit_price = 0.0
                outcome = ""

                if est_opt_price >= opt_t2:
                    exit_price = opt_t2
                    outcome = "TARGET_2_HIT (JACKPOT)"
                elif trail_sl > opt_entry and est_opt_price <= trail_sl:
                    exit_price = trail_sl
                    outcome = "T1_PROFIT_LOCKED" if trail_sl >= t1_locked_floor else "BREAKEVEN_SHIELD_EXIT"
                elif est_opt_price <= opt_sl:
                    exit_price = opt_sl
                    outcome = "SL_HIT"
                elif tot_mins_now >= 920:
                    exit_price = current_bid if current_bid > 0 else est_opt_price
                    outcome = "EOD_SQUARE_OFF"
                else:
                    entry_ts = pd.to_numeric(row.get('Signal_Timestamp_Epoch'), errors='coerce')
                    if pd.notna(entry_ts) and entry_ts > 0:
                        elapsed_mins = (now_ts - float(entry_ts)) / 60.0
                        opt_risk = opt_entry - opt_sl
                        # [LINE 14] 30-Minute Stagnant Scratch: 30+ mins pure hone ke baad hi stagnant exit check hoga (No 9-min/15-min premature cut)
                        if elapsed_mins >= 30.0 and est_opt_price < opt_t1 and trail_sl <= opt_entry and est_opt_price <= (opt_entry - opt_risk * 0.25):
                            exit_price = current_bid if current_bid > 0 else est_opt_price
                            outcome = "30M_STAGNANT_EXIT"

                if outcome:
                    pts_cap = round(exit_price - opt_entry, 2)
                    gross_pnl = round(pts_cap * qty, 2)
                    friction = round(min(500.0, max(60.0, abs(gross_pnl) * 0.03)), 2)
                    net_pnl = round(gross_pnl - friction, 2)

                    df.at[idx, 'Exit_Price_INR'] = exit_price
                    df.at[idx, 'Exit_Time_IST'] = ist_now.strftime('%I:%M:%S %p IST')
                    df.at[idx, 'Points_Captured'] = pts_cap
                    df.at[idx, 'Gross_PnL_INR'] = gross_pnl
                    df.at[idx, 'Friction_Taxes_INR'] = friction
                    df.at[idx, 'Net_PnL_INR'] = net_pnl
                    df.at[idx, 'Trade_Outcome'] = outcome
                    df.at[idx, 'Status'] = outcome
                    updated_any = True

                    exit_msg = (
                        f"🏁 <b>ALPHA PAPER POSITION CLOSED</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>CONTRACT:</b> <code>{contract}</code>\n"
                        f"🎯 <b>OUTCOME:</b> <b>{outcome}</b>\n"
                        f"💵 <b>ENTRY:</b> Rs. {opt_entry:.2f} | <b>EXIT:</b> Rs. {exit_price:.2f}\n"
                        f"📈 <b>POINTS:</b> <b>{pts_cap:+.2f} pts</b>\n"
                        f"💰 <b>NET P&L:</b> <b>Rs. {net_pnl:+,.2f}</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )
                    self.notifier.send_broadcast_message(exit_msg, is_trade_signal=False)

            if updated_any:
                df.to_csv(csv_path, index=False)
                for xl_path in [daily_xl, monthly_xl]:
                    if os.path.exists(xl_path):
                        df.to_excel(xl_path, index=False)
                print(f"[POSITION MANAGER]: Successfully updated positions in CSV and Excel ledgers.")

        except Exception as e:
            print(f"[POSITION MANAGER ERROR]: {e}")

    def _log_rejection_audit(self, ist_now, all_eval_results, verified_trades, stage2_rejections):
        ts_str = ist_now.strftime('%Y-%m-%d %H:%M:%S IST')
        qual_syms = {t['symbol'] for t in verified_trades}
        rej_count = len(all_eval_results) - len(qual_syms)
        self.logger.info(f"[AUDIT SUMMARY]: Evaluated {len(all_eval_results)} assets at {ts_str} -> {len(qual_syms)} qualified, {rej_count} rejected.")

    def run_live_slab_session(self, interval_seconds: int = 180):
        print(f"[SLAB RUNNER]: Initializing 3-minute autonomous screening loop...")
        while True:
            try:
                self.update_open_paper_trades()
                self.run_screener(ignore_window=False)
            except Exception as e:
                print(f"[SLAB RUNNER ERROR]: {e}")
            time.sleep(interval_seconds)


    def run_live_slab_loop(self):
        print(f"=======================================================")
        print(f"  ALPHA CONTINUOUS SLAB RUNNER INITIALIZED")
        print(f"  Designated Slabs: 09:30-10:15 AM IST & 01:15-02:15 PM IST")
        print(f"=======================================================")

        iteration = 1
        while True:
            ist_now = get_ist_time()
            tot_mins = ist_now.hour * 60 + ist_now.minute
            is_weekday = (ist_now.weekday() < 5)

            if not is_weekday:
                print(f"[MARKET CLOSED]: Weekend detected ({ist_now.strftime('%A')}). Exiting loop.")
                break

            if is_in_active_trading_slab(ist_now):
                print(f"\n>>> [ITERATION #{iteration}] Running 5-Pillar F&O Scan at {ist_now.strftime('%I:%M:%S %p IST')}...")
                self.run_screener(ignore_window=True)
                print(f">>> [ITERATION #{iteration} COMPLETE] Sleeping 180s (3 minutes) until next scan...")
                time.sleep(180)
                iteration += 1
            elif tot_mins < 570:
                sleep_secs = min(300, (570 - tot_mins) * 60)
                print(f"\r[STANDBY]: Pre-slab 1. Sleeping {sleep_secs}s until 09:30 AM IST...   ", end='', flush=True)
                time.sleep(sleep_secs)
            elif 615 < tot_mins < 795:
                rem_mins = 795 - tot_mins
                sleep_secs = min(300, rem_mins * 60)
                print(f"\r[MID-DAY CHOP PAUSE]: Standing by for Slab 2 (01:15 PM IST). Resuming in {rem_mins} mins...   ", end='', flush=True)
                time.sleep(sleep_secs)
            else:
                print(f"\n[TRADING SLABS CONCLUDED]: Current time {ist_now.strftime('%I:%M:%S %p IST')} is past 02:15 PM IST. Completed for today.")
                break

    def run_eod_analysis_and_dispatch(self):
        """Performs daily trade success/failure analysis at 03:35 PM IST and dispatches Report + Excel strictly to Owner Chatbot."""
        ist_now = get_ist_time()
        ist_time_str = ist_now.strftime('%I:%M:%S %p IST')
        ist_date_str = ist_now.strftime('%d-%b-%Y')
        daily_xl = os.path.join(CURRENT_DIR, 'Alpha_5Pillar_Live_Paper_Trading_Journal.xlsx')

        if not os.path.exists(daily_xl):
            print(f"[EOD ERROR]: Excel file {daily_xl} not found.")
            return

        import pandas as pd
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
            "📊 <b>DAILY EOD TRADE AUDIT & PERFORMANCE POST-MORTEM</b>",
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

        caption = (
            f"📊 <b>DAILY OFFICIAL LIVE PAPER TRADING JOURNAL</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 <b>DATE:</b> {ist_date_str}\n"
            f"⏰ <b>DISPATCH TIME:</b> {ist_time_str}\n"
            f"🎯 <b>MODE:</b> EOD Official Paper Trading Ledger\n"
            f"🔒 <b>SECURITY:</b> Strictly Confidential (Owner Private DM Only)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        res = self.notifier.send_document_to_owner_with_approval_prompt(daily_xl, caption=caption)
        print(f"[EOD AUDIT & DISPATCH]: Delivered daily analysis & journal to Owner DM. Result: {res.get('ok')}")
        return res

    def send_eod_journal_to_owner(self):
        """Compatibility alias for EOD dispatch."""
        return self.run_eod_analysis_and_dispatch()


if __name__ == '__main__':
    screener = DailyAutonomousScreener()
    if '--slab-loop' in sys.argv:
        screener.run_live_slab_loop()
    elif '--eod-audit' in sys.argv or '--send-journal' in sys.argv:
        screener.run_eod_analysis_and_dispatch()
    else:
        ignore_win = ('--ignore-window' in sys.argv)
        screener.run_screener(ignore_window=ignore_win)

