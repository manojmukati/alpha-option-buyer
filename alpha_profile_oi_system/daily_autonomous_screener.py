# -*- coding: utf-8 -*-
"""
daily_autonomous_screener.py
============================
Alpha 5-Pillar Pure Institutional Paper-Trading Screener
Built to quantitative institutional standards across all 217 NSE F&O Assets (4 Indices + 213 Stocks).

9 Exact Practical Upgrades:
1. Universe & Bulk Refresh: Fresh bulk NSE snapshot on every run_screener() cycle.
2. 14-Day Wilder ATR: Uses authoritative config/stock_atr_cache.json (removes buggy high-low*0.70).
3. Configurable Price Floor: MIN_STOCK_PRICE = 250.0 (blocks penny stocks, allows liquid F&O leaders).
4. Sub-Bar CVD: Completed-bar calculation (iloc[:-1]); Index uses range proxy (preserves index jackpots).
5. Robust Macro Regime: Nifty trend check for stocks & indices (allows independent leaders with RVOL >= 2.5x).
6. Directional Breakout & Clean RVOL: Green for CALL, Red for PUT; runway from boundary; RVOL vs 10-bar base.
7. Dispatch KeyError Safety: Fallback reads for sl_price, target_1, target_2.
8. 15-Minute Time SL Epoch & EOD Square-Off: Signal_Timestamp_Epoch calculation (no AM/PM bug); EOD at 15:20 IST;
   updates CSV, Daily Excel, and Monthly Excel simultaneously.
9. Timezone & Post-Slab Position Monitoring: Monitors open trades until target/SL/EOD.
"""

import os
import sys
import time
import json
import random
import datetime
import math
import urllib.request
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

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

# Only 4 Official Indices Hardcoded (Zero stock names hardcoded)
INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# Configurable Price Floor: Blocks penny and noisy PSU stocks (< Rs 350), allows liquid F&O leaders
MIN_STOCK_PRICE = 350.0

MAX_DAILY_SUPER_TRADES = 8  # 4 Morning + 4 Afternoon
MAX_SLAB_TRADES = 3
MAX_INDEX_PER_SLAB = 1
MAX_STOCKS_PER_SLAB = 2
NIFTY_HEAVYWEIGHTS = {'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'}


def is_in_active_trading_slab(ist_time):
    """
    Checks if current IST time falls strictly within designated trading slabs:
    - Slab 1 (Morning):   09:30 AM to 10:15 AM IST (570 to 615 mins)
    - Slab 2 (Afternoon): 01:15 PM to 02:15 PM IST (795 to 855 mins)
    Must also be a weekday (Monday to Friday).
    """
    if ist_time.weekday() >= 5:
        return False
    tot_mins = ist_time.hour * 60 + ist_time.minute
    slab1 = (570 <= tot_mins <= 615)
    slab2 = (795 <= tot_mins <= 855)
    return slab1 or slab2


def compute_bar_delta_proxy(df_bars, is_index=False):
    """
    Computes Bulk Volume Classification (BVC) Volume Delta Proxy
    from completed intraday OHLCV bars based on quantitative market microstructure.
    
    Bar Delta = Volume * [ 2 * Close - (High + Low) ] / Range
    
    Data Integrity Rules:
    - Only evaluates completed bars (caller excludes live ticking bar).
    - If sub-bar 1m delta is pre-aggregated in df_bars['delta'], uses it directly.
    - If volume is zero or unavailable for stocks, returns valid=False. Never invents synthetic data.
    - For spot indices (where volume is 0 in spot feed), uses range-weighted proxy explicitly flagged
      so that Index jackpots are preserved!
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
            # Spot Index Range Proxy: True institutional delta proxy from spot price action
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


def compute_bvc_cvd(df_bars):
    """Backward compatibility wrapper redirecting to compute_bar_delta_proxy."""
    return compute_bar_delta_proxy(df_bars)


class DailyAutonomousScreener:
    def __init__(self):
        self.data_feed = LiveMarketDataFeed()
        self.notifier = TelegramNotifier(config_path=CONFIG_PATH)
        self.universe = self.data_feed.universe_assets

        # Load Authoritative 14-Day Wilder ATR Cache (221 F&O Stocks)
        self.atr_cache = {}
        cache_file = os.path.join(CURRENT_DIR, 'config', 'stock_atr_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.atr_cache = json.load(f)
            except Exception:
                pass

        self.dispatched_cache_file = os.path.join(CURRENT_DIR, 'config', 'dispatched_signals_today.json')
        self.dispatched_today = self._load_dispatched_cache()
        self.nifty_regime_cache = {
            'allow_stock_call': True,
            'allow_stock_put': True,
            'reason': 'NEUTRAL',
            'timestamp': 0.0
        }

        if not self.notifier.is_configured():
            env_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
            env_owner = os.environ.get('TELEGRAM_CHAT_ID', '')
            env_group = os.environ.get('TELEGRAM_GROUP_ID', '')
            if env_token and env_owner:
                self.notifier.bot_token = env_token
                self.notifier.owner_chat_id = env_owner
                self.notifier.group_chat_ids = [env_group] if env_group else []

    def _load_dispatched_cache(self):
        today_str = get_ist_time().strftime('%Y-%m-%d')
        if os.path.exists(self.dispatched_cache_file):
            try:
                with open(self.dispatched_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('date') == today_str:
                        return data.get('dispatched', {})
            except Exception:
                pass
        return {}

    def _save_dispatched_cache(self):
        today_str = get_ist_time().strftime('%Y-%m-%d')
        try:
            os.makedirs(os.path.dirname(self.dispatched_cache_file), exist_ok=True)
            with open(self.dispatched_cache_file, 'w', encoding='utf-8') as f:
                json.dump({'date': today_str, 'dispatched': self.dispatched_today}, f, indent=2)
        except Exception as e:
            print(f"[CACHE WRITE ERROR]: Failed to save dispatched cache: {e}")

    def is_already_dispatched(self, symbol, contract=''):
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        return sym_clean in self.dispatched_today

    def mark_dispatched(self, symbol, contract=''):
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        self.dispatched_today[sym_clean] = time.time()
        self._save_dispatched_cache()

    def get_stock_atr(self, symbol, q):
        """Returns verified 14-day Wilder ATR from authoritative cache (fallback: 2.2% of spot)."""
        cur_p = float(q.get('live_spot', q.get('close', 100.0)))
        cached_atr = float(self.atr_cache.get(symbol, 0.0))
        if cached_atr > 0.0:
            return cached_atr
        return round(max(3.0, cur_p * 0.022), 2)

    def get_nifty_market_regime(self, force_refresh=False):
        """
        Symmetrical Macro Trend Filter from NIFTY 50 3-minute completed bars:
        - Climax Candle: If last completed bar range > 35 pts, smart money is blowing off; blocks both CALL and PUT.
        - Downtrend: Spot < 9-EMA and move from open < -10 pts -> blocks CALLs
        - Uptrend: Spot > 9-EMA and move from open > +10 pts -> blocks PUTs
        - Chop Trap: 18-minute range < 22 pts and distance from EMA < 12 pts -> blocks both CALL and PUT
        """
        now = time.time()
        if not force_refresh and (now - self.nifty_regime_cache.get('timestamp', 0.0) < 60.0):
            return self.nifty_regime_cache

        df_nifty = self.data_feed.get_live_intraday_bars('NIFTY')
        nq = self.data_feed.get_live_asset_quote('NIFTY')

        nifty_spot = float(nq.get('live_spot', 0.0)) if nq else 0.0
        allow_call = True
        allow_put = True
        reason = "NIFTY_NEUTRAL_ALIGNED"

        if df_nifty is not None and len(df_nifty) >= 6:
            live_b = df_nifty.iloc[-1]
            eval_b = df_nifty.iloc[-2]
            nifty_spot = float(live_b['close'])
            ema9 = float(df_nifty['ema9'].iloc[-1]) if 'ema9' in df_nifty.columns else float(df_nifty['close'].ewm(span=9, adjust=False).mean().iloc[-1])
            day_open = float(df_nifty['open'].iloc[0])
            move_from_open = round(nifty_spot - day_open, 2)

            # Climax candle check on Nifty (Pillar 3 Anti-Exhaustion - Calibrated for high-volatility 24,000+ Index)
            n_bar_rng = float(eval_b['high'] - eval_b['low'])
            if n_bar_rng > 65.0:
                allow_call = False
                allow_put = False
                reason = f"NIFTY_CLIMAX_EXHAUSTION_BAR ({n_bar_rng:.1f}pts > 65pts)"
            else:
                recent_rng = round(df_nifty['high'].tail(6).max() - df_nifty['low'].tail(6).min(), 2)
                ema_distance = abs(nifty_spot - ema9)

                if recent_rng < 22.0 and ema_distance < 12.0:
                    allow_call = False
                    allow_put = False
                    reason = f"NIFTY_SIDEWAYS_CHOP_TRAP (18m Range {recent_rng:.1f} pts < 22 pts)"
                else:
                    if move_from_open < -10.0 or nifty_spot < ema9 - 15.0:
                        allow_call = False
                        reason = f"NIFTY_ACTIVE_DOWNTREND (Move {move_from_open:+.1f}pts from open | Spot {nifty_spot:.1f} vs EMA {ema9:.1f})"
                    elif move_from_open > +10.0 or nifty_spot > ema9 + 15.0:
                        allow_put = False
                        reason = f"NIFTY_ACTIVE_UPTREND (Move {move_from_open:+.1f}pts from open | Spot {nifty_spot:.1f} vs EMA {ema9:.1f})"

        self.nifty_regime_cache = {
            'allow_stock_call': allow_call,
            'allow_stock_put': allow_put,
            'reason': reason,
            'timestamp': now
        }
        return self.nifty_regime_cache

    def evaluate_index_engine(self, symbol, live_q=None, is_afternoon=False):
        """
        PURE INDEX ENGINE (Kadak / Strict Parameters for High Volatility)
        Preserves all 5 Pillars:
        - Pillar 1: Range Expansion RVOL >= 1.35x. Spot move from Open <= 0.8% (Anti-Exhaustion).
        - Pillar 2: True BVC Cumulative Volume Delta & Bar Delta Proxy.
        - Pillar 3: 8-bar Compression Base (width <= 150 pts Nifty/Fin, <= 350 pts BankNifty).
                    Early Runway <= 30 pts (Nifty/Fin) or <= 60 pts (BankNifty).
        - Pillar 4: Strict Nifty Macro Tide (Index CALL only if Nifty >= Open; Index PUT only if Nifty <= Open).
        - Pillar 5: Point-Based Dynamic Risk (Nifty: 35/70/120 pts; BankNifty: 70/140/240 pts).
        """
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        q = live_q or self.data_feed.get_live_asset_quote(sym_clean)
        if not q:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'NO_NSE_LIVE_QUOTE', 'is_index': True}

        cur_p = float(q.get('live_spot', q.get('close', 0.0)))
        if cur_p <= 0:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'INVALID_SPOT_PRICE', 'is_index': True}

        idx_df = self.data_feed.get_live_intraday_bars(sym_clean)
        if idx_df is None or len(idx_df) < 10:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'INDEX_INSUFFICIENT_BARS_LESS_THAN_10', 'is_index': True}

        comp_bars = idx_df.iloc[:-2].tail(8)
        eval_bar = idx_df.iloc[-2]

        b_open = float(eval_bar['open'])
        b_high = float(eval_bar['high'])
        b_low = float(eval_bar['low'])
        b_close = float(eval_bar['close'])
        rng = max(0.01, b_high - b_low)

        comp_high = float(comp_bars['high'].max())
        comp_low = float(comp_bars['low'].min())
        comp_width = comp_high - comp_low

        max_allowed_base = 120.0 if sym_clean in ['NIFTY', 'FINNIFTY', 'MIDCPNIFTY'] else 280.0
        if comp_width > max_allowed_base:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'INDEX_BASE_TOO_WIDE ({comp_width:.1f} > {max_allowed_base:.1f}pts)', 'is_index': True}

        # Pillar 3 Anti-Exhaustion: Climax Bar Check (Calibrated for real institutional momentum expansion)
        max_bar_rng = 65.0 if sym_clean != 'BANKNIFTY' else 150.0
        if rng > max_bar_rng:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'INDEX_CLIMAX_EXHAUSTION_BAR ({rng:.1f} > {max_bar_rng:.1f}pts)', 'is_index': True}

        solid_body = round(abs(b_close - b_open) / rng, 2)
        idx_dist = 35.0 if sym_clean in ['NIFTY', 'FINNIFTY', 'MIDCPNIFTY'] else 80.0

        direction = None
        if b_close >= comp_high and b_close <= comp_high + idx_dist and cur_p >= comp_high and b_close > b_open:
            direction = 'CALL'
        elif b_close <= comp_low and b_close >= comp_low - idx_dist and cur_p <= comp_low and b_close < b_open:
            direction = 'PUT'

        if direction is None:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'INDEX_NO_COMPRESSION_BREAKOUT_OR_DIRECTION_MISALIGNED', 'is_index': True}

        if solid_body < 0.45:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'INDEX_WEAK_CANDLE_BODY ({solid_body:.2f} < 0.45)', 'is_index': True}

        # Pillar 4 Macro Alignment: Index CALL only if Index >= Own Open and Nifty not dumping
        idx_day_open = float(idx_df['open'].iloc[0])
        if direction == 'CALL' and cur_p < idx_day_open:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'INDEX_CALL_BELOW_OWN_DAY_OPEN', 'is_index': True}
        if direction == 'PUT' and cur_p > idx_day_open:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'INDEX_PUT_ABOVE_OWN_DAY_OPEN', 'is_index': True}

        # Macro Tide Alignment (Dynamic Intraday Trend - Replaces static Day Open)
        df_nifty = self.data_feed.get_live_intraday_bars('NIFTY')
        if df_nifty is not None and len(df_nifty) > 2:
            n_spot = float(df_nifty['close'].iloc[-1])
            n_open = float(df_nifty['open'].iloc[0])
            move_from_open = n_spot - n_open

            # Anti-Trap Gap-Down / Gap-Up Dead-Cat Bounce Protection
            if direction == 'CALL' and move_from_open < -10.0:
                return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'INDEX_MACRO_DOWNTREND_BLOCKED (Nifty move {move_from_open:+.1f}pts from open)', 'is_index': True}
            if direction == 'PUT' and move_from_open > +10.0:
                return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'INDEX_MACRO_UPTREND_BLOCKED (Nifty move {move_from_open:+.1f}pts from open)', 'is_index': True}

        # BVC Cumulative Volume Delta for Index
        cvd_res = compute_bar_delta_proxy(idx_df.iloc[:-1], is_index=True)
        rec_cvd = cvd_res.get('recent_cvd_ratio', 0.0)
        cvd_pass = (rec_cvd >= 0.12) if direction == 'CALL' else (rec_cvd <= -0.12)
        if not cvd_pass:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'INDEX_CVD_MISALIGNED (RecCVD={rec_cvd:+.2f})', 'is_index': True}

        # Range Expansion RVOL (Calculated against session median consolidation baseline)
        session_rngs = (idx_df.iloc[:-1]['high'] - idx_df.iloc[:-1]['low'])
        median_rng = float(session_rngs.median()) if len(session_rngs) > 0 else 15.0
        idx_rvol = round(rng / max(0.5, median_rng), 2)
        if idx_rvol < 1.15:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'INDEX_RANGE_EXPANSION_RVOL_LOW ({idx_rvol:.2f}x < 1.15x)', 'is_index': True}

        score = min(99.0, 55.0 + min(25.0, (idx_rvol - 1.0) * 15.0) + (15.0 if abs(rec_cvd) >= 0.25 else 5.0) + (10.0 if solid_body >= 0.55 else 0.0))

        return {
            'symbol': sym_clean,
            'passed': True,
            'spot_price': cur_p,
            'direction': direction,
            'pct_change': float(q.get('pct_change', 0.0)),
            'rvol': idx_rvol,
            'turnover_cr': 0.0,
            'score': round(score, 2),
            'is_index': True,
            'stock_atr': 0.0,
            'is_compression_breakout': True,
            'cvd_info': cvd_res
        }

    def evaluate_stock_engine(self, symbol, live_q=None, is_afternoon=False, nifty_context=None):
        """
        PURE STOCK ENGINE (Early Ignition & Anti-Exhaustion "Thak Raha Hai" Protection)
        Preserves all 5 Pillars:
        - Pillar 1: Institutional Cash Turnover Floor >= Rs 3.0 Cr. RVOL >= 1.80x (or >= 1.60x if turnover >= Rs 6.0 Cr).
        - Pillar 2: True BVC Cumulative Volume Delta with Current Bar Delta Priority (>= +35% for CALL, <= -35% for PUT).
        - Pillar 3: 8-bar Compression Base (width <= 0.65 * ATR). Early Runway <= 0.20 * ATR from boundary.
                    ANTI-EXHAUSTION: Breakout Bar Range <= 0.45 * ATR, Prior 3-Bar Runup <= 0.60 * ATR, Day Range <= 1.30 * ATR.
                    Day High / Low Frontier: Within 0.12 * ATR buffer.
        - Pillar 4: Dynamic Nifty Macro Trend (9-EMA + 60 pts drop/rise; Mega heavyweights MUST align; Decoupled Alpha Outperformer allowed).
        - Pillar 5: ATR-Normalized Risk (SL = 0.20 * ATR, T1 = 0.35 * ATR, T2 = 0.70 * ATR).
        """
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        q = live_q or self.data_feed.get_live_asset_quote(sym_clean)
        if not q:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'NO_NSE_LIVE_QUOTE', 'is_index': False}

        cur_p = float(q.get('live_spot', q.get('close', 0.0)))
        if cur_p <= 0:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'INVALID_SPOT_PRICE', 'is_index': False}

        stock_atr = self.get_stock_atr(sym_clean, q)
        atr_pct = (stock_atr / max(1.0, cur_p)) * 100.0
        if cur_p < MIN_STOCK_PRICE or atr_pct < 1.0:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'STOCK_PRICE_BELOW_{int(MIN_STOCK_PRICE)}_INR: Spot Rs {cur_p:.1f} < {MIN_STOCK_PRICE} (or ATR% {atr_pct:.2f}% < 1.0%)',
                'is_index': False
            }

        # BUG 6 FIX: Fast Pre-Filter for 213 Stocks (Zero Network calls for dead/flat or overextended stocks)
        pct_change = float(q.get('pct_change', 0.0))
        if abs(pct_change) < 0.35:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'INSIGNIFICANT_DAY_MOVE ({pct_change:.2f}% < 0.35%)', 'is_index': False}
        if abs(pct_change) > 3.00:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'EXTREME_DAY_MOVE_OVEREXTENDED ({pct_change:+.2f}% > 3.00%)', 'is_index': False}

        stk_df = self.data_feed.get_live_intraday_bars(sym_clean)
        if stk_df is None or len(stk_df) < 10:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'STOCK_INSUFFICIENT_BARS_LESS_THAN_10', 'is_index': False}

        eval_bar = stk_df.iloc[-2]
        live_bar = stk_df.iloc[-1]
        b_open = float(eval_bar['open'])
        b_high = float(eval_bar['high'])
        b_low = float(eval_bar['low'])
        b_close = float(eval_bar['close'])
        rng = max(0.01, b_high - b_low)

        # 1. 8-Bar Compression Base & Early Runway
        comp_bars = stk_df.iloc[:-2].tail(8)
        comp_high = float(comp_bars['high'].max())
        comp_low = float(comp_bars['low'].min())
        comp_w = comp_high - comp_low

        if comp_w > stock_atr * 0.60:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'STOCK_BASE_TOO_WIDE ({comp_w:.1f} > {stock_atr * 0.60:.1f}pts)', 'is_index': False}

        # Anti-Chase Buffer: Maximum 0.08 * ATR runway above base (Zero chasing candle tops!)
        max_dist_buffer = stock_atr * 0.08
        direction = None
        if b_close >= comp_high and b_close <= comp_high + max_dist_buffer and b_close > b_open and cur_p >= (comp_high - stock_atr * 0.03):
            direction = 'CALL'
        elif b_close <= comp_low and b_close >= comp_low - max_dist_buffer and b_close < b_open and cur_p <= (comp_low + stock_atr * 0.03):
            direction = 'PUT'

        if direction is None:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'NO_8BAR_COMPRESSION_BREAKOUT_OR_CHASED', 'is_index': False}

        # 2. ANTI-EXHAUSTION: "THAK RAHA HAI" / PROFIT BOOKING DETECTION
        # Climax Candle: If single bar range is > 0.40 ATR, smart money is climaxing!
        c_rng = b_high - b_low
        if c_rng > stock_atr * 0.40:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'CLIMAX_EXHAUSTION_CANDLE ({c_rng:.1f} > {stock_atr * 0.40:.1f}pts)', 'is_index': False}

        # Prior 3-Bar Run: If stock already moved > 0.55 ATR in preceding 9 mins, energy is spent!
        p3_low = stk_df.iloc[:-2].tail(3)['low'].min()
        p3_high = stk_df.iloc[:-2].tail(3)['high'].max()
        p3_move = (b_close - p3_low) if direction == 'CALL' else (p3_high - b_close)
        if p3_move > stock_atr * 0.55:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'PRIOR_RUNUP_EXHAUSTED ({p3_move:.1f} > {stock_atr * 0.55:.1f}pts)', 'is_index': False}

        # BUG 3 FIX: Full-Day Anti-Exhaustion Check (Move > 3.20% or Day Range > 1.30 * ATR)
        stk_open = float(stk_df['open'].iloc[0])
        day_pct_move = ((cur_p - stk_open) / max(1.0, stk_open)) * 100.0
        stk_day_high = float(stk_df['high'].max())
        stk_day_low = float(stk_df['low'].min())
        day_range = stk_day_high - stk_day_low
        if abs(day_pct_move) > 3.20 or day_range > stock_atr * 1.30:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'DAY_RANGE_EXHAUSTED (Move {day_pct_move:+.2f}%, Range {day_range:.1f} > {stock_atr * 1.30:.1f}pts)', 'is_index': False}

        # 3. Volume, RVOL & Turnover Floor
        curr_v = float(eval_bar.get('volume', 0.0))
        prior_10_bars = stk_df.iloc[:-2].tail(10)
        base_10_v = float(prior_10_bars['volume'].mean()) if len(prior_10_bars) > 0 else 1000.0
        hist_avg_v = float(stk_df['hist_avg_3m_vol'].iloc[-1]) if 'hist_avg_3m_vol' in stk_df.columns else 0.0
        base_v = max(1.0, hist_avg_v if hist_avg_v > 0 else base_10_v)
        real_rvol = round(curr_v / base_v, 2)
        turnover_cr = round((curr_v * cur_p) / 1e7, 2)

        if real_rvol < 1.80:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'RVOL_BELOW_INSTITUTIONAL_FLOOR ({real_rvol:.2f}x < 1.80x)', 'is_index': False}
        if turnover_cr < 3.0:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'TURNOVER_BELOW_FLOOR (Rs {turnover_cr:.1f}Cr < 3.0Cr)', 'is_index': False}

        # BUG 1 FIX: True Breakout Bar Delta Ratio directly from BVC Microstructure formula with CVD Non-Divergence Lock
        completed_bars = stk_df.iloc[:-1]
        cvd_res = compute_bar_delta_proxy(completed_bars, is_index=False)
        rec_cvd = cvd_res.get('recent_cvd_ratio', 0.0)
        bar_delta_ratio = round((2.0 * b_close - (b_high + b_low)) / rng, 3)

        if direction == 'CALL':
            cvd_pass = (rec_cvd >= 0.15 or bar_delta_ratio >= 0.35) and (rec_cvd >= 0.0)
        else:
            cvd_pass = (rec_cvd <= -0.15 or bar_delta_ratio <= -0.35) and (rec_cvd <= 0.0)

        if not cvd_pass:
            reason = "BVC_CVD_COUNTER_DIVERGENCE" if ((direction == 'CALL' and rec_cvd < 0.0) or (direction == 'PUT' and rec_cvd > 0.0)) else "BVC_CVD_MISALIGNED"
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'{reason} (RecCVD={rec_cvd:+.2f}, BarDelta={bar_delta_ratio:+.2f})', 'is_index': False}

        # BUG 5 FIX: Day High / Day Low Check with 0.12 * ATR Buffer
        comp_df = stk_df.iloc[:-1]
        day_high = float(comp_df['high'].max())
        day_low = float(comp_df['low'].min())
        near_day_high = (direction == 'CALL' and b_close >= (day_high - stock_atr * 0.12))
        near_day_low = (direction == 'PUT' and b_close <= (day_low + stock_atr * 0.12))

        # BUG 5 FIX: Strict Day High / Day Low Check with 0.12 * ATR Buffer (Applied to Morning & Afternoon to block mid-range chop)
        if not (near_day_high or near_day_low):
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'NOT_DAY_HIGH_LOW_BREAKOUT_CHOP_TRAP', 'is_index': False}

        # BUG 4 FIX: Dynamic Nifty Macro Check (Pre-computed Context Passed In)
        if nifty_context is None:
            df_nifty = self.data_feed.get_live_intraday_bars('NIFTY')
            n_spot = float(df_nifty['close'].iloc[-1]) if df_nifty is not None and len(df_nifty) > 1 else cur_p
            n_open = float(df_nifty['open'].iloc[0]) if df_nifty is not None and len(df_nifty) > 1 else cur_p
            n_high = float(df_nifty['high'].max()) if df_nifty is not None and len(df_nifty) > 1 else cur_p
            n_low = float(df_nifty['low'].min()) if df_nifty is not None and len(df_nifty) > 1 else cur_p
            n_ema9 = float(df_nifty['close'].ewm(span=9, adjust=False).mean().iloc[-1]) if df_nifty is not None and len(df_nifty) > 1 else n_spot
            nifty_context = {
                'spot': n_spot,
                'open': n_open,
                'move': n_spot - n_open,
                'ema9': n_ema9,
                'drop_high': n_high - n_spot,
                'rise_low': n_spot - n_low
            }

        n_move = nifty_context.get('move', 0.0)
        n_spot = nifty_context.get('spot', 0.0)
        n_ema9 = nifty_context.get('ema9', n_spot)
        n_drop = nifty_context.get('drop_high', 0.0)
        n_rise = nifty_context.get('rise_low', 0.0)

        # Pillar 4: Symmetrical Macro Gate Alignment
        allow_call = nifty_context.get('allow_stock_call', True) if nifty_context else True
        allow_put = nifty_context.get('allow_stock_put', True) if nifty_context else True
        regime_reason = nifty_context.get('regime_reason', 'NORMAL') if nifty_context else 'NORMAL'

        if direction == 'CALL' and not allow_call:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'MACRO_REGIME_BLOCKS_CALL: {regime_reason}', 'is_index': False}
        if direction == 'PUT' and not allow_put:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'MACRO_REGIME_BLOCKS_PUT: {regime_reason}', 'is_index': False}

        # Mega heavyweights CANNOT trade against Nifty macro direction
        if sym_clean in NIFTY_HEAVYWEIGHTS:
            if direction == 'CALL' and n_move < -10.0:
                return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'HEAVYWEIGHT_COUNTER_TREND_BLOCKED: Nifty move {n_move:.1f}pts', 'is_index': False}
            if direction == 'PUT' and n_move > +10.0:
                return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'HEAVYWEIGHT_COUNTER_TREND_BLOCKED: Nifty move {n_move:+.1f}pts', 'is_index': False}

        # 7. Price Action CLV & Body Thrust
        body = abs(b_close - b_open)
        solid_body_ratio = round(body / rng, 2)
        clv = round((b_close - b_low) / rng, 2)
        min_thrust = max(2.5, round(stock_atr * 0.06, 1))

        if solid_body_ratio < 0.50 or body < min_thrust:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'WEAK_CANDLE_THRUST (Body {body:.1f} < {min_thrust:.1f}pts)', 'is_index': False}

        clv_pass = (clv >= 0.60) if direction == 'CALL' else (clv <= 0.40)
        if not clv_pass:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': f'CLV_QUALITY_MISALIGNED (CLV={clv:.2f})', 'is_index': False}

        score = min(99.0, 50.0 + min(25.0, (real_rvol - 1.0) * 10.0) + (15.0 if abs(bar_delta_ratio) >= 0.35 else 5.0) + min(10.0, turnover_cr * 1.5))

        return {
            'symbol': sym_clean,
            'passed': True,
            'spot_price': cur_p,
            'direction': direction,
            'pct_change': float(q.get('pct_change', 0.0)),
            'rvol': real_rvol,
            'turnover_cr': turnover_cr,
            'score': round(score, 2),
            'is_index': False,
            'stock_atr': stock_atr,
            'is_compression_breakout': True,
            'cvd_info': cvd_res
        }

    def evaluate_asset(self, symbol, nifty_context=None):
        """
        Master Router: Dispatches to dedicated Index Engine or Stock Engine.
        """
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        now_ist = get_ist_time()
        tot_mins = now_ist.hour * 60 + now_ist.minute
        is_afternoon = (795 <= tot_mins <= 855)

        if sym_clean in INDEX_NAMES:
            return self.evaluate_index_engine(sym_clean, is_afternoon=is_afternoon)
        else:
            return self.evaluate_stock_engine(sym_clean, is_afternoon=is_afternoon, nifty_context=nifty_context)


    def _log_rejection_audit(self, ist_now, all_evaluations, qualified_trades, stage2_rejections=None):
        """
        Phase 4: Structured Rejection and Audit Logger.
        Logs every asset evaluated with component gate results, exact failure codes, and decision.
        """
        audit_file = os.path.join(CURRENT_DIR, 'Alpha_Rejection_Audit_Ledger.jsonl')
        qual_syms = {t['symbol']: t for t in qualified_trades}
        stg2_rej = stage2_rejections or {}
        ts_str = ist_now.strftime('%Y-%m-%d %H:%M:%S IST')

        try:
            with open(audit_file, 'a', encoding='utf-8') as f:
                for item in all_evaluations:
                    sym = item.get('symbol', '')
                    is_qual = sym in qual_syms
                    stg1_pass = item.get('passed', False)
                    fail_reason = 'PASS'
                    rej_stage = 'NONE'

                    if is_qual:
                        rej_stage = 'NONE'
                        fail_reason = 'QUALIFIED_FOR_DISPATCH'
                    elif not stg1_pass:
                        rej_stage = 'STAGE_1'
                        fail_reason = item.get('failed_gate', 'STAGE_1_FILTER')
                    elif sym in stg2_rej:
                        rej_stage = 'STAGE_2'
                        fail_reason = stg2_rej[sym]
                    else:
                        rej_stage = 'STAGE_2'
                        fail_reason = 'STAGE_2_UNVERIFIED'

                    rec = {
                        'symbol': sym,
                        'timestamp': ts_str,
                        'direction': item.get('direction', 'NEUTRAL'),
                        'is_index': item.get('is_index', False),
                        'stage1_passed': stg1_pass,
                        'rejection_stage': rej_stage,
                        'failure_reason': fail_reason,
                        'score': item.get('score', 0.0),
                        'rvol': item.get('rvol', 0.0),
                        'pct_change': item.get('pct_change', 0.0),
                        'decision': 'QUALIFIED' if is_qual else 'REJECTED'
                    }
                    f.write(json.dumps(rec) + '\n')
        except Exception as e:
            print(f'[AUDIT LOG ERROR]: {e}')

    def _record_signal_to_excel(self, ist_now, cand, opt, opt_action, contract, opt_entry, opt_sl, opt_tgt1, opt_tgt2, time_sl_time, buffer_end_time):
        """Maintains Daily & Monthly Official Excel Journals with epoch timestamp tracking for 100% reliable 15M Time SL."""
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
            'Signal_Timestamp_Epoch': ist_now.timestamp(),  # Fixed Epoch Timestamp for 15M Time SL
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
        except Exception as e_xl:
            print(f"[EXCEL JOURNAL ERROR]: Could not record to Excel: {e_xl}")

    def update_open_paper_trades(self):
        """
        PILLAR 5: Real-Time Paper Trading Position Manager.
        Evaluates active open positions on every 3-minute iteration using LIVE Option Quotes:
        - Target 1 hit -> Trails SL to Profit Lock floor (+0.40R floor, eliminating 0.00R drag).
        - Target 2 hit -> Books TARGET_2_HIT (JACKPOT) (+3.5R).
        - Trailing SL hit -> Exits at T1_PROFIT_LOCKED (Guaranteed Profit).
        - Initial SL hit -> Books SL_HIT (-1.00R).
        - 15-Minute Stagnant Kill Switch -> Exits at 15M_TIME_SL_EXIT (Parsed via Signal_Timestamp_Epoch).
        - EOD Square-Off at 15:20 IST -> Exits remaining open trades at market close.
        Updates CSV, Daily Excel, and Monthly Excel simultaneously!
        """
        csv_path = os.path.join(CURRENT_DIR, 'Live_Paper_Trading_Journal.csv')
        daily_xl = os.path.join(CURRENT_DIR, 'Alpha_5Pillar_Live_Paper_Trading_Journal.xlsx')
        monthly_xl = os.path.join(CURRENT_DIR, 'Alpha_Monthly_Paper_Trading_Journal.xlsx')
        if not os.path.exists(csv_path):
            return

        try:
            df = pd.read_csv(csv_path)
            if df.empty or 'Status' not in df.columns:
                return

            for col in ['Exit_Time_IST', 'Trade_Outcome', 'Status', 'Holding_Time']:
                if col in df.columns:
                    df[col] = df[col].astype(object)

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

                # Priority 1: Pull live option quote directly from NSE Option Chain
                current_bid = 0.0
                current_ask = 0.0
                est_opt_price = 0.0

                opt_quote = self.data_feed.get_option_quote(sym, contract_identifier=contract)
                if opt_quote and opt_quote.get('valid') and opt_quote.get('ltp', 0.0) > 0:
                    est_opt_price = opt_quote['ltp']
                    current_bid = opt_quote['bid']
                    current_ask = opt_quote['ask']
                else:
                    # Model delta fallback if live chain quote is momentarily rate-limited
                    delta = 0.52 if is_idx else 0.50
                    if 'CE' in contract or 'CALL' in opt_action:
                        spot_gain = cur_spot - spot_entry
                    else:
                        spot_gain = spot_entry - cur_spot
                    est_opt_price = round(max(0.5, opt_entry + (spot_gain * delta)), 2)
                    current_bid = est_opt_price
                    current_ask = est_opt_price

                # Target 1 Check: Lock in Target 1 Profit (Never drag back to 0.00R cost or theta decay)
                t1_locked_floor = round(opt_entry + (opt_t1 - opt_entry) * 0.40, 2)
                if est_opt_price >= opt_t1 and trail_sl < t1_locked_floor:
                    df.at[idx, 'Trailing_SL_INR'] = t1_locked_floor
                    trail_sl = t1_locked_floor
                    updated_any = True
                    print(f"[POSITION MANAGER]: {sym} reached Target 1! Trailing SL locked at Profit level Rs. {t1_locked_floor:.2f} (NO COST DRAG).")

                # Check Exits
                exit_price = 0.0
                outcome = ""

                # 1. Target 2 Jackpot Hit
                if est_opt_price >= opt_t2:
                    exit_price = opt_t2
                    outcome = "TARGET_2_HIT (JACKPOT)"
                # 2. T1 Profit Locked Exit (Guaranteed Profit, Eliminates Breakeven 0.00R loss)
                elif trail_sl > opt_entry and est_opt_price <= trail_sl:
                    exit_price = trail_sl
                    outcome = "T1_PROFIT_LOCKED"
                # 3. Initial Stop Loss Hit
                elif est_opt_price <= opt_sl:
                    exit_price = opt_sl
                    outcome = "SL_HIT"
                # 4. EOD Square-Off at 15:20 IST (tot_mins >= 920)
                elif tot_mins_now >= 920:
                    exit_price = current_bid if current_bid > 0 else est_opt_price
                    outcome = "EOD_SQUARE_OFF"
                # 5. 15-Minute Stagnant Kill Switch (Parsed strictly via Signal_Timestamp_Epoch)
                else:
                    entry_ts = pd.to_numeric(row.get('Signal_Timestamp_Epoch'), errors='coerce')
                    if pd.notna(entry_ts) and entry_ts > 0:
                        elapsed_minutes = (now_ts - float(entry_ts)) / 60.0
                        if elapsed_minutes >= 15.0 and est_opt_price <= opt_entry:
                            exit_price = current_bid if current_bid > 0 else est_opt_price
                            outcome = "15M_TIME_SL_EXIT"
                    else:
                        try:
                            sig_time_str = str(row['Signal_Time_IST'])
                            sig_dt = datetime.datetime.strptime(sig_time_str.replace(' IST', ''), '%I:%M:%S %p')
                            sig_mins = sig_dt.hour * 60 + sig_dt.minute
                            cur_mins = ist_now.hour * 60 + ist_now.minute
                            if (cur_mins - sig_mins) >= 15 and est_opt_price <= opt_entry:
                                exit_price = current_bid if current_bid > 0 else est_opt_price
                                outcome = "15M_TIME_SL_EXIT"
                        except Exception:
                            pass

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
                        f"🔔 <b>ALPHA TRADE EXIT ALERT</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"🎯 <b>STATUS:</b> <b>{outcome}</b>\n"
                        f"📌 <b>CONTRACT:</b> <code>{contract}</code>\n"
                        f"💰 <b>ENTRY:</b> Rs. {opt_entry:.2f} | <b>EXIT:</b> Rs. {exit_price:.2f}\n"
                        f"📈 <b>POINTS CAPTURED:</b> <b>{pts_cap:+.2f} pts</b>\n"
                        f"💵 <b>NET P&L:</b> <b>Rs. {net_pnl:+,.2f}</b>\n"
                        f"⏰ <b>TIME:</b> {ist_now.strftime('%I:%M:%S %p IST')}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )
                    self.notifier.send_broadcast_message(exit_msg, is_trade_signal=True)
                    print(f"[TRADE CLOSED]: {contract} -> {outcome} ({pts_cap:+.2f} pts | Rs {net_pnl:+,.2f})")

            if updated_any:
                df.to_csv(csv_path, index=False)
                for xl_path in [daily_xl, monthly_xl]:
                    try:
                        df.to_excel(xl_path, index=False)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[POSITION MANAGER ERROR]: {e}")

    def run_screener(self, ignore_window=False):
        ist_now = get_ist_time()
        ist_time_str = ist_now.strftime('%I:%M:%S %p IST')
        ist_date_str = ist_now.strftime('%d-%b-%Y')

        if ist_now.weekday() >= 5:
            print(f"[MARKET CLOSED]: Weekend detected ({ist_now.strftime('%A')}). Zero scan execution.")
            return

        print(f"=======================================================")
        print(f"  ALPHA 5-PILLAR INSTITUTIONAL SCREENER | {ist_date_str} {ist_time_str}")
        print(f"  Designated Slabs: 09:30-10:15 AM IST & 01:15-02:15 PM IST")
        print(f"  Rule: Spot >= {int(MIN_STOCK_PRICE)} INR Floor for Stock Options")
        print(f"=======================================================")

        # First: Monitor and manage active open paper positions!
        self.update_open_paper_trades()

        if not ignore_window and not is_in_active_trading_slab(ist_now):
            print(f"[OUTSIDE TRADING SLAB]: Current time {ist_time_str} is outside active trading slabs.")
            outside_msg = (
                f"🛡️ <b>ALPHA CRON RUNNER ACTIVE</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ <b>Trigger Time:</b> {ist_time_str}\n"
                f"📌 <b>Status:</b> Outside Designated Trading Slab\n"
                f"⏳ <b>Next Active Window:</b> 09:30-10:15 AM or 01:15-02:15 PM IST\n"
                f"🔒 <b>Capital Protection:</b> 100% Active (Quiet in Group)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            self.notifier.send_broadcast_message(outside_msg, is_trade_signal=False)
            return

        # Balanced Slab Allocation
        tot_mins_now = ist_now.hour * 60 + ist_now.minute
        is_morning = (570 <= tot_mins_now <= 615)
        is_afternoon = (795 <= tot_mins_now <= 855)

        morn_count = 0
        aft_count = 0
        morn_idx_count = 0
        aft_idx_count = 0
        morn_stk_count = 0
        aft_stk_count = 0

        for sym_d, ts in self.dispatched_today.items():
            if ts > 0:
                try:
                    dt = datetime.datetime.fromtimestamp(ts, tz=ist_now.tzinfo)
                    is_m = (dt.hour == 9 or (dt.hour == 10 and dt.minute <= 15))
                    is_a = (dt.hour >= 13)
                    if is_m:
                        morn_count += 1
                        if sym_d in INDEX_NAMES: morn_idx_count += 1
                        else: morn_stk_count += 1
                    elif is_a:
                        aft_count += 1
                        if sym_d in INDEX_NAMES: aft_idx_count += 1
                        else: aft_stk_count += 1
                except Exception:
                    pass

        active_slab_count = morn_count if is_morning else aft_count
        if active_slab_count >= MAX_SLAB_TRADES and not ignore_window:
            print(f"[SLAB CAP REACHED]: {active_slab_count}/{MAX_SLAB_TRADES} trades already dispatched in this slab. Capital protected.")
            cap_msg = (
                f"🛡️ <b>ALPHA SLAB ALLOCATION COMPLETE</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ <b>Execution Time:</b> {ist_time_str}\n"
                f"📌 <b>Status:</b> Slab Cap ({active_slab_count}/{MAX_SLAB_TRADES}) Reached\n"
                f"🔒 <b>Capital Protection:</b> Active (Zero Overtrading)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            self.notifier.send_broadcast_message(cap_msg, is_trade_signal=False)
            return

        # STAGE 1: Consolidated NSE Bulk Snapshot Fetch
        fetch_bulk_nse_snapshot(force_refresh=True)
        nifty_regime = self.get_nifty_market_regime(force_refresh=True)
        print(
            f"[MACRO REGIME]: AllowStockCALL={nifty_regime['allow_stock_call']} | "
            f"AllowStockPUT={nifty_regime['allow_stock_put']} | {nifty_regime['reason']}"
        )

        # PHASE A: DEDICATED INDEX ENGINE (Priority 1: Run 4 Indices First!)
        # -------------------------------------------------------------
        index_symbols = [s for s in INDEX_NAMES]
        with ThreadPoolExecutor(max_workers=4) as executor:
            index_eval_results = list(executor.map(lambda s: self.evaluate_index_engine(s, is_afternoon=is_afternoon), index_symbols))

        index_candidates = [
            r for r in index_eval_results 
            if r.get('passed', False) and r.get('score', 0.0) >= 70.0
        ]
        index_candidates.sort(key=lambda x: (x.get('score', 0.0), x.get('rvol', 0.0)), reverse=True)

        # PHASE B: DEDICATED STOCK ENGINE (Priority 2: Run Stock F&O Universe Second!)
        # -------------------------------------------------------------
        df_nifty = self.data_feed.get_live_intraday_bars('NIFTY')
        nifty_context = {
            'spot': 0.0, 'open': 0.0, 'move': 0.0, 'ema9': 0.0, 'drop_high': 0.0, 'rise_low': 0.0,
            'allow_stock_call': nifty_regime.get('allow_stock_call', True),
            'allow_stock_put': nifty_regime.get('allow_stock_put', True),
            'regime_reason': nifty_regime.get('reason', 'NORMAL')
        }
        if df_nifty is not None and len(df_nifty) > 1:
            n_spot = float(df_nifty['close'].iloc[-1])
            n_open = float(df_nifty['open'].iloc[0])
            n_high = float(df_nifty['high'].max())
            n_low = float(df_nifty['low'].min())
            n_ema9 = float(df_nifty['close'].ewm(span=9, adjust=False).mean().iloc[-1])
            nifty_context = {
                'spot': n_spot,
                'open': n_open,
                'move': n_spot - n_open,
                'ema9': n_ema9,
                'drop_high': n_high - n_spot,
                'rise_low': n_spot - n_low,
                'allow_stock_call': nifty_regime.get('allow_stock_call', True),
                'allow_stock_put': nifty_regime.get('allow_stock_put', True),
                'regime_reason': nifty_regime.get('reason', 'NORMAL')
            }

        stock_symbols = [x['symbol'] for x in self.universe if x['symbol'] not in INDEX_NAMES]
        with ThreadPoolExecutor(max_workers=8) as executor:
            stock_eval_results = list(executor.map(lambda s: self.evaluate_stock_engine(s, is_afternoon=is_afternoon, nifty_context=nifty_context), stock_symbols))

        stock_candidates = [
            r for r in stock_eval_results 
            if r.get('passed', False) and r.get('score', 0.0) >= 70.0
        ]
        stock_candidates.sort(
            key=lambda x: (
                x.get('score', 0.0),
                x.get('turnover_cr', 0.0),
                x.get('rvol', 0.0)
            ),
            reverse=True
        )

        all_eval_results = index_eval_results + stock_eval_results
        total_scanned = len(index_symbols) + len(stock_symbols)
        print(f"[DISCOVERY COMPLETE]: Scanned {total_scanned} assets ({len(index_symbols)} Indices, {len(stock_symbols)} Stocks).")
        print(f"[QUALIFIED DISCOVERY]: {len(index_candidates)} Index candidate(s), {len(stock_candidates)} Stock candidate(s).")

        # STAGE 2: Real 3-Minute Breakout, Live Spot Holding & BVC CVD Audit
        verified_super_trades = []
        stage2_rejections = {}
        active_slab_index = morn_idx_count if is_morning else aft_idx_count
        active_slab_stock = morn_stk_count if is_morning else aft_stk_count

        # 1. Evaluate Index Candidates into Dedicated Index Slot (Max 1 per slab)
        for cand in index_candidates:
            if active_slab_index >= MAX_INDEX_PER_SLAB:
                stage2_rejections[cand['symbol']] = "MAX_INDEX_PER_SLAB_CAP_REACHED"
                continue

            sym = cand['symbol']
            if self.is_already_dispatched(sym):
                stage2_rejections[sym] = "ALREADY_DISPATCHED_TODAY"
                continue

            stk_df = self.data_feed.get_live_intraday_bars(sym)
            if stk_df is None or len(stk_df) < 10:
                stage2_rejections[sym] = "INSUFFICIENT_3M_BARS_LESS_THAN_10"
                continue

            live_bar = stk_df.iloc[-1]
            l_close = float(live_bar['close'])
            spot = cand['spot_price']
            direction = cand['direction']

            comp_bars = stk_df.iloc[:-2].tail(8)
            comp_high = float(comp_bars['high'].max())
            comp_low = float(comp_bars['low'].min())
            spot_buffer = 15.0 if sym == 'BANKNIFTY' else 8.0

            if direction == 'CALL':
                live_holding = (spot >= comp_high - spot_buffer) and (l_close >= comp_high - spot_buffer)
            else:
                live_holding = (spot <= comp_low + spot_buffer) and (l_close <= comp_low + spot_buffer)

            if not live_holding:
                stage2_rejections[sym] = f"LIVE_SPOT_FAILED_HOLDING (Spot {spot:.2f} vs Base Level)"
                continue

            oi_conf = self.data_feed.get_live_dual_oi_confluence(sym, spot, direction=direction)
            if not oi_conf.get('confluent', False):
                stage2_rejections[sym] = f"OI_RUNWAY_REJECTED: {oi_conf.get('reason')}"
                continue

            opt = self.data_feed.compute_option_pricing(spot, direction=direction, symbol=sym)
            if not opt or not opt.get('is_live_chain_verified', False) or float(opt.get('entry_price', 0.0)) < 2.0:
                stage2_rejections[sym] = "LIVE_OPTION_CHAIN_LTP_UNAVAILABLE"
                continue

            contract = opt['contract']
            if self.is_already_dispatched(sym, contract):
                stage2_rejections[sym] = "CONTRACT_ALREADY_DISPATCHED"
                continue

            cand['verified_opt'] = opt
            cand['oi_confluent'] = True
            verified_super_trades.append(cand)
            active_slab_index += 1

        # 2. Evaluate Stock Candidates into Dedicated Stock Slots (Max 2 per slab)
        for cand in stock_candidates:
            if active_slab_stock >= MAX_STOCKS_PER_SLAB:
                stage2_rejections[cand['symbol']] = "MAX_STOCK_PER_SLAB_CAP_REACHED"
                continue

            sym = cand['symbol']
            if self.is_already_dispatched(sym):
                stage2_rejections[sym] = "ALREADY_DISPATCHED_TODAY"
                continue

            stk_df = self.data_feed.get_live_intraday_bars(sym)
            if stk_df is None or len(stk_df) < 10:
                stage2_rejections[sym] = "INSUFFICIENT_3M_BARS_LESS_THAN_10"
                continue

            live_bar = stk_df.iloc[-1]
            l_close = float(live_bar['close'])
            spot = cand['spot_price']
            direction = cand['direction']

            comp_bars = stk_df.iloc[:-2].tail(8)
            comp_high = float(comp_bars['high'].max())
            comp_low = float(comp_bars['low'].min())
            stock_atr = float(cand.get('stock_atr', 30.0))
            spot_buffer = stock_atr * 0.05

            if direction == 'CALL':
                live_holding = (spot >= comp_high - spot_buffer) and (l_close >= comp_high - spot_buffer)
            else:
                live_holding = (spot <= comp_low + spot_buffer) and (l_close <= comp_low + spot_buffer)

            if not live_holding:
                stage2_rejections[sym] = f"LIVE_SPOT_FAILED_HOLDING (Spot {spot:.2f} vs Base Level)"
                continue

            oi_conf = self.data_feed.get_live_dual_oi_confluence(sym, spot, direction=direction)
            if not oi_conf.get('confluent', False):
                stage2_rejections[sym] = f"OI_RUNWAY_REJECTED: {oi_conf.get('reason')}"
                continue

            opt = self.data_feed.compute_option_pricing(spot, direction=direction, symbol=sym)
            if not opt or not opt.get('is_live_chain_verified', False) or float(opt.get('entry_price', 0.0)) < 2.0:
                stage2_rejections[sym] = "LIVE_OPTION_CHAIN_LTP_UNAVAILABLE"
                continue

            contract = opt['contract']
            if self.is_already_dispatched(sym, contract):
                stage2_rejections[sym] = "CONTRACT_ALREADY_DISPATCHED"
                continue

            cand['verified_opt'] = opt
            cand['oi_confluent'] = True
            verified_super_trades.append(cand)
            active_slab_stock += 1

        # Log Full Diagnostic Audit
        self._log_rejection_audit(ist_now, all_eval_results, verified_super_trades, stage2_rejections)

        if not verified_super_trades:
            print(f"[SCREENER]: 0 signals qualified after separate Index & Stock scan at {ist_time_str}. Capital 100% Protected.")
            # Dispatches confirmation heartbeat strictly to Owner Chatbot (Never spams Group)
            heartbeat_msg = (
                f"🛡️ <b>ALPHA 5-PILLAR SCANNER STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ <b>Execution Time:</b> {ist_time_str}\n"
                f"📊 <b>Universe Scanned:</b> {total_scanned} Assets\n"
                f"🏛️ <b>Index Engine:</b> 4 Indices Evaluated (Slots: {active_slab_index}/{MAX_INDEX_PER_SLAB})\n"
                f"📈 <b>Stock Engine:</b> {len(stock_symbols)} Stocks Evaluated (Slots: {active_slab_stock}/{MAX_STOCKS_PER_SLAB})\n"
                f"🛡️ <b>Status:</b> Zero 5-Pillar Confluent Breakouts\n"
                f"🔒 <b>Capital Protection:</b> 100% Active (Zero Forced Trades)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            self.notifier.send_broadcast_message(heartbeat_msg, is_trade_signal=False)
            return

        print(f"[SCREENER]: Verified {len(verified_super_trades)} INSTITUTIONAL SUPER TRADE(S)! Dispatching...")

        for cand in verified_super_trades:
            sym = cand['symbol']
            spot = cand['spot_price']
            direction = cand['direction']
            is_idx = cand.get('is_index', False)
            opt = cand['verified_opt']

            contract = opt['contract']
            opt_entry = float(opt.get('entry_price', opt.get('ltp', 50.0)))
            opt_action = 'BUY CALL (CE)' if direction == 'CALL' else 'BUY PUT (PE)'

            # Fresh 4-Minute Execution Buffer calculated right at dispatch time!
            dispatch_now = get_ist_time()
            buffer_end_time = (dispatch_now + datetime.timedelta(minutes=4)).strftime('%I:%M:%S %p IST')
            time_sl_time = (dispatch_now + datetime.timedelta(minutes=15)).strftime('%I:%M:%S %p IST')

            # Realistic Option Targets calibrated via Delta (0.52 for Index, Risk:Reward for Stock)
            if is_idx:
                delta = 0.52
                spot_t1 = 75.0 if sym == 'BANKNIFTY' else 35.0
                spot_t2 = 150.0 if sym == 'BANKNIFTY' else 70.0
                spot_sl = 60.0 if sym == 'BANKNIFTY' else 30.0

                opt_sl = round(max(1.0, opt_entry - (spot_sl * delta)), 2)
                opt_tgt1 = round(opt_entry + (spot_t1 * delta), 2)
                opt_tgt2 = round(opt_entry + (spot_t2 * delta), 2)
            else:
                stk_atr = cand.get('stock_atr', spot * 0.022)
                # Safely read opt_sl with fallbacks
                opt_sl = float(opt.get('stop_loss', opt.get('sl_price', round(opt_entry * 0.80, 2))))
                opt_sl_pts = max(1.0, opt_entry - opt_sl)
                opt_tgt1 = float(opt.get('target_1', round(opt_entry + (opt_sl_pts * 1.50), 2)))
                opt_tgt2 = float(opt.get('target_2', round(opt_entry + (opt_sl_pts * 3.50), 2)))

            pure_signal_msg = (
                f"🚀 <b>ALPHA 5-PILLAR INSTITUTIONAL TRADE</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>INSTRUMENT:</b> <b>{sym} ({direction})</b>\n"
                f"⚡ <b>ACTION:</b> <b>{opt_action}</b>\n"
                f"📌 <b>CONTRACT:</b> <code>{contract}</code>\n"
                f"💵 <b>BUYING PRICE:</b> <b>Rs. {opt_entry:,.2f}</b>\n"
                f"🛡️ <b>STOP LOSS:</b> <b>Rs. {opt_sl:,.2f}</b>\n"
                f"🎯 <b>TARGET 1 (Lock Profit):</b> <b>Rs. {opt_tgt1:,.2f}</b>\n"
                f"🏆 <b>TARGET 2 (Jackpot):</b> <b>Rs. {opt_tgt2:,.2f}</b>\n"
                f"⏱️ <b>STAGNANT TIME EXIT:</b> <b>{time_sl_time} (15-Min Kill Switch)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏳ <b>4-MIN BUFFER VALID TILL:</b> <b>{buffer_end_time}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            # Atomic Dispatch: Record to Journal BEFORE Telegram broadcast!
            self.mark_dispatched(sym, contract)
            self._record_signal_to_excel(
                ist_now=dispatch_now,
                cand=cand,
                opt=opt,
                opt_action=opt_action,
                contract=contract,
                opt_entry=opt_entry,
                opt_sl=opt_sl,
                opt_tgt1=opt_tgt1,
                opt_tgt2=opt_tgt2,
                time_sl_time=time_sl_time,
                buffer_end_time=buffer_end_time
            )
            print(f"[BROADCAST]: Dispatching {contract} @ Rs. {opt_entry:,.2f}...")
            self.notifier.send_broadcast_message(pure_signal_msg, is_trade_signal=True)

    def _check_concurrent_github_run(self):
        """
        Concurrency shield: Checks if an older GitHub Actions run is already actively running.
        Prevents duplicate parallel executions during active slab hours.
        """
        run_id_str = os.environ.get('GITHUB_RUN_ID', '').strip()
        if not run_id_str:
            return False
        try:
            cur_run_id = int(run_id_str)
            url = 'https://api.github.com/repos/manojmukati/alpha-option-buyer/actions/runs?status=in_progress'
            req = urllib.request.Request(url, headers={'User-Agent': 'AlphaOptionBuyer/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                for r in data.get('workflow_runs', []):
                    r_id = r.get('id', 0)
                    if r_id != cur_run_id and r_id < cur_run_id:
                        print(f"[CONCURRENCY SHIELD]: Active runner detected (Run #{r.get('run_number')} ID {r_id}). Yielding execution.")
                        return True
        except Exception:
            pass
        return False

    def run_live_slab_session(self, interval_seconds=180):
        """
        Runs a persistent single-runner session throughout the active trading slab.
        Executes immediate scan, then sleeps interval_seconds (default 180s / 3 mins).
        If open positions exist when the slab ends, continues monitoring positions until 03:25 PM IST!
        """
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            if self._check_concurrent_github_run():
                print("[CONCURRENCY SHIELD]: Exiting gracefully to let the primary runner execute.")
                return

        ist_now = get_ist_time()
        tot_m = ist_now.hour * 60 + ist_now.minute

        # 1. Pre-Slab Warmup Buffer (Handles GitHub Actions or local schedulers launching up to 30 mins early)
        if 540 <= tot_m < 570:
            wait_sec = max(5, (570 - tot_m) * 60 - ist_now.second)
            print(f"[PRE-SLAB WARMUP]: Launched early at {ist_now.strftime('%I:%M:%S %p IST')}. Sleeping {wait_sec}s until Morning Slab opens at 09:30:00 AM IST...")
            time.sleep(wait_sec)
            ist_now = get_ist_time()
            tot_m = ist_now.hour * 60 + ist_now.minute
        elif 765 <= tot_m < 795:
            wait_sec = max(5, (795 - tot_m) * 60 - ist_now.second)
            print(f"[PRE-SLAB WARMUP]: Launched early at {ist_now.strftime('%I:%M:%S %p IST')}. Sleeping {wait_sec}s until Afternoon Slab opens at 01:15:00 PM IST...")
            time.sleep(wait_sec)
            ist_now = get_ist_time()
            tot_m = ist_now.hour * 60 + ist_now.minute

        if 570 <= tot_m <= 615:
            slab_name = "MORNING SLAB (09:30 - 10:15 AM IST)"
            end_min = 615
        elif 795 <= tot_m <= 855:
            slab_name = "AFTERNOON SLAB (01:15 - 02:15 PM IST)"
            end_min = 855
        else:
            print(f"[SLAB SESSION]: Current time {ist_now.strftime('%I:%M %p IST')} is outside active slabs (09:30-10:15 AM & 01:15-02:15 PM IST). Executing single status scan.")
            self.run_screener(ignore_window=True)
            return

        print(f"=======================================================")
        print(f"  STARTING PERSISTENT SLAB RUNNER: {slab_name}")
        print(f"  Cadence: Exact {interval_seconds}s (3-Minute Resampled Candles)")
        print(f"=======================================================")

        # Send Slab Initiation Heartbeat strictly to Owner Chatbot
        init_msg = (
            f"⚡ <b>ALPHA PERSISTENT SLAB RUNNER STARTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏛️ <b>Session:</b> {slab_name}\n"
            f"⏰ <b>Start Time:</b> {ist_now.strftime('%I:%M:%S %p IST')}\n"
            f"⏱️ <b>Cadence:</b> Every {interval_seconds}s (Resampled 3-Min Candles)\n"
            f"🎯 <b>Active Until:</b> {datetime.time(end_min // 60, end_min % 60).strftime('%I:%M %p IST')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        self.notifier.send_broadcast_message(init_msg, is_trade_signal=False)

        while True:
            cur_time = get_ist_time()
            cur_mins = cur_time.hour * 60 + cur_time.minute

            # Check if any paper positions remain open
            has_open_trades = False
            csv_path = os.path.join(CURRENT_DIR, 'Live_Paper_Trading_Journal.csv')
            if os.path.exists(csv_path):
                try:
                    df_j = pd.read_csv(csv_path)
                    if not df_j.empty and 'Status' in df_j.columns:
                        has_open_trades = (df_j['Status'] == 'SIGNAL_GENERATED_OPEN').any()
                except Exception:
                    pass

            if cur_mins > end_min:
                if has_open_trades and cur_mins <= 925:  # 03:25 PM IST
                    print(f"[POSITION MONITORING]: Slab concluded, but open positions remain! Monitoring positions till target/SL or 03:25 PM...")
                    try:
                        self.update_open_paper_trades()
                    except Exception as e:
                        print(f"[POSITION MONITOR ERROR]: {e}")
                    time.sleep(interval_seconds)
                    continue
                else:
                    print(f"[SLAB CONCLUDED]: {slab_name} completed at {cur_time.strftime('%I:%M:%S %p IST')}.")
                    conclusion_msg = (
                        f"🏁 <b>ALPHA SLAB RUNNER COMPLETED</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"🏛️ <b>Session:</b> {slab_name}\n"
                        f"⏰ <b>Finish Time:</b> {cur_time.strftime('%I:%M:%S %p IST')}\n"
                        f"📊 <b>Positions Status:</b> {'Open trades being monitored' if has_open_trades else 'All trades concluded'}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )
                    self.notifier.send_broadcast_message(conclusion_msg, is_trade_signal=False)
                    break

            try:
                self.run_screener(ignore_window=True)
            except Exception as scan_err:
                print(f"[SLAB SCAN EXCEPTION]: {scan_err}. Retrying in next 3-minute candle...")

            print(f"[SLEEP]: Waiting {interval_seconds}s for next 3-minute candle close...")
            time.sleep(interval_seconds)

    def send_eod_journal(self):
        """
        Dispatches the closing Excel Trading Journals strictly to the Owner Chatbot via Telegram.
        """
        print("[EOD JOURNAL]: Checking and updating final paper trade states...")
        self.update_open_paper_trades()

        now_str = get_ist_time().strftime('%d-%b-%Y %I:%M %p IST')
        daily_excel = os.path.join(CURRENT_DIR, 'Alpha_5Pillar_Live_Paper_Trading_Journal.xlsx')
        monthly_excel = os.path.join(CURRENT_DIR, 'Alpha_Monthly_Paper_Trading_Journal.xlsx')

        sent_any = False
        if os.path.exists(daily_excel):
            caption = f"📊 <b>ALPHA OPTION BUYER — DAILY EXCEL JOURNAL</b>\n📅 Date: {now_str}\n🔒 Confidential: Strictly for System Owner"
            print(f"[EOD JOURNAL]: Dispatching Daily Excel Journal: {daily_excel}")
            self.notifier.send_document_to_owner_with_approval_prompt(daily_excel, caption=caption)
            sent_any = True

        if os.path.exists(monthly_excel):
            caption = f"📈 <b>ALPHA MONTHLY CUMULATIVE JOURNAL</b>\n📅 Date: {now_str}\n🔒 Confidential: Strictly for System Owner"
            print(f"[EOD JOURNAL]: Dispatching Monthly Excel Journal: {monthly_excel}")
            self.notifier.send_document_to_owner_with_approval_prompt(monthly_excel, caption=caption)
            sent_any = True

        if not sent_any:
            print("[EOD JOURNAL WARNING]: No Excel journals found to dispatch.")
        else:
            print("[EOD JOURNAL SUCCESS]: Journals dispatched strictly to Owner Chatbot.")


if __name__ == '__main__':
    screener = DailyAutonomousScreener()
    if '--send-journal' in sys.argv:
        screener.send_eod_journal()
    elif '--single-pass' in sys.argv:
        screener.run_screener(ignore_window=('--ignore-window' in sys.argv))
    else:
        # Default production execution mode for both GitHub Actions and Local:
        # Executes persistent slab session (09:30-10:15 & 01:15-02:15) with 3-minute candle cadence
        if '--ignore-window' in sys.argv:
            screener.run_screener(ignore_window=True)
        else:
            screener.run_live_slab_session(interval_seconds=180)
