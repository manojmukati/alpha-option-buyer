# -*- coding: utf-8 -*-
"""
daily_autonomous_screener.py
============================
Alpha 5-Pillar Pure Institutional Paper-Trading Screener
Built to institutional standards across all 217 NSE F&O Assets (4 Indices + 213 Stocks).

Key Institutional Upgrades:
1. True Cumulative Volume Delta (CVD) via Bulk Volume Classification (BVC) across all intraday bars.
2. Zero hardcoded stock names. Only 4 official indices: ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'].
3. All live quotes, option chains, OI buildup/unwinding, and LTP pulled directly from NSE.
4. Akamai anti-blocking engine: 2-step cookie priming, bulk single-call snapshots, sequential option chain locks.
5. True 8-bar compression base requiring >= 10 bars (8 base + 1 breakout + 1 live ticking).
6. Live spot holding re-confirmation before dispatch (eliminates false breakouts).
7. Local consolidation base runway check (allows clean afternoon breakouts without whole-day range choking).
8. Real-time paper trade position manager: monitors open trades on every 3m bar for T1, T2, Trailing Cost SL,
   Initial SL, and 15m Time SL, recording realized P&L to CSV & Excel.
9. Full diagnostic Rejection Audit Ledger logging exact Stage 1 & Stage 2 failure reasons.
"""

import os
import sys
import time
import json
import random
import datetime
import math
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

MAX_DAILY_SUPER_TRADES = 8  # 4 Morning + 4 Afternoon
MAX_SLAB_TRADES = 4
MAX_INDEX_PER_SLAB = 2
MAX_STOCKS_PER_SLAB = 2


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


def compute_bvc_cvd(df_bars):
    """
    Computes Bulk Volume Classification (BVC) Cumulative Volume Delta (CVD)
    from intraday OHLCV bars based on quantitative market microstructure.
    
    Bar Delta = Volume * [ (Close - Low) - (High - Close) ] / (High - Low)
              = Volume * [ 2 * Close - (High + Low) ] / Range
              
    Session CVD = Sum(Bar Delta) across all session bars.
    Session CVD Imbalance = Session CVD / Total Traded Volume.
    Recent 3-Bar CVD Imbalance = Sum(Last 3 Deltas) / Sum(Last 3 Volumes).
    """
    if df_bars is None or len(df_bars) == 0:
        return {
            'session_cvd_ratio': 0.0,
            'recent_cvd_ratio': 0.0,
            'session_cvd_raw': 0.0,
            'cvd_direction': 'NEUTRAL'
        }

    highs = df_bars['high'].values.astype(float)
    lows = df_bars['low'].values.astype(float)
    closes = df_bars['close'].values.astype(float)
    vols = df_bars['volume'].values.astype(float) if 'volume' in df_bars.columns else np.zeros_like(closes)

    rngs = np.maximum(0.01, highs - lows)
    tot_vol = np.sum(vols)

    if tot_vol > 0:
        # Standard Institutional BVC Volume Delta
        deltas = vols * ((2.0 * closes - (highs + lows)) / rngs)
        session_cvd_raw = float(np.sum(deltas))
        session_cvd_ratio = round(session_cvd_raw / max(1.0, tot_vol), 3)

        rec_n = min(3, len(df_bars))
        rec_vol = float(np.sum(vols[-rec_n:]))
        rec_delta = float(np.sum(deltas[-rec_n:]))
        recent_cvd_ratio = round(rec_delta / max(1.0, rec_vol), 3)
    else:
        # Spot Index Fallback (when Volume = 0 in index spot feed): Range-Weighted Delta
        deltas = (2.0 * closes - (highs + lows))
        tot_rng = float(np.sum(rngs))
        session_cvd_raw = float(np.sum(deltas))
        session_cvd_ratio = round(session_cvd_raw / max(0.5, tot_rng), 3)

        rec_n = min(3, len(df_bars))
        rec_rng = float(np.sum(rngs[-rec_n:]))
        rec_delta = float(np.sum(deltas[-rec_n:]))
        recent_cvd_ratio = round(rec_delta / max(0.5, rec_rng), 3)

    cvd_dir = 'NEUTRAL'
    if recent_cvd_ratio >= 0.15 and session_cvd_ratio >= 0.0:
        cvd_dir = 'CALL'
    elif recent_cvd_ratio <= -0.15 and session_cvd_ratio <= 0.0:
        cvd_dir = 'PUT'

    return {
        'session_cvd_ratio': session_cvd_ratio,
        'recent_cvd_ratio': recent_cvd_ratio,
        'session_cvd_raw': session_cvd_raw,
        'cvd_direction': cvd_dir
    }


class DailyAutonomousScreener:
    def __init__(self):
        self.data_feed = LiveMarketDataFeed()
        self.notifier = TelegramNotifier(config_path=CONFIG_PATH)
        self.universe = self.data_feed.universe_assets

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
        cur_p = float(q.get('live_spot', q.get('close', 100.0)))
        cached_atr = float(self.atr_cache.get(symbol, 0.0))
        if cached_atr > 0.0:
            return cached_atr
        # Institutional ATR% baseline: 2.2% of spot
        return round(max(3.0, cur_p * 0.022), 2)

    def get_nifty_market_regime(self, force_refresh=False):
        """
        Symmetrical Macro Trend Filter from NIFTY 50 3-minute bars:
        - Downtrend: Spot < 9-EMA, down >= 35 pts from open, and drop >= 60 pts from high -> blocks CALLs
        - Uptrend: Spot > 9-EMA, up >= 35 pts from open, and rise >= 60 pts from low -> blocks PUTs
        """
        now = time.time()
        if not force_refresh and (now - self.nifty_regime_cache.get('timestamp', 0.0) < 60.0):
            return self.nifty_regime_cache

        df_nifty = self.data_feed.get_live_intraday_bars('NIFTY')
        nq = self.data_feed.get_live_asset_quote('NIFTY')

        nifty_spot = float(nq.get('live_spot', 0.0)) if nq else 0.0
        nifty_open = float(nq.get('open', nifty_spot)) if nq else nifty_spot
        nifty_high = float(nq.get('high', nifty_spot)) if nq else nifty_spot
        nifty_low = float(nq.get('low', nifty_spot)) if nq else nifty_spot

        allow_call = True
        allow_put = True
        reason = "NIFTY_NEUTRAL_ALIGNED"

        if df_nifty is not None and len(df_nifty) >= 6:
            live_b = df_nifty.iloc[-1]
            nifty_spot = float(live_b['close'])
            ema9 = float(df_nifty['ema9'].iloc[-1]) if 'ema9' in df_nifty.columns else float(df_nifty['close'].ewm(span=9).mean().iloc[-1])
            day_open = float(df_nifty['open'].iloc[0])
            day_high = float(df_nifty['high'].max())
            day_low = float(df_nifty['low'].min())

            drop_from_high = round(day_high - nifty_spot, 2)
            rise_from_low = round(nifty_spot - day_low, 2)
            down_from_open = (nifty_spot < day_open - 35.0)
            up_from_open = (nifty_spot > day_open + 35.0)

            recent_rng = round(df_nifty['high'].tail(6).max() - df_nifty['low'].tail(6).min(), 2)
            ema_distance = abs(nifty_spot - ema9)

            if recent_rng < 30.0 and ema_distance < 15.0:
                allow_call = False
                allow_put = False
                reason = f"NIFTY_SIDEWAYS_CHOP_TRAP (18m Range {recent_rng:.1f} pts < 30 pts)"
            else:
                if (nifty_spot < ema9) and down_from_open and drop_from_high >= 60.0:
                    allow_call = False
                    reason = f"NIFTY_ACTIVE_DOWNTREND (Spot {nifty_spot:.1f} < 9EMA {ema9:.1f} | Drop={drop_from_high:.1f}pts)"
                elif (nifty_spot > ema9) and up_from_open and rise_from_low >= 60.0:
                    allow_put = False
                    reason = f"NIFTY_ACTIVE_UPTREND (Spot {nifty_spot:.1f} > 9EMA {ema9:.1f} | Rise={rise_from_low:.1f}pts)"

        self.nifty_regime_cache = {
            'allow_stock_call': allow_call,
            'allow_stock_put': allow_put,
            'reason': reason,
            'timestamp': now
        }
        return self.nifty_regime_cache

    def evaluate_asset(self, symbol):
        """
        STAGE 1: Opportunity Discovery across All 217 Assets.
        - Zero hardcoded stock names.
        - No arbitrary ₹150 price floor (all valid NSE F&O stocks scanned).
        - Volatility calibrated via ATR% >= 1.0% of spot price.
        - Sweet spot scoring (0.60% to 2.20% move = maximum points).
        - Preliminary BVC CVD and RVOL scoring.
        """
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        is_idx = sym_clean in INDEX_NAMES

        q = self.data_feed.get_live_asset_quote(sym_clean)
        if not q:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': 'NO_NSE_LIVE_QUOTE',
                'is_index': is_idx
            }

        cur_p = float(q.get('live_spot', q.get('close', 0.0)))
        if cur_p <= 0:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': 'INVALID_SPOT_PRICE',
                'is_index': is_idx
            }

        pct_chg = float(q.get('pct_change', 0.0))
        rvol = float(q.get('rvol', 1.0))
        stock_atr = 0.0 if is_idx else self.get_stock_atr(sym_clean, q)

        # Gate 0: Institutional High-Value F&O Universe Filter (Spot >= 1000.0 INR and ATR% >= 1.0%)
        # Strictly enforces institutional liquidity standards (BlackRock Aladdin level):
        # Blocks cheap penny/mid-cap stocks (< 1000 INR) with erratic option bid-ask spreads and retail chop.
        # Only trades liquid leaders >= 1000 INR (TCS, RELIANCE, DIXON, BAJFINANCE, PIDILITIND, etc.)
        if not is_idx:
            atr_pct = (stock_atr / max(1.0, cur_p)) * 100.0
            if cur_p < 1000.0 or atr_pct < 1.0:
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': f'STOCK_PRICE_BELOW_1000_INR: Spot Rs {cur_p:.1f} < 1000.0 (or ATR% {atr_pct:.2f}% < 1.0%)',
                    'is_index': False
                }

        # -------------------------------------------------------------
        # STAGE 1 CANDLE DISCOVERY FOR INDICES
        # -------------------------------------------------------------
        if is_idx:
            idx_df = self.data_feed.get_live_intraday_bars(sym_clean)
            if idx_df is None or len(idx_df) < 10:
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': 'INDEX_INSUFFICIENT_BARS_LESS_THAN_10',
                    'is_index': True
                }

            comp_bars = idx_df.iloc[:-2].tail(8)
            eval_bar = idx_df.iloc[-2]
            live_bar = idx_df.iloc[-1]

            b_open = float(eval_bar['open'])
            b_high = float(eval_bar['high'])
            b_low = float(eval_bar['low'])
            b_close = float(eval_bar['close'])
            rng = max(0.01, b_high - b_low)

            comp_high = float(comp_bars['high'].max())
            comp_low = float(comp_bars['low'].min())
            comp_width = comp_high - comp_low

            max_allowed_base = 180.0 if sym_clean in ['NIFTY', 'FINNIFTY', 'MIDCPNIFTY'] else 420.0
            if comp_width > max_allowed_base:
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': f'INDEX_BASE_TOO_WIDE ({comp_width:.1f} > {max_allowed_base:.1f}pts)',
                    'is_index': True
                }

            clv = round((b_close - b_low) / rng, 2)
            solid_body = round(abs(b_close - b_open) / rng, 2)
            min_pts = 35.0 if sym_clean in ['NIFTY', 'FINNIFTY', 'MIDCPNIFTY'] else 80.0
            idx_dist = 40.0 if sym_clean in ['NIFTY', 'FINNIFTY', 'MIDCPNIFTY'] else 90.0

            direction = None
            is_breakout = False
            if b_close >= comp_high and b_close <= comp_high + idx_dist and cur_p >= comp_high:
                direction = 'CALL'
                is_breakout = True
            elif b_close <= comp_low and b_close >= comp_low - idx_dist and cur_p <= comp_low:
                direction = 'PUT'
                is_breakout = True

            if not is_breakout or direction is None:
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': 'INDEX_NO_COMPRESSION_BREAKOUT',
                    'is_index': True
                }

            # BVC Cumulative Volume Delta for Index (Order flow aggression)
            cvd_res = compute_bvc_cvd(idx_df)
            rec_cvd = cvd_res['recent_cvd_ratio']
            sess_cvd = cvd_res['session_cvd_ratio']

            cvd_pass = (rec_cvd >= 0.15) if direction == 'CALL' else (rec_cvd <= -0.15)
            if not cvd_pass:
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': f'INDEX_CVD_MISALIGNED (RecCVD={rec_cvd:+.2f})',
                    'is_index': True
                }

            # Afternoon Midday Chop Shield (01:15 to 02:15 PM)
            if not is_morning:
                day_h = float(idx_df['high'].max())
                day_l = float(idx_df['low'].min())
                day_rng = max(1.0, day_h - day_l)
                pos_pct = (cur_p - day_l) / day_rng * 100.0
                if 35.0 <= pos_pct <= 65.0:
                    return {
                        'symbol': sym_clean,
                        'passed': False,
                        'failed_gate': f'MIDDAY_CHOP_ZONE ({pos_pct:.1f}% of Day Range)',
                        'is_index': True
                    }
                if direction == 'CALL' and pos_pct > 88.0:
                    return {
                        'symbol': sym_clean,
                        'passed': False,
                        'failed_gate': f'CALL_EXHAUSTION_ZONE ({pos_pct:.1f}% of Day Range)',
                        'is_index': True
                    }
                if direction == 'PUT' and pos_pct < 12.0:
                    return {
                        'symbol': sym_clean,
                        'passed': False,
                        'failed_gate': f'PUT_EXHAUSTION_ZONE ({pos_pct:.1f}% of Day Range)',
                        'is_index': True
                    }

            # Range Expansion RVOL (1.20x threshold captures genuine Index breakouts)
            prior_rng = (comp_bars['high'] - comp_bars['low']).mean()
            idx_rvol = round(rng / max(0.5, prior_rng), 2) if prior_rng > 0 else 2.50
            if idx_rvol < 1.20:
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': f'INDEX_RANGE_EXPANSION_RVOL_LOW ({idx_rvol:.2f}x < 1.20x)',
                    'is_index': True
                }

            score = min(99.0, 50.0 + min(25.0, (idx_rvol - 1.0) * 10.0) + (15.0 if abs(rec_cvd) >= 0.30 else 5.0) + (10.0 if solid_body >= 0.60 else 0.0))

            return {
                'symbol': sym_clean,
                'passed': True,
                'spot_price': cur_p,
                'direction': direction,
                'pct_change': pct_chg,
                'rvol': idx_rvol,
                'score': round(score, 2),
                'is_index': True,
                'stock_atr': 0.0,
                'is_compression_breakout': True,
                'cvd_info': cvd_res
            }

        # -------------------------------------------------------------
        # STAGE 1 CANDLE DISCOVERY FOR STOCKS
        # -------------------------------------------------------------
        # Sweet Spot Scoring: 0.60% to 2.20% move = 30.0 maximum points!
        base_score = 40.0
        abs_p = abs(pct_chg)
        if abs_p < 0.60:
            mom_pts = round((abs_p / 0.60) * 22.0, 1)
        elif abs_p <= 2.20:
            mom_pts = 30.0
        else:
            mom_pts = max(10.0, round(30.0 - (abs_p - 2.20) * 25.0, 1))

        vol_pts = min(30.0, (rvol - 1.0) * 12.0) if rvol > 1.0 else 0.0
        score = round(min(99.0, base_score + mom_pts + vol_pts), 2)

        # Preliminary direction from price momentum (Stage 2 will strictly verify 3m candle breakout!)
        tentative_direction = 'CALL' if pct_chg >= 0 else 'PUT'

        return {
            'symbol': sym_clean,
            'passed': True,
            'spot_price': cur_p,
            'direction': tentative_direction,
            'pct_change': pct_chg,
            'rvol': rvol,
            'score': score,
            'is_index': False,
            'stock_atr': stock_atr
        }

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
        """Maintains Daily & Monthly Official Excel Journals for every qualified signal dispatched."""
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
        Evaluates active open positions on every 3-minute iteration:
        - Target 1 hit -> Trails SL to Cost (Breakeven lock).
        - Target 2 hit -> Books TARGET_2_HIT (JACKPOT) (+3.5R).
        - Trailing SL hit -> Exits at COST_TRAILED_EXIT (0.00R loss).
        - Initial SL hit -> Books SL_HIT (-1.00R).
        - 15-Minute Stagnant Kill Switch -> Exits at 15M_TIME_SL_EXIT.
        Updates CSV and Excel journals in place and broadcasts exit notifications!
        """
        csv_path = os.path.join(CURRENT_DIR, 'Live_Paper_Trading_Journal.csv')
        daily_xl = os.path.join(CURRENT_DIR, 'Alpha_5Pillar_Live_Paper_Trading_Journal.xlsx')
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
                if not q:
                    continue

                cur_spot = float(q.get('live_spot', q.get('close', spot_entry)))
                if cur_spot <= 0 or spot_entry <= 0:
                    continue

                # Estimate current option price using genuine delta
                delta = 0.52 if is_idx else 0.50
                if 'CE' in contract or 'CALL' in opt_action:
                    spot_gain = cur_spot - spot_entry
                else:
                    spot_gain = spot_entry - cur_spot

                est_opt_price = round(max(0.5, opt_entry + (spot_gain * delta)), 2)

                # Try live option chain quote if available
                oc_data = self.data_feed._oc_cache.get(sym)
                if oc_data and 'records' in oc_data:
                    for c_item in oc_data['records'].get('data', []):
                        for side in ['CE', 'PE']:
                            if c_item.get(side, {}).get('identifier') == contract or c_item.get(side, {}).get('strikePrice') == row.get('Spot_Entry'):
                                ltp_val = float(c_item.get(side, {}).get('lastPrice', 0.0))
                                if ltp_val > 0:
                                    est_opt_price = ltp_val
                                    break

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
                # 2. T1 Profit Locked Exit (Eliminates Breakeven 0.00R loss)
                elif trail_sl > opt_entry and est_opt_price <= trail_sl:
                    exit_price = trail_sl
                    outcome = "T1_PROFIT_LOCKED"
                # 3. Initial Stop Loss Hit
                elif est_opt_price <= opt_sl:
                    exit_price = opt_sl
                    outcome = "SL_HIT"
                # 4. 15-Minute Stagnant Kill Switch
                else:
                    try:
                        sig_time_str = str(row['Signal_Time_IST'])
                        # Parse time
                        sig_dt = datetime.datetime.strptime(sig_time_str.split(' ')[0], '%I:%M:%S')
                        sig_mins = sig_dt.hour * 60 + sig_dt.minute
                        cur_mins = ist_now.hour * 60 + ist_now.minute
                        if (cur_mins - sig_mins) >= 15 and est_opt_price <= opt_entry:
                            exit_price = est_opt_price
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
                    self.notifier.send_broadcast_message(exit_msg, is_trade_signal=False)
                    print(f"[TRADE CLOSED]: {contract} -> {outcome} ({pts_cap:+.2f} pts | Rs {net_pnl:+,.2f})")

            if updated_any:
                df.to_csv(csv_path, index=False)
                try:
                    df.to_excel(daily_xl, index=False)
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
        print(f"=======================================================")

        # First: Monitor and manage active open paper positions!
        self.update_open_paper_trades()

        if not ignore_window and not is_in_active_trading_slab(ist_now):
            print(f"[OUTSIDE TRADING SLAB]: Current time {ist_time_str} is outside active trading slabs. Maintaining complete silence.")
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
            return

        # STAGE 1: Consolidated NSE Bulk Snapshot Fetch
        fetch_bulk_nse_snapshot(force_refresh=True)
        nifty_regime = self.get_nifty_market_regime(force_refresh=True)
        print(
            f"[MACRO REGIME]: AllowStockCALL={nifty_regime['allow_stock_call']} | "
            f"AllowStockPUT={nifty_regime['allow_stock_put']} | {nifty_regime['reason']}"
        )

        symbols = [x['symbol'] for x in self.universe]
        with ThreadPoolExecutor(max_workers=8) as executor:
            eval_results = list(executor.map(self.evaluate_asset, symbols))

        # Stage 1 Qualification Filter
        stage1_passed = [
            r for r in eval_results 
            if r.get('passed', False) and (
                r.get('score', 0.0) >= 70.0 or 
                (float(r.get('rvol', 0.0)) >= 2.0 and abs(float(r.get('pct_change', 0.0))) >= 0.50)
            )
        ]

        # Prioritize Indices first, then high Setup Strength Score + RVOL
        stage1_passed.sort(
            key=lambda x: (
                1 if x.get('is_index', False) else 0,
                x.get('score', 0.0),
                x.get('rvol', 0.0)
            ),
            reverse=True
        )

        print(f"[STAGE-1 COMPLETE]: Scanned {len(symbols)} assets. {len(stage1_passed)} candidates passed discovery.")

        # STAGE 2: Real 3-Minute Breakout, Live Spot Holding & BVC CVD Audit
        verified_super_trades = []
        stage2_rejections = {}
        active_slab_index = morn_idx_count if is_morning else aft_idx_count
        active_slab_stock = morn_stk_count if is_morning else aft_stk_count
        remaining_slab_slots = max(0, MAX_SLAB_TRADES - active_slab_count)
        slab_cap = min(MAX_SLAB_TRADES, remaining_slab_slots)

        for cand in stage1_passed:
            if len(verified_super_trades) >= slab_cap:
                break

            sym = cand['symbol']
            is_idx = cand.get('is_index', False)

            if is_idx and active_slab_index >= MAX_INDEX_PER_SLAB:
                stage2_rejections[sym] = "MAX_INDEX_PER_SLAB_CAP_REACHED"
                continue
            if not is_idx and active_slab_stock >= MAX_STOCKS_PER_SLAB:
                stage2_rejections[sym] = "MAX_STOCK_PER_SLAB_CAP_REACHED"
                continue

            if self.is_already_dispatched(sym):
                stage2_rejections[sym] = "ALREADY_DISPATCHED_TODAY"
                continue

            # Index verification (already verified 3m breakout & CVD in Stage 1)
            if is_idx:
                direction = cand['direction']
                if direction == 'CALL' and not nifty_regime.get('allow_stock_call', True):
                    stage2_rejections[sym] = f"MACRO_TIDE_BLOCKED_INDEX_CALL: {nifty_regime.get('reason')}"
                    continue
                if direction == 'PUT' and not nifty_regime.get('allow_stock_put', True):
                    stage2_rejections[sym] = f"MACRO_TIDE_BLOCKED_INDEX_PUT: {nifty_regime.get('reason')}"
                    continue
            else:
                # ---------------------------------------------------------
                # STAGE 2 STOCK 3-MINUTE AUDIT
                # ---------------------------------------------------------
                stk_df = self.data_feed.get_live_intraday_bars(sym)
                if stk_df is None or len(stk_df) < 10:
                    stage2_rejections[sym] = "INSUFFICIENT_3M_BARS_LESS_THAN_10"
                    continue

                eval_bar = stk_df.iloc[-2]
                live_bar = stk_df.iloc[-1]
                b_open = float(eval_bar['open'])
                b_high = float(eval_bar['high'])
                b_low = float(eval_bar['low'])
                b_close = float(eval_bar['close'])

                l_open = float(live_bar['open'])
                l_high = float(live_bar['high'])
                l_low = float(live_bar['low'])
                l_close = float(live_bar['close'])

                live_q = self.data_feed.get_live_asset_quote(sym)
                spot = float(live_q.get('live_spot', l_close))
                cand['spot_price'] = spot

                # User Rule: Strictly High-Value Stocks (Spot >= 1000 Rs)
                if spot < 1000.0:
                    stage2_rejections[sym] = f"STOCK_PRICE_BELOW_1000_INR (Spot {spot:.1f} < 1000.0)"
                    continue

                stock_atr = float(cand.get('stock_atr', self.get_stock_atr(sym, live_q)))
                rng = max(0.01, b_high - b_low)

                # 1. 8-Bar Compression Base
                comp_bars = stk_df.iloc[:-2].tail(8)
                comp_high = float(comp_bars['high'].max())
                comp_low = float(comp_bars['low'].min())
                max_dist_buffer = stock_atr * 0.20

                # Direction strictly determined by 3-minute breakout level!
                direction = None
                if b_close >= comp_high and b_close <= comp_high + max_dist_buffer:
                    direction = 'CALL'
                elif b_close <= comp_low and b_close >= comp_low - max_dist_buffer:
                    direction = 'PUT'

                if direction is None:
                    stage2_rejections[sym] = "NO_8BAR_COMPRESSION_BREAKOUT"
                    continue

                # 2. Day High / Day Low Breakout Check (Ensures genuine extreme expansion)
                day_high = float(stk_df['high'].max())
                day_low = float(stk_df['low'].min())
                if direction == 'CALL' and b_close < (day_high - stock_atr * 0.08):
                    stage2_rejections[sym] = "NOT_DAY_HIGH_BREAKOUT"
                    continue
                if direction == 'PUT' and b_close > (day_low + stock_atr * 0.08):
                    stage2_rejections[sym] = "NOT_DAY_LOW_BREAKOUT"
                    continue

                # 3. Exhaustion from Open Check (Prevents chasing exhausted moves > 3.2%)
                day_open = float(stk_df['open'].iloc[0])
                pct_from_open = abs((spot - day_open) / day_open * 100.0)
                if pct_from_open > 3.20:
                    stage2_rejections[sym] = f"EXHAUSTED_MOVE_FROM_OPEN ({pct_from_open:.1f}% > 3.2%)"
                    continue

                # 4. Live Spot Holding Re-Confirmation (Eliminates false breakouts!)
                spot_buffer = stock_atr * 0.05
                if direction == 'CALL':
                    live_holding = (spot >= comp_high - spot_buffer) and ((l_high - spot) <= stock_atr * 0.25)
                else:
                    live_holding = (spot <= comp_low + spot_buffer) and ((spot - l_low) <= stock_atr * 0.25)

                if not live_holding:
                    stage2_rejections[sym] = f"LIVE_SPOT_FAILED_HOLDING (Spot {spot:.2f} vs Base Level)"
                    continue

                # 5. Local Base Runway Check (Measures expansion from local base, NOT whole day!)
                base_mean = float(comp_bars['close'].mean())
                local_expansion = abs(spot - base_mean)
                if local_expansion > (stock_atr * 0.70):
                    stage2_rejections[sym] = f"LOCAL_RUNWAY_EXHAUSTED ({local_expansion:.1f} > 70% of ATR)"
                    continue

                # 6. Wyckoff Effort vs Result & Body Strength
                body = abs(b_close - b_open)
                solid_body_ratio = round(body / rng, 2)
                clv = round((b_close - b_low) / rng, 2)
                min_thrust = max(4.0, round(stock_atr * 0.08, 1))

                if solid_body_ratio < 0.55 or body < min_thrust:
                    stage2_rejections[sym] = f"WEAK_CANDLE_THRUST (Body {body:.1f} < {min_thrust:.1f}pts)"
                    continue

                # 7. Price Action CLV Quality
                clv_pass = (clv >= 0.65) if direction == 'CALL' else (clv <= 0.35)
                if not clv_pass:
                    stage2_rejections[sym] = f"CLV_QUALITY_MISALIGNED (CLV={clv:.2f})"
                    continue

                # 8. True Bulk Volume Classification (BVC) Cumulative Volume Delta
                cvd_res = compute_bvc_cvd(stk_df)
                rec_cvd = cvd_res['recent_cvd_ratio']
                sess_cvd = cvd_res['session_cvd_ratio']

                cvd_pass = (rec_cvd >= 0.15) if direction == 'CALL' else (rec_cvd <= -0.15)
                if not cvd_pass:
                    stage2_rejections[sym] = f"BVC_CVD_MISALIGNED (RecCVD={rec_cvd:+.2f})"
                    continue

                # 9. True Institutional RVOL & Absolute Turnover (Aladdin Model)
                # Measures volume against clean morning baseline (09:15-10:00 AM) to eliminate self-inflation
                curr_3m_v = float(eval_bar.get('volume', 0.0))
                morning_bars = stk_df.iloc[:15]
                morning_base_v = float(morning_bars['volume'].mean()) if len(morning_bars) > 0 else 1000.0
                real_3m_rvol = round(curr_3m_v / max(1.0, morning_base_v), 2)
                turnover_cr = round((curr_3m_v * spot) / 1e7, 2)

                # Institutional Floor: Minimum Rs 2.0 Cr turnover and RVOL >= 2.0x (or ultra-liquidity >= 4.0 Cr)
                if turnover_cr < 2.0:
                    stage2_rejections[sym] = f"TURNOVER_BELOW_INSTITUTIONAL_FLOOR (Rs. {turnover_cr:.1f}Cr < 2.0Cr)"
                    continue
                if real_3m_rvol < 2.0 and turnover_cr < 4.0:
                    stage2_rejections[sym] = f"RVOL_INSUFFICIENT ({real_3m_rvol:.2f}x < 2.0x, Turn Rs. {turnover_cr:.1f}Cr)"
                    continue

                cand['direction'] = direction
                cand['rvol'] = real_3m_rvol
                cand['turnover_cr'] = turnover_cr
                cand['cvd_info'] = cvd_res

            # -------------------------------------------------------------
            # STAGE 3: LIVE OPTION CHAIN, OI CONFLUENCE & REAL LTP AUDIT
            # -------------------------------------------------------------
            direction = cand['direction']
            oi_conf = self.data_feed.get_live_dual_oi_confluence(sym, cand['spot_price'], direction=direction)
            if not oi_conf.get('confluent', False):
                stage2_rejections[sym] = f"OI_RUNWAY_REJECTED: {oi_conf.get('reason')}"
                continue

            opt = self.data_feed.compute_option_pricing(cand['spot_price'], direction=direction, symbol=sym)
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

            if is_idx: active_slab_index += 1
            else: active_slab_stock += 1

        # Log Full Diagnostic Audit
        self._log_rejection_audit(ist_now, eval_results, verified_super_trades, stage2_rejections)

        if not verified_super_trades:
            print(f"[SCREENER]: 0 signals qualified after full 217 scan & live OI audit at {ist_time_str}. Capital 100% Protected.")
            return

        print(f"[SCREENER]: Verified {len(verified_super_trades)} INSTITUTIONAL SUPER TRADE(S)! Dispatching...")

        for cand in verified_super_trades:
            sym = cand['symbol']
            spot = cand['spot_price']
            direction = cand['direction']
            is_idx = cand.get('is_index', False)
            opt = cand['verified_opt']

            contract = opt['contract']
            opt_entry = float(opt['entry_price'])
            opt_action = 'BUY CALL (CE)' if direction == 'CALL' else 'BUY PUT (PE)'

            # Fresh 4-Minute Execution Buffer calculated right at dispatch time!
            dispatch_now = get_ist_time()
            buffer_end_time = (dispatch_now + datetime.timedelta(minutes=4)).strftime('%I:%M:%S %p IST')
            time_sl_time = (dispatch_now + datetime.timedelta(minutes=15)).strftime('%I:%M:%S %p IST')

            # Realistic Option Targets calibrated via Delta (0.52 for Index, Risk:Reward for Stock)
            if is_idx:
                delta = 0.52
                spot_t1 = 140.0 if sym == 'BANKNIFTY' else (70.0 if sym in ['NIFTY', 'FINNIFTY'] else 40.0)
                spot_t2 = 240.0 if sym == 'BANKNIFTY' else (120.0 if sym in ['NIFTY', 'FINNIFTY'] else 70.0)
                spot_sl = 70.0 if sym == 'BANKNIFTY' else (35.0 if sym in ['NIFTY', 'FINNIFTY'] else 20.0)

                opt_sl = round(max(1.0, opt_entry - (spot_sl * delta)), 2)
                opt_tgt1 = round(opt_entry + (spot_t1 * delta), 2)
                opt_tgt2 = round(opt_entry + (spot_t2 * delta), 2)
            else:
                stk_atr = cand.get('stock_atr', spot * 0.022)
                spot_t1 = round(stk_atr * 0.35, 1)
                spot_t2 = round(stk_atr * 0.70, 1)
                spot_sl = round(stk_atr * 0.20, 1)

                opt_sl = round(max(1.0, opt_entry * 0.80), 2)  # 20% Option SL
                opt_sl_pts = opt_entry - opt_sl
                opt_tgt1 = round(opt_entry + (opt_sl_pts * 1.50), 2)  # 1:1.5 R:R
                opt_tgt2 = round(opt_entry + (opt_sl_pts * 3.50), 2)  # 1:3.5 R:R Jackpot

            pure_signal_msg = (
                f"🚀 <b>ALPHA 5-PILLAR INSTITUTIONAL TRADE</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>INSTRUMENT:</b> <b>{sym} ({direction})</b>\n"
                f"⚡ <b>ACTION:</b> <b>{opt_action}</b>\n"
                f"📌 <b>CONTRACT:</b> <code>{contract}</code>\n"
                f"💵 <b>BUYING PRICE:</b> <b>Rs. {opt_entry:,.2f}</b>\n"
                f"🛡️ <b>STOP LOSS:</b> <b>Rs. {opt_sl:,.2f}</b>\n"
                f"🎯 <b>TARGET 1 (Lock Cost):</b> <b>Rs. {opt_tgt1:,.2f}</b>\n"
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

    def run_live_slab_session(self, interval_seconds=180):
        """
        Runs a persistent single-runner session throughout the active trading slab.
        Executes immediate scan, then sleeps interval_seconds (default 180s / 3 mins).
        """
        ist_now = get_ist_time()
        tot_m = ist_now.hour * 60 + ist_now.minute

        if 570 <= tot_m <= 615:
            slab_name = "MORNING SLAB (09:30 - 10:15 AM IST)"
            end_min = 615
        elif 795 <= tot_m <= 855:
            slab_name = "AFTERNOON SLAB (01:15 - 02:15 PM IST)"
            end_min = 855
        else:
            print(f"[SLAB SESSION]: Current time {ist_now.strftime('%I:%M %p IST')} is outside active slabs (09:30-10:15 AM & 01:15-02:15 PM IST). No trades dispatched.")
            return

        print(f"=======================================================")
        print(f"  STARTING PERSISTENT SLAB RUNNER: {slab_name}")
        print(f"  Cadence: Exact {interval_seconds}s (3-Minute Resampled Candles)")
        print(f"=======================================================")

        while True:
            cur_time = get_ist_time()
            cur_mins = cur_time.hour * 60 + cur_time.minute
            if cur_mins > end_min:
                print(f"[SLAB CONCLUDED]: {slab_name} completed at {cur_time.strftime('%I:%M:%S %p IST')}.")
                break

            self.run_screener(ignore_window=True)
            print(f"[SLEEP]: Waiting {interval_seconds}s for next 3-minute candle close...")
            time.sleep(interval_seconds)


if __name__ == '__main__':
    screener = DailyAutonomousScreener()
    if '--slab-session' in sys.argv:
        screener.run_live_slab_session(interval_seconds=180)
    else:
        ignore_win = ('--ignore-window' in sys.argv)
        screener.run_screener(ignore_window=ignore_win)
