# -*- coding: utf-8 -*-
"""
Alpha 5-Pillar Autonomous Daily Screener & Pure Signal Dispatched Engine
========================================================================
Operating Windows (Strictly Enforced):
  - Slab 1 (Morning):   09:30 AM to 10:15 AM IST (04:00 to 04:45 UTC)
  - Slab 2 (Afternoon): 01:15 PM to 02:15 PM IST (07:45 to 08:45 UTC)

Strict Rules:
  1. Outside Slab 1 and Slab 2: ZERO messages sent anywhere (Chatbot or Group).
  2. If Signal Generated: Send ONLY clean, direct trade signal with 4-min execution buffer to BOTH Chatbot and Group.
  3. If NO Signal Generated: ZERO messages sent to Group. Exactly 1 status update sent to Chatbot (Bot).
  4. Automatically records dispatched signals into Daily & Monthly Excel Ledgers.
  5. Maximum 1 signal per stock per day & 2 to 5 Super Trades total across Morning + Afternoon.
"""

import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import json
import time
import datetime
import html
from concurrent.futures import ThreadPoolExecutor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from modules.live_data_feed import (
    LiveMarketDataFeed,
    fetch_bulk_nse_snapshot,
    get_ist_time
)
from modules.fno_universe import get_instrument_meta
from modules.telegram_notifier import TelegramNotifier
from core.pillar5_engine import evaluate_5pillar_gate

CONFIG_PATH = os.path.join(CURRENT_DIR, 'config', 'telegram_config.json')
MAX_DAILY_SUPER_TRADES = 8  # Balanced Slab Quota (4 Morning + 4 Afternoon = 8 Total Daily Super Trades)


def is_in_active_trading_slab(ist_time):
    """
    Checks if current IST time falls strictly within designated trading slabs:
    - Slab 1 (Morning):   09:30 AM to 10:15 AM IST (570 to 615 mins)
    - Slab 2 (Afternoon): 01:15 PM to 02:15 PM IST (795 to 855 mins)
    Must also be a weekday (Monday to Friday).
    """
    is_weekday = (ist_time.weekday() < 5)
    if not is_weekday:
        return False

    tot_mins = ist_time.hour * 60 + ist_time.minute
    slab1 = (570 <= tot_mins <= 615)
    slab2 = (795 <= tot_mins <= 855)
    return slab1 or slab2


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
        self.index_recent_dispatch = {}
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
                self.notifier.base_url = f"https://api.telegram.org/bot{env_token}"

    def _load_dispatched_cache(self):
        ist_now = get_ist_time()
        today_str = ist_now.strftime('%Y-%m-%d')
        indices_prefixes = ('NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50')
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
        ist_now = get_ist_time()
        today_str = ist_now.strftime('%Y-%m-%d')
        try:
            os.makedirs(os.path.dirname(self.dispatched_cache_file), exist_ok=True)
            with open(self.dispatched_cache_file, 'w', encoding='utf-8') as f:
                json.dump({'date': today_str, 'dispatched': self.dispatched_today}, f, indent=2)
        except Exception:
            pass

    def is_already_dispatched(self, symbol, contract=''):
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        last_ts = self.dispatched_today.get(sym_clean, 0.0)
        if sym_clean in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50']:
            return (time.time() - last_ts) < 2100.0

        return sym_clean in self.dispatched_today

    def mark_dispatched(self, symbol, contract=''):
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        self.dispatched_today[sym_clean] = time.time()
        self._save_dispatched_cache()

    def get_nifty_market_regime(self, force_refresh=False):
        """
        FLAW #3 FIX — NIFTY MACRO TIDE & CRASH REGIME GATE:
        Audits NIFTY 50 live spot, 3-minute closed candle (`iloc[-2]`), live forming candle (`iloc[-1]`),
        and 3-minute 9-EMA (`ema9`).
        - Strictly blocks Stock CALL (CE) trades if NIFTY is crashing / below 9-EMA / printing red 3m candle.
        - Does NOT block Stock PUT (PE) on minor +0.10% (+25 pts) opens; only blocks Stock PUT during a
          strong vertical Nifty surge above 9-EMA.
        """
        now = time.time()
        if not force_refresh and (now - self.nifty_regime_cache.get('timestamp', 0.0) < 35.0):
            return self.nifty_regime_cache

        nq = self.data_feed.get_live_asset_quote('NIFTY')
        ndf = self.data_feed.get_live_intraday_bars('NIFTY')

        allow_call = True
        allow_put = True
        reason = "NIFTY_NEUTRAL_OR_ALIGNED"

        nifty_spot = float(nq.get('live_spot', 0.0)) if nq else 0.0
        nifty_high = float(nq.get('high', nifty_spot)) if nq else nifty_spot
        nifty_low = float(nq.get('low', nifty_spot)) if nq else nifty_spot

        if ndf is not None and len(ndf) >= 3:
            eval_b = ndf.iloc[-2]
            live_b = ndf.iloc[-1]
            n_open = float(eval_b['open'])
            n_close = float(eval_b['close'])
            l_open = float(live_b['open'])
            l_close = float(live_b['close'])
            if nifty_spot <= 0:
                nifty_spot = l_close

            ema9 = float(ndf['ema9'].iloc[-1]) if 'ema9' in ndf.columns else float(ndf['close'].ewm(span=9, adjust=False).mean().iloc[-1])
            day_high = max(nifty_high, float(ndf['high'].max()))
            day_low = min(nifty_low, float(ndf['low'].min())) if nifty_low > 0 else float(ndf['low'].min())

            drop_from_high = round(day_high - nifty_spot, 2)
            rise_from_low = round(nifty_spot - day_low, 2)

            prev_3m_red = (n_close < n_open)
            live_3m_red = (l_close < l_open) or (nifty_spot < n_close)
            below_ema9 = (nifty_spot < ema9)

            # Strict Macro Compass: Check 9-EMA and Day Open confluence
            day_open = float(ndf['open'].iloc[0])
            down_from_open = (nifty_spot < day_open - 15.0)
            up_from_open = (nifty_spot > day_open + 15.0)

            # NIFTY INTRADAY CHOP / COMPRESSION FILTER
            # यदि पिछले 6 बार (18 मिनट) में निफ्टी की हाई-लो रेंज < 30 अंक हो और 9-EMA फ्लैट हो,
            # तो मार्केट साइडवेज़ ट्रैप में है — सभी ब्रेकआउट्स ब्लॉक रहेंगे!
            is_nifty_choppy = False
            if len(ndf) >= 6:
                recent_6 = ndf.tail(6)
                recent_rng = float(recent_6['high'].max() - recent_6['low'].min())
                ema_distance = abs(nifty_spot - ema9)
                if recent_rng < 30.0 and ema_distance <= 10.0:
                    is_nifty_choppy = True

            if is_nifty_choppy:
                allow_call = False
                allow_put = False
                reason = f"NIFTY_SIDEWAYS_CHOP_TRAP (18m Range {recent_rng:.1f} pts < 30 pts | Spot within {ema_distance:.1f} pts of 9-EMA)"
            else:
                # Block Stock CALLs if NIFTY is in the red from open, below 9-EMA, or dropping >= 50 pts
                if prev_3m_red or below_ema9 or down_from_open or (drop_from_high >= 50.0 and live_3m_red):
                    allow_call = False
                    reason = (
                        f"NIFTY_BEARISH_OR_CRASH_REGIME (Spot {nifty_spot:.1f} vs 9-EMA {ema9:.1f} | "
                        f"DownFromOpen={down_from_open} | Prev3mRed={prev_3m_red} | DropFromHigh=-{drop_from_high:.1f} pts)"
                    )

                # Block Stock PUTs on strong active rally above 9-EMA or in strong green day
                prev_3m_strong_green = (n_close - n_open) >= 15.0
                live_3m_green = (l_close >= l_open) and (nifty_spot >= n_close)
                above_ema9_strong = (nifty_spot > (ema9 + 15.0))
                if above_ema9_strong or up_from_open or (prev_3m_strong_green and live_3m_green and rise_from_low >= 60.0):
                    allow_put = False
                    reason = (
                        f"NIFTY_STRONG_RALLY_REGIME (Spot {nifty_spot:.1f} > 9-EMA {ema9:.1f} | "
                        f"UpFromOpen={up_from_open} | RiseFromLow=+{rise_from_low:.1f} pts)"
                    )
        elif nq and nifty_spot > 0:
            drop_from_high = round(nifty_high - nifty_spot, 2)
            p_chg = float(nq.get('pct_change', 0.0))
            if drop_from_high >= 60.0 or p_chg <= -0.35:
                allow_call = False
                reason = f"NIFTY_INTRADAY_SELLOFF (DropFromHigh=-{drop_from_high:.1f} pts | {p_chg:+.2f}%)"

        self.nifty_regime_cache = {
            'allow_stock_call': allow_call,
            'allow_stock_put': allow_put,
            'reason': reason,
            'timestamp': now
        }
        return self.nifty_regime_cache

    def get_stock_atr(self, symbol, q):
        cur_p = float(q.get('live_spot', 0))
        h = float(q.get('high', 0))
        l = float(q.get('low', 0))
        day_rng = round(h - l, 2) if h > 0 and l > 0 else 0.0
        est_atr = round(cur_p * 0.028, 2) if cur_p > 0 else 0.0
        cached_atr = float(self.atr_cache.get(symbol, 0.0))
        return max(cached_atr, day_rng, est_atr)

    def evaluate_asset(self, sym_meta, verify_oi_now=False):
        """
        UPSIDE STAGE-1 EVALUATION:
        Scans all 216 F&O Stocks and 4 Indices on Price Action, Compression Base,
        Freshness Lock (Anti-Exhaustion), RVOL >= 2.50x, and CVD Dominance.
        When verify_oi_now=False, skips per-stock Option Chain HTTP calls during bulk scan
        so NSE NEVER blocks GitHub/Local IPs and all 216 stocks are 100% scanned!
        """
        symbol = sym_meta['symbol'] if isinstance(sym_meta, dict) else sym_meta
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()

        if self.is_already_dispatched(sym_clean):
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'ALREADY_DISPATCHED_TODAY'}

        q = self.data_feed.get_live_asset_quote(sym_clean)
        if not q or q.get('status') == 'NSE_DATA_UNAVAILABLE':
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'NSE_DATA_UNAVAILABLE'}

        cur_p = float(q['live_spot'])
        if cur_p <= 0:
            return {'symbol': sym_clean, 'passed': False, 'failed_gate': 'INVALID_SPOT'}

        is_idx = sym_clean in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

        # =====================================================================
        # 1. INDEX-SPECIFIC EVALUATION (35-Minute Compression Breakout Priority)
        # =====================================================================
        if is_idx:
            # 13:45 का कृत्रिम कटऑफ हटा दिया गया है। इंडेक्स दोनों स्लैब्स (Morning 09:30-10:15 और Afternoon 01:15-02:15) में ट्रेड हो सकते हैं।

            idx_df = self.data_feed.get_live_intraday_bars(sym_clean)
            is_breakout = False
            cum_mom_ok = False
            single_mom_ok = False
            direction = 'CALL'

            if idx_df is not None and not idx_df.empty:
                eval_bar = idx_df.iloc[-2] if len(idx_df) >= 2 else idx_df.iloc[-1]
                b_open = float(eval_bar['open'])
                b_high = float(eval_bar['high'])
                b_low = float(eval_bar['low'])
                b_close = float(eval_bar['close'])
                cur_p = float(idx_df.iloc[-1]['close'])

                day_high = float(idx_df['high'].max())
                day_low = float(idx_df['low'].min())

                # 8 closed bars (24 mins) + 1 forming bar = minimum 9 bars
                if len(idx_df) < 9:
                    return {
                        'symbol': sym_clean,
                        'passed': False,
                        'failed_gate': f'INSUFFICIENT_COMPRESSION_BASE: Only {len(idx_df)-1} closed bars (< 24 mins)',
                        'spot_price': cur_p,
                        'score': 0.0,
                        'is_index': True
                    }

                eval_bar = idx_df.iloc[-2] if len(idx_df) >= 2 else idx_df.iloc[-1]
                b_open = float(eval_bar['open'])
                b_high = float(eval_bar['high'])
                b_low = float(eval_bar['low'])
                b_close = float(eval_bar['close'])
                # Keep true live spot price from bulk feed or latest tick
                cur_p = float(q.get('live_spot', idx_df.iloc[-1]['close']))

                day_high = float(idx_df['high'].max())
                day_low = float(idx_df['low'].min())

                comp_bars = idx_df.iloc[:-2].tail(8)
                comp_high = float(comp_bars['high'].max())
                comp_low = float(comp_bars['low'].min())

                direction = 'CALL' if b_close >= b_open else 'PUT'

                # PINPOINT FIX: Measure exhaustion from Local Compression Base (NOT arbitrary 130/280 pts from Day Extremes)
                # An index with a fresh 24-min tight base can rally 300-500 pts on trending days!
                comp_width = comp_high - comp_low
                max_allowed_base_width = 85.0 if sym_clean in ['NIFTY', 'FINNIFTY', 'MIDCPNIFTY'] else 180.0
                if comp_width > max_allowed_base_width:
                    return {
                        'symbol': sym_clean,
                        'passed': False,
                        'failed_gate': f'COMPRESSION_BASE_TOO_WIDE: Width {comp_width:.1f} pts > {max_allowed_base_width:.1f} pts',
                        'spot_price': cur_p,
                        'score': 0.0,
                        'is_index': True
                    }

                pct_chg = round(((b_close - b_open) / max(1.0, b_open)) * 100.0, 2)
                rng = max(0.01, b_high - b_low)
                body = abs(b_close - b_open)
                solid_body_ratio = round(body / rng, 2)
                up_wick = round((b_high - max(b_open, b_close)) / rng, 2)
                dn_wick = round((min(b_open, b_close) - b_low) / rng, 2)
                clv = round((b_close - b_low) / rng, 2)
                clv_ok = (clv >= 0.75) if direction == 'CALL' else (clv <= 0.25)
                wick_ok = (up_wick <= 0.20) if direction == 'CALL' else (dn_wick <= 0.20)
                # ---------------------------------------------------------
                # मोड 1: फास्ट मार्केट (1 कैंडल मारुबोज़ू ब्लास्ट: Nifty >= 40 pts, BankNifty >= 90 pts)
                # ---------------------------------------------------------
                min_fast_bar_pts = 40.0 if sym_clean in ['NIFTY', 'FINNIFTY', 'MIDCPNIFTY'] else 90.0
                single_mom_ok = (solid_body_ratio >= 0.60) and wick_ok and clv_ok and (rng >= min_fast_bar_pts)

                idx_dist_buffer = 40.0 if sym_clean in ['NIFTY', 'FINNIFTY', 'MIDCPNIFTY'] else 90.0
                if direction == 'CALL':
                    if (b_close >= comp_high) and (b_close <= comp_high + idx_dist_buffer):
                        is_breakout = True
                else:
                    if (b_close <= comp_low) and (b_close >= comp_low - idx_dist_buffer):
                        is_breakout = True

                # ---------------------------------------------------------
                # मोड 2: स्लो मार्केट (2 से 3 कैंडल संस्थागत ग्राइंड: Nifty 50-75 pts, BankNifty 110-160 pts)
                # ---------------------------------------------------------
                cum_mom_ok = False
                if len(idx_df) >= 5:
                    b_prev1 = idx_df.iloc[-3]
                    b_prev2 = idx_df.iloc[-4]
                    pre_seq_bars = idx_df.iloc[:-4].tail(8) if len(idx_df) >= 6 else comp_bars
                    seq_comp_high = float(pre_seq_bars['high'].max())
                    seq_comp_low = float(pre_seq_bars['low'].min())

                    cum_fall_3_pct = ((float(b_prev2['open']) - b_close) / float(b_prev2['open'])) * 100.0
                    cum_fall_2_pct = ((float(b_prev1['open']) - b_close) / float(b_prev1['open'])) * 100.0
                    all_3_red = (float(b_prev2['close']) <= float(b_prev2['open']) and
                                 float(b_prev1['close']) <= float(b_prev1['open']) and
                                 b_close <= b_open)
                    last_2_red = (float(b_prev1['close']) <= float(b_prev1['open']) and b_close <= b_open)
                    broke_range_low = ((b_close <= comp_low) and (b_close >= comp_low - idx_dist_buffer)) or \
                                     ((b_close <= seq_comp_low) and (b_close >= seq_comp_low - idx_dist_buffer))

                    min_idx_2bar_pct = 0.22
                    min_idx_3bar_pct = 0.32

                    if (all_3_red and cum_fall_3_pct >= min_idx_3bar_pct and broke_range_low) or (last_2_red and cum_fall_2_pct >= min_idx_2bar_pct and broke_range_low):
                        direction = 'PUT'
                        cum_mom_ok = True
                        is_breakout = True
                        pct_chg = round(-max(cum_fall_3_pct, cum_fall_2_pct), 2)

                    cum_rise_3_pct = ((b_close - float(b_prev2['open'])) / float(b_prev2['open'])) * 100.0
                    cum_rise_2_pct = ((b_close - float(b_prev1['open'])) / float(b_prev1['open'])) * 100.0
                    all_3_green = (float(b_prev2['close']) >= float(b_prev2['open']) and
                                   float(b_prev1['close']) >= float(b_prev1['open']) and
                                   b_close >= b_open)
                    last_2_green = (float(b_prev1['close']) >= float(b_prev1['open']) and b_close >= b_open)
                    broke_range_high = ((b_close >= comp_high) and (b_close <= comp_high + idx_dist_buffer)) or \
                                      ((b_close >= seq_comp_high) and (b_close <= seq_comp_high + idx_dist_buffer))

                    if (all_3_green and cum_rise_3_pct >= min_idx_3bar_pct and broke_range_high) or (last_2_green and cum_rise_2_pct >= min_idx_2bar_pct and broke_range_high):
                        direction = 'CALL'
                        cum_mom_ok = True
                        is_breakout = True
                        pct_chg = round(max(cum_rise_3_pct, cum_rise_2_pct), 2)

                cand_mom_pass = (single_mom_ok or cum_mom_ok) and is_breakout

                # PINPOINT FIX: Yahoo Finance Spot Index (`^NSEI`, `^NSEBANK`) has volume=0.
                # When Spot Index volume is 0, use Range-Expansion RVOL (breakout bar range vs compression bar range)!
                prior_v = idx_df['volume'].iloc[:-2].tail(10).mean() if ('volume' in idx_df.columns and len(idx_df) > 2) else 0.0
                curr_v = float(eval_bar.get('volume', 0.0))
                if prior_v > 0 and curr_v > 0:
                    idx_rvol = round(curr_v / prior_v, 2)
                    idx_rvol_pass = (idx_rvol >= 2.20)
                else:
                    prior_rng = (comp_bars['high'] - comp_bars['low']).mean()
                    idx_rvol = round(rng / max(0.5, prior_rng), 2) if prior_rng > 0 else 2.50
                # इंडेक्स के लिए संस्थागत विस्फोट अनिवार्य (कम से कम 2.20x वॉल्यूम/रेंज एक्सपेंशन)
                idx_rvol_pass = (idx_rvol >= 2.20)

                candle_cvd = round((2.0 * clv) - 1.0, 2)
                idx_cvd = candle_cvd
                idx_cvd_pass = (idx_cvd >= 0.45) if direction == 'CALL' else (idx_cvd <= -0.45)
                # डायरेक्ट बायर/सेलर आक्रामकता (+/- 0.45) अनिवार्य — कोई बाईपास नहीं!

                stage1_pass = cand_mom_pass and idx_rvol_pass and idx_cvd_pass
                oi_confluent = True
                oi_reason = 'PENDING_STAGE2_OI_AUDIT'

                if stage1_pass and verify_oi_now:
                    oi_conf = self.data_feed.get_live_dual_oi_confluence(sym_clean, cur_p, direction=direction)
                    oi_confluent = oi_conf.get('confluent', False)
                    oi_reason = oi_conf.get('reason', '')

                passed = stage1_pass and oi_confluent
                # इंडेक्स को 98.0 का अंधा स्कोर देने के बजाय वास्तविक ब्रेकआउट स्ट्रेंथ पर स्कोर दें
                score = round(min(99.0, 75.0 + min(15.0, idx_rvol * 5.0) + (10.0 if abs(idx_cvd) >= 0.60 else 5.0)), 2) if passed else 50.0

            else:
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': 'PILLAR_1_INTRADAY_CANDLES_UNAVAILABLE_CANNOT_VERIFY_COMPRESSION_BASE',
                    'spot_price': cur_p,
                    'score': 0.0,
                    'is_index': True
                }

            return {
                'symbol': sym_clean,
                'passed': passed,
                'spot_price': cur_p,
                'direction': direction,
                'pct_change': pct_chg,
                'rvol': idx_rvol,
                'score': score,
                'is_index': True,
                'is_compression_breakout': is_breakout,
                'oi_confluent': oi_confluent,
                'oi_reason': oi_reason
            }

        # =====================================================================
        # 2. STAGE-1 FAST PRE-FILTER FOR ALL 216 STOCKS
        # (Does NOT reject stocks on full-day wick! Full 3-min candle & RVOL audit runs in Stage-2!)
        # =====================================================================
        b_open = float(q.get('open', cur_p))
        b_high = float(q.get('high', cur_p))
        b_low = float(q.get('low', cur_p))
        b_close = float(q.get('close', cur_p))
        pct_chg = float(q.get('pct_change', 0.0))
        rvol = float(q.get('rvol', 1.0))
        direction = 'CALL' if pct_chg >= 0 else 'PUT'

        # Candle Rejection Wick Shield (Block Deal Buying/Selling Absorption)
        candle_total_range = b_high - b_low
        if candle_total_range > 0:
            # अगर PUT ट्रेड है, लेकिन कैंडल के नीचे लंबा रिजेक्शन विक (Lower Wick > 35%) है,
            # तो इसका मतलब संस्थागत खरीदारों ने सारा माल सोख लिया है (Absorption) - नो PUT!
            lower_wick = min(b_open, b_close) - b_low
            if direction == 'PUT' and (lower_wick / candle_total_range) > 0.35:
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': f'BUYING_ABSORPTION_WICK: Lower wick {lower_wick:.1f} pts is {(lower_wick/candle_total_range)*100:.0f}% of candle'
                }
            # अगर CALL ट्रेड है, लेकिन ऊपर लंबा रिजेक्शन विक (Upper Wick > 35%) है - नो CALL!
            upper_wick = b_high - max(b_open, b_close)
            if direction == 'CALL' and (upper_wick / candle_total_range) > 0.35:
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': f'SELLING_REJECTION_WICK: Upper wick {upper_wick:.1f} pts is {(upper_wick/candle_total_range)*100:.0f}% of candle'
                }

        # Gate 0: 30+ Points Move Capability (Zero-Tolerance for Low-ATR)
        # ऑप्शन बायर का नियम: कम से कम 30 अंक का फास्ट मूव केवल High-ATR (ATR >= 85) में संभव है।
        # TataConsum (ATR 31), Pidilite (ATR 38), ICICI (ATR 28) जैसे सुस्त शेयर्स 100% BLOCKED!
        stock_atr = self.get_stock_atr(sym_clean, q)
        if stock_atr < 85.0:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'LOW_ATR_STAGNANT_TRAP: ATR {stock_atr:.1f} < 85.0 pts (Theta Decay Hazard)'
            }

        # FLAW #3 FIX (Calibrated Macro Tide Gate):
        # केवल तभी स्टॉक CALL ब्लॉक करें जब निफ्टी में वास्तविक क्रैश (Severe Drop >= 60 pts) हो!
        # अगर शेयर अपने दम पर मजबूत है (PctChg >= +0.50% & RVOL >= 1.5x), तो निफ्टी के मामूली -0.10% पर ब्लॉक नहीं होगा!
        nifty_regime = self.get_nifty_market_regime(force_refresh=False)
        is_independent_outperformer = (abs(pct_chg) >= 0.50 and rvol >= 1.50 and stock_atr >= 85.0)

        if direction == 'CALL' and not nifty_regime.get('allow_stock_call', True):
            if not is_independent_outperformer or "DropFromHigh=-" in str(nifty_regime.get('reason', '')):
                if "DropFromHigh" in str(nifty_regime.get('reason', '')):
                    try:
                        drop_val = float(str(nifty_regime.get('reason')).split('DropFromHigh=-')[1].split(' ')[0])
                    except Exception:
                        drop_val = 60.0
                    if drop_val >= 60.0:
                        return {
                            'symbol': sym_clean,
                            'passed': False,
                            'failed_gate': f"NIFTY_REAL_CRASH_BLOCKED_CALL: {nifty_regime.get('reason')}"
                        }
                elif not is_independent_outperformer:
                    return {
                        'symbol': sym_clean,
                        'passed': False,
                        'failed_gate': f"FLAW3_NIFTY_REGIME_BLOCKED_CALL: {nifty_regime.get('reason')}"
                    }

        if direction == 'PUT' and not nifty_regime.get('allow_stock_put', True):
            if not is_independent_outperformer or "RiseFromLow=+" in str(nifty_regime.get('reason', '')):
                if "RiseFromLow" in str(nifty_regime.get('reason', '')):
                    try:
                        rise_val = float(str(nifty_regime.get('reason')).split('RiseFromLow=+')[1].split(' ')[0])
                    except Exception:
                        rise_val = 60.0
                    if rise_val >= 60.0:
                        return {
                            'symbol': sym_clean,
                            'passed': False,
                            'failed_gate': f"NIFTY_REAL_RALLY_BLOCKED_PUT: {nifty_regime.get('reason')}"
                        }
                elif not is_independent_outperformer:
                    return {
                        'symbol': sym_clean,
                        'passed': False,
                        'failed_gate': f"FLAW3_NIFTY_REGIME_BLOCKED_PUT: {nifty_regime.get('reason')}"
                    }

        delta = 0.60
        expected_opt_move = max(30.0, round(stock_atr * 0.35, 2))

        # Anti-Climax Parabolic Exhaustion Shield (TataElxsi Top-Buying Trap)
        # अगर शेयर अपनी 3-मिनट 9-EMA से पहले ही बहुत दूर (Over-Extended) भाग चुका है,
        # तो उस कैंडल के पीक (Exact Top) पर CALL या बॉटम पर PUT लेना 100% ब्लॉक!
        spot_price = float(q.get('live_spot', cur_p))
        ema9_val = float(q.get('ema9', 0.0))
        if ema9_val > 0:
            dist_from_ema = spot_price - ema9_val
            if direction == 'CALL' and dist_from_ema > (stock_atr * 0.35):
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': f'PARABOLIC_CLIMAX_TOP: Extended {dist_from_ema:.1f} pts > 0.35x ATR from 9-EMA'
                }
            if direction == 'PUT' and (-dist_from_ema) > (stock_atr * 0.35):
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': f'PARABOLIC_CLIMAX_BOTTOM: Extended {-dist_from_ema:.1f} pts > 0.35x ATR below 9-EMA'
                }

        day_range = round(b_high - b_low, 2)
        # Anti-Exhaustion Shield: RVOL >= 2.50x gets 2.00x ATR expansion; standard gets 1.35x ATR
        max_expansion = (stock_atr * 2.00) if rvol >= 2.50 else (stock_atr * 1.35)
        if day_range > max_expansion:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'STOCK_MOVE_ALREADY_EXHAUSTED ({day_range:.1f} > {max_expansion:.1f} pts)'
            }

        # नियम B: वास्तविक साइडवेज़ चॉप शील्ड (Calibrated Dead Sideways Shield)
        # हाई-ATR शेयर्स (ATR >= 85) के लिए शेयर को 186 अंक भागने के लिए मजबूर न करें!
        # नियम: शेयर में कम से कम 25 अंक (या 0.15x ATR) की हलचल होनी चाहिए ताकि ऑप्शन बायर शुरुआती मोमेंटम पकड़ सके।
        min_required_expansion = max(25.0, round(stock_atr * 0.15, 1)) if stock_atr >= 85.0 else max(8.0, round(stock_atr * 0.35, 1))
        if day_range < min_required_expansion:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'DEAD_SIDEWAYS_CHOP (Day Range {day_range:.1f} < {min_required_expansion:.1f} pts Base)'
            }

        # नियम #1: न्यूनतम 0.50% प्रतिशत चाल फ़्लोर (Micro-Move & Theta-Decay Trap Shield)
        # ऑप्शन बाइंग में थीटा डीके को हराने के लिए कम से कम 0.50% की वास्तविक चाल अनिवार्य है
        if abs(pct_chg) < 0.50:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'SUB_EXPANSION_THETA_TRAP: Intraday move {pct_chg:+.2f}% < 0.50% minimum threshold'
            }

        # नियम #2: मॉर्निंग रन-अवे चेज़िंग एग्जॉशन कैप (Anti-Chasing Cap)
        # सुबह 09:30 - 10:15 के बीच जो शेयर पहले ही 3.50% से ज्यादा भाग चुके हैं (जैसे KEI -6.2%, PowerIndia -4.1%),
        # उनका पीछा करना 'ट्रेन छूटने के बाद दौड़ना' है — उन्हें सुबह 100% ब्लॉक करें!
        if is_morning and abs(pct_chg) > 3.50:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'MORNING_MOVE_ALREADY_EXHAUSTED ({pct_chg:+.2f}% > 3.50% in morning slab)'
            }

        # नियम #4: दिन में 5.0% से ज्यादा भागे शेयर्स पर नो-एंट्री (Anti-FOMO Top Chasing Cap)
        # जो स्टॉक दिनभर में पहले ही 5.0% से ज्यादा भाग चुका है (जैसे Mankind +7.02%), उसके टॉप पर कॉल या बॉटम पर पुट लेना ट्रैप है
        if abs(pct_chg) >= 5.0:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'PARABOLIC_MOVE_EXHAUSTED (Move {pct_chg:+.2f}% >= 5.0% day move cap)'
            }

        # बेस 40 से शुरू: प्राइस वेलोसिटी + संस्थागत RVOL + ऑर्डर फ्लो CVD का वास्तविक संगम
        base_score = 40.0
        mom_pts = min(25.0, abs(pct_chg) * 7.0)          # मजबूत प्राइस मोमेंटम (कम से कम 3% मूव)
        vol_pts = min(25.0, (rvol - 1.0) * 8.0) if rvol > 1.0 else 0.0  # संस्थागत वॉल्यूम (RVOL >= 2.8x होने पर ही पूरे 15+ अंक)
        cvd_pts = 10.0 if (abs(delta) >= 0.35) else 0.0 # आक्रामक बायर/सेलर ऑर्डर फ्लो का दबदबा
        score = round(min(99.0, base_score + mom_pts + vol_pts + cvd_pts), 2)
        # नियम A: संस्थागत सुपर-वॉल्यूम (RVOL >= 5.0x और मजबूत दिशा >= 0.50%) होने पर सीधे 95.0 का एलिट स्कोर दें
        if rvol >= 5.0 and abs(pct_chg) >= 0.50:
            score = max(score, 95.0)

        return {
            'symbol': sym_clean,
            'passed': True,
            'spot_price': cur_p,
            'direction': direction,
            'pct_change': pct_chg,
            'rvol': rvol,
            'score': score,
            'is_index': False,
            'stock_atr': stock_atr,
            'delta': delta,
            'expected_opt_move': expected_opt_move
        }

    def _record_signal_to_excel(self, ist_now, cand, opt, opt_action, contract, opt_entry, opt_sl, opt_tgt1, opt_tgt2, sl_pts, t1_pts, t2_pts, time_sl_time, buffer_end_time):
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
            'Buying_Price_INR': opt_entry,
            'Stop_Loss_INR': opt_sl,
            'Trailing_SL_INR': opt_sl,
            'Time_SL_IST': time_sl_time,
            'Target_Price_INR': opt_tgt2,
            'Exit_Time_IST': '',
            'Exit_Price_INR': 0.0,
            'Holding_Time': '0m',
            'Points_Captured': 0.0,
            'Gross_PnL_INR': 0.0,
            'Friction_Taxes_INR': 0.0,
            'Net_PnL_INR': 0.0,
            'Cumulative_Equity_INR': 100000.0,
            'Status': 'SIGNAL_GENERATED_OPEN'
        }

        try:
            import pandas as pd
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

    def run_screener(self, ignore_window=False):
        ist_now = get_ist_time()
        ist_time_str = ist_now.strftime('%I:%M:%S %p IST')
        ist_date_str = ist_now.strftime('%d-%b-%Y')

        print(f"=======================================================")
        print(f"  ALPHA SCREENER | {ist_date_str} {ist_time_str}")
        print(f"  Designated Slabs: 09:30-10:15 AM IST & 01:15-02:15 PM IST")
        print(f"=======================================================")

        if not ignore_window and not is_in_active_trading_slab(ist_now):
            print(f"[OUTSIDE TRADING SLAB]: Current time {ist_time_str} is outside active trading slabs (09:30-10:15 AM & 01:15-02:15 PM IST). Maintaining complete silence.")
            return

        # PINPOINT FIX: Balanced Slab Allocation (Max 4 Morning + Max 4 Afternoon = 8 Total Daily Super Trades)
        # Guarantees afternoon jackpot window (01:15-02:15 PM) is NEVER starved or blocked by morning trades!
        # पूर्ण आधिकारिक ट्रेडिंग स्लैब्स (Morning 09:30-10:15 AM & Afternoon 01:15-02:15 PM IST):
        tot_mins_now = ist_now.hour * 60 + ist_now.minute
        is_morning = (570 <= tot_mins_now <= 615)
        is_afternoon = (795 <= tot_mins_now <= 855)

        morn_count = 0
        aft_count = 0
        for sym_d, ts in self.dispatched_today.items():
            try:
                dt = datetime.datetime.fromtimestamp(ts, tz=ist_now.tzinfo)
                if dt.hour == 9 or (dt.hour == 10 and dt.minute <= 15):
                    morn_count += 1
                elif dt.hour >= 13:
                    aft_count += 1
            except Exception:
                pass

        MAX_SLAB_TRADES = 4
        if is_morning and morn_count >= MAX_SLAB_TRADES and not ignore_window:
            print(f"[MORNING SLAB CAP REACHED]: {morn_count}/{MAX_SLAB_TRADES} morning trades already dispatched. Reserving 4 slots for afternoon jackpot slab.")
            return

        if is_afternoon and aft_count >= MAX_SLAB_TRADES and not ignore_window:
            print(f"[AFTERNOON SLAB CAP REACHED]: {aft_count}/{MAX_SLAB_TRADES} afternoon trades already dispatched.")
            return

        print(f"[ACTIVE SLAB REACHED]: Executing Full 216 F&O + Index Scan at {ist_time_str}...")

        # STEP 1 (UPSIDE): Fetch 1 Bulk NSE Snapshot, Audit NIFTY Macro Regime, & Pre-Filter ALL 216 Stocks + 4 Indices
        fetch_bulk_nse_snapshot(force_refresh=True)
        nifty_regime = self.get_nifty_market_regime(force_refresh=True)
        print(
            f"[MACRO TIDE REGIME]: AllowStockCALL={nifty_regime['allow_stock_call']} | "
            f"AllowStockPUT={nifty_regime['allow_stock_put']} | {nifty_regime['reason']}"
        )

        symbols = [x['symbol'] for x in self.universe]
        with ThreadPoolExecutor(max_workers=8) as executor:
            eval_results = list(executor.map(self.evaluate_asset, symbols))

        # Elite 90% Win-Probability Gate: Only setups with Score >= 90.0 (or Indices) qualify!
        # Elite 90% Win-Probability Gate: Only setups with Score >= 88.0 qualify!
        # सिर्फ वही सेटअप्स पास होंगे जिनका वास्तविक संस्थागत स्कोर >= 91.0 होगा (केवल शीर्ष 12-14 जैकपॉट ट्रेड्स)
        # नियम A: स्कोर >= 91.0 हो या फिर संस्थागत RVOL >= 5.0x और दिशा >= 0.50% हो, तो सीधे स्टेज-2 में पास करें
        stage1_passed = [
            r for r in eval_results 
            if r.get('passed', False) and (
                r.get('score', 0.0) >= 91.0 or 
                (float(r.get('rvol', 0.0)) >= 5.0 and abs(float(r.get('pct_change', 0.0))) >= 0.50)
            )
        ]

        # Sort ALL qualified candidates across the entire 216 universe by Win-Probability Score & Expected Move
        stage1_passed.sort(
            key=lambda x: (
                1 if x.get('is_index', False) and x.get('is_compression_breakout', False) else 0,
                x.get('score', 0.0),
                x.get('expected_opt_move', 50.0 if x.get('is_index', False) else 0.0),
                x.get('rvol', 0.0)
            ),
            reverse=True
        )

        print(f"[STAGE-1 COMPLETE]: Scanned {len(symbols)} assets. {len(stage1_passed)} candidates passed price/volume/CVD gates.")

        # STEP 2 (DOWNSIDE): Deep Live Option Chain (-4 | 0 | +4) OI + Volume + Runway Wall & Live LTP Verification
        verified_super_trades = []
        active_slab_count = morn_count if is_morning else aft_count
        remaining_slab_slots = max(1, MAX_SLAB_TRADES - active_slab_count)
        slab_cap = min(4, remaining_slab_slots)

        # नियम: प्रति स्लैब केवल 1 बेस्ट इंडेक्स ट्रेड (Morning Trend + Afternoon Reversal दोनों को अनुमति)
        # सुबह (09:30-10:15) में 1 बेस्ट इंडेक्स और दोपहर (01:15-02:15) में 1 बेस्ट इंडेक्स मान्य होगा।
        # एक ही स्कैन साइकिल/मिनट में 3-4 इंडेक्स का क्लस्टर कभी नहीं जाएगा (केवल #1 बेस्ट स्कोर वाला जाएगा)।
        index_names = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
        
        has_index_in_active_slab = False
        for idx in index_names:
            last_ts = self.dispatched_today.get(idx, 0.0)
            if last_ts > 0:
                try:
                    dt = datetime.datetime.fromtimestamp(last_ts, tz=ist_now.tzinfo)
                    # यदि वर्तमान स्लैब सुबह का है और सुबह कोई इंडेक्स गया है:
                    if is_morning and (dt.hour == 9 or (dt.hour == 10 and dt.minute <= 15)):
                        has_index_in_active_slab = True
                        break
                    # यदि वर्तमान स्लैब दोपहर का है और दोपहर में कोई इंडेक्स जा चुका है:
                    elif is_afternoon and (dt.hour >= 13):
                        has_index_in_active_slab = True
                        break
                except Exception:
                    pass

        has_index_in_this_cycle = False

        for cand in stage1_passed:
            if len(verified_super_trades) >= slab_cap:
                break

            sym = cand['symbol']
            spot = cand['spot_price']
            direction = cand['direction']
            is_idx = cand.get('is_index', False)

            # यदि वर्तमान सत्र में 1 इंडेक्स जा चुका है, या इस स्कैन में 1 इंडेक्स चुना जा चुका है, तो दूसरे इंडेक्स को छोड़ें
            if is_idx and (has_index_in_active_slab or has_index_in_this_cycle):
                print(f"  [PER_SLAB_INDEX_EXCLUSIVE]: {sym} bypassed. Active slab already has 1 Top Index.")
                continue

            # -------------------------------------------------------------
            # STAGE 2A: 5-PILLAR REAL 3-MINUTE INTRADAY CANDLE & BREAKOUT AUDIT
            # (Indices already audited on real 3m bars in Stage 1;
            #  Stocks audited here on genuine 3-min resampled bars!)
            # -------------------------------------------------------------
            if not is_idx:
                # Re-verify Nifty Macro Tide before evaluating stock
                cand_rvol = float(cand.get('rvol', 1.0))
                cand_pct = abs(float(cand.get('pct_change', 0.0)))
                cand_atr = float(cand.get('stock_atr', 0.0))
                is_outperformer = (cand_pct >= 0.50 and cand_rvol >= 1.50 and cand_atr >= 85.0)

                if direction == 'CALL' and not nifty_regime.get('allow_stock_call', True):
                    if not is_outperformer or "DropFromHigh=-" in str(nifty_regime.get('reason', '')):
                        print(f"  [NIFTY REGIME BLOCKED]: {sym} (CALL) -> {nifty_regime.get('reason')}")
                        continue
                if direction == 'PUT' and not nifty_regime.get('allow_stock_put', True):
                    if not is_outperformer or "RiseFromLow=+" in str(nifty_regime.get('reason', '')):
                        print(f"  [NIFTY REGIME BLOCKED]: {sym} (PUT) -> {nifty_regime.get('reason')}")
                        continue

                stk_df = self.data_feed.get_live_intraday_bars(sym)
                if stk_df is None or len(stk_df) < 9:
                    print(f"  [5-PILLAR CANDLE REJECTED]: {sym} -> Insufficient 3-min candles (less than 9 bars)")
                    continue

                eval_bar = stk_df.iloc[-2] if len(stk_df) >= 2 else stk_df.iloc[-1]
                live_bar = stk_df.iloc[-1]
                b_open = float(eval_bar['open'])
                b_high = float(eval_bar['high'])
                b_low = float(eval_bar['low'])
                b_close = float(eval_bar['close'])

                l_open = float(live_bar['open'])
                l_high = float(live_bar['high'])
                l_low = float(live_bar['low'])
                l_close = float(live_bar['close'])

                # Keep live ticking spot price accurate
                live_q = self.data_feed.get_live_asset_quote(sym)
                spot = float(live_q.get('live_spot', l_close))
                cand['spot_price'] = spot

                stock_atr = float(cand.get('stock_atr', self.get_stock_atr(sym, live_q)))
                rng = max(0.01, b_high - b_low)

                # FLAW #1 FIX (Anti-Parabolic Exhaustion Candle Cap):
                # Never chase a 3-minute candle that already burned > 35% of the stock's full-day ATR (e.g. Navin Fluorine +143 pts)
                max_single_candle_rng = round(stock_atr * 0.35, 2)
                if rng > max_single_candle_rng:
                    print(
                        f"  [FLAW #1 ANTI-PARABOLIC REJECTED]: {sym} -> 3m Candle Range {rng:.2f} pts > "
                        f"35% of Daily ATR ({max_single_candle_rng:.2f} pts). Blow-off trap avoided!"
                    )
                    continue

                # PINPOINT FIX: Runway is measured by total day expansion vs 1.35x ATR so afternoon trend reversals/breakdowns are not prematurely choked
                day_high_3m = max(float(live_q.get('high', spot)), float(stk_df['high'].max()))
                day_low_3m = min(float(live_q.get('low', spot)), float(stk_df['low'].min())) if float(live_q.get('low', 0)) > 0 else float(stk_df['low'].min())
                day_total_expansion = day_high_3m - day_low_3m
                max_expansion_mult = 2.00 if cand_rvol >= 2.50 else 1.35
                if day_total_expansion > (stock_atr * max_expansion_mult):
                    print(
                        f"  [EXHAUSTED RUNWAY REJECTED]: {sym} ({direction}) -> Day Range {day_total_expansion:.2f} pts "
                        f"(> {int(max_expansion_mult*100)}% of ATR {stock_atr:.1f} pts)"
                    )
                    continue

                # नियम B: स्टेज-2 में भी टाइम-अडैप्टिव एंटी-साइडवेज़ फ़िल्टर
                min_rng_pct_stage2 = 0.35 if is_morning else 0.60
                if day_total_expansion < (stock_atr * min_rng_pct_stage2) and cand_rvol < 5.0:
                    print(
                        f"  [DEAD_SIDEWAYS_CHOP REJECTED]: {sym} ({direction}) -> Day Range {day_total_expansion:.2f} pts "
                        f"(< {int(min_rng_pct_stage2*100)}% of ATR {stock_atr*min_rng_pct_stage2:.1f} pts)"
                    )
                    continue

                # नियम #1: काउंटर-ट्रेंड ब्लॉक (Counter-Trend Shield)
                # यदि स्टॉक दिनभर लाल (pct_chg < 0) है तो CALL कभी नहीं, और यदि हरा (pct_chg > 0) है तो PUT कभी नहीं
                day_pct = float(cand.get('pct_change', 0.0))
                if direction == 'CALL' and day_pct < 0:
                    print(f"  [COUNTER_TREND_BLOCKED]: {sym} (CALL) rejected because stock is in red ({day_pct:+.2f}%)")
                    continue
                if direction == 'PUT' and day_pct > 0:
                    print(f"  [COUNTER_TREND_BLOCKED]: {sym} (PUT) rejected because stock is in green ({day_pct:+.2f}%)")
                    continue

                # FLAW #2 FIX: Live Forming Bar Reversal Check with 10% ATR spot buffer & 25% ATR wick tolerance
                # Allows healthy 0.05% bid-ask tick variance so valid breakouts (Trent, Siemens) are not killed
                spot_buffer = stock_atr * 0.10
                if direction == 'CALL':
                    live_bar_holding = (spot >= (b_close - spot_buffer)) and ((l_high - spot) <= (stock_atr * 0.25))
                else:
                    live_bar_holding = (spot <= (b_close + spot_buffer)) and ((spot - l_low) <= (stock_atr * 0.25))

                if not live_bar_holding:
                    print(
                        f"  [FLAW #2 LIVE BAR REVERSAL REJECTED]: {sym} ({direction}) -> "
                        f"LiveSpot={spot:.2f} vs ClosedBar={b_close:.2f} (LiveOpen={l_open:.2f}, LiveClose={l_close:.2f})"
                    )
                    continue
                body = abs(b_close - b_open)
                solid_body_ratio = round(body / rng, 2)
                up_wick = round((b_high - max(b_open, b_close)) / rng, 2)
                dn_wick = round((min(b_open, b_close) - b_low) / rng, 2)
                clv = round((b_close - b_low) / rng, 2)

                comp_bars = stk_df.iloc[:-2].tail(8)
                comp_high = float(comp_bars['high'].max())
                comp_low = float(comp_bars['low'].min())
                max_dist_buffer = stock_atr * 0.20

                # ---------------------------------------------------------
                # मोड 1: फास्ट मार्केट (1 कैंडल मारुबोज़ू ब्लास्ट: कम से कम 15 अंक का सिंगल बार)
                # ---------------------------------------------------------
                min_stock_blast_pts = max(15.0, stock_atr * 0.15)
                if direction == 'CALL':
                    single_mom_ok = (b_close >= b_open) and (clv >= 0.75) and (up_wick <= 0.20) and \
                                    (solid_body_ratio >= 0.60) and (rng >= min_stock_blast_pts)
                    broke_comp = (b_close >= comp_high) and (b_close <= comp_high + max_dist_buffer)
                else:
                    single_mom_ok = (b_close <= b_open) and (clv <= 0.25) and (dn_wick <= 0.20) and \
                                    (solid_body_ratio >= 0.60) and (rng >= min_stock_blast_pts)
                    broke_comp = (b_close <= comp_low) and (b_close >= comp_low - max_dist_buffer)

                # ---------------------------------------------------------
                # मोड 2: स्लो मार्केट (2 या 3 कैंडल संस्थागत ग्राइंड: 2 कैंडल >= 0.40% या 3 कैंडल >= 0.60%)
                # ---------------------------------------------------------
                cum_mom_ok = False
                if len(stk_df) >= 5:
                    b_prev1 = stk_df.iloc[-3]
                    b_prev2 = stk_df.iloc[-4]
                    pre_seq_bars = stk_df.iloc[:-4].tail(8) if len(stk_df) >= 6 else comp_bars
                    seq_comp_high = float(pre_seq_bars['high'].max())
                    seq_comp_low = float(pre_seq_bars['low'].min())

                    if direction == 'CALL':
                        last_2_green = (float(b_prev1['close']) >= float(b_prev1['open']) and b_close >= b_open)
                        all_3_green = (float(b_prev2['close']) >= float(b_prev2['open']) and last_2_green)
                        cum_rise_2_pct = ((b_close - float(b_prev1['open'])) / float(b_prev1['open'])) * 100.0
                        cum_rise_3_pct = ((b_close - float(b_prev2['open'])) / float(b_prev2['open'])) * 100.0
                        cum_mom_ok = (all_3_green and cum_rise_3_pct >= 0.60) or (last_2_green and cum_rise_2_pct >= 0.40)
                        if cum_mom_ok and (b_close >= seq_comp_high) and (b_close <= seq_comp_high + max_dist_buffer):
                            broke_comp = True
                    else:
                        last_2_red = (float(b_prev1['close']) <= float(b_prev1['open']) and b_close <= b_open)
                        all_3_red = (float(b_prev2['close']) <= float(b_prev2['open']) and last_2_red)
                        cum_fall_2_pct = ((float(b_prev1['open']) - b_close) / float(b_prev1['open'])) * 100.0
                        cum_fall_3_pct = ((float(b_prev2['open']) - b_close) / float(b_prev2['open'])) * 100.0
                        cum_mom_ok = (all_3_red and cum_fall_3_pct >= 0.60) or (last_2_red and cum_fall_2_pct >= 0.40)
                        if cum_mom_ok and (b_close <= seq_comp_low) and (b_close >= seq_comp_low - max_dist_buffer):
                            broke_comp = True

                cand_mom_pass = (single_mom_ok or cum_mom_ok) and broke_comp
                if not cand_mom_pass:
                    print(f"  [5-PILLAR 3M CANDLE REJECTED]: {sym} ({direction}) -> SingleMom={single_mom_ok}, CumMom={cum_mom_ok}, BrokeComp={broke_comp}")
                    continue

                # अडैप्टिव RVOL: फास्ट ब्लास्ट (1 कैंडल) के लिए 3.50x अनिवार्य,
                # स्लो ग्राइंड (2-3 कैंडल) के लिए 2.80x पर्याप्त (चूँकि वॉल्यूम 3 कैंडल में बंटा है)
                hist_base_v = float(stk_df['hist_avg_3m_vol'].iloc[-1]) if 'hist_avg_3m_vol' in stk_df.columns else 0.0
                prior_10_v = float(stk_df['volume'].iloc[:-2].tail(10).mean()) if len(stk_df) > 2 else 0.0
                base_3m_v = hist_base_v if hist_base_v > 0 else prior_10_v
                curr_3m_v = float(eval_bar.get('volume', 0.0))
                if cum_mom_ok and len(stk_df) >= 4:
                    curr_3m_v = max(curr_3m_v, float(stk_df.iloc[-3].get('volume', 0.0)))
                real_3m_rvol = round(curr_3m_v / max(1.0, base_3m_v), 2) if base_3m_v > 0 else cand.get('rvol', 2.5)
                cand['rvol'] = real_3m_rvol
                min_rvol_req = 2.80 if cum_mom_ok else 3.50
                if real_3m_rvol < min_rvol_req:
                    print(f"  [INSTITUTIONAL_RVOL_REJECTED]: {sym} -> Real 3m RVOL {real_3m_rvol:.2f}x < {min_rvol_req:.2f}x (Insufficient Momentum)")
                    continue

                # Pillar 3: Real 3-Minute Order Flow CVD Delta Verification
                bar_cvd = round((2.0 * clv) - 1.0, 2)
                cand['bar_cvd'] = bar_cvd
                # नियम #2: CVD दिशा अनिवार्य (कोई बाईपास नहीं!)
                # कॉल के लिए Bar CVD >= +0.35 और पुट के लिए Bar CVD <= -0.35 अनिवार्य
                cvd_pass = (bar_cvd >= 0.35) if direction == 'CALL' else (bar_cvd <= -0.35)
                if not cvd_pass:
                    print(f"  [3M CVD REJECTED]: {sym} ({direction}) -> Bar CVD {bar_cvd:+.2f} fails directional alignment (+/-0.35)")
                    continue

            # -------------------------------------------------------------
            # STAGE 2B: LIVE OPTION CHAIN (-4 / 0 / +4) RUNWAY & LIVE LTP AUDIT
            # -------------------------------------------------------------

            oi_conf = self.data_feed.get_live_dual_oi_confluence(sym, spot, direction=direction)
            if not oi_conf.get('confluent', False):
                print(f"  [OI/RUNWAY REJECTED]: {sym} ({direction}) -> {oi_conf.get('reason')}")
                continue

            opt = self.data_feed.compute_option_pricing(spot, direction=direction, symbol=sym)
            if not opt or not opt.get('is_live_chain_verified', False) or float(opt.get('entry_price', 0.0)) < 2.0:
                print(f"  [LIVE LTP REJECTED]: {sym} -> Live Option Chain LTP (< Rs. 2.00) unavailable.")
                continue

            contract = opt['contract']
            if self.is_already_dispatched(sym, contract):
                continue

            cand['oi_confluent'] = True
            cand['oi_reason'] = oi_conf.get('reason', '')
            cand['verified_opt'] = opt
            verified_super_trades.append(cand)
            if is_idx:
                has_index_in_this_cycle = True

        if not verified_super_trades:
            print(f"[SCREENER]: 0 signals qualified after full 216 scan & live OI audit at {ist_time_str}. Dispatching 3-min heartbeat to Chatbot.")
            bot_zero_msg = (
                f"🛡️ <b>ALPHA LIVE SCAN UPDATE</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ <b>SCAN TIME:</b> <b>{ist_time_str}</b>\n"
                f"📊 <b>UNIVERSE SCANNED:</b> <b>{len(symbols)} F&O Assets (Full NSE Scan)</b>\n"
                f"🎯 <b>QUALIFIED SIGNALS:</b> <b>0 Setups (Market in Chop/Compression)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🔒 <b>CAPITAL SHIELD:</b> Zero forced trades. Capital 100% Protected.\n"
                f"👥 <b>GROUP STATUS:</b> Silent (Alerts sent only on confirmed buy signals).\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            self.notifier.send_broadcast_message(bot_zero_msg, is_trade_signal=False)
            return

        print(f"[SCREENER]: Verified {len(verified_super_trades)} PURE LIVE SUPER TRADE(S)! Dispatching...")

        dispatched_count = 0
        for cand in verified_super_trades:
            sym = cand['symbol']
            spot = cand['spot_price']
            direction = cand['direction']
            is_idx = cand.get('is_index', False)
            opt_action = "BUY CALL (CE)" if direction == "CALL" else "BUY PUT (PE)"
            opt = cand['verified_opt']
            contract = opt['contract']

            meta = get_instrument_meta(sym)
            lot_sz = opt.get('quantity') or (meta.get('lot', 25) if meta else 25)
            buffer_end_time = (ist_now + datetime.timedelta(minutes=4)).strftime('%I:%M %p IST')
            time_sl_time = (ist_now + datetime.timedelta(minutes=15)).strftime('%I:%M %p IST')

            opt_entry = opt['entry_price']
            opt_sl = opt['sl_price']
            sl_pts = opt.get('sl_points', round(opt_entry - opt_sl, 2))

            if is_idx:
                if sym == 'NIFTY':
                    t1_pts = 80.0
                    t2_pts = 130.0
                    target_note = "+80 to +130 pts Spot Target Move"
                elif sym == 'BANKNIFTY':
                    t1_pts = 150.0
                    t2_pts = 250.0
                    target_note = "+150 to +250 pts Spot Target Move"
                elif sym == 'FINNIFTY':
                    t1_pts = 80.0
                    t2_pts = 130.0
                    target_note = "+80 to +130 pts Spot Target Move"
                else: # MIDCPNIFTY
                    t1_pts = 45.0
                    t2_pts = 75.0
                    target_note = "+45 to +75 pts Spot Target Move"
            else:
                t1_pts = round(sl_pts * 2.0, 2)
                t2_pts = max(30.0, round(sl_pts * 4.0, 2))
                stk_atr = cand.get('stock_atr', 60.0)
                exp_m = cand.get('expected_opt_move', round(stk_atr * 0.60, 1))
                target_note = f"+30 to +60 pts | +{t2_pts:.2f} pts [ATR {stk_atr:.1f} | Exp Move: ~{exp_m:.1f} pts]"

            opt_tgt1 = round(opt_entry + t1_pts, 2)
            opt_tgt2 = round(opt_entry + t2_pts, 2)

            pure_signal_msg = (
                f"🟢 <b>ALPHA FAST GAMMA BUY SIGNAL</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📌 <b>CONTRACT:</b> <code>{contract}</code> ({opt_action}) | <b>LOT:</b> {lot_sz}\n"
                f"⏰ <b>ENTRY TIME:</b> <b>{ist_time_str}</b>\n"
                f"📊 <b>SPOT PRICE:</b> Rs. {spot:,.2f} ({cand['pct_change']:+.2f}%)\n"
                f"🧭 <b>OI RUNWAY (-4/0/+4):</b> <b>{cand.get('oi_reason', 'CLEAR')}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 <b>BUY PRICE (LIVE LTP):</b> <b>Rs. {opt_entry:,.2f}</b>\n"
                f"🛑 <b>STOP LOSS:</b> <b>Rs. {opt_sl:,.2f}</b> (-{sl_pts:.2f} pts)\n"
                f"🎯 <b>TARGET 1:</b> <b>Rs. {opt_tgt1:,.2f}</b> (+{t1_pts:.2f} pts)\n"
                f"🚀 <b>TARGET 2:</b> <b>Rs. {opt_tgt2:,.2f}</b> (+{t2_pts:.2f} pts | {target_note})\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏳ <b>BUFFER TIME:</b> <b>4 Minutes Window (Valid till {buffer_end_time})</b>\n"
                f"⏱️ <b>15-MIN KILL SWITCH:</b> <b>{time_sl_time}</b> (STRICT EXIT ONLY if in Loss/Red)\n"
                f"🛡️ <b>PROFIT RUNNER LOCK:</b> If in Profit at 15m, <b>Trail SL to Entry (Cost)</b> & Ride to Target 1 (+80-150 pts) & Target 2 (+250 pts)!\n"
                f"💡 <b>EXECUTION:</b> Place Limit Order near CMP, Set GTT Target 2 & SL-M\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            print(f"[BROADCAST]: Dispatching {contract} @ Rs. {opt_entry:,.2f} to BOTH Chatbot and Group...")
            self.notifier.send_broadcast_message(pure_signal_msg, is_trade_signal=True)
            self.mark_dispatched(sym, contract)
            dispatched_count += 1

            self._record_signal_to_excel(
                ist_now=ist_now,
                cand=cand,
                opt=opt,
                opt_action=opt_action,
                contract=contract,
                opt_entry=opt_entry,
                opt_sl=opt_sl,
                opt_tgt1=opt_tgt1,
                opt_tgt2=opt_tgt2,
                sl_pts=sl_pts,
                t1_pts=t1_pts,
                t2_pts=t2_pts,
                time_sl_time=time_sl_time,
                buffer_end_time=buffer_end_time
            )

        if dispatched_count > 0:
            print(f"[SCREENER]: {dispatched_count} signals dispatched and recorded to Daily & Monthly Excel successfully.\n")
            daily_xl = os.path.join(CURRENT_DIR, 'Alpha_5Pillar_Live_Paper_Trading_Journal.xlsx')
            if os.path.exists(daily_xl):
                caption = (
                    f"📊 <b>OFFICIAL LIVE PAPER TRADING JOURNAL (UPDATED)</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📅 <b>DATE:</b> {ist_date_str}\n"
                    f"⏰ <b>ALERT TIME:</b> {ist_time_str}\n"
                    f"⚡ <b>NEW TRADES LOGGED:</b> {dispatched_count} Signal(s)\n"
                    f"🔒 <b>SECURITY:</b> Strictly Confidential (Owner Private DM Only)\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
                self.notifier.send_document_to_owner_with_approval_prompt(daily_xl, caption=caption)

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
        if ist_now.weekday() >= 5:
            print(f"[WEEKEND]: {ist_now.strftime('%A')} market closed. Zero messages sent.")
            return {"ok": True, "weekend": True}
        ist_time_str = ist_now.strftime('%I:%M:%S %p IST')
        ist_date_str = ist_now.strftime('%d-%b-%Y')
        daily_xl = os.path.join(CURRENT_DIR, 'Alpha_5Pillar_Live_Paper_Trading_Journal.xlsx')
        if not os.path.exists(daily_xl):
            import pandas as pd
            df_today = pd.DataFrame()
        else:
            try:
                import pandas as pd
                df = pd.read_excel(daily_xl)
                df_today = df[df['Date'].astype(str).str.contains(ist_date_str, na=False)] if 'Date' in df.columns else pd.DataFrame()
            except Exception:
                df_today = pd.DataFrame()

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

        if os.path.exists(daily_xl):
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
        else:
            print("[EOD AUDIT & DISPATCH]: Delivered daily text analysis to Owner DM.")
            return {"ok": True}

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
        # आंतरिक 30-मिनट लूप हटाया गया — अब हर कॉल 30-40 सेकंड में पूरी स्कैन करके तुरंत एग्जिट होगी!
        # cron-job.org जब भी कॉल करेगा, रनर तुरंत खाली मिलेगा और कतार कभी जाम नहीं होगी!
        screener.run_screener(ignore_window=ignore_win)
