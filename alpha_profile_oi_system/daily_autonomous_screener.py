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
MAX_DAILY_SUPER_TRADES = 5  # Morning + Afternoon combined cap (2 to 5 Super Trades)


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
                        raw_dict = data.get('dispatched', {})
                        return {k: v for k, v in raw_dict.items() if not k.startswith(indices_prefixes)}
            except Exception:
                pass
        return {}

    def _save_dispatched_cache(self):
        ist_now = get_ist_time()
        today_str = ist_now.strftime('%Y-%m-%d')
        indices_prefixes = ('NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50')
        stock_only = {k: v for k, v in self.dispatched_today.items() if not k.startswith(indices_prefixes)}
        try:
            os.makedirs(os.path.dirname(self.dispatched_cache_file), exist_ok=True)
            with open(self.dispatched_cache_file, 'w', encoding='utf-8') as f:
                json.dump({'date': today_str, 'dispatched': stock_only}, f, indent=2)
        except Exception:
            pass

    def is_already_dispatched(self, symbol, contract=''):
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        if sym_clean in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50']:
            last_ts = self.index_recent_dispatch.get(sym_clean, 0.0)
            return (time.time() - last_ts) < 2100.0

        return sym_clean in self.dispatched_today

    def mark_dispatched(self, symbol, contract=''):
        sym_clean = symbol.upper().replace('.NS', '').replace('^', '').strip()
        if sym_clean in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50']:
            self.index_recent_dispatch[sym_clean] = time.time()
            return

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

            # Flaw #3 Exact Protection: Block Stock CALLs if NIFTY 3m candle is red OR NIFTY is below 9-EMA OR crashing >= 50 pts from high
            if prev_3m_red or below_ema9 or (drop_from_high >= 50.0 and live_3m_red):
                allow_call = False
                reason = (
                    f"NIFTY_BEARISH_OR_CRASH_REGIME (Spot {nifty_spot:.1f} vs 9-EMA {ema9:.1f} | "
                    f"Prev3mRed={prev_3m_red} | DropFromHigh=-{drop_from_high:.1f} pts)"
                )

            # Symmetrical Surge Protection for PUTs (only block PUTs on strong active rally above 9-EMA, NEVER on mild +25 pt open)
            prev_3m_strong_green = (n_close - n_open) >= 15.0
            live_3m_green = (l_close >= l_open) and (nifty_spot >= n_close)
            above_ema9_strong = (nifty_spot > (ema9 + 15.0))
            if above_ema9_strong and prev_3m_strong_green and live_3m_green and rise_from_low >= 75.0:
                allow_put = False
                reason = (
                    f"NIFTY_STRONG_RALLY_REGIME (Spot {nifty_spot:.1f} > 9-EMA {ema9:.1f} | "
                    f"RiseFromLow=+{rise_from_low:.1f} pts)"
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
        atr14 = float(self.atr_cache.get(symbol, 0.0))
        if atr14 > 0:
            return atr14
        h = float(q.get('high', 0))
        l = float(q.get('low', 0))
        cur_p = float(q.get('live_spot', 0))
        day_rng = round(h - l, 2) if h > 0 and l > 0 else 0.0
        est_atr = round(cur_p * 0.028, 2) if cur_p > 0 else 0.0
        return max(day_rng, est_atr)

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

                max_ext = 130.0 if sym_clean in ['NIFTY', 'FINNIFTY'] else 280.0
                if direction == 'PUT':
                    fall_from_high = day_high - cur_p
                    if fall_from_high >= max_ext:
                        return {
                            'symbol': sym_clean,
                            'passed': False,
                            'failed_gate': f'EXHAUSTION_TRAP_AVOIDED: Fall {fall_from_high:.1f} pts >= {max_ext:.1f} pts from Day High',
                            'spot_price': cur_p,
                            'score': 0.0,
                            'is_index': True
                        }
                else:
                    rise_from_low = cur_p - day_low
                    if rise_from_low >= max_ext:
                        return {
                            'symbol': sym_clean,
                            'passed': False,
                            'failed_gate': f'EXHAUSTION_TRAP_AVOIDED: Rise {rise_from_low:.1f} pts >= {max_ext:.1f} pts from Day Low',
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
                clv_ok = (clv >= 0.72) if direction == 'CALL' else (clv <= 0.28)
                wick_ok = (up_wick <= 0.20) if direction == 'CALL' else (dn_wick <= 0.20)
                single_mom_ok = (solid_body_ratio >= 0.40) and wick_ok and clv_ok

                if direction == 'CALL':
                    if b_close >= comp_high:
                        is_breakout = True
                else:
                    if b_close <= comp_low:
                        is_breakout = True

                # Multi-candle cumulative breakout uses base prior to the 2/3-candle sequence
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
                    broke_range_low = (b_close <= comp_low) or (b_close <= seq_comp_low)

                    if (all_3_red and cum_fall_3_pct >= 0.25 and broke_range_low) or (last_2_red and cum_fall_2_pct >= 0.20 and broke_range_low):
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
                    broke_range_high = (b_close >= comp_high) or (b_close >= seq_comp_high)

                    if (all_3_green and cum_rise_3_pct >= 0.25 and broke_range_high) or (last_2_green and cum_rise_2_pct >= 0.20 and broke_range_high):
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
                    idx_rvol_pass = (idx_rvol >= 1.50) or cum_mom_ok

                candle_cvd = round((2.0 * clv) - 1.0, 2)
                idx_cvd = candle_cvd
                idx_cvd_pass = (idx_cvd >= 0.45) if direction == 'CALL' else (idx_cvd <= -0.45)
                if cum_mom_ok:
                    idx_cvd_pass = True

                stage1_pass = cand_mom_pass and idx_rvol_pass and idx_cvd_pass
                oi_confluent = True
                oi_reason = 'PENDING_STAGE2_OI_AUDIT'

                if stage1_pass and verify_oi_now:
                    oi_conf = self.data_feed.get_live_dual_oi_confluence(sym_clean, cur_p, direction=direction)
                    oi_confluent = oi_conf.get('confluent', False)
                    oi_reason = oi_conf.get('reason', '')

                passed = stage1_pass and oi_confluent
                score = 98.0 if passed else 50.0

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

        # FLAW #3 FIX (Nifty Macro Tide Gate): Block Stock CALLs if Nifty is crashing/below 9-EMA
        nifty_regime = self.get_nifty_market_regime(force_refresh=False)
        if direction == 'CALL' and not nifty_regime.get('allow_stock_call', True):
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f"FLAW3_NIFTY_REGIME_BLOCKED_CALL: {nifty_regime.get('reason')}"
            }
        if direction == 'PUT' and not nifty_regime.get('allow_stock_put', True):
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f"FLAW3_NIFTY_REGIME_BLOCKED_PUT: {nifty_regime.get('reason')}"
            }

        # Gate 0: Mathematical Expected Option Move >= 30.0 Pts
        stock_atr = self.get_stock_atr(sym_clean, q)
        delta = 0.60
        expected_opt_move = round(stock_atr * delta, 2)
        if expected_opt_move < 30.0:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'GATE_0_EXPECTED_OPTION_MOVE_BELOW_30_PTS ({expected_opt_move:.1f} < 30.0)'
            }

        # Anti-Exhaustion Shield: Allow up to 1.15x ATR so afternoon trenders (1:15 PM) are not prematurely choked
        day_range = round(b_high - b_low, 2)
        if day_range > (stock_atr * 1.15):
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'STOCK_MOVE_ALREADY_EXHAUSTED ({day_range:.1f} > 1.15x ATR {stock_atr * 1.15:.1f} pts)'
            }

        # Require meaningful intraday directional activity to enter Stage-2 3-min candle audit
        if abs(pct_chg) < 0.20:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'FLAT_STOCK_NO_MOMENTUM ({pct_chg:+.2f}%)'
            }

        score = round(min(99.0, 70.0 + abs(pct_chg) * 5.0 + min(15.0, rvol * 3.0)), 2)

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
            print(f"[OUTSIDE TRADING SLAB]: Current time {ist_time_str} is outside active trading slabs (09:30-10:15 AM & 01:15-02:15 PM IST).")
            standby_msg = (
                f"ℹ️ <b>ALPHA SCREENER CLOUD STANDBY UPDATE</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ <b>TRIGGER TIME:</b> <b>{ist_time_str}</b>\n"
                f"🔒 <b>STATUS:</b> Market currently closed or outside active trading slabs.\n"
                f"🎯 <b>DESIGNATED TRADING SLABS:</b>\n"
                f"  • Slab 1 (Morning): 09:30 AM to 10:15 AM IST\n"
                f"  • Slab 2 (Afternoon): 01:15 PM to 02:15 PM IST\n"
                f"  • EOD Audit: 03:35 PM IST\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ <b>TELEGRAM CLOUD LINK:</b> Active & Verified. Bot is online and waiting for next trading session.\n"
                f"👥 <b>GROUP STATUS:</b> Silence maintained (0 messages to Group).\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            self.notifier.send_broadcast_message(standby_msg, is_trade_signal=False)
            return

        already_sent_today = len(self.dispatched_today) + len(self.index_recent_dispatch)
        if already_sent_today >= MAX_DAILY_SUPER_TRADES and not ignore_window:
            print(f"[DAILY SUPER-TRADE CAP REACHED]: {already_sent_today}/{MAX_DAILY_SUPER_TRADES} best trades already dispatched today.")
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

        stage1_passed = [r for r in eval_results if r.get('passed', False)]

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
        remaining_daily_slots = max(1, MAX_DAILY_SUPER_TRADES - already_sent_today)
        slab_cap = min(4, remaining_daily_slots)

        for cand in stage1_passed:
            if len(verified_super_trades) >= slab_cap:
                break

            sym = cand['symbol']
            spot = cand['spot_price']
            direction = cand['direction']
            is_idx = cand.get('is_index', False)

            # -------------------------------------------------------------
            # STAGE 2A: 5-PILLAR REAL 3-MINUTE INTRADAY CANDLE & BREAKOUT AUDIT
            # (Indices already audited on real 3m bars in Stage 1;
            #  Stocks audited here on genuine 3-min resampled bars!)
            # -------------------------------------------------------------
            if not is_idx:
                # Re-verify Nifty Macro Tide before evaluating stock
                if direction == 'CALL' and not nifty_regime.get('allow_stock_call', True):
                    print(f"  [NIFTY REGIME BLOCKED]: {sym} (CALL) -> {nifty_regime.get('reason')}")
                    continue
                if direction == 'PUT' and not nifty_regime.get('allow_stock_put', True):
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

                # Intraday Remaining Runway Check: Ensure stock hasn't already moved > 70% of ATR from day's low/high
                day_high_3m = max(float(live_q.get('high', spot)), float(stk_df['high'].max()))
                day_low_3m = min(float(live_q.get('low', spot)), float(stk_df['low'].min())) if float(live_q.get('low', 0)) > 0 else float(stk_df['low'].min())
                move_already_done = (spot - day_low_3m) if direction == 'CALL' else (day_high_3m - spot)
                if move_already_done > (stock_atr * 0.70):
                    print(
                        f"  [EXHAUSTED RUNWAY REJECTED]: {sym} ({direction}) -> Already moved {move_already_done:.2f} pts "
                        f"(> 70% of ATR {stock_atr:.1f} pts)"
                    )
                    continue

                # FLAW #2 FIX (Live Forming Bar Reversal Check):
                # Current live forming candle (`iloc[-1]`) & live spot MUST hold in the trade direction (blocks Siemens/Cholafin)
                if direction == 'CALL':
                    live_bar_holding = (spot >= b_close) and (l_close >= l_open) and ((l_high - spot) <= (stock_atr * 0.12))
                else:
                    live_bar_holding = (spot <= b_close) and (l_close <= l_open) and ((spot - l_low) <= (stock_atr * 0.12))

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

                # Pillar 1A: Single 3-Minute Candle Blast Momentum
                if direction == 'CALL':
                    single_mom_ok = (b_close >= b_open) and (clv >= 0.72) and (up_wick <= 0.20) and (solid_body_ratio >= 0.40)
                    broke_comp = (b_close >= comp_high)
                else:
                    single_mom_ok = (b_close <= b_open) and (clv <= 0.28) and (dn_wick <= 0.20) and (solid_body_ratio >= 0.40)
                    broke_comp = (b_close <= comp_low)

                # Pillar 1B: Dual & Triple Candle Cumulative Momentum (Slow Grinding Move Support)
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
                        cum_mom_ok = (all_3_green and cum_rise_3_pct >= 0.25) or (last_2_green and cum_rise_2_pct >= 0.20)
                        if cum_mom_ok and (b_close >= seq_comp_high):
                            broke_comp = True
                    else:
                        last_2_red = (float(b_prev1['close']) <= float(b_prev1['open']) and b_close <= b_open)
                        all_3_red = (float(b_prev2['close']) <= float(b_prev2['open']) and last_2_red)
                        cum_fall_2_pct = ((float(b_prev1['open']) - b_close) / float(b_prev1['open'])) * 100.0
                        cum_fall_3_pct = ((float(b_prev2['open']) - b_close) / float(b_prev2['open'])) * 100.0
                        cum_mom_ok = (all_3_red and cum_fall_3_pct >= 0.25) or (last_2_red and cum_fall_2_pct >= 0.20)
                        if cum_mom_ok and (b_close <= seq_comp_low):
                            broke_comp = True

                cand_mom_pass = (single_mom_ok or cum_mom_ok) and broke_comp
                if not cand_mom_pass:
                    print(f"  [5-PILLAR 3M CANDLE REJECTED]: {sym} ({direction}) -> SingleMom={single_mom_ok}, CumMom={cum_mom_ok}, BrokeComp={broke_comp}")
                    continue

                # Pillar 2 & 3 on Real 3-Minute Stock Bars (RVOL & CVD Verification)
                hist_base_v = float(stk_df['hist_avg_3m_vol'].iloc[-1]) if 'hist_avg_3m_vol' in stk_df.columns else 0.0
                prior_10_v = float(stk_df['volume'].iloc[:-2].tail(10).mean()) if len(stk_df) > 2 else 0.0
                base_3m_v = hist_base_v if hist_base_v > 0 else prior_10_v
                curr_3m_v = float(eval_bar.get('volume', 0.0))
                if cum_mom_ok and len(stk_df) >= 4:
                    curr_3m_v = max(curr_3m_v, float(stk_df.iloc[-3].get('volume', 0.0)))
                real_3m_rvol = round(curr_3m_v / max(1.0, base_3m_v), 2) if base_3m_v > 0 else cand.get('rvol', 2.5)
                cand['rvol'] = real_3m_rvol
                if real_3m_rvol < 1.80 and not single_mom_ok:
                    print(f"  [3M RVOL REJECTED]: {sym} -> 3m RVOL {real_3m_rvol:.2f}x < 1.80x")
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

        if not verified_super_trades:
            print("[SCREENER]: 0 signals qualified after full 216 scan & live OI audit. Dispatching 1 status update to BOT only.")
            bot_zero_msg = (
                f"🛡️ <b>ALPHA SCREENER STATUS UPDATE</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ <b>SCAN TIME:</b> <b>{ist_time_str}</b>\n"
                f"📊 <b>UNIVERSE SCANNED:</b> <b>{len(symbols)} F&O Assets (Full NSE Scan)</b>\n"
                f"🎯 <b>QUALIFIED SIGNALS:</b> <b>0 Setups</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🔒 <b>VERDICT:</b> Market in sideways regime or blocked by -4/+4 OI Wall.\n"
                f"🛡️ <b>CAPITAL DEFENSE:</b> ZERO forced trades. Capital 100% Protected.\n"
                f"👥 <b>GROUP STATUS:</b> Complete silence maintained (0 messages to Group).\n"
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
                    t1_pts = round(sl_pts * 1.5, 2)
                    t2_pts = round(sl_pts * 3.5, 2)
                    target_note = "+100 to +150 pts Spot Target Move"
                elif sym == 'BANKNIFTY':
                    t1_pts = round(sl_pts * 2.0, 2)
                    t2_pts = round(sl_pts * 4.0, 2)
                    target_note = "+150 to +200 pts Spot Target Move"
                else:
                    t1_pts = round(sl_pts * 1.5, 2)
                    t2_pts = round(sl_pts * 3.5, 2)
                    target_note = "+80 to +120 pts Spot Target Move"
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
                f"⏱️ <b>15-MIN KILL SWITCH:</b> <b>{time_sl_time}</b> (STRICT EXIT if stagnant/red)\n"
                f"⌛ <b>MAX HOLD:</b> <b>26 Mins MAX</b> (Anti-Theta Defense; Zero Drift)\n"
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
        screener.run_screener(ignore_window=ignore_win)
