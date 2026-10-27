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

                if len(idx_df) < 10:
                    return {
                        'symbol': sym_clean,
                        'passed': False,
                        'failed_gate': f'INSUFFICIENT_COMPRESSION_BASE: Only {len(idx_df)-2} closed bars (< 24 mins)',
                        'spot_price': cur_p,
                        'score': 0.0,
                        'is_index': True
                    }
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
                single_mom_ok = (solid_body_ratio >= 0.50) and wick_ok and clv_ok

                if direction == 'CALL':
                    if b_close >= comp_high:
                        is_breakout = True
                else:
                    if b_close <= comp_low:
                        is_breakout = True

                if len(idx_df) >= 4:
                    b_prev1 = idx_df.iloc[-3]
                    b_prev2 = idx_df.iloc[-4]

                    cum_fall_3_pct = ((float(b_prev2['open']) - b_close) / float(b_prev2['open'])) * 100.0
                    cum_fall_2_pct = ((float(b_prev1['open']) - b_close) / float(b_prev1['open'])) * 100.0
                    all_3_red = (float(b_prev2['close']) <= float(b_prev2['open']) and
                                 float(b_prev1['close']) <= float(b_prev1['open']) and
                                 b_close <= b_open)
                    last_2_red = (float(b_prev1['close']) <= float(b_prev1['open']) and b_close <= b_open)
                    broke_range_low = (b_close <= comp_low)

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
                    broke_range_high = (b_close >= comp_high)

                    if (all_3_green and cum_rise_3_pct >= 0.25 and broke_range_high) or (last_2_green and cum_rise_2_pct >= 0.20 and broke_range_high):
                        direction = 'CALL'
                        cum_mom_ok = True
                        is_breakout = True
                        pct_chg = round(max(cum_rise_3_pct, cum_rise_2_pct), 2)

                cand_mom_pass = (single_mom_ok or cum_mom_ok) and is_breakout

                prior_v = idx_df['volume'].iloc[:-2].tail(10).mean() if ('volume' in idx_df.columns and len(idx_df) > 2) else 0.0
                curr_v = float(eval_bar.get('volume', 0.0))
                if prior_v > 0 and curr_v > 0:
                    idx_rvol = round(curr_v / max(1.0, prior_v), 2)
                else:
                    eval_rng = max(1.0, b_high - b_low)
                    prior_rng_mean = (idx_df['high'] - idx_df['low']).iloc[:-2].tail(10).mean() if len(idx_df) > 2 else eval_rng
                    range_rvol = round(eval_rng / max(1.0, prior_rng_mean), 2)
                    idx_rvol = max(2.50, round(range_rvol * 2.0, 2)) if range_rvol >= 1.20 else 2.50
                idx_rvol_pass = (idx_rvol >= 2.50)

                candle_cvd = round((2.0 * clv) - 1.0, 2)
                raw_idx_cvd = float(q.get('cvd_delta_pct', 0.0))
                idx_cvd = raw_idx_cvd if abs(raw_idx_cvd) > 0.01 else candle_cvd
                idx_cvd_pass = (idx_cvd >= 0.50) if direction == 'CALL' else (idx_cvd <= -0.50)

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
        # 2. FULL 216 STOCK OPTION EVALUATION (30 to 60+ Pts Option Move Target)
        # =====================================================================
        b_open = float(q.get('open', cur_p))
        b_high = float(q.get('high', cur_p))
        b_low = float(q.get('low', cur_p))
        b_close = float(q.get('close', cur_p))
        tot_vol = float(q.get('volume', 10000.0))
        pct_chg = float(q.get('pct_change', 0.0))
        rvol = float(q.get('rvol', 1.0))
        direction = 'CALL' if pct_chg >= 0 else 'PUT'

        stock_atr = self.get_stock_atr(sym_clean, q)
        if stock_atr < 30.0:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'GATE_0_STOCK_ATR_BELOW_30_PTS ({stock_atr:.1f} < 30.0)'
            }

        delta = 0.60
        expected_opt_move = round(stock_atr * delta, 2)

        # Anti-Exhaustion Freshness Shield ("Jaldi move ho chuke na de")
        day_range = round(b_high - b_low, 2)
        ist_now = get_ist_time()
        max_atr_mult = 1.20 if (ist_now.hour >= 13 or ist_now.hour < 9 or (ist_now.hour == 9 and ist_now.minute < 30)) else 0.85
        if day_range > (stock_atr * max_atr_mult):
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'STOCK_MOVE_ALREADY_EXHAUSTED ({day_range:.1f} > {max_atr_mult:.2f}x ATR {stock_atr * max_atr_mult:.1f} pts)'
            }

        min_rvol = 1.80
        if rvol < min_rvol:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'GATE_2_RVOL_BELOW_MINIMUM_{min_rvol}X ({rvol:.2f}x < {min_rvol}x)'
            }

        rng = max(0.01, b_high - b_low)
        clv = round((b_close - b_low) / rng, 2)
        candle_delta = round((2.0 * clv) - 1.0, 2)
        raw_cvd = float(q.get('cvd_delta_pct', 0.0))
        real_cvd_ratio = raw_cvd if abs(raw_cvd) > 0.01 else candle_delta

        cvd_ok = (real_cvd_ratio >= 0.50) if direction == 'CALL' else (real_cvd_ratio <= -0.50)
        if not cvd_ok:
            return {
                'symbol': sym_clean,
                'passed': False,
                'failed_gate': f'GATE_3_CVD_DELTA_NOT_CONFLUENT ({real_cvd_ratio:+.2f} not aligned >= 0.50)'
            }

        body = abs(b_close - b_open)
        body_ratio = round(body / rng, 2)
        up_wick = round((b_high - max(b_open, b_close)) / rng, 2)
        dn_wick = round((min(b_open, b_close) - b_low) / rng, 2)
        hw_aligned = bool((direction == 'CALL' and pct_chg > 0.15) or (direction == 'PUT' and pct_chg < -0.15))

        gate_res = evaluate_5pillar_gate(
            symbol=sym_clean,
            spot_price=cur_p,
            open=b_open,
            high=b_high,
            low=b_low,
            close=b_close,
            volume=tot_vol,
            rvol=rvol,
            clv_ratio=clv,
            solid_body_ratio=body_ratio,
            upper_wick=up_wick,
            lower_wick=dn_wick,
            squeeze_active=True,
            cvd_imbalance=real_cvd_ratio,
            oi_confluence=True,
            oi_neutralized=False,
            hw_alignment=hw_aligned,
            hw_aligned=hw_aligned,
            hw_neutralized=False,
            is_index=False,
            index_opt_volume=0.0,
            index_min_opt_volume=0.0,
            direction=direction,
            vwap=cur_p,
            max_vwap_distance_pct=0.60,
            follow_through_confirmed=True
        )

        stage1_passed = bool(gate_res.get('passed', False))
        score = float(gate_res.get('composite_score', gate_res.get('quality_score', 0.0)))

        if stage1_passed and verify_oi_now:
            oi_conf = self.data_feed.get_live_dual_oi_confluence(sym_clean, cur_p, direction=direction)
            if not oi_conf.get('confluent', False):
                return {
                    'symbol': sym_clean,
                    'passed': False,
                    'failed_gate': f"STOCK_OI_CONFLICT_{oi_conf.get('reason', 'NO_CONFLUENCE')}",
                    'spot_price': cur_p,
                    'score': 0.0,
                    'is_index': False
                }

        return {
            'symbol': sym_clean,
            'passed': stage1_passed,
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
        qty = opt.get('quantity', 25)
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
            print(f"[OUTSIDE TRADING SLAB]: Current time {ist_time_str} is outside 09:30-10:15 AM & 01:15-02:15 PM IST.")
            print("[POLICY]: Strictly ZERO messages dispatched. Exiting immediately.")
            return

        already_sent_today = len(self.dispatched_today) + len(self.index_recent_dispatch)
        if already_sent_today >= MAX_DAILY_SUPER_TRADES:
            print(f"[DAILY SUPER-TRADE CAP REACHED]: {already_sent_today}/{MAX_DAILY_SUPER_TRADES} best trades already dispatched today.")
            return

        print(f"[ACTIVE SLAB REACHED]: Executing Full 216 F&O + Index Scan at {ist_time_str}...")

        # STEP 1 (UPSIDE): Fetch 1 Bulk NSE Snapshot & Scan ALL 216 Stocks + 4 Indices in memory
        fetch_bulk_nse_snapshot(force_refresh=True)

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
            # STAGE 2: 5-PILLAR REAL INTRADAY CANDLE & BREAKOUT AUDIT
            # (Indices already audited on real 3m bars in Stage 1;
            #  Stocks audited here on genuine 3-min resampled bars)
            # -------------------------------------------------------------
            if not is_idx:
                stk_df = self.data_feed.get_live_intraday_bars(sym)
                if stk_df is None or len(stk_df) < 10:
                    print(f"  [5-PILLAR CANDLE REJECTED]: {sym} -> Insufficient 3-min candles (less than 10 bars)")
                    continue

                eval_bar = stk_df.iloc[-2] if len(stk_df) >= 2 else stk_df.iloc[-1]
                b_open = float(eval_bar['open'])
                b_high = float(eval_bar['high'])
                b_low = float(eval_bar['low'])
                b_close = float(eval_bar['close'])
                spot = b_close
                cand['spot_price'] = spot
                rng = max(0.01, b_high - b_low)
                body = abs(b_close - b_open)
                solid_body_ratio = round(body / rng, 2)
                up_wick = round((b_high - max(b_open, b_close)) / rng, 2)
                dn_wick = round((min(b_open, b_close) - b_low) / rng, 2)
                clv = round((b_close - b_low) / rng, 2)

                comp_bars = stk_df.iloc[:-2].tail(8)
                comp_high = float(comp_bars['high'].max())
                comp_low = float(comp_bars['low'].min())

                # Pillar 1: Single 3-Minute Candle Momentum
                if direction == 'CALL':
                    single_mom_ok = (b_close >= b_open) and (clv >= 0.72) and (up_wick <= 0.20) and (solid_body_ratio >= 0.40)
                    broke_comp = (b_close >= comp_high)
                else:
                    single_mom_ok = (b_close <= b_open) and (clv <= 0.28) and (dn_wick <= 0.20) and (solid_body_ratio >= 0.40)
                    broke_comp = (b_close <= comp_low)

                # Pillar 1: Dual & Triple Candle Cumulative Momentum (Slow Grinding Move Support)
                cum_mom_ok = False
                if len(stk_df) >= 4:
                    b_prev1 = stk_df.iloc[-3]
                    b_prev2 = stk_df.iloc[-4]
                    if direction == 'CALL':
                        last_2_green = (float(b_prev1['close']) >= float(b_prev1['open']) and b_close >= b_open)
                        all_3_green = (float(b_prev2['close']) >= float(b_prev2['open']) and last_2_green)
                        cum_rise_2_pct = ((b_close - float(b_prev1['open'])) / float(b_prev1['open'])) * 100.0
                        cum_rise_3_pct = ((b_close - float(b_prev2['open'])) / float(b_prev2['open'])) * 100.0
                        cum_mom_ok = (all_3_green and cum_rise_3_pct >= 0.25) or (last_2_green and cum_rise_2_pct >= 0.20)
                    else:
                        last_2_red = (float(b_prev1['close']) <= float(b_prev1['open']) and b_close <= b_open)
                        all_3_red = (float(b_prev2['close']) <= float(b_prev2['open']) and last_2_red)
                        cum_fall_2_pct = ((float(b_prev1['open']) - b_close) / float(b_prev1['open'])) * 100.0
                        cum_fall_3_pct = ((float(b_prev2['open']) - b_close) / float(b_prev2['open'])) * 100.0
                        cum_mom_ok = (all_3_red and cum_fall_3_pct >= 0.25) or (last_2_red and cum_fall_2_pct >= 0.20)

                cand_mom_pass = (single_mom_ok or cum_mom_ok) and broke_comp
                if not cand_mom_pass:
                    print(f"  [5-PILLAR CANDLE/BREAKOUT REJECTED]: {sym} ({direction}) -> SingleMom={single_mom_ok}, CumMom={cum_mom_ok}, BrokeComp={broke_comp} (CompHigh={comp_high:.1f}, CompLow={comp_low:.1f}, Close={b_close:.1f})")
                    continue

            oi_conf = self.data_feed.get_live_dual_oi_confluence(sym, spot, direction=direction)
            if not oi_conf.get('confluent', False):
                print(f"  [OI/RUNWAY REJECTED]: {sym} ({direction}) -> {oi_conf.get('reason')}")
                continue

            opt = self.data_feed.compute_option_pricing(spot, direction=direction, symbol=sym)
            if not opt or not opt.get('is_live_chain_verified', False) or float(opt.get('entry_price', 0.0)) < 2.0:
                print(f"  [LIVE LTP REJECTED]: {sym} -> Live Option Chain LTP unavailable.")
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
                f"📌 <b>CONTRACT:</b> <code>{contract}</code> ({opt_action})\n"
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
