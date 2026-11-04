import os
import glob
import json
import numpy as np
import pandas as pd

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPORT_DIR, "data_cache_latest_5d_1m")
CONFIG_DIR = os.path.join(os.path.dirname(REPORT_DIR), "alpha_profile_oi_system", "config")

INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
MIN_STOCK_PRICE = 650.0
MIN_STOCK_ATR = 18.0  # Allows SBICARD (18.8), HDFCBANK (22.0), RELIANCE (30.5), eliminates BEL (7.5), ITC (5.0), ONGC (6.1), BHEL (12.5)

MAX_BASE_WIDTH_ATR = 0.65
MAX_BREAKOUT_BUFFER_ATR = 0.20  # Unchoked institutional ignition
MAX_CLIMAX_BAR_ATR = 0.65
MAX_PRIOR_3BAR_ATR = 0.80
MAX_DAY_RANGE_ATR = 2.20
MAX_DAY_PCT_MOVE = 4.50

# Active position caps
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
        with open(cfg_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            res = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    res[k] = float(v.get('atr_wilder_14', 30.0))
                else:
                    res[k] = float(v)
            return res
    return {}

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
        return {'allow_call': True, 'allow_put': True, 'reason': 'START', 'move': 0.0}
    eval_b = nifty_day.iloc[b_idx - 1]
    live_b = nifty_day.iloc[b_idx]
    n_spot = float(live_b['close'])
    ema9 = float(nifty_day['ema9'].iloc[b_idx])
    day_open = float(nifty_day['open'].iloc[0])
    move_from_open = n_spot - day_open

    n_last_bar_rng = float(eval_b['high'] - eval_b['low'])
    if n_last_bar_rng > 65.0:
        return {'allow_call': False, 'allow_put': False, 'reason': 'NIFTY_CLIMAX_EXHAUSTION', 'move': move_from_open}

    recent_rng = nifty_day['high'].iloc[max(0, b_idx - 5):b_idx + 1].max() - nifty_day['low'].iloc[max(0, b_idx - 5):b_idx + 1].min()
    ema_dist = abs(n_spot - ema9)
    if recent_rng < 22.0 and ema_dist < 12.0:
        return {'allow_call': False, 'allow_put': False, 'reason': 'CHOP_TRAP', 'move': move_from_open}

    allow_call = (move_from_open >= -20.0) and (n_spot >= ema9 - 15.0)
    allow_put = (move_from_open <= +20.0) and (n_spot <= ema9 + 15.0)
    return {'allow_call': allow_call, 'allow_put': allow_put, 'reason': 'NORMAL', 'move': move_from_open}

def check_stock_signal_pro(sym, df_today, bar_idx, stock_atr, regime):
    if bar_idx < 10: return None
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

    comp_bars = df_today.iloc[:bar_idx - 1].tail(8)
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

    # Macro Alignment: Anti-Counter Trend
    if direction == 'CALL' and not regime['allow_call']: return None
    if direction == 'PUT' and not regime['allow_put']: return None
    if direction == 'PUT' and regime['move'] > 5.0: return None
    if direction == 'CALL' and regime['move'] < -5.0: return None

    if rng > stock_atr * MAX_CLIMAX_BAR_ATR: return None

    p3_low = df_today.iloc[max(0, bar_idx - 4):bar_idx - 1]['low'].min()
    p3_high = df_today.iloc[max(0, bar_idx - 4):bar_idx - 1]['high'].max()
    p3_move = (b_close - p3_low) if direction == 'CALL' else (p3_high - b_close)
    if p3_move > stock_atr * MAX_PRIOR_3BAR_ATR: return None

    # Institutional Turnover Floor: Minimum Rs 6.0 Cr
    curr_v = float(eval_b.get('volume', 0.0))
    prior_10_bars = df_today.iloc[:bar_idx - 1].tail(10)
    base_10_v = float(prior_10_bars['volume'].mean()) if len(prior_10_bars) > 0 else 1000.0
    real_3m_rvol = round(curr_v / max(1.0, base_10_v), 2)
    turnover_cr = round((curr_v * spot) / 1e7, 2)

    if real_3m_rvol < 1.80 or turnover_cr < 6.0:
        return None

    cvd_res = compute_bar_delta_proxy(df_today.iloc[:bar_idx], is_index=False)
    rec_cvd = cvd_res['recent_cvd_ratio']
    bar_delta_ratio = round((2.0 * b_close - (b_high + b_low)) / rng, 3)

    if direction == 'CALL':
        cvd_pass = (rec_cvd >= 0.15 or bar_delta_ratio >= 0.35) and (rec_cvd >= 0.0)
    else:
        cvd_pass = (rec_cvd <= -0.15 or bar_delta_ratio <= -0.35) and (rec_cvd <= 0.0)

    if not cvd_pass: return None

    near_hl = (direction == 'CALL' and b_close >= stk_day_high - stock_atr * 0.15) or \
              (direction == 'PUT' and b_close <= stk_day_low + stock_atr * 0.15)
    if not near_hl: return None

    body = abs(b_close - b_open)
    solid_body_ratio = round(body / rng, 2)
    clv = round((b_close - b_low) / rng, 2)
    if solid_body_ratio < 0.48: return None

    clv_pass = (clv >= 0.58) if direction == 'CALL' else (clv <= 0.42)
    if not clv_pass: return None

    score = min(99.0, 55.0 + min(25.0, (real_3m_rvol - 1.0) * 10.0) + (15.0 if abs(bar_delta_ratio) >= 0.35 else 5.0) + min(10.0, turnover_cr * 1.5))
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
        'score': round(score, 2),
        'type': 'PURE_STOCK_CALL' if direction == 'CALL' else 'PURE_STOCK_PUT',
        'base_high': comp_high,
        'base_low': comp_low
    }

def check_index_signal_pro(sym, df_today, bar_idx, regime):
    if bar_idx < 10: return None
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
    comp_high = float(comp_bars['high'].max())
    comp_low = float(comp_bars['low'].min())
    comp_width = comp_high - comp_low

    max_allowed_base = 120.0 if sym != 'BANKNIFTY' else 350.0
    if comp_width > max_allowed_base: return None

    idx_dist = 35.0 if sym != 'BANKNIFTY' else 80.0
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
    cvd_pass = (rec_cvd >= 0.12) if direction == 'CALL' else (rec_cvd <= -0.12)
    if not cvd_pass: return None

    session_rngs = df_today.iloc[:bar_idx]['high'] - df_today.iloc[:bar_idx]['low']
    median_rng = float(session_rngs.median()) if len(session_rngs) > 0 else 15.0
    idx_rvol = round(rng / max(0.5, median_rng), 2)
    if idx_rvol < 1.15: return None

    idx_pct_move = ((cur_p - idx_day_open) / max(1.0, idx_day_open)) * 100.0
    if abs(idx_pct_move) > 1.10: return None

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

class ActiveTrade:
    def __init__(self, cand, entry_idx):
        self.cand = cand
        self.entry_idx = entry_idx
        self.sym = cand['symbol']
        self.direction = cand['direction']
        self.entry_p = float(cand['entry_price'])
        self.is_idx = cand.get('is_index', False)
        self.is_bank = (self.sym == 'BANKNIFTY')
        self.atr = cand.get('stock_atr', 30.0)
        self.step = 0
        self.is_closed = False
        self.exit_p = self.entry_p
        self.exit_reason = None
        self.exit_time = None
        self.t1_hit = False
        self.is_risk_free = False

        if self.is_idx:
            is_mid = (self.sym == 'MIDCPNIFTY')
            self.spot_t1 = 120.0 if self.is_bank else (60.0 if is_mid else 90.0)
            self.spot_t2 = 250.0 if self.is_bank else (120.0 if is_mid else 160.0)
            self.spot_sl = 60.0 if self.is_bank else (30.0 if is_mid else 40.0)
        else:
            # Pro Option Buyer Point Targets: 25 to 50+ pts T1, 55 to 100+ pts T2!
            self.spot_t1 = max(25.0, min(50.0, round(self.atr * 0.35, 1)))
            self.spot_t2 = max(55.0, min(100.0, round(self.atr * 0.75, 1)))
            self.spot_sl = max(15.0, min(30.0, round(self.atr * 0.18, 1)))

    def update(self, b_high, b_low, b_close, ts):
        if self.is_closed: return
        self.step += 1
        fav_move = (b_high - self.entry_p) if self.direction == 'CALL' else (self.entry_p - b_low)
        adv_move = (self.entry_p - b_low) if self.direction == 'CALL' else (b_high - self.entry_p)
        cur_gain = (b_close - self.entry_p) if self.direction == 'CALL' else (self.entry_p - b_close)

        if fav_move >= max(20.0, self.spot_t1 * 0.40):
            self.is_risk_free = True

        if not self.t1_hit and fav_move >= self.spot_t1:
            self.t1_hit = True
            self.is_risk_free = True

        if self.t1_hit:
            if fav_move >= self.spot_t2:
                self.exit_p = (self.entry_p + self.spot_t2) if self.direction == 'CALL' else (self.entry_p - self.spot_t2)
                self.exit_reason = "TARGET_2_HIT (JACKPOT)"
                self.exit_time = ts
                self.is_closed = True
                return
            lock_pts = self.spot_t1 * 0.40
            locked_sl = (self.entry_p + lock_pts) if self.direction == 'CALL' else (self.entry_p - lock_pts)
            if (self.direction == 'CALL' and b_low <= locked_sl) or (self.direction == 'PUT' and b_high >= locked_sl):
                self.exit_p = locked_sl
                self.exit_reason = "T1_PROFIT_LOCKED_EXIT"
                self.exit_time = ts
                self.is_closed = True
                return
        else:
            if adv_move >= self.spot_sl:
                self.exit_p = (self.entry_p - self.spot_sl) if self.direction == 'CALL' else (self.entry_p + self.spot_sl)
                self.exit_reason = "SL_HIT"
                self.exit_time = ts
                self.is_closed = True
                return
            # Pro Time SL Scratch:
            # 1. At 15m (step == 4): Scratch ONLY if trade is breaking down severely (loss > 35% of SL)
            if self.step == 4 and cur_gain < -(self.spot_sl * 0.35):
                self.exit_p = b_close
                self.exit_reason = "15M_TIME_SL_EXIT"
                self.exit_time = ts
                self.is_closed = True
                return
            # 2. At 25-30m (step >= 8): If trade is still dead/negative without any profit traction, scratch it!
            if self.step >= 8 and cur_gain <= 0.0:
                self.exit_p = b_close
                self.exit_reason = "30M_DEAD_SCRATCH"
                self.exit_time = ts
                self.is_closed = True
                return

    def close_eod(self, final_close, ts):
        if not self.is_closed:
            self.exit_p = final_close
            self.exit_reason = "EOD_EXIT"
            self.exit_time = ts
            self.is_closed = True

    def get_summary(self, d_str):
        pts_captured = round((self.exit_p - self.entry_p) if self.direction == 'CALL' else (self.entry_p - self.exit_p), 2)
        sl_denom = max(0.5, self.spot_sl)
        rr_realized = round(pts_captured / sl_denom, 2) if pts_captured != 0 else 0.0
        return {
            'date': d_str,
            'symbol': self.sym,
            'direction': self.direction,
            'entry_price': self.entry_p,
            'eval_time': self.cand['eval_time'],
            'exit_time': self.exit_time,
            'is_index': self.is_idx,
            'stock_atr': self.atr,
            'rvol': self.cand['rvol'],
            'turnover_cr': self.cand['turnover_cr'],
            'pct_chg': self.cand['pct_chg'],
            'score': self.cand['score'],
            'type': self.cand['type'],
            'exit_price': self.exit_p,
            'points_captured': pts_captured,
            'spot_sl': self.spot_sl,
            'spot_t1': self.spot_t1,
            'spot_t2': self.spot_t2,
            'exit_reason': self.exit_reason,
            'realized_r': rr_realized
        }

if __name__ == "__main__":
    print("Loading latest resampled 3m data...")
    asset_data = load_resampled_3m_data()
    atrs = load_verified_atrs()
    all_dates = sorted(list(set(d for df in asset_data.values() for d in df.index.date)))
    stock_syms = [s for s in asset_data.keys() if s not in INDEX_NAMES]
    all_executed_trades = []

    print(f"Loaded {len(asset_data)} active assets. Backtest dates: {[str(d) for d in all_dates]}")

    for d in all_dates:
        nifty_day = asset_data['NIFTY'][asset_data['NIFTY'].index.date == d].copy()
        if nifty_day.empty: continue
        nifty_day['ema9'] = nifty_day['close'].ewm(span=9, adjust=False).mean()

        active_trades = []
        daily_traded_symbols = set()
        time_index = nifty_day.index

        for b_idx in range(10, len(time_index)):
            curr_t = time_index[b_idx]
            tot_m = curr_t.hour * 60 + curr_t.minute
            is_m = (570 <= tot_m <= 615)
            is_a = (795 <= tot_m <= 855)

            # 1. Update active trades using exact timestamp matching!
            for tr in active_trades:
                if not tr.is_closed:
                    df_asset = asset_data[tr.sym][asset_data[tr.sym].index.date == d]
                    if curr_t in df_asset.index:
                        b = df_asset.loc[curr_t]
                        if isinstance(b, pd.DataFrame): b = b.iloc[0]
                        tr.update(float(b['high']), float(b['low']), float(b['close']), curr_t)

            # Remove closed trades from active list
            active_trades = [tr for tr in active_trades if not tr.is_closed]

            if not (is_m or is_a): continue

            regime = get_nifty_regime(nifty_day, b_idx)
            if not (regime['allow_call'] or regime['allow_put']): continue

            # Count active positions with capital at risk (Risk-free runners with locked profit don't consume slots!)
            act_stocks = sum(1 for tr in active_trades if (not tr.is_idx) and (not tr.is_risk_free))
            act_broad_idx = sum(1 for tr in active_trades if tr.is_idx and (not tr.is_bank) and (not tr.is_risk_free))
            act_bank_idx = sum(1 for tr in active_trades if tr.is_idx and tr.is_bank and (not tr.is_risk_free))

            # 2. Check Index Candidates
            for isym in INDEX_NAMES:
                if isym not in asset_data or isym in daily_traded_symbols: continue
                is_bank = (isym == 'BANKNIFTY')
                if is_bank and act_bank_idx >= MAX_ACTIVE_BANK_INDEX: continue
                if (not is_bank) and act_broad_idx >= MAX_ACTIVE_BROAD_INDEX: continue

                idf = asset_data[isym][asset_data[isym].index.date == d]
                if len(idf) <= b_idx: continue
                sig = check_index_signal_pro(isym, idf, b_idx, regime)
                if sig:
                    trade = ActiveTrade(sig, b_idx)
                    active_trades.append(trade)
                    all_executed_trades.append(trade)
                    daily_traded_symbols.add(isym)
                    if is_bank: act_bank_idx += 1
                    else: act_broad_idx += 1

            # 3. Check Stock Candidates (Ranked by institutional score!)
            if act_stocks < MAX_ACTIVE_STOCKS:
                stock_cands = []
                for ssym in stock_syms:
                    if ssym in daily_traded_symbols: continue
                    sdf = asset_data[ssym][asset_data[ssym].index.date == d]
                    if len(sdf) <= b_idx: continue
                    stock_atr = atrs.get(ssym, 30.0)
                    sig = check_stock_signal_pro(ssym, sdf, b_idx, stock_atr, regime)
                    if sig:
                        stock_cands.append(sig)

                # Sort by institutional score descending!
                stock_cands.sort(key=lambda x: x['score'], reverse=True)
                slots_available = MAX_ACTIVE_STOCKS - act_stocks
                for chosen in stock_cands[:slots_available]:
                    trade = ActiveTrade(chosen, b_idx)
                    active_trades.append(trade)
                    all_executed_trades.append(trade)
                    daily_traded_symbols.add(chosen['symbol'])
                    act_stocks += 1

        # End of day: close remaining active trades
        for tr in active_trades:
            if not tr.is_closed:
                df_asset = asset_data[tr.sym][asset_data[tr.sym].index.date == d]
                if len(df_asset) > 0:
                    final_close = float(df_asset.iloc[-1]['close'])
                    tr.close_eod(final_close, df_asset.index[-1])

    trade_summaries = [tr.get_summary(str(tr.cand['eval_time'].date())) for tr in all_executed_trades]
    df_res = pd.DataFrame(trade_summaries)

    csv_path = os.path.join(REPORT_DIR, "Alpha_Latest_5Day_Ending_Today_Backtest.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved authoritative backtest CSV to: {csv_path}")

    print("\n" + "="*80)
    print("PRO-TRADER SEPARATED ENGINE (TRUE HIGH-MOMENTUM WINNERS)")
    print("="*80)
    print(f"Total Dispatched Trades: {len(df_res)}")
    wins = df_res[df_res['points_captured'] > 0]
    losses = df_res[df_res['points_captured'] <= 0]
    scratches = df_res[df_res['exit_reason'].str.contains('SCRATCH|TIME_SL', na=False)]
    full_sl = df_res[df_res['exit_reason'] == 'SL_HIT']

    gross_p = wins['realized_r'].sum()
    gross_l = abs(losses['realized_r'].sum())
    net_p = gross_p - gross_l
    pf = round(gross_p / max(0.01, gross_l), 2)

    print(f"Wins: {len(wins)} ({len(wins)/max(1, len(df_res))*100:.1f}%)")
    print(f"Scratches: {len(scratches)}")
    print(f"Full Stoploss: {len(full_sl)}")
    print(f"Gross Profit: +{gross_p:.2f}R | Gross Loss: -{gross_l:.2f}R | Net Profit: +{net_p:.2f}R | Profit Factor: {pf}")
    print("\n--- ALL EXECUTED TRADES ---")
    for _, r in df_res.iterrows():
        e_str = r['eval_time'].strftime('%Y-%m-%d %H:%M')
        print(f"{e_str} | {r['symbol']:<12} | {r['direction']:<4} | En: {r['entry_price']:8.2f} | Ex: {r['exit_price']:8.2f} | Pts: {r['points_captured']:+7.2f} | R: {r['realized_r']:+5.2f}R | {r['exit_reason']}")
