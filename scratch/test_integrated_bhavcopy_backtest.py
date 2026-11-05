import sys, os, glob, json, datetime
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('scratch'))
import numpy as np
import pandas as pd
from execute_latest_5d_backtest import (
    load_resampled_3m_data, load_verified_atrs, get_nifty_regime,
    check_stock_signal_pro, check_index_signal_pro, ActiveTrade,
    INDEX_NAMES, STOCK_SECTOR_MAP, MAX_ACTIVE_STOCKS, MAX_ACTIVE_BROAD_INDEX, MAX_ACTIVE_BANK_INDEX
)

# Load Bhavcopy Cache
bhav_cache = {}
def get_bhav(date_obj):
    # date_obj: datetime.date
    d_fmt = date_obj.strftime('%d-%m-%Y')
    p = os.path.join('alpha_profile_oi_system/bhavcopy_cache', f'fno_bhav_{d_fmt}.parquet')
    if p not in bhav_cache:
        if os.path.exists(p):
            bhav_cache[p] = pd.read_parquet(p)
        else:
            bhav_cache[p] = None
    return bhav_cache[p]

def check_oi_runway_bhavcopy(sym, date_obj, spot, direction, target_price=None, is_index=False):
    df_bhav = get_bhav(date_obj)
    if df_bhav is None:
        return {'confluent': True, 'reason': 'NO_BHAVCOPY'}
    
    sub = df_bhav[(df_bhav['TckrSymb'] == sym) & (df_bhav['OptnTp'].isin(['CE', 'PE']))]
    if sub.empty:
        return {'confluent': True, 'reason': 'EMPTY_BHAVCOPY_OPTIONS'}
    
    # Nearest active expiry
    cur_exp = sub['XpryDt'].min()
    exp_df = sub[sub['XpryDt'] == cur_exp]
    
    strike_data = {}
    for _, row in exp_df.iterrows():
        stk = float(row['StrkPric'])
        opt_tp = str(row['OptnTp']).upper().strip()
        oi = float(row['OpnIntrst'])
        chg_oi = float(row['ChngInOpnIntrst'])
        vol = float(row['TtlTradgVol'])
        
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

    sorted_strikes = sorted(strike_data.keys())
    if not sorted_strikes:
        return {'confluent': True, 'reason': 'NO_STRIKES'}

    atm_strike = min(sorted_strikes, key=lambda s: abs(s - spot))
    atm_idx = sorted_strikes.index(atm_strike)

    lower_4 = sorted_strikes[max(0, atm_idx - 4) : atm_idx]
    upper_4 = sorted_strikes[atm_idx + 1 : min(len(sorted_strikes), atm_idx + 5)]
    window_9 = lower_4 + [atm_strike] + upper_4

    strike_step = (sorted_strikes[1] - sorted_strikes[0]) if len(sorted_strikes) > 1 else 50.0

    atm_info = strike_data[atm_strike]
    tot_window_ce_oi = sum(strike_data[s]['ce_oi'] for s in window_9)
    tot_window_pe_oi = sum(strike_data[s]['pe_oi'] for s in window_9)

    tot_upper_ce_chg = sum(strike_data[s]['ce_chg_oi'] for s in upper_4)
    tot_upper_pe_chg = sum(strike_data[s]['pe_chg_oi'] for s in upper_4)
    tot_upper_ce_vol = sum(strike_data[s]['ce_vol'] for s in upper_4)
    tot_upper_ce_oi = sum(strike_data[s]['ce_oi'] for s in upper_4)

    tot_lower_pe_chg = sum(strike_data[s]['pe_chg_oi'] for s in lower_4)
    tot_lower_ce_chg = sum(strike_data[s]['ce_chg_oi'] for s in lower_4)
    tot_lower_pe_vol = sum(strike_data[s]['pe_vol'] for s in lower_4)
    tot_lower_pe_oi = sum(strike_data[s]['pe_oi'] for s in lower_4)

    support_pe_chg = atm_info['pe_chg_oi'] + tot_lower_pe_chg
    resistance_ce_chg = atm_info['ce_chg_oi'] + tot_upper_ce_chg

    # Share units thresholds
    min_wall_oi = 2500000.0 if is_index else 40000.0
    monster_standing_oi = 5000000.0 if is_index else 80000.0
    ambush_wall_min = 1500000.0 if is_index else 35000.0

    if direction == 'CALL':
        target_barrier = float(target_price) if target_price else (spot + (200.0 if is_index else 100.0))
        candidate_wall_strikes = [s for s in sorted_strikes if (s >= (atm_strike - strike_step * 0.2) and s <= target_barrier + strike_step * 0.6)]

        # 1. Ambush Detection: upper 4 strikes call writing heavily dominating
        is_otm_ce_ambush = (tot_upper_ce_chg >= ambush_wall_min) and (tot_upper_ce_chg > 2.0 * max(3000.0, support_pe_chg)) and (atm_info['ce_chg_oi'] >= 0)
        gamma_squeeze_call = (tot_upper_ce_vol >= 6.0 * max(1.0, tot_upper_ce_oi))

        if is_otm_ce_ambush and not gamma_squeeze_call:
            return {'confluent': False, 'reason': f'UPPER_OTM_CALL_WRITING_AMBUSH (+{int(tot_upper_ce_chg)} CE vs +{int(support_pe_chg)} PE)'}

        # 2. Strict ATM Gate
        atm_ce_chg = atm_info['ce_chg_oi']
        atm_pe_chg = atm_info['pe_chg_oi']
        atm_blocked = (atm_ce_chg > 0) and (atm_ce_chg > max(1000.0, atm_pe_chg * 1.15))
        if atm_blocked:
            return {'confluent': False, 'reason': f'ATM_CALL_WRITERS_FIGHTING (+{int(atm_ce_chg)} CE vs +{int(atm_pe_chg)} PE at {int(atm_strike)})'}

        # 3. Target Corridor Resistance Wall Check
        res_wall_strike = None
        for up_s in candidate_wall_strikes:
            st_d = strike_data[up_s]
            ce_oi = st_d['ce_oi']
            pe_oi = st_d['pe_oi']
            ce_chg = st_d['ce_chg_oi']
            pe_chg = st_d['pe_chg_oi']

            # If call writers are unwinding (ce_chg < 0) or PE writers dominate (pe_chg >= ce_chg), this strike is supported, NOT a wall!
            if ce_chg < 0 or pe_chg >= ce_chg:
                continue

            if ce_oi < min_wall_oi: continue

            ce_conc = ce_oi / max(1.0, tot_window_ce_oi)
            is_concentrated = (ce_conc >= 0.18) or (ce_oi >= 1.25 * max(1.0, atm_info['ce_oi']))
            is_standing_wall = (ce_oi >= monster_standing_oi)
            is_ratio_heavy = (ce_oi > 1.5 * max(1000.0, pe_oi)) or (is_standing_wall and ce_oi > 1.20 * max(500.0, pe_oi))

            if (is_concentrated or is_standing_wall) and is_ratio_heavy:
                if st_d['ce_vol'] >= 6.0 * ce_oi: continue
                res_wall_strike = up_s
                break

        confluent = (res_wall_strike is None)
        reason = f'CALL_RUNWAY_CLEAR_ATM_{int(atm_strike)}' if confluent else f'CALL_BLOCKED_BY_WALL_AT_{int(res_wall_strike)}'
        return {'confluent': confluent, 'reason': reason, 'atm_strike': atm_strike}

    else:
        target_barrier = float(target_price) if target_price else (spot - (200.0 if is_index else 100.0))
        candidate_wall_strikes = [s for s in sorted_strikes if (target_barrier - strike_step * 0.6 <= s <= atm_strike + strike_step * 0.2)]

        is_otm_pe_ambush = (tot_lower_pe_chg >= ambush_wall_min) and (tot_lower_pe_chg > 2.0 * max(3000.0, resistance_ce_chg)) and (atm_info['pe_chg_oi'] >= 0)
        gamma_squeeze_put = (tot_lower_pe_vol >= 6.0 * max(1.0, tot_lower_pe_oi))

        if is_otm_pe_ambush and not gamma_squeeze_put:
            return {'confluent': False, 'reason': f'LOWER_OTM_PUT_WRITING_AMBUSH (+{int(tot_lower_pe_chg)} PE vs +{int(resistance_ce_chg)} CE)'}

        atm_pe_chg = atm_info['pe_chg_oi']
        atm_ce_chg = atm_info['ce_chg_oi']
        atm_blocked = (atm_pe_chg > 0) and (atm_pe_chg > max(1000.0, atm_ce_chg * 1.15))
        if atm_blocked:
            return {'confluent': False, 'reason': f'ATM_PUT_WRITERS_FIGHTING (+{int(atm_pe_chg)} PE vs +{int(atm_ce_chg)} CE at {int(atm_strike)})'}

        sup_wall_strike = None
        for dn_s in sorted(candidate_wall_strikes, reverse=True):
            st_d = strike_data[dn_s]
            pe_oi = st_d['pe_oi']
            ce_oi = st_d['ce_oi']
            pe_chg = st_d['pe_chg_oi']
            ce_chg = st_d['ce_chg_oi']

            # If put writers are unwinding (pe_chg < 0) or CE writers dominate (ce_chg >= pe_chg), this strike is resisted, NOT a floor!
            if pe_chg < 0 or ce_chg >= pe_chg:
                continue

            if pe_oi < min_wall_oi: continue

            pe_conc = pe_oi / max(1.0, tot_window_pe_oi)
            is_concentrated = (pe_conc >= 0.18) or (pe_oi >= 1.25 * max(1.0, atm_info['pe_oi']))
            is_standing_wall = (pe_oi >= monster_standing_oi)
            is_ratio_heavy = (pe_oi > 1.5 * max(1000.0, ce_oi)) or (is_standing_wall and pe_oi > 1.20 * max(500.0, ce_oi))

            if (is_concentrated or is_standing_wall) and is_ratio_heavy:
                if st_d['pe_vol'] >= 6.0 * pe_oi: continue
                sup_wall_strike = dn_s
                break

        confluent = (sup_wall_strike is None)
        reason = f'PUT_RUNWAY_CLEAR_ATM_{int(atm_strike)}' if confluent else f'PUT_BLOCKED_BY_WALL_AT_{int(sup_wall_strike)}'
        return {'confluent': confluent, 'reason': reason, 'atm_strike': atm_strike}

# Run Full 5-Day Dynamic Simulation with True Bhavcopy -4 to +4 OI Confluence
print("Loading 3m intraday data and verified ATRs...")
asset_data = load_resampled_3m_data()
atrs = load_verified_atrs()
all_dates = sorted(list(set([d for df in asset_data.values() for d in df.index.date])))
backtest_dates = all_dates[-5:]
print('Loaded active assets:', len(asset_data), 'Backtest dates:', backtest_dates)

all_executed_trades = []

for d in backtest_dates:
    nifty_day = asset_data['NIFTY'][asset_data['NIFTY'].index.date == d].copy()
    if len(nifty_day) < 15: continue
    nifty_day['ema9'] = nifty_day['close'].ewm(span=9, adjust=False).mean()
    
    daily_traded_symbols = set()
    active_trades = []
    sector_last_trade = {}
    stock_syms = [s for s in asset_data.keys() if s not in INDEX_NAMES]
    time_index = nifty_day.index

    for b_idx in range(10, len(time_index)):
        curr_t = time_index[b_idx]
        tot_m = curr_t.hour * 60 + curr_t.minute
        is_m = (570 <= tot_m <= 615)
        is_a = (795 <= tot_m <= 855)

        # 1. Update active trades
        for tr in active_trades:
            if not tr.is_closed:
                df_asset = asset_data[tr.sym][asset_data[tr.sym].index.date == d]
                if curr_t in df_asset.index:
                    b = df_asset.loc[curr_t]
                    if isinstance(b, pd.DataFrame): b = b.iloc[0]
                    tr.update(float(b['high']), float(b['low']), float(b['close']), curr_t)

        active_trades = [tr for tr in active_trades if not tr.is_closed]
        if not (is_m or is_a): continue

        regime = get_nifty_regime(nifty_day, b_idx)
        if not (regime['allow_call'] or regime['allow_put']): continue

        act_stocks = sum(1 for tr in active_trades if (not tr.is_idx) and (not tr.is_risk_free))
        act_broad_idx = sum(1 for tr in active_trades if tr.is_idx and (not tr.is_bank) and (not tr.is_risk_free))
        act_bank_idx = sum(1 for tr in active_trades if tr.is_idx and tr.is_bank and (not tr.is_risk_free))

        # Check Index
        for isym in INDEX_NAMES:
            if isym not in asset_data or isym in daily_traded_symbols: continue
            is_bank = (isym == 'BANKNIFTY')
            if is_bank and act_bank_idx >= MAX_ACTIVE_BANK_INDEX: continue
            if (not is_bank) and act_broad_idx >= MAX_ACTIVE_BROAD_INDEX: continue

            idf = asset_data[isym][asset_data[isym].index.date == d]
            if len(idf) <= b_idx: continue
            sig = check_index_signal_pro(isym, idf, b_idx, regime)
            if sig:
                # TRUE OPTION CHAIN CONFLUENCE CHECK
                t_price = (sig['entry_price'] + sig.get('spot_t2', 150.0)) if sig['direction'] == 'CALL' else (sig['entry_price'] - sig.get('spot_t2', 150.0))
                oi_res = check_oi_runway_bhavcopy(isym, d, sig['entry_price'], sig['direction'], target_price=t_price, is_index=True)
                if not oi_res['confluent']:
                    continue
                trade = ActiveTrade(sig, b_idx)
                active_trades.append(trade)
                all_executed_trades.append(trade)
                daily_traded_symbols.add(isym)
                if is_bank: act_bank_idx += 1
                else: act_broad_idx += 1

        # Check Stocks
        if act_stocks < MAX_ACTIVE_STOCKS:
            stock_cands = []
            for ssym in stock_syms:
                if ssym in daily_traded_symbols: continue
                sdf = asset_data[ssym][asset_data[ssym].index.date == d]
                if len(sdf) <= b_idx: continue
                stock_atr = atrs.get(ssym, 30.0)
                sig = check_stock_signal_pro(ssym, sdf, b_idx, stock_atr, regime)
                if sig:
                    # TRUE OPTION CHAIN CONFLUENCE CHECK
                    t_price = (sig['entry_price'] + sig.get('spot_t2', 50.0)) if sig['direction'] == 'CALL' else (sig['entry_price'] - sig.get('spot_t2', 50.0))
                    oi_res = check_oi_runway_bhavcopy(ssym, d, sig['entry_price'], sig['direction'], target_price=t_price, is_index=False)
                    if not oi_res['confluent']:
                        continue
                    stock_cands.append(sig)

            stock_cands.sort(key=lambda x: x['score'], reverse=True)
            slots_available = MAX_ACTIVE_STOCKS - act_stocks
            chosen_batch = []
            for cand in stock_cands:
                if len(chosen_batch) >= slots_available: break
                sec = STOCK_SECTOR_MAP.get(cand['symbol'], cand['symbol'])
                last_sec_t = sector_last_trade.get(sec)
                if last_sec_t and (curr_t - last_sec_t).total_seconds() < 1800: continue
                chosen_batch.append(cand)
                sector_last_trade[sec] = curr_t

            for chosen in chosen_batch:
                trade = ActiveTrade(chosen, b_idx)
                active_trades.append(trade)
                all_executed_trades.append(trade)
                daily_traded_symbols.add(chosen['symbol'])
                act_stocks += 1

    for tr in active_trades:
        if not tr.is_closed:
            df_asset = asset_data[tr.sym][asset_data[tr.sym].index.date == d]
            if len(df_asset) > 0:
                b_close = float(df_asset['close'].iloc[-1])
                tr.close_eod(b_close, df_asset.index[-1])

results = [t.get_summary(str(t.cand['eval_time'].date())) for t in all_executed_trades]
df_res = pd.DataFrame(results)

wins = df_res[df_res['realized_r'] > 0.0]
losses = df_res[df_res['realized_r'] < 0.0]
scratches = df_res[df_res['exit_reason'].str.contains('15M|SCRATCH', na=False)]
full_sl = df_res[df_res['exit_reason'] == 'SL_HIT']

gross_p = wins['realized_r'].sum()
gross_l = abs(losses['realized_r'].sum())
net_p = gross_p - gross_l
pf = (gross_p / gross_l) if gross_l > 0 else 999.0
win_rate = (len(wins) / len(df_res)) * 100.0 if len(df_res) > 0 else 0.0

print('\n' + '='*85)
print('PRO-TRADER DYNAMIC 5-DAY BACKTEST WITH TRUE -4 TO +4 BHAVCOPY OI ENGINE')
print('='*85)
print(f'Total Dispatched Trades: {len(df_res)}')
print(f'Wins: {len(wins)} ({win_rate:.1f}%)')
print(f'Scratches: {len(scratches)}')
print(f'Full Stoploss: {len(full_sl)}')
print(f'Gross Profit: +{gross_p:.2f}R | Gross Loss: -{gross_l:.2f}R | Net Profit: +{net_p:.2f}R | Profit Factor: {pf:.2f}')

print('\n--- ALL EXECUTED TRADES ---')
for r in results:
    e_str = r['eval_time'].strftime('%Y-%m-%d %H:%M')
    print(f"{e_str} | {r['symbol']:12s} | {r['direction']:4s} | En: {r['entry_price']:8.2f} | Ex: {r['exit_price']:8.2f} | Pts: {r['points_captured']:+7.2f} | R: {r['realized_r']:+5.2f}R | {r['exit_reason']}")

# Save CSV
df_res.to_csv('scratch/Alpha_Latest_5Day_Ending_Today_Backtest.csv', index=False)
print("\nUpdated authoritative CSV saved to scratch/Alpha_Latest_5Day_Ending_Today_Backtest.csv")
