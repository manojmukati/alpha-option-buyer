import os, sys
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import numpy as np
import subprocess

print('======================================================================')
print('=== EXHAUSTIVE 20-POINT INSTITUTIONAL PROP-DESK QUANTITATIVE AUDIT ===')
print('======================================================================')

from alpha_profile_oi_system.daily_autonomous_screener import (
    verify_multi_candle_momentum,
    verify_bhavcopy_oi_confluence,
    DailyAutonomousScreener
)
from alpha_profile_oi_system.modules.live_data_feed import LiveMarketDataFeed

results = []

# --- Point 1: PIIND Thin-Book Rejection on Bar 1 ---
dates = pd.date_range('2026-11-19 09:15', periods=8, freq='3min')
df_piind = pd.DataFrame({
    'open': [1620, 1622, 1621, 1623, 1622, 1625, 1630, 1632],
    'high': [1623, 1624, 1623, 1625, 1624, 1650, 1652, 1635],
    'low':  [1619, 1620, 1620, 1621, 1621, 1624, 1625, 1628],
    'close':[1622, 1621, 1623, 1622, 1624, 1640, 1633, 1630],
    'volume':[2000, 2500, 1800, 2200, 2100, 12000, 3000, 1500]
}, index=dates)
comp_piind = df_piind.iloc[:-2].tail(5)
p1, r1, s1 = verify_multi_candle_momentum(df_piind, 'CALL', comp_piind, daily_atr=67.4, lot_size=250.0, sym_clean='PIIND', is_index=False)
assert not p1 and r1 in ['THIN_BOOK_BASE_LIQUIDITY_VETO', 'INSUFFICIENT_15M_TURNOVER_DEPTH', 'WEAK_CANDLE_BODY_REJECTION_VETO'], f'Test 1 failed: {r1}'
results.append(('Point 1: PIIND Thin-Book Bar-1 Veto', True, str(r1)))

# --- Point 2: MARUTI 1-Candle Ultra-Confirmation (+750 pt surge) ---
df_m1 = pd.DataFrame({
    'open': [12800, 12820, 12840, 12850, 12880, 12890, 12900, 13160],
    'high': [12830, 12850, 12860, 12890, 12900, 12910, 13160, 13170],
    'low':  [12790, 12810, 12830, 12840, 12870, 12880, 12890, 13150],
    'close':[12820, 12840, 12850, 12880, 12895, 12905, 13155, 13165],
    'volume':[50000, 60000, 55000, 58000, 62000, 50000, 220000, 1000]
}, index=dates)
for i in range(len(df_m1)): df_m1.loc[df_m1.index[i], 'hist_slot_vol'] = 60000.0
comp_m1 = df_m1.iloc[:-2].tail(5)
p2, r2, s2 = verify_multi_candle_momentum(df_m1, 'CALL', comp_m1, daily_atr=240.0, lot_size=50.0, sym_clean='MARUTI', is_index=False)
assert p2 and r2 == 'TIER_1_ONE_CANDLE_ULTRA_CONFIRMED', f'Test 2 failed: {r2}'
results.append(('Point 2: MARUTI 1-Candle Ultra-Confirmation', True, f'{r2} ({s2} pts)'))

# --- Point 3: M&M Opening Drive Ultra-Confirmation (+156 pt surge) ---
df_mm = pd.DataFrame({
    'open': [2900, 2910, 2915, 2920, 2930, 2935, 2940, 2995],
    'high': [2915, 2920, 2925, 2935, 2940, 2942, 2990, 3000],
    'low':  [2895, 2905, 2910, 2915, 2925, 2930, 2938, 2990],
    'close':[2910, 2915, 2920, 2930, 2938, 2940, 2988, 2998],
    'volume':[40000, 45000, 42000, 48000, 50000, 45000, 160000, 2000]
}, index=dates)
for i in range(len(df_mm)): df_mm.loc[df_mm.index[i], 'hist_slot_vol'] = 45000.0
comp_mm = df_mm.iloc[:-2].tail(5)
p3, r3, s3 = verify_multi_candle_momentum(df_mm, 'CALL', comp_mm, daily_atr=65.0, lot_size=200.0, sym_clean='M&M', is_index=False)
assert p3 and r3 == 'TIER_1_ONE_CANDLE_ULTRA_CONFIRMED', f'Test 3 failed: {r3}'
results.append(('Point 3: M&M Opening Drive Ultra-Confirmation', True, f'{r3} ({s3} pts)'))

# --- Point 4: NIFTY 1-Candle Ultra-Confirmation ---
df_nifty = pd.DataFrame({
    'open': [23500, 23510, 23505, 23515, 23520, 23525, 23530, 23630],
    'high': [23515, 23520, 23520, 23525, 23530, 23532, 23625, 23640],
    'low':  [23495, 23505, 23500, 23510, 23515, 23520, 23528, 23620],
    'close':[23510, 23505, 23515, 23520, 23528, 23530, 23622, 23635],
    'volume':[0, 0, 0, 0, 0, 0, 0, 0]
}, index=dates)
comp_nifty = df_nifty.iloc[:-2].tail(5)
p4, r4, s4 = verify_multi_candle_momentum(df_nifty, 'CALL', comp_nifty, daily_atr=180.0, lot_size=25.0, sym_clean='NIFTY', is_index=True)
assert p4 and r4 == 'TIER_1_ONE_CANDLE_ULTRA_CONFIRMED', f'Test 4 failed: {r4}'
results.append(('Point 4: NIFTY Index Ultra-Confirmation', True, f'{r4} ({s4} pts)'))

# --- Point 5: BANKNIFTY Put Breakdown Tier 1 Ultra ---
df_bnf = pd.DataFrame({
    'open': [51200, 51180, 51190, 51170, 51160, 51150, 51140, 50800],
    'high': [51210, 51195, 51200, 51185, 51175, 51160, 51145, 50810],
    'low':  [51180, 51170, 51165, 51150, 51140, 51135, 50820, 50780],
    'close':[51190, 51175, 51170, 51160, 51150, 51140, 50830, 50790],
    'volume':[0, 0, 0, 0, 0, 0, 0, 0]
}, index=dates)
comp_bnf = df_bnf.iloc[:-2].tail(5)
p5, r5, s5 = verify_multi_candle_momentum(df_bnf, 'PUT', comp_bnf, daily_atr=450.0, lot_size=15.0, sym_clean='BANKNIFTY', is_index=True)
assert p5 and r5 == 'TIER_1_ONE_CANDLE_ULTRA_CONFIRMED', f'Test 5 failed: {r5}'
results.append(('Point 5: BANKNIFTY Put Breakdown Ultra', True, f'{r5} ({s5} pts)'))

# --- Point 6: 2-Candle Follow-Through Confirmation (Tier 2) ---
df_t2 = pd.DataFrame({
    'open': [2500, 2505, 2510, 2508, 2512, 2515, 2560, 2570],
    'high': [2510, 2515, 2518, 2515, 2520, 2565, 2575, 2580],
    'low':  [2495, 2500, 2505, 2502, 2510, 2512, 2555, 2565],
    'close':[2505, 2512, 2510, 2512, 2518, 2562, 2572, 2575],
    'volume':[25000, 28000, 26000, 24000, 27000, 80000, 70000, 15000]
}, index=dates)
for i in range(len(df_t2)): df_t2.loc[df_t2.index[i], 'hist_slot_vol'] = 26000.0
comp_t2 = df_t2.iloc[:-2].tail(5)
p6, r6, s6 = verify_multi_candle_momentum(df_t2, 'CALL', comp_t2, daily_atr=50.0, lot_size=250.0, sym_clean='XYZ', is_index=False)
assert p6 and 'FOLLOW_THROUGH' in r6, f'Test 6 failed: {r6}'
results.append(('Point 6: 2-Candle Follow-Through (Tier 2)', True, f'{r6} ({s6} pts)'))

# --- Point 7: Lower Strike Call Writing Ambush Veto ---
import time
feed = LiveMarketDataFeed()
feed._oc_cache['AMBUSH_TEST'] = {
    'records': {
        'expiryDates': ['28-Nov-2024'],
        'data': [
            {'strikePrice': 1000 + i*20, 'expiryDate': '28-Nov-2024',
             'CE': {'openInterest': 1000, 'changeinOpenInterest': 800 if i < 4 else -100, 'totalTradedVolume': 500, 'lastPrice': 50, 'buyPrice1': 49.5, 'sellPrice1': 50.5},
             'PE': {'openInterest': 1000, 'changeinOpenInterest': 50, 'totalTradedVolume': 100, 'lastPrice': 50, 'buyPrice1': 49.5, 'sellPrice1': 50.5}}
            for i in range(10)
        ]
    }
}
feed._last_oc_fetch['AMBUSH_TEST'] = time.time()
oi_ambush = feed.get_live_dual_oi_confluence('AMBUSH_TEST', 1080.0, 'CALL', is_index=False)
assert not oi_ambush['confluent'] and oi_ambush['reason'] == 'LOWER_STRIKE_CALL_WRITING_AMBUSH', f'Test 7 failed: {oi_ambush}'
results.append(('Point 7: Lower Strike Call Writing Ambush Veto', True, oi_ambush['reason']))

# --- Point 8: Upper Strike Put Writing Ambush Veto ---
feed._oc_cache['PUT_AMBUSH_TEST'] = {
    'records': {
        'expiryDates': ['28-Nov-2024'],
        'data': [
            {'strikePrice': 1000 + i*20, 'expiryDate': '28-Nov-2024',
             'CE': {'openInterest': 1000, 'changeinOpenInterest': 50, 'totalTradedVolume': 100, 'lastPrice': 50, 'buyPrice1': 49.5, 'sellPrice1': 50.5},
             'PE': {'openInterest': 1000, 'changeinOpenInterest': 800 if i >= 5 else -100, 'totalTradedVolume': 500, 'lastPrice': 50, 'buyPrice1': 49.5, 'sellPrice1': 50.5}}
            for i in range(10)
        ]
    }
}
feed._last_oc_fetch['PUT_AMBUSH_TEST'] = time.time()
oi_put_ambush = feed.get_live_dual_oi_confluence('PUT_AMBUSH_TEST', 1080.0, 'PUT', is_index=False)
assert not oi_put_ambush['confluent'] and oi_put_ambush['reason'] == 'UPPER_STRIKE_PUT_WRITING_AMBUSH', f'Test 8 failed: {oi_put_ambush}'
results.append(('Point 8: Upper Strike Put Writing Ambush Veto', True, oi_put_ambush['reason']))

# --- Point 9: Institutional Gamma Squeeze Trigger ---
feed._oc_cache['GAMMA_TEST'] = {
    'records': {
        'expiryDates': ['28-Nov-2024'],
        'data': [
            {'strikePrice': 1000 + i*20, 'expiryDate': '28-Nov-2024',
             'CE': {'openInterest': 1000, 'changeinOpenInterest': -120 if i in [4, 5] else -20, 'totalTradedVolume': 3500 if i in [4, 5] else 500, 'lastPrice': 50, 'buyPrice1': 49.5, 'sellPrice1': 50.5},
             'PE': {'openInterest': 1000, 'changeinOpenInterest': 200 if i < 4 else 0, 'totalTradedVolume': 200, 'lastPrice': 50, 'buyPrice1': 49.5, 'sellPrice1': 50.5}}
            for i in range(10)
        ]
    }
}
feed._last_oc_fetch['GAMMA_TEST'] = time.time()
oi_gamma = feed.get_live_dual_oi_confluence('GAMMA_TEST', 1080.0, 'CALL', is_index=False)
assert oi_gamma['confluent'] and oi_gamma['is_gamma_squeeze'], f'Test 9 failed: {oi_gamma}'
results.append(('Point 9: Institutional Gamma Squeeze Acceptance', True, f"{oi_gamma['reason']} (Score: {oi_gamma['score']})"))

# --- Point 10: Institutional Panic Squeeze Trigger ---
feed._oc_cache['PANIC_TEST'] = {
    'records': {
        'expiryDates': ['28-Nov-2024'],
        'data': [
            {'strikePrice': 1000 + i*20, 'expiryDate': '28-Nov-2024',
             'CE': {'openInterest': 1000, 'changeinOpenInterest': 200 if i >= 5 else 0, 'totalTradedVolume': 200, 'lastPrice': 50, 'buyPrice1': 49.5, 'sellPrice1': 50.5},
             'PE': {'openInterest': 1000, 'changeinOpenInterest': -120 if i in [3, 4] else -20, 'totalTradedVolume': 3500 if i in [3, 4] else 500, 'lastPrice': 50, 'buyPrice1': 49.5, 'sellPrice1': 50.5}}
            for i in range(10)
        ]
    }
}
feed._last_oc_fetch['PANIC_TEST'] = time.time()
oi_panic = feed.get_live_dual_oi_confluence('PANIC_TEST', 1080.0, 'PUT', is_index=False)
assert oi_panic['confluent'] and oi_panic['is_panic_squeeze'], f'Test 10 failed: {oi_panic}'
results.append(('Point 10: Institutional Panic Squeeze Acceptance', True, f"{oi_panic['reason']} (Score: {oi_panic['score']})"))

# --- Point 11: Stock Retail Noise Unwinding Rejection (4 lots) ---
feed._oc_cache['RETAIL_NOISE_TEST'] = {
    'records': {
        'expiryDates': ['28-Nov-2024'],
        'data': [
            {'strikePrice': 1000 + i*20, 'expiryDate': '28-Nov-2024',
             'CE': {'openInterest': 2000, 'changeinOpenInterest': -4, 'totalTradedVolume': 10, 'lastPrice': 50, 'buyPrice1': 49.5, 'sellPrice1': 50.5},
             'PE': {'openInterest': 2000, 'changeinOpenInterest': 200 if i < 4 else 0, 'totalTradedVolume': 200, 'lastPrice': 50, 'buyPrice1': 49.5, 'sellPrice1': 50.5}}
            for i in range(10)
        ]
    }
}
feed._last_oc_fetch['RETAIL_NOISE_TEST'] = time.time()
oi_retail = feed.get_live_dual_oi_confluence('RETAIL_NOISE_TEST', 1080.0, 'CALL', is_index=False)
assert not oi_retail['confluent'] and 'UNWINDING' in oi_retail['reason'], f'Test 11 failed: {oi_retail}'
results.append(('Point 11: Stock Retail Noise (4 lots) Rejection', True, oi_retail['reason']))

# --- Point 12: Index Retail Noise Unwinding Rejection (150 contracts) ---
feed._oc_cache['NIFTY'] = {
    'records': {
        'expiryDates': ['28-Nov-2024'],
        'data': [
            {'strikePrice': 24000 + i*50, 'expiryDate': '28-Nov-2024',
             'CE': {'openInterest': 50000, 'changeinOpenInterest': -150, 'totalTradedVolume': 5000, 'lastPrice': 150, 'buyPrice1': 149.5, 'sellPrice1': 150.5},
             'PE': {'openInterest': 50000, 'changeinOpenInterest': 12000 if i < 4 else 0, 'totalTradedVolume': 15000, 'lastPrice': 150, 'buyPrice1': 149.5, 'sellPrice1': 150.5}}
            for i in range(10)
        ]
    }
}
feed._last_oc_fetch['NIFTY'] = time.time()
oi_idx_ret = feed.get_live_dual_oi_confluence('NIFTY', 24200.0, 'CALL', is_index=True)
assert not oi_idx_ret['confluent'] and 'UNWINDING' in oi_idx_ret['reason'], f'Test 12 failed: {oi_idx_ret}'
results.append(('Point 12: Index Retail Noise (150 lots) Rejection', True, oi_idx_ret['reason']))

# --- Point 13: Index Institutional Unwinding Acceptance (6,000 contracts) ---
feed._oc_cache['NIFTY'] = {
    'records': {
        'expiryDates': ['28-Nov-2024'],
        'data': [
            {'strikePrice': 24000 + i*50, 'expiryDate': '28-Nov-2024',
             'CE': {'openInterest': 50000, 'changeinOpenInterest': -6000 if i in [4, 5] else -1000, 'totalTradedVolume': 15000, 'lastPrice': 150, 'buyPrice1': 149.5, 'sellPrice1': 150.5},
             'PE': {'openInterest': 50000, 'changeinOpenInterest': 12000 if i < 4 else 0, 'totalTradedVolume': 15000, 'lastPrice': 150, 'buyPrice1': 149.5, 'sellPrice1': 150.5}}
            for i in range(10)
        ]
    }
}
feed._last_oc_fetch['NIFTY'] = time.time()
oi_idx_inst = feed.get_live_dual_oi_confluence('NIFTY', 24200.0, 'CALL', is_index=True)
assert oi_idx_inst['confluent'], f'Test 13 failed: {oi_idx_inst}'
results.append(('Point 13: Index Institutional Unwinding (6,000 lots) Confirmed', True, f"{oi_idx_inst['reason']} (Score: {oi_idx_inst['score']})"))

# --- Point 14: Option Dynamic SL Proportional Scaling (Maruti high premium) ---
feed._oc_cache['MARUTI_PRICING'] = {
    'records': {
        'expiryDates': ['28-Nov-2024'],
        'data': [
            {'strikePrice': 13000 + i*100, 'expiryDate': '28-Nov-2024',
             'CE': {'openInterest': 1000, 'changeinOpenInterest': -120 if i in [4, 5] else -20, 'totalTradedVolume': 3500 if i in [4, 5] else 500, 'lastPrice': 320.0, 'buyPrice1': 318.0, 'sellPrice1': 322.0, 'impliedVolatility': 22.0},
             'PE': {'openInterest': 1000, 'changeinOpenInterest': 200 if i < 4 else 0, 'totalTradedVolume': 200, 'lastPrice': 50.0, 'buyPrice1': 49.5, 'sellPrice1': 50.5, 'impliedVolatility': 20.0}}
            for i in range(10)
        ]
    }
}
feed._last_oc_fetch['MARUTI_PRICING'] = time.time()
prc_maruti = feed.compute_option_pricing(13400.0, 'CALL', symbol='MARUTI_PRICING', is_index=False)
assert prc_maruti['sl_points'] == 25.0 and prc_maruti['sl_price'] == 297.0, f'Test 14 failed: {prc_maruti}'
results.append(('Point 14: Dynamic Proportional SL on Maruti (322 entry -> 25 pt SL)', True, f"SL: {prc_maruti['sl_points']} pts, Price: {prc_maruti['sl_price']}"))

# --- Point 15: Option SL Fixed on Normal Stock (Entry <= 60) ---
feed._oc_cache['SBIN_PRICING'] = {
    'records': {
        'expiryDates': ['28-Nov-2024'],
        'data': [
            {'strikePrice': 780 + i*5, 'expiryDate': '28-Nov-2024',
             'CE': {'openInterest': 2000, 'changeinOpenInterest': -150 if i in [4, 5] else -20, 'totalTradedVolume': 5000 if i in [4, 5] else 500, 'lastPrice': 20.0, 'buyPrice1': 19.8, 'sellPrice1': 20.2, 'impliedVolatility': 25.0},
             'PE': {'openInterest': 2000, 'changeinOpenInterest': 300 if i < 4 else 0, 'totalTradedVolume': 500, 'lastPrice': 20.0, 'buyPrice1': 19.8, 'sellPrice1': 20.2, 'impliedVolatility': 25.0}}
            for i in range(10)
        ]
    }
}
feed._last_oc_fetch['SBIN_PRICING'] = time.time()
prc_norm = feed.compute_option_pricing(800.0, 'CALL', symbol='SBIN_PRICING', is_index=False)
assert prc_norm['sl_points'] == 5.0 and prc_norm['sl_price'] == 15.2, f'Test 15 failed: {prc_norm}'
results.append(('Point 15: Fixed 5 pt SL on Normal Option (20.2 entry -> 5 pt SL)', True, f"SL: {prc_norm['sl_points']} pts, Price: {prc_norm['sl_price']}"))

# --- Point 16: Option Spread Scaled for High-Premium Contract ---
feed._oc_cache['MARUTI_SPREAD'] = {
    'records': {
        'expiryDates': ['28-Nov-2024'],
        'data': [
            {'strikePrice': 13000 + i*100, 'expiryDate': '28-Nov-2024',
             'CE': {'openInterest': 1000, 'changeinOpenInterest': -120 if i in [4, 5] else -20, 'totalTradedVolume': 3500 if i in [4, 5] else 500, 'lastPrice': 318.0, 'buyPrice1': 316.5, 'sellPrice1': 320.0, 'impliedVolatility': 22.0},
             'PE': {'openInterest': 1000, 'changeinOpenInterest': 200 if i < 4 else 0, 'totalTradedVolume': 200, 'lastPrice': 50.0, 'buyPrice1': 49.5, 'sellPrice1': 50.5, 'impliedVolatility': 20.0}}
            for i in range(10)
        ]
    }
}
feed._last_oc_fetch['MARUTI_SPREAD'] = time.time()
prc_spread = feed.compute_option_pricing(13400.0, 'CALL', symbol='MARUTI_SPREAD', is_index=False)
assert prc_spread['is_liquid'] and prc_spread['is_live_chain_verified'], f'Test 16 failed: {prc_spread}'
results.append(('Point 16: Spread Risk Scaled for High-Premium Options (Spread Rs 3.50 allowed)', True, 'LIQUID_AND_VERIFIED'))

# --- Point 17: Bhavcopy Parquet Volume Unit Integrity ---
bhav_res = verify_bhavcopy_oi_confluence('MARUTI', '2026-11-19', 13200.0, 'CALL', target_price=13300.0, is_index=False)
assert bhav_res is not None and 'score' in bhav_res, f'Test 17 failed: {bhav_res}'
results.append(('Point 17: Bhavcopy Confluence Unit Integrity Tested', True, f"Status: {bhav_res.get('reason')} (Score: {bhav_res.get('score')})"))

# --- Point 18: Screener Class Instantiation ---
screener = DailyAutonomousScreener()
assert screener is not None
results.append(('Point 18: DailyAutonomousScreener Instantiation', True, 'SUCCESS'))

# --- Point 19: LiveDataFeed Class Instantiation ---
feed_inst = LiveMarketDataFeed()
assert feed_inst is not None
results.append(('Point 19: LiveMarketDataFeed Instantiation', True, 'SUCCESS'))

# --- Point 20: Telegram Files Untouched Verification ---
git_status = subprocess.check_output('git status --porcelain alpha_profile_oi_system/telegram* telegram*', shell=True).decode('utf-8').strip()
assert git_status == '', f'Telegram files were modified: {git_status}'
results.append(('Point 20: Telegram Notifier & Config 100% UNTOUCHED', True, 'VERIFIED_ZERO_MODIFICATIONS'))

print('\n=== AUDIT SUMMARY RESULTS ===')
for pt, status, note in results:
    badge = 'PASS' if status else 'FAIL'
    print(f'[{badge}] {pt} -> {note}')

print(f'\nTOTAL CHECKS: {len(results)}/20 PASSED WITH ZERO ERRORS!')
