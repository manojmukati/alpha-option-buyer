# Dynamic FNO Universe from Official NSE Servers
import os
import json
import datetime
import pandas as pd

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fno_universe_cache.json')

DEFAULT_FALLBACK_UNIVERSE = [
    {'symbol': 'NIFTY', 'lot': 50, 'step': 50.0, 'sector': 'INDEX', 'base_p': 24150.0, 'type': 'INDEX'},
    {'symbol': 'BANKNIFTY', 'lot': 15, 'step': 100.0, 'sector': 'INDEX', 'base_p': 51200.0, 'type': 'INDEX'},
    {'symbol': 'FINNIFTY', 'lot': 25, 'step': 50.0, 'sector': 'INDEX', 'base_p': 23500.0, 'type': 'INDEX'},
    {'symbol': 'MIDCPNIFTY', 'lot': 50, 'step': 25.0, 'sector': 'INDEX', 'base_p': 12800.0, 'type': 'INDEX'},
    {'symbol': 'NIFTYNXT50', 'lot': 10, 'step': 100.0, 'sector': 'INDEX', 'base_p': 72000.0, 'type': 'INDEX'}
]

def determine_step_size(price, is_index=False, symbol=''):
    if is_index:
        if symbol == 'NIFTY': return 50.0
        if symbol == 'BANKNIFTY': return 100.0
        if symbol == 'FINNIFTY': return 50.0
        if symbol == 'MIDCPNIFTY': return 25.0
        if symbol == 'NIFTYNXT50': return 100.0
        return 50.0
    if price < 100: return 2.5
    elif price < 250: return 5.0
    elif price < 500: return 10.0
    elif price < 1500: return 20.0
    elif price < 3000: return 50.0
    else: return 100.0

def fetch_dynamic_fno_from_nse():
    try:
        from nselib import derivatives
        now = datetime.datetime.now()
        for days_back in range(0, 10):
            test_dt = (now - datetime.timedelta(days=days_back)).strftime('%d-%m-%Y')
            try:
                df = derivatives.derivative_data.fno_bhav_copy(test_dt)
                if df is not None and not df.empty and 'TckrSymb' in df.columns:
                    # STRICTLY FILTER FUTURES (STF / IDF) to get real underlying SPOT prices
                    fut_df = df[df['FinInstrmTp'].isin(['STF', 'IDF'])].copy()
                    fut_df['ClsPric'] = pd.to_numeric(fut_df['ClsPric'], errors='coerce').fillna(0.0)
                    fut_df['OpnPric'] = pd.to_numeric(fut_df['OpnPric'], errors='coerce').fillna(0.0)
                    fut_df['NewBrdLotQty'] = pd.to_numeric(fut_df['NewBrdLotQty'], errors='coerce').fillna(500)
                    
                    unique_df = fut_df.drop_duplicates(subset=['TckrSymb'])
                    universe = []
                    index_syms = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50']
                    
                    for idx_sym in index_syms:
                        row = unique_df[unique_df['TckrSymb'] == idx_sym]
                        lot = int(row['NewBrdLotQty'].iloc[0]) if not row.empty else (50 if idx_sym=='NIFTY' else (15 if idx_sym=='BANKNIFTY' else 25))
                        cls_p = float(row['ClsPric'].iloc[0]) if not row.empty and float(row['ClsPric'].iloc[0]) > 0 else (24000.0 if idx_sym=='NIFTY' else 52000.0)
                        universe.append({
                            'symbol': idx_sym,
                            'lot': lot,
                            'step': determine_step_size(cls_p, is_index=True, symbol=idx_sym),
                            'sector': 'INDEX',
                            'base_p': cls_p,
                            'type': 'INDEX'
                        })
                        
                    for _, row in unique_df.iterrows():
                        sym = str(row['TckrSymb']).strip().upper()
                        if sym in index_syms:
                            continue
                        try:
                            lot = int(row['NewBrdLotQty'])
                            cls_p = float(row['ClsPric']) if float(row['ClsPric']) > 0 else float(row['OpnPric'])
                        except Exception:
                            lot = 500
                            cls_p = 1000.0
                            
                        universe.append({
                            'symbol': sym,
                            'lot': lot,
                            'step': determine_step_size(cls_p, is_index=False, symbol=sym),
                            'sector': 'EQUITY_FNO',
                            'base_p': cls_p,
                            'type': 'STOCK'
                        })
                        
                    if len(universe) >= 100:
                        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                            json.dump(universe, f, indent=2)
                        return universe
            except Exception:
                continue
    except Exception:
        pass
    return None

def get_fno_universe(force_refresh=False):
    if not force_refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Verify that prices are real spot prices (> 50 for large stocks)
                if len(data) >= 100 and data[0]['base_p'] > 5000:
                    return data
        except Exception:
            pass
    dynamic_data = fetch_dynamic_fno_from_nse()
    if dynamic_data and len(dynamic_data) >= 100:
        return dynamic_data
    return DEFAULT_FALLBACK_UNIVERSE

def get_instrument_meta(symbol):
    u = get_fno_universe()
    for item in u:
        if item['symbol'].upper() == symbol.upper():
            return item
    return {'symbol': symbol, 'lot': 500, 'step': 10.0, 'sector': 'STOCK', 'base_p': 1000.0, 'type': 'STOCK'}
