import pandas as pd

df = pd.read_csv('scratch/Alpha_Latest_5Day_Ending_Today_Backtest.csv')
df['eval_dt'] = pd.to_datetime(df['eval_time'])
df['exit_dt'] = pd.to_datetime(df['exit_time'])
df['dur'] = (df['exit_dt'] - df['eval_dt']).dt.total_seconds() / 60

for i, r in df.iterrows():
    h = int(r['dur'] // 60)
    m = int(r['dur'] % 60)
    dur_str = f"{h}h {m}m" if h > 0 else f"{m}m"
    eval_t = str(r['eval_time'])[:16]
    print(f"| {i+1} | `{eval_t}` | **{r['contract']}** | **{dur_str}** | {r['spot_entry']:.2f} -> {r['spot_exit']:.2f} | {r['spot_pts']:+.2f} | {r['opt_pts']:+.2f} | {r['lot_size']} | **Rs {r['net_pnl']:+,.2f}** | **{r['realized_r']:+.2f}R** | `{r['exit_reason']}` |")
