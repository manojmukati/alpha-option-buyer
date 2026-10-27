# Alpha Profile & OI Confluence Trading System 🚀
**High Win-Probability Options & Futures System**

Yeh system **5 Institutional Pillars** ko combine karta hai high-accuracy trade signals generate karne ke liye:
1. **OI Cluster Analysis (+/- 6 Strikes around ATM):** Call vs Put Open Interest, PCR, Shift aur Unwinding Squeeze Detection.
2. **Volume Delta & Cumulative Volume Delta (CVD):** Aggressive Buy/Sell Volume, Delta Spikes, aur Passive Absorption/Divergence.
3. **Volume Profile Analysis:** Point of Control (POC), Value Area High (VAH), Value Area Low (VAL), aur HVN/LVN nodes.
4. **Market Profile (TPO Analysis):** Initial Balance (IB 9:15-10:15 AM), Day Types, aur P-Shape / b-Shape / D-Shape profiles.
5. **PBD (Price Behavior Dynamics):** Previous Day High (PDH), Low (PDL), Close (PDC), aur Opening Range auction types.

---

## Folder Structure
`
alpha_profile_oi_system/
├── core/
│   ├── data_feed.py            # Real-time / simulation data feeder & option chain generator
│   ├── confluence_engine.py    # Multi-pillar scoring engine (Weighted Confluence Score)
│   ├── risk_manager.py         # Dynamic Stop Loss, 1:2.5+ Target, and Time-Stop rule
│   └── order_selector.py       # Chooses Option Buy (ITM) vs Hedged Option Sell vs Futures
├── modules/
│   ├── oi_analyzer.py          # +/- 6 strike cluster, PCR, buildup & unwinding detector
│   ├── volume_delta.py         # Bar Delta, CVD, CVD Slope & Divergence
│   ├── volume_profile.py       # VAH, VAL, POC, and Volume at Price
│   ├── market_profile.py       # TPO, Initial Balance (IB), P/b/D shape classifier
│   └── pbd_engine.py           # PDH, PDL, PDC Breakouts & Compression
├── backtester/
│   ├── backtest_engine.py      # Multi-session simulation & trade lifecycle manager
│   └── metrics.py              # Win Rate, Profit Factor, Expectancy, Payoff Ratio
├── ui/
│   └── terminal_view.py        # Rich interactive CLI dashboard with ASCII charts
├── run_system.py               # Main Live Scanner & Simulation runner
├── run_backtest.py             # Backtesting execution script
├── requirements.txt            # Python dependencies
└── README.md                   # Complete Strategy Guide
`

---

## Strategy Rules for High Win Probability

### 1. Option Buying (High Momentum Scalp/Breakout)
* **Conditions:**
  * Market Profile: **P-Shape (Bullish)** ya **b-Shape (Bearish)** ya **Trend Day**.
  * Volume Profile: Price trading **Above VAH** (for Call) or **Below VAL** (for Put).
  * Volume Delta: **CVD Surge** + **Delta Spike (> 2x Sigma)**.
  * OI Cluster: **Call Unwinding** (Short Covering Squeeze) for CE / **Put Unwinding** for PE.
  * PBD: Clean sustained breakout of **PDH** or breakdown of **PDL**.
* **Execution:**
  * Buy **ITM Strike (Delta 0.65 - 0.70)**.
  * **Time-Stop:** Agar 3-5 candles me momentum continue nahi hota, cost-to-cost exit kar jao (Theta protection).
  * **Risk-Reward:** Minimum 1:2.5 Target.

### 2. Hedged Option Selling (Range & Theta Decay)
* **Conditions:**
  * Market Profile: **D-Shape Balanced Curve**.
  * Volume Profile: Price oscillating **Inside Value Area (between VAL and VAH)**.
  * OI Cluster: High Call & Put writing at $\pm 2$ strikes.
* **Execution:**
  * Execute **Iron Condor** ya **Credit Spreads** (Sell near strikes, Buy far OTM hedge).
  * Capture 60-70% Theta decay.

---

## Kaise Run Karein:
`ash
# Backtest run karne ke liye:
python run_backtest.py

# Live Dashboard / Scanner simulation dekhne ke liye:
python run_system.py
`