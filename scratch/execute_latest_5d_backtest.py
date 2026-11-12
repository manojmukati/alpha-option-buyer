# -*- coding: utf-8 -*-
"""
scratch/execute_latest_5d_backtest.py
=====================================
Unified Authoritative 5-Day Option Buyer Backtest Engine.
Directly executes DailyAutonomousScreener().run_authoritative_5day_backtest()
to guarantee ZERO DISCREPANCY between production screener and backtest.
"""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha_profile_oi_system.daily_autonomous_screener import DailyAutonomousScreener

if __name__ == '__main__':
    screener = DailyAutonomousScreener()
    screener.run_authoritative_5day_backtest()
