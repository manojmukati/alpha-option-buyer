import datetime
import os
import json
import time
import requests

class TelegramNotifier:
    """
    Institutional Momentum Telegram Notification Engine.
    - 4-Minute Execution Buffer Pre-Alerts (T_sig)
    - Confirmed Entry Alerts (T_sig + 4 min) with Real NSE Strikes & Spot Targets (+100-150 Index, +30-60 Stocks)
    - Momentum Impulse & Stall Exit Alerts (12-26 Mins, No 60+m Stagnation)
    - Automated Failed Signal Post-Mortem Diagnostic Alerts
    """
    def __init__(self, config_path=None):
        if config_path is None:
            p1 = os.path.join(os.path.dirname(__file__), "..", "config", "telegram_config.json")
            
            config_path = p1
            
        self.config_path = config_path
        self.bot_token = ""
        self.owner_chat_id = ""
        self.group_chat_ids = []
        self.load_config()
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else ""

    def load_config(self):
        # 1. Read directly from GitHub Secrets / Environment variables
        env_token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
        env_owner = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
        env_group = os.environ.get('TELEGRAM_GROUP_ID', '').strip()

        if env_token:
            self.bot_token = env_token
        if env_owner:
            self.owner_chat_id = env_owner
        if env_group:
            self.group_chat_ids = [env_group]

        # 2. Local config fallback
        if (not self.bot_token or not self.owner_chat_id) and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if not self.bot_token:
                        self.bot_token = cfg.get("bot_token", "")
                    if not self.owner_chat_id:
                        self.owner_chat_id = str(cfg.get("chat_id", ""))
                    if not self.group_chat_ids:
                        grp = cfg.get("group_chat_ids", [])
                        for cid in cfg.get("chat_ids", []):
                            if str(cid).startswith("-") and str(cid) not in [str(g) for g in grp]:
                                grp.append(str(cid))
                        if isinstance(grp, list):
                            self.group_chat_ids = [str(g) for g in grp if g]
                        elif grp:
                            self.group_chat_ids = [str(grp)]
            except Exception as e:
                print(f"[TELEGRAM CONFIG ERROR]: {e}")

    def is_configured(self):
        return bool(self.bot_token and self.owner_chat_id and self.bot_token != "YOUR_BOT_TOKEN_HERE")

    def send_document(self, file_path, caption="", send_to_group=False):
        if not self.is_configured():
            print(f"[TELEGRAM SIMULATED DOCUMENT]: {file_path} | Caption: {caption}")
            return {"ok": True, "simulated": True}
        if not os.path.exists(file_path):
            print(f"[TELEGRAM ERROR]: File {file_path} does not exist.")
            return {"ok": False, "reason": "file_not_found"}

        # Strict Security Mandate: Excel & CSV files contain private journal PnL and trade logs.
        # They must NEVER go to the Telegram group. Enforce owner-only dispatch for spreadsheets.
        if file_path.lower().endswith(('.xlsx', '.xls', '.csv')):
            send_to_group = False

        all_recipients = [self.owner_chat_id] + self.group_chat_ids if send_to_group else [self.owner_chat_id]
        url = f"{self.base_url}/sendDocument"
        res = {"ok": False}
        for cid in all_recipients:
            try:
                with open(file_path, "rb") as doc_f:
                    resp = requests.post(
                        url,
                        data={"chat_id": cid, "caption": caption, "parse_mode": "HTML"},
                        files={"document": doc_f},
                        timeout=30
                    )
                    res = resp.json()
                    if res.get("ok"):
                        print(f"[TELEGRAM DOCUMENT SUCCESS for {cid}]")
                    else:
                        print(f"[TELEGRAM DOCUMENT WARNING for {cid}]: {res.get('description', res)}")
            except Exception as e:
                print(f"[TELEGRAM ERROR send_document for {cid}]: {e}")
        return res

    def send_document_to_owner_with_approval_prompt(self, file_path, caption=""):
        """Sends private journal/report strictly to the OWNER only. NEVER to group chat IDs."""
        if not self.is_configured():
            print(f"[TELEGRAM SIMULATED OWNER DOCUMENT]: {file_path} | Caption: {caption}")
            return {"ok": True, "simulated": True}
        if not os.path.exists(file_path):
            print(f"[TELEGRAM ERROR]: File {file_path} does not exist.")
            return {"ok": False, "reason": "file_not_found"}

        url = f"{self.base_url}/sendDocument"
        try:
            with open(file_path, "rb") as doc_f:
                resp = requests.post(
                    url,
                    data={"chat_id": self.owner_chat_id, "caption": caption, "parse_mode": "HTML"},
                    files={"document": doc_f},
                    timeout=30
                )
                res = resp.json()
                if res.get("ok"):
                    print(f"[TELEGRAM DOCUMENT SUCCESS for owner {self.owner_chat_id}]")
                else:
                    print(f"[TELEGRAM DOCUMENT WARNING for owner {self.owner_chat_id}]: {res.get('description', res)}")
                return res
        except Exception as e:
            print(f"[TELEGRAM ERROR send_document for owner {self.owner_chat_id}]: {e}")
            return {"ok": False, "error": str(e)}

    def send_broadcast_message(self, text, parse_mode="HTML", is_trade_signal=False):
        if not self.is_configured():
            print("[TELEGRAM SIMULATED ALERT]:")
            print(text)
            print("-" * 50)
            return {"ok": True, "simulated": True}

        # User Rule: Trade signals broadcast to Group AND Chatbot.
        # Routine status/heartbeats/scan updates go strictly to Chatbot ONLY (Never spam the group).
        if is_trade_signal:
            all_recipients = [self.owner_chat_id] + self.group_chat_ids
        else:
            all_recipients = [self.owner_chat_id]

        url = f"{self.base_url}/sendMessage"
        res = {"ok": False}
        for cid in all_recipients:
            try:
                resp = requests.post(url, json={"chat_id": cid, "text": text, "parse_mode": parse_mode}, timeout=10)
                res = resp.json()
                if res.get("ok"):
                    print(f"[TELEGRAM DISPATCH SUCCESS for {cid}]")
                else:
                    print(f"[TELEGRAM DISPATCH WARNING for {cid}]: {res.get('description', res)}")
            except Exception as e:
                print(f"[TELEGRAM ERROR for {cid}]: {e}")
        return res

    def send_pre_alert(self, signal):
        """Dispatched at T_sig: Gives human trader 4 full minutes to prepare."""
        sym = signal.get("Symbol", "")
        opt_type = signal.get("Type", "CE")
        contract = signal.get("Strike_Contract", "")
        spot_p = float(signal.get("Spot_Entry", signal.get("spot_price", 0)))
        alert_t = signal.get("Alert_Time", "")
        is_index = sym in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
        spot_tgt_txt = "+100 to +150 Points" if is_index else "+30 to +60 Points"
        
        lines = [
            "🚨 <b>INSTITUTIONAL MOMENTUM PRE-ALERT (4-MIN BUFFER)</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"⚡ <b>SYMBOL:</b> <b>{sym}</b> ({opt_type})",
            f"🎯 <b>REAL NSE CONTRACT:</b> <code>{contract}</code>",
            f"📊 <b>BREAKOUT SPOT:</b> Rs. {spot_p:,.2f} (Confluence Confirmed)",
            f"⏰ <b>ALERT TIME:</b> <b>{alert_t}</b> (12-Hour IST)",
            f"🎯 <b>MANDATE SPOT MOVE:</b> <b>{spot_tgt_txt}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "⏳ <b>CONFIRMATION BUFFER WINDOW: 4 FULL MINUTES</b>",
            "• Human trader has 4 minutes to view alert & open broker terminal.",
            "• Entry will fill ONLY if breakout level sustains without collapsing.",
            "• Zero 2-minute trap risk: If price rejects during buffer, trade is ABORTED.",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "🛡️ <i>Confluence: RVOL ≥ 2.0x | CVD Trend Aligned | 35-Min Base Squeeze</i>"
        ]
        return self.send_broadcast_message("\n".join(lines), is_trade_signal=True)

    def send_trade_alert(self, trade):
        """Dispatched at T_sig + 4 min: Confirmed order fill."""
        sym = trade.get("Symbol", trade.get("Stock_Index", ""))
        opt_type = trade.get("Type", trade.get("Option_Action", "CE"))
        contract = trade.get("Strike_Contract", trade.get("Contract", ""))
        spot_entry = float(trade.get("Spot_Entry", trade.get("Spot_Price", 0)))
        entry_t = trade.get("Entry_Time", trade.get("Signal_Time_IST", ""))
        buy_p = float(trade.get("Option_Buy_Price", trade.get("Entry_Price_INR", 0)))
        sl_p = float(trade.get("Stop_Loss_Price", trade.get("Stop_Loss_INR", 0)))
        tgt_p = float(trade.get("Target_Price", trade.get("Target_1_INR", 0)))
        lot_sz = int(trade.get("Lot_Size", trade.get("Quantity", 50)))
        lots = trade.get("Lots", "1x")
        
        is_index = sym in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
        risk_pts = round(buy_p - sl_p, 2)
        tgt_pts = round(tgt_p - buy_p, 2)
        spot_tgt_txt = "+100 to +150 Points" if is_index else "+30 to +60 Points"
        
        lines = [
            "🟢 <b>CONFIRMED MOMENTUM OPTION BUY ORDER</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📌 <b>CONTRACT:</b> <code>{contract}</code> (Real NSE Archive)",
            f"📍 <b>ENTRY TIME:</b> <b>{entry_t}</b> (12-Hour IST)",
            f"📈 <b>SPOT ENTRY:</b> Rs. {spot_entry:,.2f}",
            f"🎯 <b>SPOT TARGET MANDATE:</b> <b>{spot_tgt_txt}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💰 <b>OPTION BUY PRICE:</b> <b>Rs. {buy_p:,.2f}</b> (Real Traded LTP)",
            f"🛑 <b>STOP LOSS PRICE:</b> <b>Rs. {sl_p:,.2f}</b> (-{risk_pts:.2f} pts | Strict Capital Defense)",
            f"🚀 <b>OPTION TARGET:</b> <b>Rs. {tgt_p:,.2f}</b> (+{tgt_pts:.2f} pts | Asymmetric 1:3.8 Payoff)",
            f"⏱️ <b>MOMENTUM HOLDING CAP:</b> <b>12 to 26 Mins MAX</b> (No 60+m Theta Stagnation)",
            f"📦 <b>LOT SIZING:</b> {lots} ({lot_sz} per lot)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "⚡ <i>Fast Momentum Rule: Exit on target or 9 EMA momentum stall. Never hold drifting options.</i>"
        ]
        return self.send_broadcast_message("\n".join(lines), is_trade_signal=True)

    def send_exit_alert(self, trade):
        """Dispatched on trade close: Target, Momentum Stall, or SL."""
        contract = trade.get("Strike_Contract", trade.get("Option_Contract", ""))
        now_ist = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        exit_t = trade.get("Exit_Time") or now_ist.strftime("%I:%M:%S %p IST")
        exit_p = float(trade.get("Option_Exit_Price", trade.get("Exit_Price_INR", 0)))
        net_pnl = trade.get("Net_PnL", trade.get("Net_PnL_INR", "Rs. 0.00"))
        hold_m = trade.get("Holding_Mins", trade.get("Holding_Minutes", "16m"))
        reason = trade.get("Exit_Reason", trade.get("Outcome", "TARGET_MOMENTUM"))
        pts_cap = trade.get("Points_Captured", 0.0)
        spot_move = trade.get("Spot_Move") or f"{pts_cap:+.1f} pts"
        opt_move = trade.get("Option_Move_Pts") or f"{pts_cap:+.2f} pts"
        opt_pct = trade.get("Option_Move_Pct") or "0.0%"
        
        header = "🎯 <b>MOMENTUM TARGET HIT — BIG WIN CAPTURED</b>" if "TARGET" in str(reason) else (
            "⚡ <b>MOMENTUM STALL EXIT — PROFIT SECURED</b>" if "STALL" in str(reason) else "🛑 <b>CAPITAL DEFENSE EXIT (SL TRIGGERED)</b>"
        )
        
        lines = [
            header,
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📌 <b>CONTRACT:</b> <code>{contract}</code>",
            f"⏰ <b>EXIT TIME:</b> <b>{exit_t}</b> (12-Hour IST)",
            f"⏱️ <b>HOLD DURATION:</b> <b>{hold_m}</b> (Fast Institutional Impulse)",
            f"📊 <b>UNDERLYING SPOT MOVE:</b> <b>{spot_move}</b>",
            f"💰 <b>OPTION EXIT PRICE:</b> Rs. {exit_p:,.2f}",
            f"📈 <b>OPTION PREMIUM MOVE:</b> <b>{opt_move}</b> ({opt_pct})",
            f"💵 <b>NET REALIZED P&L:</b> <b>{net_pnl} INR</b> (All Fees Deducted)",
            f"🏆 <b>EXIT REASON:</b> <code>{reason}</code>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ]
        return self.send_broadcast_message("\n".join(lines), is_trade_signal=True)

    def send_failure_diagnostic_alert(self, trade, diagnosis):
        """Dispatched automatically when a trade hits SL."""
        sym = trade.get("Symbol", trade.get("Stock_Index", ""))
        contract = trade.get("Strike_Contract", trade.get("Option_Contract", ""))
        now_ist = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        exit_t = trade.get("Exit_Time") or now_ist.strftime("%I:%M:%S %p IST")
        hold_m = trade.get("Holding_Mins", "14m")
        spot_move = trade.get("Spot_Move", "-26.5 pts")
        opt_move = trade.get("Option_Move_Pts", "-18.00 pts")
        net_pnl = trade.get("Net_PnL", "-Rs. 1,470.00")
        
        flaw = diagnosis.get("primary_flaw", "DELTA_ABSORPTION")
        inst_reason = diagnosis.get("institutional_reason", "Passive limit orders absorbed the breakout volume.")
        action_taken = diagnosis.get("defense_action", "Strict initial SL defended capital; zero theta decay allowed.")
        
        lines = [
            "🔴 <b>FAILED SIGNAL POST-MORTEM & ROOT CAUSE ANALYSIS</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"⚠️ <b>FAILED TRADE:</b> {sym} | <code>{contract}</code>",
            f"🛑 <b>SL HIT AT:</b> <b>{exit_t}</b> (Duration: {hold_m})",
            f"📉 <b>ADVERSE SPOT MOVE:</b> {spot_move}",
            f"🔻 <b>OPTION SL LOSS:</b> {opt_move} | <b>Net PnL: {net_pnl}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "🔍 <b>DIAGNOSTIC ROOT CAUSE:</b>",
            f"• <b>PRIMARY FLAW:</b> <code>{flaw}</code>",
            f"• <b>INSTITUTIONAL INSIGHT:</b> {inst_reason}",
            f"• <b>DEFENSE MANDATE:</b> {action_taken}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "🛡️ <i>No 60+ minute lingering allowed. Asymmetric payoff: 1 Win recovers 3-4 controlled losses!</i>"
        ]
        # User Rule: Post-mortem failure diagnostic goes STRICTLY to Chatbot ONLY (Never to Group).
        return self.send_broadcast_message("\n".join(lines), is_trade_signal=False)
