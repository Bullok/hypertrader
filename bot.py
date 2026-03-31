import json
import time
import eth_account
import requests
import pandas as pd
import numpy as np
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from datetime import datetime, timezone

with open("config.json") as f:
    config = json.load(f)

wallet = eth_account.Account.from_key(config["secret_key"])
address = config["account_address"]
exchange = Exchange(wallet, constants.MAINNET_API_URL, account_address=address)

SYMBOL = "BTC"
RISK_PCT = 0.015
LEVERAGE = 3
DONCHIAN_PERIOD = 20
ATR_PERIOD = 14
ADX_PERIOD = 14
ATR_SL_MULT = 2.0
ATR_TP_MULT = 4.0
MAX_FUNDING = 0.0005
MAX_WICK_MULT = 3.0
DAILY_DD_LIMIT = 0.03
WEEKLY_DD_LIMIT = 0.07

state = {"daily_start_equity": None, "weekly_start_equity": None, "daily_stop": False, "weekly_risk_halved": False, "last_day": None, "last_week": None}

def get_equity():
    r = requests.post("https://api.hyperliquid.xyz/info", json={"type": "userState", "user": address}, headers={"Content-Type": "application/json"})
    return 17.0  # saldo fisso temporaneo

def get_position():
    from hyperliquid.info import Info
    _info = Info(constants.MAINNET_API_URL, skip_ws=True)
    state = _info.user_state(address)
    for pos in state.get("assetPositions", []):
        if pos["position"]["coin"] == SYMBOL:
            szi = float(pos["position"]["szi"])
            if szi != 0:
                return "long" if szi > 0 else "short", abs(szi), float(pos["position"]["entryPx"])
    return None, 0.0, 0.0

def get_candles():
    now = int(time.time() * 1000)
    start = now - 100 * 3600 * 1000
    r = requests.post("https://api.hyperliquid.xyz/info", json={"type": "candleSnapshot", "req": {"coin": SYMBOL, "interval": "1h", "startTime": start, "endTime": now}}, headers={"Content-Type": "application/json"})
    df = pd.DataFrame(r.json(), columns=["t","T","s","i","o","c","h","l","v","n"])
    return df.astype({"o": float, "h": float, "l": float, "c": float, "v": float})

def calc_atr(df):
    h, l, c = df["h"], df["l"], df["c"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean()

def calc_adx(df):
    h, l = df["h"], df["l"]
    pdm = h.diff().clip(lower=0)
    mdm = (-l.diff()).clip(lower=0)
    pdm = pdm.where(pdm > mdm, 0.0)
    mdm = mdm.where(mdm >= pdm, 0.0)
    atr = calc_atr(df)
    atr_safe = atr.replace(0, np.nan)
    pdi = 100 * pdm.ewm(span=ADX_PERIOD, adjust=False).mean() / atr_safe
    mdi = 100 * mdm.ewm(span=ADX_PERIOD, adjust=False).mean() / atr_safe
    denom = (pdi + mdi).replace(0, np.nan)
    dx = (100 * (pdi - mdi).abs() / denom).fillna(0)
    return dx.ewm(span=ADX_PERIOD, adjust=False).mean()

def get_funding():
    r = requests.post("https://api.hyperliquid.xyz/info", json={"type": "metaAndAssetCtxs"}, headers={"Content-Type": "application/json"})
    data = r.json()
    for i, a in enumerate(data[0]["universe"]):
        if a["name"] == SYMBOL:
            return float(data[1][i].get("funding", 0))
    return 0.0

def calc_size(equity, atr, price, risk_pct):
    if atr == 0:
        return 0.0
    return round((equity * risk_pct) / (atr * ATR_SL_MULT / price) / price, 5)

def check_drawdown(equity):
    now = datetime.now(timezone.utc)
    today = now.date()
    week = now.isocalendar()[1]
    if state["last_day"] != today:
        state["daily_start_equity"] = equity
        state["daily_stop"] = False
        state["last_day"] = today
    if state["last_week"] != week:
        state["weekly_start_equity"] = equity
        state["weekly_risk_halved"] = False
        state["last_week"] = week
    if state["daily_start_equity"] and state["daily_start_equity"] > 0:
        if (state["daily_start_equity"] - equity) / state["daily_start_equity"] >= DAILY_DD_LIMIT:
            state["daily_stop"] = True
            print("STOP giornaliero raggiunto")
    if state["weekly_start_equity"] and state["weekly_start_equity"] > 0:
        if (state["weekly_start_equity"] - equity) / state["weekly_start_equity"] >= WEEKLY_DD_LIMIT:
            if not state["weekly_risk_halved"]:
                state["weekly_risk_halved"] = True
                print("Rischio dimezzato per drawdown settimanale")
    return RISK_PCT * 0.5 if state["weekly_risk_halved"] else RISK_PCT

def run():
    print("Bot avviato")
    exchange.update_leverage(LEVERAGE, SYMBOL, is_cross=True)
    print("Capitale: " + str(round(get_equity(), 2)) + " USDC")
    while True:
        try:
            now = datetime.now(timezone.utc)
            print("\n[" + now.strftime("%H:%M:%S") + "] Analisi...")
            df = get_candles().iloc[:-1]
            close = df["c"].iloc[-1]
            high_prev = df["h"].iloc[-DONCHIAN_PERIOD-1:-1].max()
            low_prev = df["l"].iloc[-DONCHIAN_PERIOD-1:-1].min()
            atr = calc_atr(df).iloc[-1]
            adx = calc_adx(df).iloc[-1]
            last = df.iloc[-1]
            big_wick = max(last["h"]-max(last["o"],last["c"]), min(last["o"],last["c"])-last["l"]) > MAX_WICK_MULT * atr
            funding = get_funding()
            equity = get_equity()
            risk = check_drawdown(equity)
            print("Equity: " + str(round(equity,2)) + " | Close: " + str(round(close,1)) + " | ADX: " + str(round(adx,1)) + " | ATR: " + str(round(atr,1)))
            pos_side, pos_size, entry_px = get_position()
            if pos_side:
                print("Posizione: " + pos_side + " | Entry: " + str(entry_px))
                if adx < 20:
                    in_profit = (pos_side == "long" and close > entry_px) or (pos_side == "short" and close < entry_px)
                    if in_profit:
                        print("ADX < 20 in profitto - chiudo")
                        exchange.market_close(SYMBOL)
            elif not state["daily_stop"]:
                long_sig = close > high_prev and adx > 25 and not big_wick and abs(funding) <= MAX_FUNDING
                short_sig = close < low_prev and adx > 25 and not big_wick and abs(funding) <= MAX_FUNDING
                if adx < 20:
                    print("Laterale - nessun trade")
                elif long_sig:
                    size = calc_size(equity, atr, close, risk)
                    print("LONG | Size: " + str(size) + " | SL: " + str(round(close-atr*ATR_SL_MULT,1)) + " | TP: " + str(round(close+atr*ATR_TP_MULT,1)))
                    print(exchange.market_open(SYMBOL, True, size))
                elif short_sig:
                    size = calc_size(equity, atr, close, risk)
                    print("SHORT | Size: " + str(size) + " | SL: " + str(round(close+atr*ATR_SL_MULT,1)) + " | TP: " + str(round(close-atr*ATR_TP_MULT,1)))
                    print(exchange.market_open(SYMBOL, False, size))
                else:
                    print("Nessun segnale")
        except Exception as e:
            print("Errore: " + str(e))
        time.sleep(60)

if __name__ == "__main__":
    run()
