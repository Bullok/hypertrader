import time, requests, numpy as np, pandas as pd, schedule
from datetime import datetime, timezone
import os, sys, json

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

KEY    = os.environ.get("KEY", "")
WALLET = os.environ.get("WALLET", "")

BASE_URL = constants.MAINNET_API_URL

wallet   = Account.from_key(KEY) if KEY else None
info     = Info(BASE_URL)
exchange = Exchange(wallet, BASE_URL) if wallet else None

CONFIG = {
    "coin"              : "BTC",
    "interval"          : "1h",
    "leverage"          : 5,
    "ema_fast"          : 9,
    "ema_slow"          : 21,
    "ema_trend"         : 50,
    "adx_threshold"     : 25.0,
    "adx_len"           : 14,
    "atr_len"           : 14,
    "sl_atr"            : 2.0,
    "tp_atr"            : 4.0,
    "risk_per_trade"    : 0.02,
    "max_notional_pct"  : 3.0,
    "min_trade_usdc"    : 20.0,
    "cooldown_candles"  : 3,
    "funding_long_max"  : 0.0005,
    "funding_short_min" : -0.0005,
}

STATE = {
    "start_value"        : 0.0,
    "stop_day"           : False,
    "candles_since_close": 0,
    "breakeven_set"      : False,
}

def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def log(msg):
    print(f"[{now()}] {msg}", flush=True)

def post_info(payload):
    r = requests.post(f"{BASE_URL}/info", json=payload, timeout=15)
    try:
        return r.json()
    except Exception:
        return {}

def get_account():
    user_state = post_info({"type": "clearinghouseState",     "user": WALLET})
    spot_state = post_info({"type": "spotClearinghouseState", "user": WALLET})
    val = float(next((b["total"] for b in spot_state.get("balances", []) if b["coin"] == "USDC"), 0))
    pos = next(
        (p["position"] for p in user_state.get("assetPositions", [])
         if float(p["position"]["szi"]) != 0),
        None
    )
    return val, pos

def get_funding():
    meta = post_info({"type": "metaAndAssetCtxs"})
    idx  = next(i for i, x in enumerate(meta[0]["universe"]) if x["name"] == CONFIG["coin"])
    return float(meta[1][idx]["funding"])

def get_candles():
    data = post_info({
        "type": "candleSnapshot",
        "req": {
            "coin"      : CONFIG["coin"],
            "interval"  : CONFIG["interval"],
            "startTime" : int((time.time() - 3600 * 150) * 1000),
        }
    })
    return data

def set_leverage():
    try:
        res = exchange.update_leverage(CONFIG["leverage"], CONFIG["coin"], is_cross=False)
        log(f"Leva impostata: {res}")
    except Exception as e:
        log(f"Errore set_leverage: {e}")

def place_order(is_buy, size, price, order_type):
    try:
        size = round(size, 4)
        if order_type == "market":
            limit_px = int(round(price * 1.01 if is_buy else price * 0.99))
            res = exchange.order(CONFIG["coin"], is_buy, size, limit_px, {"limit": {"tif": "Ioc"}})
        elif order_type == "sl":
            trigger_px = float(int(round(price)))
            res = exchange.order(CONFIG["coin"], is_buy, size, trigger_px,
                {"trigger": {"triggerPx": trigger_px, "isMarket": True, "tpsl": "sl"}}, reduce_only=True)
        elif order_type == "tp":
            trigger_px = float(int(round(price)))
            res = exchange.order(CONFIG["coin"], is_buy, size, trigger_px,
                {"trigger": {"triggerPx": trigger_px, "isMarket": False, "tpsl": "tp"}}, reduce_only=True)
        else:
            res = None
        return res
    except Exception as e:
        log(f"Errore place_order: {e}")
        return None

def cancel_all_orders():
    try:
        open_orders = post_info({"type": "openOrders", "user": WALLET})
        for o in open_orders:
            if o.get("coin") == CONFIG["coin"]:
                exchange.cancel(CONFIG["coin"], o["oid"])
        log("Ordini pendenti cancellati")
    except Exception as e:
        log(f"Errore cancel ordini: {e}")

def market_close():
    val, pos = get_account()
    if pos:
        cancel_all_orders()
        time.sleep(0.5)
        is_long  = float(pos["szi"]) > 0
        size     = abs(float(pos["szi"]))
        mids     = info.all_mids()
        price    = float(mids[CONFIG["coin"]])
        close_px = int(round(price * 0.99 if is_long else price * 1.01))
        res = exchange.order(CONFIG["coin"], not is_long, size, close_px,
            {"limit": {"tif": "Ioc"}}, reduce_only=True)
        log(f"Posizione chiusa: {res}")
        STATE["candles_since_close"] = 0
        STATE["breakeven_set"]       = False

def move_sl_to_breakeven(is_long, entry, size):
    try:
        cancel_all_orders()
        time.sleep(0.5)
        place_order(not is_long, size, entry, "sl")
        STATE["breakeven_set"] = True
        log(f"SL spostato a breakeven @ ${entry:.0f}")
    except Exception as e:
        log(f"Errore breakeven: {e}")

def calc_ema(series, period):
    k = 2 / (period + 1)
    result = [None] * len(series)
    for i in range(len(series)):
        if i < period - 1:
            continue
        if i == period - 1:
            result[i] = sum(series[i-period+1:i+1]) / period
            continue
        result[i] = series[i] * k + result[i-1] * (1 - k)
    return result

def compute(candles):
    df = pd.DataFrame(candles, columns=["t","o","h","l","c","v","n"])
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    closes = df["close"].tolist()
    df["ema_fast"]  = calc_ema(closes, CONFIG["ema_fast"])
    df["ema_slow"]  = calc_ema(closes, CONFIG["ema_slow"])
    df["ema_trend"] = calc_ema(closes, CONFIG["ema_trend"])

    a = CONFIG["atr_len"]
    df["tr"] = np.maximum(df["high"]-df["low"],
               np.maximum(abs(df["high"]-df["close"].shift(1)),
                          abs(df["low"] -df["close"].shift(1))))
    df["atr"] = df["tr"].ewm(alpha=1/a, adjust=False).mean()

    adx_len = CONFIG["adx_len"]
    df["+dm"] = np.where((df["high"]-df["high"].shift(1))>(df["low"].shift(1)-df["low"]),
                          np.maximum(df["high"]-df["high"].shift(1),0),0)
    df["-dm"] = np.where((df["low"].shift(1)-df["low"])>(df["high"]-df["high"].shift(1)),
                          np.maximum(df["low"].shift(1)-df["low"],0),0)
    atr14 = df["tr"].ewm(alpha=1/adx_len, adjust=False).mean()
    pdi   = 100 * df["+dm"].ewm(alpha=1/adx_len, adjust=False).mean() / atr14
    mdi   = 100 * df["-dm"].ewm(alpha=1/adx_len, adjust=False).mean() / atr14
    dx    = 100 * abs(pdi-mdi) / (pdi+mdi+1e-10)
    df["adx"] = dx.ewm(alpha=1/adx_len, adjust=False).mean()

    df = df.dropna().reset_index(drop=True)
    return df

def run():
    log("=" * 60)
    log("HyperTrader v3.0 — MAINNET — EMA Cross + ADX")
    log(f"Coin: {CONFIG['coin']} | Interval: {CONFIG['interval']} | Leva: {CONFIG['leverage']}x")
    log(f"EMA {CONFIG['ema_fast']}/{CONFIG['ema_slow']}/{CONFIG['ema_trend']} | ADX soglia: {CONFIG['adx_threshold']} | Risk: {CONFIG['risk_per_trade']*100:.0f}%")
    log("=" * 60)

    try:
        candles = get_candles()
        if not candles or len(candles) < 60:
            log("Errore: candele insufficienti")
            return

        df      = compute(candles)
        row     = df.iloc[-1]
        prev    = df.iloc[-2]
        funding = get_funding()
        val, pos = get_account()

        log(f"Balance: ${val:.2f} USDC")
        log(f"Close=${row['close']:.0f} | EMA{CONFIG['ema_fast']}={row['ema_fast']:.0f} | "
            f"EMA{CONFIG['ema_slow']}={row['ema_slow']:.0f} | EMA{CONFIG['ema_trend']}={row['ema_trend']:.0f} | "
            f"ADX={row['adx']:.1f} | ATR={row['atr']:.0f} | Funding={funding:.4%}")

        if pos:
            is_long  = float(pos["szi"]) > 0
            entry    = float(pos["entryPx"])
            size     = abs(float(pos["szi"]))
            pnl      = (row["close"] - entry) * size * (1 if is_long else -1)
            notional = size * row["close"]

            log(f"HOLD {'LONG' if is_long else 'SHORT'} | Entry=${entry:.0f} | PnL=${pnl:+.2f}")

            tp_hit = (is_long  and row["close"] >= entry + row["atr"] * CONFIG["tp_atr"]) or \
                     (not is_long and row["close"] <= entry - row["atr"] * CONFIG["tp_atr"])
            if tp_hit and not STATE["breakeven_set"]:
                move_sl_to_breakeven(is_long, entry, size)

            ema_against = (is_long  and row["ema_fast"] < row["ema_slow"]) or \
                          (not is_long and row["ema_fast"] > row["ema_slow"])
            min_profit_usdc = notional * 0.05 / 100
            if ema_against and row["adx"] < 20 and pnl > min_profit_usdc:
                market_close()
                log(f"CHIUSURA ANTICIPATA — EMA invertita + ADX<20 + PnL=${pnl:+.2f}")
            return

        STATE["candles_since_close"] += 1
        if STATE["candles_since_close"] <= CONFIG["cooldown_candles"]:
            remaining = CONFIG["cooldown_candles"] - STATE["candles_since_close"] + 1
            log(f"COOLDOWN — ancora {remaining} candele di attesa")
            return

        bull_cross = (row["ema_fast"]  > row["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"])
        bear_cross = (row["ema_fast"]  < row["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"])
        trend_up   = row["close"] > row["ema_trend"]
        trend_down = row["close"] < row["ema_trend"]
        adx_ok     = row["adx"] > CONFIG["adx_threshold"]

        log(f"Segnali — EMA cross: {'BULL' if bull_cross else 'BEAR' if bear_cross else 'none'} | "
            f"Trend: {'UP' if trend_up else 'DOWN'} | ADX ok: {adx_ok}")

        long_s  = bull_cross and trend_up  and adx_ok
        short_s = bear_cross and trend_down and adx_ok

        if long_s and funding > CONFIG["funding_long_max"]:
            long_s = False
            log("Funding troppo alto — blocca LONG")
        if short_s and funding < CONFIG["funding_short_min"]:
            short_s = False
            log("Funding troppo basso — blocca SHORT")

        if not long_s and not short_s:
            log("NO TRADE — nessun segnale valido")
            return

        is_long = long_s
        atr     = row["atr"]
        price   = row["close"]
        sl      = price - atr * CONFIG["sl_atr"] if is_long else price + atr * CONFIG["sl_atr"]
        tp      = price + atr * CONFIG["tp_atr"] if is_long else price - atr * CONFIG["tp_atr"]

        risk = val * CONFIG["risk_per_trade"]
        size = round(min(risk / (abs(price - sl) + 1e-10), val * CONFIG["max_notional_pct"] / price), 4)

        if size * price < CONFIG["min_trade_usdc"]:
            log(f"Size troppo piccola (${size*price:.2f}) — skip")
            return

        set_leverage()
        res = place_order(is_long, size, price, "market")
        log(f"{'LONG' if is_long else 'SHORT'} APERTO | Entry~${price:.0f} | SL=${sl:.0f} | TP=${tp:.0f} | "
            f"Size={size} BTC | Rischio=${risk:.2f} USDC | Leva={CONFIG['leverage']}x")
        log(f"Risposta: {res}")

        time.sleep(1)
        STATE["breakeven_set"] = False
        place_order(not is_long, size, sl, "sl")
        place_order(not is_long, round(size * 0.6, 4), tp, "tp")

    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"ERRORE: {e}")

def reset_day():
    STATE["stop_day"]    = False
    STATE["start_value"] = 0.0
    log("Reset giornaliero — nuovo giorno")

schedule.every().hour.at(":01").do(run)
schedule.every().day.at("00:00").do(reset_day)

if __name__ == "__main__":
    run()
    log("Scheduler attivo — ciclo automatico ogni ora")
    while True:
        schedule.run_pending()
        time.sleep(30)
