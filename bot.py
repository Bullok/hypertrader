import time, requests, numpy as np, pandas as pd, schedule
from datetime import datetime, timezone
import os, sys, json

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

# — config ————————————————————————————————————————————
KEY    = os.environ.get("KEY", "")
WALLET = os.environ.get("WALLET", "")

BASE_URL = constants.MAINNET_API_URL

wallet  = Account.from_key(KEY) if KEY else None
info    = Info(BASE_URL)
exchange = Exchange(wallet, BASE_URL) if wallet else None

CONFIG = {
    "coin"              : "BTC",
    "interval"          : "1h",
    "leverage"          : 5,
    "adx_threshold"     : 30.0,
    "donchian_len"      : 20,
    "adx_len"           : 14,
    "atr_len"           : 14,
    "sl_atr"            : 2.0,
    "tp_atr"            : 4.0,
    "risk_per_trade"    : 0.02,
    "max_notional_pct"  : 3.0,
    "min_trade_usdc"    : 20.0,
    "max_dd_day"        : 0.03,
    "min_profit_close"  : 0.05,
    "cooldown_candles"  : 2,
    "funding_long_max"  : 0.0005,
    "funding_short_min" : -0.0005,
}

STATE = {
    "start_value"        : 0.0,
    "stop_day"           : False,
    "last_close_time"    : 0,
    "candles_since_close": 0,
    "breakeven_set"      : False,
}

# — helpers ————————————————————————————————————————————
def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def log(msg):
    print(f"[{now()}] {msg}", flush=True)

# — API helpers ————————————————————————————————————————
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

def get_asset_index():
    meta = post_info({"type": "metaAndAssetCtxs"})
    return next(i for i, x in enumerate(meta[0]["universe"]) if x["name"] == CONFIG["coin"])

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
            "startTime" : int((time.time() - 3600 * 100) * 1000),
        }
    })
    return data

# — trading functions ——————————————————————————————————
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
            limit_px = round(price * 1.01 if is_buy else price * 0.99)
            res = exchange.order(
                CONFIG["coin"], is_buy, size, limit_px,
                {"limit": {"tif": "Ioc"}}
            )
        elif order_type == "sl":
            trigger_px = int(round(price))
            res = exchange.order(
                CONFIG["coin"], is_buy, size, trigger_px,
                {"trigger": {"triggerPx": str(trigger_px), "isMarket": True, "tpsl": "sl"}},
                reduce_only=True
            )
        elif order_type == "tp":
            trigger_px = int(round(price))
            res = exchange.order(
                CONFIG["coin"], is_buy, size, trigger_px,
                {"trigger": {"triggerPx": str(trigger_px), "isMarket": False, "tpsl": "tp"}},
                reduce_only=True
            )
        else:
            res = None
        return res
    except Exception as e:
        log(f"Errore place_order: {e}")
        return None

def cancel_all_orders():
    try:
        open_orders = post_info({"type": "openOrders", "user": WALLET})
        asset_idx   = get_asset_index()
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
        is_long = float(pos["szi"]) > 0
        size    = abs(float(pos["szi"]))
        mids    = info.all_mids()
        price   = float(mids[CONFIG["coin"]])
        close_px = int(round(price * 0.99 if is_long else price * 1.01))
        res = exchange.order(
            CONFIG["coin"], not is_long, size, close_px,
            {"limit": {"tif": "Ioc"}},
            reduce_only=True
        )
        log(f"Posizione chiusa: {res}")
        STATE["last_close_time"]     = time.time()
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

# — indicators ————————————————————————————————————————
def compute(candles):
    df = pd.DataFrame(candles, columns=["t","o","h","l","c","v","n"])
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    n  = CONFIG["donchian_len"]
    df["dh"] = df["high"].shift(1).rolling(n).max()
    df["dl"] = df["low"].shift(1).rolling(n).min()

    # ADX
    a  = CONFIG["adx_len"]
    df["tr"]  = np.maximum(df["high"]-df["low"],
                np.maximum(abs(df["high"]-df["close"].shift(1)),
                           abs(df["low"] -df["close"].shift(1))))
    df["+dm"] = np.where((df["high"]-df["high"].shift(1)) > (df["low"].shift(1)-df["low"]),
                          np.maximum(df["high"]-df["high"].shift(1), 0), 0)
    df["-dm"] = np.where((df["low"].shift(1)-df["low"]) > (df["high"]-df["high"].shift(1)),
                          np.maximum(df["low"].shift(1)-df["low"], 0), 0)
    atr14      = df["tr"].ewm(alpha=1/a, adjust=False).mean()
    df["atr"]  = atr14
    pdi        = 100 * df["+dm"].ewm(alpha=1/a, adjust=False).mean() / atr14
    mdi        = 100 * df["-dm"].ewm(alpha=1/a, adjust=False).mean() / atr14
    dx         = 100 * abs(pdi-mdi) / (pdi+mdi+1e-10)
    df["adx"]  = dx.ewm(alpha=1/a, adjust=False).mean()

    df = df.dropna().reset_index(drop=True)
    return df

# — main loop —————————————————————————————————————————
def run():
    log("=" * 60)
    log("HyperTrader v2.0 — MAINNET")
    log(f"Coin: {CONFIG['coin']} | Interval: {CONFIG['interval']} | Leva: {CONFIG['leverage']}x")
    log(f"ADX soglia: {CONFIG['adx_threshold']} | Risk/trade: {CONFIG['risk_per_trade']*100:.0f}%")
    log(f"Cooldown: {CONFIG['cooldown_candles']} candele | Min trade: ${CONFIG['min_trade_usdc']}")
    log("=" * 60)

    try:
        candles = get_candles()
        if not candles:
            log("Errore: nessuna candela ricevuta")
            return

        df      = compute(candles)
        row     = df.iloc[-1]
        funding = get_funding()
        val, pos = get_account()

        log(f"Balance: ${val:.2f} USDC")
        log(f"Close=${row['close']:.0f} | DH={row['dh']:.0f} | DL={row['dl']:.0f} | "
            f"ADX={row['adx']:.1f} | ATR={row['atr']:.0f} | Funding={funding:.4%}")

        # — gestione posizione aperta ————————————————
        if pos:
            is_long = float(pos["szi"]) > 0
            entry   = float(pos["entryPx"])
            size    = abs(float(pos["szi"]))
            pnl     = (row["close"] - entry) * size * (1 if is_long else -1)
            notional= size * row["close"]

            log(f"HOLD {'LONG' if is_long else 'SHORT'} | Entry=${entry:.0f} | PnL=${pnl:+.2f}")

            # TP parziale breakeven
            tp_hit = (is_long  and row["close"] >= entry + row["atr"] * CONFIG["tp_atr"]) or \
                     (not is_long and row["close"] <= entry - row["atr"] * CONFIG["tp_atr"])
            if tp_hit and not STATE["breakeven_set"]:
                move_sl_to_breakeven(is_long, entry, size)

            # Chiusura anticipata ADX debole + profitto minimo
            min_profit_usdc = notional * CONFIG["min_profit_close"] / 100
            if row["adx"] < 20 and pnl > min_profit_usdc:
                market_close()
                log(f"CHIUSURA ANTICIPATA — ADX<20 + PnL={pnl:+.2f} > soglia ${min_profit_usdc:.2f}")
            return

        # — cooldown check ————————————————————————————
        STATE["candles_since_close"] += 1
        if STATE["candles_since_close"] <= CONFIG["cooldown_candles"]:
            remaining = CONFIG["cooldown_candles"] - STATE["candles_since_close"] + 1
            log(f"COOLDOWN — ancora {remaining} candele di attesa")
            return

        # — segnali ——————————————————————————————————
        long_s  = bool(row["close"] >= row["dh"] * 0.998 and row["adx"] > CONFIG["adx_threshold"])
        short_s = bool(row["close"] <= row["dl"] * 1.002 and row["adx"] > CONFIG["adx_threshold"])

        if long_s and funding > CONFIG["funding_long_max"]:
            long_s = False
            log("Funding troppo alto — blocca LONG")
        if short_s and funding < CONFIG["funding_short_min"]:
            short_s = False
            log("Funding troppo basso — blocca SHORT")

        if not long_s and not short_s:
            log("NO TRADE — nessun segnale valido")
            return

        # — sizing ————————————————————————————————————
        is_long = long_s
        atr     = row["atr"]
        price   = row["close"]
        sl      = price - atr * CONFIG["sl_atr"] if is_long else price + atr * CONFIG["sl_atr"]
        tp      = price + atr * CONFIG["tp_atr"] if is_long else price - atr * CONFIG["tp_atr"]

        risk    = val * CONFIG["risk_per_trade"]
        size    = round(min(risk / (abs(price - sl) + 1e-10),
                            val * CONFIG["max_notional_pct"] / price), 4)

        if size * price < CONFIG["min_trade_usdc"]:
            log(f"Size troppo piccola (${size*price:.2f}) — skip")
            return

        # — apre trade ————————————————————————————————
        set_leverage()
        res = place_order(is_long, size, price, "market")
        log(
            f"{'LONG' if is_long else 'SHORT'} APERTO | "
            f"Entry~${price:.0f} | SL=${sl:.0f} | TP=${tp:.0f} | "
            f"Size={size} BTC | Rischio=${risk:.2f} USDC | Leva={CONFIG['leverage']}x"
        )
        log(f"Risposta exchange: {res}")

        time.sleep(1)
        STATE["breakeven_set"] = False
        place_order(not is_long, size, sl, "sl")
        place_order(not is_long, round(size * 0.6, 4), tp, "tp")

    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"ERRORE: {e}")

def reset_day():
    STATE["stop_day"]   = False
    STATE["start_value"] = 0.0
    log("Reset giornaliero — nuovo giorno")

# — scheduler —————————————————————————————————————————
schedule.every().hour.at(":01").do(run)
schedule.every().day.at("00:00").do(reset_day)

if __name__ == "__main__":
    run()
    log("Scheduler attivo — ciclo automatico ogni ora")
    while True:
        schedule.run_pending()
        time.sleep(30)
