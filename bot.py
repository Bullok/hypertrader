import time, json, hashlib, requests, numpy as np, pandas as pd, schedule
from datetime import datetime
from eth_account import Account
from eth_account.messages import encode_defunct
from datetime import timezone

# ── config ───────────────────────────────────────────────────────────────────
import os, sys

KEY      = os.environ.get("KEY", "")
WALLET   = os.environ.get("WALLET", "")
BASE_URL = "https://api.hyperliquid-testnet.xyz"

ACCT = Account.from_key(KEY)

CONFIG = {
    "coin"            : "BTC",
    "interval"        : "1h",
    "leverage"        : 3,
    "adx_threshold"   : 25.0,
    "donchian_len"    : 20,
    "adx_len"         : 14,
    "atr_len"         : 14,
    "sl_atr"          : 2.0,
    "tp_atr"          : 4.0,
    "risk_per_trade"  : 0.02,
    "max_notional_pct": 3.0,
    "min_trade_usdc"  : 10.0,
    "max_dd_day"      : 0.03,
}

STATE = {
    "start_value": 0.0,
    "stop_day"   : False,
}

# ── helpers ───────────────────────────────────────────────────────────────────
def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def log(msg):
    print(f"[{now()}] {msg}", flush=True)

def post_info(payload):
    r = requests.post(f"{BASE_URL}/info", json=payload, timeout=15)
    return r.json()

def get_asset_index():
    r = post_info({"type": "metaAndAssetCtxs"})
    return next(i for i, x in enumerate(r[0]["universe"]) if x["name"] == CONFIG["coin"])

# ── FIX: le due operazioni (balance USDC e candele) sono ora separate ─────────
def get_candles():
    r = post_info({
        "type": "candleSnapshot",
        "req": {
            "coin"     : CONFIG["coin"],
            "interval" : CONFIG["interval"],
            "startTime": 0,
            "endTime"  : int(time.time() * 1000)
        }
    })
    candles = []
    for c in r[-60:]:
        if isinstance(c, dict):
            candles.append([c["t"], c["o"], c["h"], c["l"], c["c"], c.get("v", 0)])
        else:
            candles.append(c)
    return candles

def get_funding():
    r   = post_info({"type": "metaAndAssetCtxs"})
    idx = next(i for i, x in enumerate(r[0]["universe"]) if x["name"] == CONFIG["coin"])
    return float(r[1][idx]["funding"])

# ── FIX: get_account calcola val internamente, non dipende da get_candles ─────
def get_account():
    r   = post_info({"type": "clearinghouseState",     "user": WALLET})
    r2  = post_info({"type": "spotClearinghouseState", "user": WALLET})
    val = next((float(b["total"]) for b in r2.get("balances", []) if b["coin"] == "USDC"), 0.0)
    pos = next(
        (p["position"] for p in r.get("assetPositions", [])
         if float(p["position"]["szi"]) != 0),
        None
    )
    return val, pos

def sign_and_post(action):
    ts      = int(time.time() * 1000)
    payload = {"action": action, "nonce": ts, "vaultAddress": None}
    msg_str = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    h       = hashlib.sha256(msg_str.encode()).digest()
    signed  = ACCT.sign_message(encode_defunct(h))
    body    = {
        **payload,
        "signature": {
            "r": hex(signed.r),
            "s": hex(signed.s),
            "v": signed.v
        }
    }
    r = requests.post(f"{BASE_URL}/exchange", json=body, timeout=15)
    return r.json()

def set_leverage():
    sign_and_post({
        "type"    : "updateLeverage",
        "asset"   : get_asset_index(),
        "isCross" : False,
        "leverage": CONFIG["leverage"]
    })

def place_order(is_buy, size, price, order_type):
    if order_type == "market":
        t      = {"market": {}}
        reduce = False
    else:
        t = {
            "trigger": {
                "triggerPx": str(round(price, 1)),
                "isMarket" : order_type == "sl",
                "tpsl"     : order_type
            }
        }
        reduce = True

    return sign_and_post({
        "type"    : "order",
        "orders"  : [{
            "a": get_asset_index(),
            "b": is_buy,
            "p": str(round(price, 1)),
            "s": str(round(size, 4)),
            "r": reduce,
            "t": t
        }],
        "grouping": "na"
    })

def market_close():
    val, pos = get_account()
    if pos:
        is_long = float(pos["szi"]) > 0
        size    = abs(float(pos["szi"]))
        place_order(not is_long, size, 0, "market")

def compute(candles):
    df = pd.DataFrame(
        candles,
        columns=["time", "open", "high", "low", "close", "volume"]
    )
    df = df.astype({c: float for c in ["open", "high", "low", "close", "volume"]})

    n        = CONFIG["donchian_len"]
    df["dh"] = df["high"].shift(1).rolling(n).max()
    df["dl"] = df["low"].shift(1).rolling(n).min()

    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs()
        )
    )
    df["atr"] = df["tr"].ewm(span=CONFIG["atr_len"], adjust=False).mean()

    up  = df["high"].diff()
    dn  = -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)

    atr_s     = df["tr"].ewm(span=CONFIG["adx_len"], adjust=False).mean()
    pdi       = 100 * pd.Series(pdm, index=df.index).ewm(span=CONFIG["adx_len"], adjust=False).mean() / atr_s
    ndi       = 100 * pd.Series(ndm, index=df.index).ewm(span=CONFIG["adx_len"], adjust=False).mean() / atr_s
    dx        = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    df["adx"] = dx.ewm(span=CONFIG["adx_len"], adjust=False).mean()

    return df.iloc[-2]

# ── main loop ─────────────────────────────────────────────────────────────────
def run():
    if STATE["stop_day"]:
        log("STOP DAY attivo — ciclo saltato")
        return

    try:
        candles  = get_candles()
        funding  = get_funding()
        val, pos = get_account()

        if STATE["start_value"] == 0.0:
            STATE["start_value"] = val

        dd = (STATE["start_value"] - val) / STATE["start_value"]
        if dd >= CONFIG["max_dd_day"]:
            STATE["stop_day"] = True
            log(f"STOP DAY — drawdown {dd*100:.1f}% >= 3% | Capitale: ${val:.2f}")
            return

        row = compute(candles)
        log(
            f"Close=${row['close']:.0f} | "
            f"DH={row['dh']:.0f} | DL={row['dl']:.0f} | "
            f"ADX={row['adx']:.1f} | ATR={row['atr']:.0f} | "
            f"Funding={funding*100:.4f}%"
        )

        if pos:
            pnl     = float(pos["unrealizedPnl"])
            is_long = float(pos["szi"]) > 0
            log(
                f"HOLD {'LONG' if is_long else 'SHORT'} | "
                f"Entry=${float(pos['entryPx']):.0f} | "
                f"PnL={pnl:+.2f} USDC"
            )
            if row["adx"] < 20 and pnl > 0:
                market_close()
                log(f"CHIUSURA ANTICIPATA — ADX<20 + profitto | PnL={pnl:+.2f}")
            return

        long_s  = bool(row["close"] > row["dh"] and row["adx"] > CONFIG["adx_threshold"])
        short_s = bool(row["close"] < row["dl"] and row["adx"] > CONFIG["adx_threshold"])

        if long_s and funding > 0.0005:
            long_s = False
            log("Funding troppo alto — blocca LONG")
        if short_s and funding < -0.0005:
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
        risk    = val * CONFIG["risk_per_trade"]
        size    = round(min(risk / abs(price - sl), val * CONFIG["max_notional_pct"] / price), 4)

        if size * price < CONFIG["min_trade_usdc"]:
            log("Size troppo piccola — skip")
            return

        set_leverage()
        res = place_order(is_long, size, price, "market")
        log(
            f"{'LONG' if is_long else 'SHORT'} APERTO | "
            f"Entry~${price:.0f} | SL=${sl:.0f} | TP=${tp:.0f} | "
            f"Size={size} BTC | Rischio=${risk:.2f} USDC"
        )
        log(f"Risposta exchange: {res}")

        time.sleep(1)
        place_order(not is_long, size,               sl, "sl")
        place_order(not is_long, round(size*0.6, 4), tp, "tp")

    except Exception as e:
        log(f"ERRORE: {e}")

def reset_day():
    STATE["stop_day"]    = False
    STATE["start_value"] = 0.0
    log("Reset giornaliero — nuovo giorno")

def test_trade(direction="long"):
    """Apre un trade di test da 0.001 BTC per verificare connessione exchange."""
    try:
        log(f"TEST TRADE — apertura {direction.upper()} di test (0.001 BTC)")
        val, pos = get_account()
        log(f"Capitale letto: ${val:.2f} USDC")
        if pos:
            log("Posizione già aperta — chiudo prima")
            market_close()
            time.sleep(2)
        set_leverage()
        is_long = direction.lower() == "long"
        res     = place_order(is_long, 0.001, 0, "market")
        log(f"Risposta exchange: {res}")
        if "response" in res:
            log("TEST OK — trade aperto correttamente")
        else:
            log(f"TEST FALLITO — risposta inattesa: {res}")
    except Exception as e:
        log(f"TEST ERRORE: {e}")

# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log("=" * 50)
    log("HyperTrader AI v5.2 — TESTNET AVVIATO")
    log("Capitale: ~900 USDC | Leva: 3x | Risk: 2% | BTC/USDC 1H")
    log("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        direction = sys.argv[2] if len(sys.argv) > 2 else "long"
        test_trade(direction)
    else:
        schedule.every().hour.at(":01").do(run)
        schedule.every().day.at("00:01").do(reset_day)
        run()
        log("Scheduler attivo — ciclo automatico ogni ora")
        while True:
            schedule.run_pending()
            time.sleep(30)
