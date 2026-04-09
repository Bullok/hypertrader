import time, requests, numpy as np, pandas as pd, schedule

from datetime import datetime, timezone
import os, sys, json, hashlib
from eth_account import Account
from eth_account.messages import encode_defunct

# ── config ───────────────────────────────────────────────────────────────────

KEY    = os.environ.get("KEY", "")
WALLET = os.environ.get("WALLET", "")
ACCT   = Account.from_key(KEY) if KEY else None

BASE_URL = "https://api.hyperliquid.xyz"

CONFIG = {
    "coin"             : "BTC",
    "interval"         : "1h",
    "leverage"         : 5,          # ← alzato da 3 a 5 (coerente col backtest)
    "adx_threshold"    : 30.0,       # ← alzato da 25 a 30 (segnali più puliti)
    "donchian_len"     : 20,
    "adx_len"          : 14,
    "atr_len"          : 14,
    "sl_atr"           : 2.0,
    "tp_atr"           : 4.0,
    "risk_per_trade"   : 0.02,       # 2% del capitale per trade
    "max_notional_pct" : 3.0,
    "min_trade_usdc"   : 20.0,       # ← alzato da 10 a 20 (fee-safe)
    "max_dd_day"       : 0.03,
    "min_profit_close" : 0.05,       # ← NUOVO: profitto minimo (%) per chiusura anticipata
    "cooldown_candles" : 2,          # ← NUOVO: candele di cooldown dopo chiusura
    "funding_long_max" : 0.0005,
    "funding_short_min": -0.0005,
}

STATE = {
    "start_value"     : 0.0,
    "stop_day"        : False,
    "last_close_time" : 0,           # timestamp ultimo close (per cooldown)
    "candles_since_close": 0,        # contatore candele dal'ultima chiusura
    "breakeven_set"   : False,       # flag SL a breakeven già spostato
}

# ── helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def log(msg):
    print(f"[{now()}] {msg}", flush=True)

def post_info(payload):
    r = requests.post(f"{BASE_URL}/info", json=payload, timeout=15)
    return r.json()

# ── signing ───────────────────────────────────────────────────────────────────

def sign_and_post(action):
    nonce   = int(time.time() * 1000)
    payload = {
        "action"      : action,
        "nonce"       : nonce,
        "agentAddress": WALLET
    }
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

# ── trading functions ─────────────────────────────────────────────────────────

def get_asset_index():
    meta = post_info({"type": "metaAndAssetCtxs"})
    return next(i for i, x in enumerate(meta[0]["universe"]) if x["name"] == CONFIG["coin"])

def get_candles():
    data = post_info({
        "type": "candleSnapshot",
        "req": {
            "coin"     : CONFIG["coin"],
            "interval" : CONFIG["interval"],
            "startTime": 0,
            "endTime"  : int(time.time() * 1000)
        }
    })
    candles = []
    for c in data[-60:]:
        if isinstance(c, dict):
            candles.append([c["t"], c["o"], c["h"], c["l"], c["c"], c.get("v", 0)])
        else:
            candles.append(c)
    return candles

def get_funding():
    meta = post_info({"type": "metaAndAssetCtxs"})
    idx  = next(i for i, x in enumerate(meta[0]["universe"]) if x["name"] == CONFIG["coin"])
    return float(meta[1][idx]["funding"])

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

def set_leverage():
    res = sign_and_post({
        "type"    : "updateLeverage",
        "asset"   : get_asset_index(),
        "isCross" : False,
        "leverage": CONFIG["leverage"]
    })
    log(f"Leva impostata: {res}")

def place_order(is_buy, size, price, order_type):
    if order_type == "market":
        t      = {"market": {}}
        reduce = False
    elif order_type == "sl":
        t      = {"trigger": {"triggerPx": str(round(price, 1)), "isMarket": True,  "tpsl": "sl"}}
        reduce = True
    elif order_type == "tp":
        t      = {"trigger": {"triggerPx": str(round(price, 1)), "isMarket": False, "tpsl": "tp"}}
        reduce = True
    else:
        t      = {"market": {}}
        reduce = False

    return sign_and_post({
        "type"    : "order",
        "orders"  : [{
            "a": get_asset_index(),
            "b": is_buy,
            "p": str(round(price, 1)) if price else "0",
            "s": str(round(size, 4)),
            "r": reduce,
            "t": t
        }],
        "grouping": "na"
    })

def cancel_all_orders():
    """Cancella tutti gli ordini aperti (SL/TP pendenti)."""
    try:
        open_orders = post_info({"type": "openOrders", "user": WALLET})
        asset_idx   = get_asset_index()
        for o in open_orders:
            if o.get("coin") == CONFIG["coin"]:
                sign_and_post({
                    "type"  : "cancel",
                    "cancels": [{"a": asset_idx, "o": o["oid"]}]
                })
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
        place_order(not is_long, size, 0, "market")
        STATE["last_close_time"]      = time.time()
        STATE["candles_since_close"]  = 0
        STATE["breakeven_set"]        = False

def move_sl_to_breakeven(is_long, entry, size):
    """Sposta lo SL a breakeven dopo il TP parziale."""
    try:
        cancel_all_orders()
        time.sleep(0.5)
        place_order(not is_long, size, entry, "sl")
        STATE["breakeven_set"] = True
        log(f"SL spostato a breakeven @ ${entry:.0f}")
    except Exception as e:
        log(f"Errore breakeven: {e}")

def compute(candles):
    df = pd.DataFrame(
        candles,
        columns=["time", "open", "high", "low", "close", "volume"]
    )
    df = df.astype({c: float for c in ["open", "high", "low", "close", "volume"]})

    n         = CONFIG["donchian_len"]
    df["dh"]  = df["high"].shift(1).rolling(n).max()
    df["dl"]  = df["low"].shift(1).rolling(n).min()

    df["tr"]  = np.maximum(
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

    atr_s = df["tr"].ewm(span=CONFIG["adx_len"], adjust=False).mean()
    pdi   = 100 * pd.Series(pdm, index=df.index).ewm(span=CONFIG["adx_len"], adjust=False).mean() / (atr_s + 1e-10).replace(0, 1e-10).replace(0, 1e-10)
    ndi   = 100 * pd.Series(ndm, index=df.index).ewm(span=CONFIG["adx_len"], adjust=False).mean() / (atr_s + 1e-10).replace(0, 1e-10).replace(0, 1e-10)
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-10).replace(0, np.nan)
    df["adx"] = dx.ewm(span=CONFIG["adx_len"], adjust=False).mean()

    return df.iloc[-2]

# ── main loop ─────────────────────────────────────────────────────────────────

def run():
    if STATE["stop_day"]:
        log("STOP DAY attivo — ciclo saltato")
        return

    try:
        candles = get_candles()
        funding = get_funding()
        val, pos = get_account()

        if STATE["start_value"] == 0.0:
            STATE["start_value"] = val

        dd = (STATE["start_value"] - val) / STATE["start_value"] if STATE["start_value"] > 0 else 0
        if dd >= CONFIG["max_dd_day"]:
            STATE["stop_day"] = True
            log(f"STOP DAY — drawdown {dd*100:.1f}% >= 3% | Capitale: ${val:.2f}")
            return

        # Incrementa cooldown counter
        if STATE["candles_since_close"] < CONFIG["cooldown_candles"]:
            STATE["candles_since_close"] += 1

        row = compute(candles)
        log(
            f"Close=${row['close']:.0f} | "
            f"DH={row['dh']:.0f} | DL={row['dl']:.0f} | "
            f"ADX={row['adx']:.1f} | ATR={row['atr']:.0f} | "
            f"Funding={funding*100:.4f}%"
        )

        # ── gestione posizione aperta ─────────────────────────────────────────
        if pos:
            pnl     = float(pos["unrealizedPnl"])
            is_long = float(pos["szi"]) > 0
            entry   = float(pos["entryPx"])
            size    = abs(float(pos["szi"]))
            notional = size * row["close"]

            log(
                f"HOLD {'LONG' if is_long else 'SHORT'} | "
                f"Entry=${entry:.0f} | "
                f"PnL={pnl:+.2f} USDC | "
                f"Notional=${notional:.2f}"
            )

            # Breakeven SL dopo TP parziale (se non già fatto)
            if not STATE["breakeven_set"]:
                tp_hit = (is_long  and row["close"] >= entry + row["atr"] * CONFIG["tp_atr"]) or \
                         (not is_long and row["close"] <= entry - row["atr"] * CONFIG["tp_atr"])
                if tp_hit:
                    log("TP parziale raggiunto — sposto SL a breakeven")
                    move_sl_to_breakeven(is_long, entry, size)

            # Chiusura anticipata: ADX debole + profitto MINIMO sufficiente a coprire fee
            min_profit_usdc = notional * CONFIG["min_profit_close"] / 100
            if row["adx"] < 20 and pnl > min_profit_usdc:
                market_close()
                log(f"CHIUSURA ANTICIPATA — ADX<20 + PnL={pnl:+.2f} > soglia ${min_profit_usdc:.2f}")
            return

        # ── cooldown check ────────────────────────────────────────────────────
        if STATE["candles_since_close"] < CONFIG["cooldown_candles"]:
            log(f"COOLDOWN — ancora {CONFIG['cooldown_candles'] - STATE['candles_since_close']} candele di attesa")
            return

        # ── segnali ───────────────────────────────────────────────────────────
        long_s  = bool(row["close"] > row["dh"] and row["adx"] > CONFIG["adx_threshold"])
        short_s = bool(row["close"] < row["dl"] and row["adx"] > CONFIG["adx_threshold"])

        if long_s  and funding >  CONFIG["funding_long_max"]:
            long_s  = False
            log("Funding troppo alto — blocca LONG")
        if short_s and funding <  CONFIG["funding_short_min"]:
            short_s = False
            log("Funding troppo basso — blocca SHORT")

        if not long_s and not short_s:
            log("NO TRADE — nessun segnale valido")
            return

        # ── sizing ────────────────────────────────────────────────────────────
        is_long = long_s
        atr     = row["atr"]
        price   = row["close"]
        sl      = price - atr * CONFIG["sl_atr"] if is_long else price + atr * CONFIG["sl_atr"]
        tp      = price + atr * CONFIG["tp_atr"] if is_long else price - atr * CONFIG["tp_atr"]

        risk    = val * CONFIG["risk_per_trade"]
        size    = round(min(risk / (abs(price - sl) + 1e-10), val * CONFIG["max_notional_pct"] / price), 4)

        if size * price < CONFIG["min_trade_usdc"]:
            log(f"Size troppo piccola (${size*price:.2f}) — skip")
            return

        # ── apre trade ────────────────────────────────────────────────────────
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
    STATE["stop_day"]             = False
    STATE["start_value"]          = 0.0
    STATE["candles_since_close"]  = CONFIG["cooldown_candles"]  # reset cooldown a fine giornata
    log("Reset giornaliero — nuovo giorno")

# ── backtesting ───────────────────────────────────────────────────────────────

def backtest(candles_raw, capital=900.0, leverage=5):
    """
    Backtesting completo sulla logica del bot v2.
    Restituisce un dict con tutte le metriche.
    """
    df = pd.DataFrame(candles_raw, columns=["time", "open", "high", "low", "close", "volume"])
    df = df.astype({c: float for c in ["open", "high", "low", "close", "volume"]})

    n         = CONFIG["donchian_len"]
    df["dh"]  = df["high"].shift(1).rolling(n).max()
    df["dl"]  = df["low"].shift(1).rolling(n).min()
    df["tr"]  = np.maximum(
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
    atr_s      = df["tr"].ewm(span=CONFIG["adx_len"], adjust=False).mean()
    pdi        = 100 * pd.Series(pdm, index=df.index).ewm(span=CONFIG["adx_len"], adjust=False).mean() / (atr_s + 1e-10).replace(0, 1e-10).replace(0, 1e-10)
    ndi        = 100 * pd.Series(ndm, index=df.index).ewm(span=CONFIG["adx_len"], adjust=False).mean() / (atr_s + 1e-10).replace(0, 1e-10).replace(0, 1e-10)
    dx         = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-10).replace(0, np.nan)
    df["adx"]  = dx.ewm(span=CONFIG["adx_len"], adjust=False).mean()
    df         = df.dropna().reset_index(drop=True)

    FEE_RATE   = 0.00035   # 0.035% taker Hyperliquid
    equity     = capital
    trades     = []
    cooldown   = 0
    pos        = None      # dict: {side, entry, sl, tp, size, equity_at_open}

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        # ── check posizione aperta ────────────────────────────────────────────
        if pos:
            hit_sl = (pos["side"] == "long"  and row["low"]  <= pos["sl"]) or \
                     (pos["side"] == "short" and row["high"] >= pos["sl"])
            hit_tp = (pos["side"] == "long"  and row["high"] >= pos["tp"]) or \
                     (pos["side"] == "short" and row["low"]  <= pos["tp"])

            close_price = None
            close_reason = None

            if hit_sl:
                close_price  = pos["sl"]
                close_reason = "SL"
            elif hit_tp:
                close_price  = pos["tp"]
                close_reason = "TP"
            elif prev["adx"] < 20:
                pnl_unrealized = (row["close"] - pos["entry"]) * pos["size"] * (1 if pos["side"] == "long" else -1)
                min_profit     = pos["size"] * row["close"] * CONFIG["min_profit_close"] / 100
                if pnl_unrealized > min_profit:
                    close_price  = row["close"]
                    close_reason = "ADX_WEAK"

            if close_price:
                pnl_gross = (close_price - pos["entry"]) * pos["size"] * (1 if pos["side"] == "long" else -1)
                fee_open  = pos["entry"]  * pos["size"] * FEE_RATE
                fee_close = close_price   * pos["size"] * FEE_RATE
                pnl_net   = pnl_gross - fee_open - fee_close

                equity += pnl_net
                trades.append({
                    "entry_time"  : pos["entry_time"],
                    "close_time"  : row["time"],
                    "side"        : pos["side"],
                    "entry"       : pos["entry"],
                    "close"       : close_price,
                    "size"        : pos["size"],
                    "pnl_net"     : round(pnl_net, 4),
                    "reason"      : close_reason,
                    "equity_after": round(equity, 2),
                })
                pos      = None
                cooldown = CONFIG["cooldown_candles"]
            continue

        # ── cooldown ──────────────────────────────────────────────────────────
        if cooldown > 0:
            cooldown -= 1
            continue

        # ── segnali ───────────────────────────────────────────────────────────
        long_s  = prev["close"] > prev["dh"] and prev["adx"] > CONFIG["adx_threshold"]
        short_s = prev["close"] < prev["dl"] and prev["adx"] > CONFIG["adx_threshold"]

        if not long_s and not short_s:
            continue

        side    = "long" if long_s else "short"
        entry   = row["open"]   # fill al open della candela successiva
        atr     = prev["atr"]
        sl      = entry - atr * CONFIG["sl_atr"] if side == "long" else entry + atr * CONFIG["sl_atr"]
        tp      = entry + atr * CONFIG["tp_atr"] if side == "long" else entry - atr * CONFIG["tp_atr"]

        risk    = equity * CONFIG["risk_per_trade"]
        size    = min(risk / abs(entry - sl), equity * CONFIG["max_notional_pct"] / entry)
        size    = round(size, 4)

        if size * entry < CONFIG["min_trade_usdc"]:
            continue

        pos = {
            "side"       : side,
            "entry"      : entry,
            "sl"         : sl,
            "tp"         : tp,
            "size"       : size,
            "entry_time" : row["time"],
        }

    # ── metriche ─────────────────────────────────────────────────────────────
    if not trades:
        print("Nessun trade generato nel periodo.")
        return {}

    tdf        = pd.DataFrame(trades)
    wins       = tdf[tdf["pnl_net"] > 0]
    losses     = tdf[tdf["pnl_net"] <= 0]
    total_pnl  = tdf["pnl_net"].sum()
    win_rate   = len(wins) / len(tdf) * 100
    avg_win    = wins["pnl_net"].mean()   if len(wins)   else 0
    avg_loss   = losses["pnl_net"].mean() if len(losses) else 0
    profit_factor = abs(wins["pnl_net"].sum() / losses["pnl_net"].sum()) if len(losses) else float("inf")

    equity_curve  = [capital] + list(tdf["equity_after"])
    peak          = pd.Series(equity_curve).cummax()
    dd_series     = (pd.Series(equity_curve) - peak) / peak * 100
    max_dd        = dd_series.min()

    print("\n" + "="*55)
    print(f"  BACKTEST — HyperTrader v2 | Leva {leverage}x")
    print("="*55)
    print(f"  Capitale iniziale : ${capital:.2f}")
    print(f"  Capitale finale   : ${equity:.2f}")
    print(f"  PnL totale        : ${total_pnl:+.2f} ({total_pnl/capital*100:+.1f}%)")
    print(f"  N. trade          : {len(tdf)}")
    print(f"  Win rate          : {win_rate:.1f}%")
    print(f"  Avg win           : ${avg_win:+.2f}")
    print(f"  Avg loss          : ${avg_loss:+.2f}")
    print(f"  Profit factor     : {profit_factor:.2f}")
    print(f"  Max drawdown      : {max_dd:.1f}%")
    print("="*55)

    by_reason = tdf.groupby("reason")["pnl_net"].agg(["count", "sum", "mean"])
    print("\n  Uscite per motivo:")
    print(by_reason.to_string())
    print()

    return {
        "trades": tdf,
        "equity_final": equity,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
    }

def run_backtest_from_api():
    """Scarica dati storici dall'API e lancia il backtest."""
    log("Scarico dati storici per backtest (1H, ultimi 500 candles)...")

    # Chiede più candele possibili
    data = post_info({
        "type": "candleSnapshot",
        "req": {
            "coin"     : CONFIG["coin"],
            "interval" : CONFIG["interval"],
            "startTime": 0,
            "endTime"  : int(time.time() * 1000)
        }
    })

    candles = []
    for c in data:
        if isinstance(c, dict):
            candles.append([c["t"], c["o"], c["h"], c["l"], c["c"], c.get("v", 0)])
        else:
            candles.append(c)

    log(f"Candele caricate: {len(candles)}")
    backtest(candles)

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

        if res and "response" in str(res):
            log("TEST OK — trade aperto correttamente")
        else:
            log(f"TEST FALLITO — risposta inattesa: {res}")
    except Exception as e:
        log(f"TEST ERRORE: {e}")

# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 55)
    log("HyperTrader v2.0 — TESTNET")
    log(f"Coin: {CONFIG['coin']} | Interval: {CONFIG['interval']} | Leva: {CONFIG['leverage']}x")
    log(f"ADX soglia: {CONFIG['adx_threshold']} | Risk/trade: {CONFIG['risk_per_trade']*100:.0f}%")
    log(f"Cooldown: {CONFIG['cooldown_candles']} candele | Min trade: ${CONFIG['min_trade_usdc']}")
    log("=" * 55)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "test":
            direction = sys.argv[2] if len(sys.argv) > 2 else "long"
            test_trade(direction)
        elif cmd == "backtest":
            run_backtest_from_api()
        else:
            log(f"Comando sconosciuto: {cmd}. Usa: test | backtest")
    else:
        schedule.every().hour.at(":01").do(run)
        schedule.every().day.at("00:01").do(reset_day)
        run()
        log("Scheduler attivo — ciclo automatico ogni ora")
        while True:
            schedule.run_pending()
            time.sleep(30)
