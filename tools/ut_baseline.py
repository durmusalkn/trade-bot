#!/usr/bin/env python3
"""
Golden-output baseline aracı (ADIM 0).

Amaç: sinyal matematiğine (UT_Stop / UT_Pos / UT_Buy / UT_Sell) kazara dokunulmasını
yakalamak. Sentetik assert testleri bunu yakalayamaz; tek güvenilir yol gerçek veriyle
üretilmiş bir referans çıktıyı dondurup refactor sonrası bire bir karşılaştırmaktır.

Akış:

    1) python tools/ut_baseline.py freeze  --interval 1h        # ham df'i diske dondur
    2) python tools/ut_baseline.py capture --interval 1h -o baseline/1h-before.json
    ... refactor ...
    3) python tools/ut_baseline.py capture --interval 1h -o baseline/1h-after.json
    4) python tools/ut_baseline.py compare baseline/1h-before.json baseline/1h-after.json

Karşılaştırma alanları üç kovaya ayrılır:

    key   : candle_key  -> HİÇBİR adımda değişmemeli (eski state anahtarları bozulur)
    math  : close/ha_close/atr/ut_stop/ut_pos/ut_buy/ut_sell -> ADIM 5 sonrası BOŞ olmalı
    time  : close_time_utc/age_sec/is_fresh/candle_label -> yalnızca ADIM 6'da değişmeli

Ağ erişimi olmayan ortamlarda `synth` alt komutu deterministik sahte veri üretir; bu
yalnızca aracın kendi doğruluğunu sınamak içindir, gerçek baseline yerine geçmez.

Donmuş ham veri baseline/raw/ altında pickle olarak tutulur ve .gitignore'dadır;
üretilen JSON baseline'lar repoya commit edilir.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hourly_ut_bot as bot  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_RAW_DIR = REPO_ROOT / "baseline" / "raw"
DEFAULT_BARS = 50

KEY_FIELDS = ("candle_key",)
MATH_FIELDS = ("close", "ha_close", "atr", "ut_stop", "ut_pos", "ut_buy", "ut_sell")
TIME_FIELDS = ("close_time_utc", "age_sec", "is_fresh", "candle_label")
ALL_FIELDS = KEY_FIELDS + MATH_FIELDS + TIME_FIELDS

# ADIM 5'ten (config refactor) sonra bu fonksiyon config nesnesi kuracak şekilde
# güncellenecek. Şu an bot modülü global mutasyonla yapılandırılıyor.
DAILY_DEFAULTS = dict(period="3y", max_alert_age=pd.Timedelta(hours=16), chart_bars=120)
HOURLY_DEFAULTS = dict(period="730d", max_alert_age=pd.Timedelta(minutes=50), chart_bars=120)


def configure_bot(interval):
    """Bot modülünü verilen periyoda göre yapılandırır (daily_ut_bot.py ile aynı mantık)."""
    defaults = DAILY_DEFAULTS if interval.endswith("d") else HOURLY_DEFAULTS
    bot.INTERVAL = interval
    bot.PERIOD = defaults["period"]
    bot.PREPOST = True
    bot.IS_DAILY = interval.endswith("d")
    bot.BAR_DURATION = bot.get_bar_duration()
    bot.MAX_ALERT_AGE = defaults["max_alert_age"]
    bot.CHART_BARS = defaults["chart_bars"]
    return defaults


def raw_path(raw_dir, symbol, interval):
    return Path(raw_dir) / f"{symbol}_{interval}.pkl"


def load_raw(raw_dir, symbol, interval):
    path = raw_path(raw_dir, symbol, interval)
    if not path.exists():
        raise FileNotFoundError(
            f"Donmuş veri yok: {path}\n"
            f"Önce şunu çalıştır: python tools/ut_baseline.py freeze --interval {interval}"
        )
    return pd.read_pickle(path)


# ==========================================
# FREEZE — ham veriyi diske dondur
# ==========================================
def cmd_freeze(args):
    import yfinance as yf

    defaults = configure_bot(args.interval)
    period = args.period or defaults["period"]
    out_dir = Path(args.raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for symbol in args.symbols:
        try:
            df = yf.Ticker(symbol).history(period=period, interval=args.interval, prepost=True)
        except Exception as e:  # ağ/proxy hataları da buraya düşer
            failures.append((symbol, repr(e)))
            print(f"✗ {symbol}: indirilemedi ({e})")
            continue
        if df.empty:
            failures.append((symbol, "boş df"))
            print(f"✗ {symbol}: boş df döndü")
            continue
        path = raw_path(out_dir, symbol, args.interval)
        df.to_pickle(path)
        print(f"✓ {symbol}: {len(df)} bar donduruldu -> {path}")

    if failures:
        print(f"\nUYARI: {len(failures)} sembol indirilemedi; baseline eksik olur.")
        return 1
    return 0


# ==========================================
# SYNTH — ağsız ortamda aracı sınamak için deterministik sahte veri
# ==========================================
ET = "America/New_York"
# Ampirik ızgara: her işlem günü istisnasız 17 bar.
GRID = [
    (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
    (9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30),
    (16, 0), (17, 0), (18, 0), (19, 0),
]


def synth_index(days, end_date):
    stamps = []
    day = pd.Timestamp(end_date)
    collected = 0
    while collected < days:
        if day.weekday() < 5:
            for hh, mm in GRID:
                stamps.append(pd.Timestamp(
                    year=day.year, month=day.month, day=day.day, hour=hh, minute=mm, tz=ET
                ))
            collected += 1
        day -= pd.Timedelta(days=1)
    return pd.DatetimeIndex(sorted(stamps))


def cmd_synth(args):
    configure_bot(args.interval)
    out_dir = Path(args.raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol in args.symbols:
        # hash() süreç başına rastgeleleştirilir; crc32 deterministik tohum verir.
        rng = np.random.default_rng(zlib.crc32(symbol.encode()))
        if args.interval.endswith("d"):
            idx = pd.DatetimeIndex([
                ts for ts in pd.date_range(end=args.end, periods=int(args.days * 1.6), freq="D", tz=ET)
                if ts.weekday() < 5
            ][-args.days:])
        else:
            idx = synth_index(args.days, args.end)
        n = len(idx)
        steps = rng.normal(0.0, 1.2, n).cumsum()
        close = 100.0 + steps
        open_ = close - rng.normal(0.0, 0.4, n)
        high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.5, n))
        low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.5, n))
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close,
             "Volume": rng.integers(1_000, 100_000, n)},
            index=idx,
        )
        path = raw_path(out_dir, symbol, args.interval)
        df.to_pickle(path)
        print(f"✓ {symbol}: {n} sentetik bar -> {path}")
    return 0


# ==========================================
# CAPTURE — donmuş veriden baseline üret
# ==========================================
def reference_now(frames):
    """Deterministik referans 'şimdi': en son barın kapanışı + 8 dk (cron :08 taklidi)."""
    last = max(bot.to_utc(df.index[-1]) for df in frames.values())
    return last + bot.BAR_DURATION + pd.Timedelta(minutes=8)


def bar_record(candle_time, row, now_utc):
    close_time = bot.to_utc(bot.candle_close_time(candle_time))
    return {
        "bar_time_utc": bot.to_utc(candle_time).isoformat(),
        "candle_key": bot.candle_key(candle_time),
        "close": float(row["Close"]),
        "ha_close": float(row["HA_Close"]),
        "atr": None if pd.isna(row["ATR"]) else float(row["ATR"]),
        "ut_stop": float(row["UT_Stop"]),
        "ut_pos": float(row["UT_Pos"]),
        "ut_buy": bool(row["UT_Buy"]),
        "ut_sell": bool(row["UT_Sell"]),
        "close_time_utc": close_time.isoformat(),
        "age_sec": (now_utc - close_time).total_seconds(),
        "is_fresh": bool(bot.is_fresh_signal(candle_time, now_utc=now_utc)),
        "candle_label": bot.format_candle_time(candle_time),
    }


def cmd_capture(args):
    configure_bot(args.interval)

    frames = {s: load_raw(args.raw_dir, s, args.interval) for s in args.symbols}
    now_utc = pd.Timestamp(args.now).tz_convert("UTC") if args.now else reference_now(frames)

    out = {
        "schema_version": SCHEMA_VERSION,
        "interval": args.interval,
        "reference_now_utc": now_utc.isoformat(),
        "bars_per_symbol": args.bars,
        "ut_params": {
            "key_value": bot.UT_KEY_VALUE,
            "atr_period": bot.UT_ATR_PERIOD,
            "use_ha": bot.UT_USE_HA,
            "prepost": bot.PREPOST,
            "max_alert_age_sec": bot.MAX_ALERT_AGE.total_seconds(),
        },
        "symbols": {},
    }

    for symbol, df in frames.items():
        calc = bot.calculate_ut_bot(
            df, key_value=bot.UT_KEY_VALUE, atr_period=bot.UT_ATR_PERIOD, use_ha=bot.UT_USE_HA
        )
        closed = bot.get_closed_bars(calc)
        tail = closed.tail(args.bars)
        out["symbols"][symbol] = [bar_record(ts, row, now_utc) for ts, row in tail.iterrows()]
        print(f"• {symbol:7s} {len(tail)} kapanmış bar (toplam {len(calc)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print(f"✓ Baseline yazıldı: {out_path}")
    return 0


# ==========================================
# COMPARE — iki baseline'ı karşılaştır
# ==========================================
def bucket_of(field):
    if field in KEY_FIELDS:
        return "key"
    if field in MATH_FIELDS:
        return "math"
    return "time"


def cmd_compare(args):
    with open(args.before, encoding="utf-8") as f:
        before = json.load(f)
    with open(args.after, encoding="utf-8") as f:
        after = json.load(f)

    diffs = {"key": [], "math": [], "time": [], "structure": []}

    if before.get("reference_now_utc") != after.get("reference_now_utc"):
        diffs["structure"].append(
            f"reference_now_utc farklı: {before.get('reference_now_utc')} != "
            f"{after.get('reference_now_utc')} (yaş alanları anlamsız karşılaştırılır)"
        )
    if before.get("ut_params") != after.get("ut_params"):
        diffs["structure"].append(f"ut_params farklı: {before.get('ut_params')} != {after.get('ut_params')}")

    symbols = sorted(set(before["symbols"]) | set(after["symbols"]))
    for symbol in symbols:
        b_bars = {r["bar_time_utc"]: r for r in before["symbols"].get(symbol, [])}
        a_bars = {r["bar_time_utc"]: r for r in after["symbols"].get(symbol, [])}
        only_b = sorted(set(b_bars) - set(a_bars))
        only_a = sorted(set(a_bars) - set(b_bars))
        for ts in only_b:
            diffs["structure"].append(f"{symbol} {ts}: yalnızca 'before' içinde")
        for ts in only_a:
            diffs["structure"].append(f"{symbol} {ts}: yalnızca 'after' içinde")

        for ts in sorted(set(b_bars) & set(a_bars)):
            for field in ALL_FIELDS:
                bv, av = b_bars[ts].get(field), a_bars[ts].get(field)
                if bv != av:
                    diffs[bucket_of(field)].append(f"{symbol} {ts} {field}: {bv!r} -> {av!r}")

    for bucket in ("structure", "key", "math", "time"):
        rows = diffs[bucket]
        if not rows:
            print(f"✓ {bucket}: fark yok")
            continue
        print(f"✗ {bucket}: {len(rows)} fark")
        for row in rows[: args.max_rows]:
            print(f"    {row}")
        if len(rows) > args.max_rows:
            print(f"    ... ({len(rows) - args.max_rows} satır daha)")

    fatal = diffs["structure"] + diffs["key"] + diffs["math"]
    if fatal:
        print("\nSONUÇ: DURDUR — sinyal matematiği veya anahtar alanlar değişmiş. COMMIT ETME.")
        return 1
    if diffs["time"]:
        if args.allow_time_diff:
            print("\nSONUÇ: yalnızca zaman alanları değişti (beklenen: ADIM 6).")
            return 0
        print("\nSONUÇ: zaman alanları değişti; beklenen buysa --allow-time-diff ile çalıştır.")
        return 2
    print("\nSONUÇ: baseline birebir aynı.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="UT Bot golden-output baseline aracı")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--symbols", nargs="+", default=bot.WATCHLIST)
        p.add_argument("--interval", default="1h")
        p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))

    p_freeze = sub.add_parser("freeze", help="yfinance'ten ham df indir ve diske dondur")
    add_common(p_freeze)
    p_freeze.add_argument("--period", default=None)
    p_freeze.set_defaults(func=cmd_freeze)

    p_synth = sub.add_parser("synth", help="ağsız ortamda aracı sınamak için sahte veri üret")
    add_common(p_synth)
    p_synth.add_argument("--days", type=int, default=120)
    p_synth.add_argument("--end", default="2025-06-30")
    p_synth.set_defaults(func=cmd_synth)

    p_cap = sub.add_parser("capture", help="donmuş veriden baseline JSON üret")
    add_common(p_cap)
    p_cap.add_argument("--bars", type=int, default=DEFAULT_BARS)
    p_cap.add_argument("--now", default=None, help="referans 'şimdi' (ISO, tz'li)")
    p_cap.add_argument("-o", "--out", required=True)
    p_cap.set_defaults(func=cmd_capture)

    p_cmp = sub.add_parser("compare", help="iki baseline JSON'unu karşılaştır")
    p_cmp.add_argument("before")
    p_cmp.add_argument("after")
    p_cmp.add_argument("--allow-time-diff", action="store_true")
    p_cmp.add_argument("--max-rows", type=int, default=20)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
