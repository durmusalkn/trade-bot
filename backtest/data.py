"""
Backtest verisi: hisse evreni, günlük OHLCV ve bilanço tarihleri.

Veriler backtest/cache/ altına parquet olarak yazılır; aynı gün tekrar
çalıştırıldığında Yahoo'ya yeniden gidilmez.
"""
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
NDX_WIKI = "https://en.wikipedia.org/wiki/Nasdaq-100"
DATA_START = "2008-06-01"
BENCHMARK = "SPY"


def _yahoo_symbol(symbol):
    return symbol.strip().replace(".", "-")


def load_universe():
    """S&P 500 + Nasdaq-100 bugünkü üyeleri -> {sembol: sektör}."""
    path = os.path.join(CACHE_DIR, "universe.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return dict(zip(df["symbol"], df["sector"]))

    sp = pd.read_csv(io.StringIO(requests.get(SP500_CSV, timeout=30).text))
    sectors = {_yahoo_symbol(s): sec for s, sec in zip(sp["Symbol"], sp["GICS Sector"])}

    try:
        html = requests.get(NDX_WIKI, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text
        for table in pd.read_html(io.StringIO(html)):
            cols = {str(c).lower(): c for c in table.columns}
            ticker_col = cols.get("ticker") or cols.get("symbol")
            if ticker_col is None or len(table) < 90:
                continue
            sector_col = next((c for k, c in cols.items() if "sector" in k), None)
            for _, row in table.iterrows():
                sym = _yahoo_symbol(str(row[ticker_col]))
                sectors.setdefault(sym, str(row[sector_col]) if sector_col else "Unknown")
            break
    except Exception as exc:  # Nasdaq-100 listesi opsiyonel
        print(f"Nasdaq-100 listesi alınamadı, sadece S&P 500 kullanılıyor: {exc}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.DataFrame({"symbol": list(sectors), "sector": list(sectors.values())}).to_csv(path, index=False)
    return sectors


def load_prices(symbols, batch_size=100, cache_name="prices.parquet", benchmarks=(BENCHMARK,)):
    """Günlük OHLCV -> {'Open': df, 'High': df, ...}; satır=tarih, sütun=sembol.

    auto_adjust=False: fiyatlar split'e göre düzeltilmiş, temettüye göre düzeltilmemiş
    (botun gördüğü fiyatlarla aynı).
    """
    path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(path):
        wide = pd.read_parquet(path)
    else:
        frames = []
        symbols = sorted(set(symbols) | set(benchmarks))
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            for attempt in range(3):
                try:
                    df = yf.download(batch, start=DATA_START, auto_adjust=False, progress=False,
                                     threads=True, group_by="column")
                    break
                except Exception as exc:
                    print(f"İndirme hatası ({attempt + 1}/3): {exc}")
                    time.sleep(5)
            else:
                continue
            frames.append(df[["Open", "High", "Low", "Close", "Volume"]])
            print(f"Fiyat indirildi: {min(i + batch_size, len(symbols))}/{len(symbols)}")
        wide = pd.concat(frames, axis=1)
        wide.index = pd.to_datetime(wide.index).tz_localize(None)
        os.makedirs(CACHE_DIR, exist_ok=True)
        wide.astype("float64").to_parquet(path)

    fields = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        fields[field] = wide[field].sort_index()
    # Hiç verisi olmayan sembolleri at
    valid = fields["Close"].columns[fields["Close"].notna().sum() > 260]
    return {k: v[valid] for k, v in fields.items()}


def _fetch_earnings(symbol):
    for attempt in range(3):
        try:
            df = yf.Ticker(symbol).get_earnings_dates(limit=100)
            if df is None or df.empty:
                return symbol, []
            idx = pd.DatetimeIndex(pd.to_datetime(df.index))
            if idx.tz is not None:
                idx = idx.tz_convert("America/New_York").tz_localize(None)
            return symbol, list(idx)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return symbol, None


def load_earnings(symbols, workers=8):
    """Bilanço açıklama zamanları -> {sembol: [Timestamp (New York saati)]}.

    Alınamayan semboller sözlükte yer almaz (kapsam raporda belirtilir).
    """
    path = os.path.join(CACHE_DIR, "earnings.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["time"])
        out = {s: list(g["time"]) for s, g in df.groupby("symbol")}
        for s in df.loc[df["time"].isna(), "symbol"]:
            out[s] = []
        return out

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (symbol, times) in enumerate(pool.map(_fetch_earnings, symbols), 1):
            if times is None:
                continue
            rows.extend({"symbol": symbol, "time": t} for t in times)
            if not times:
                rows.append({"symbol": symbol, "time": pd.NaT})
            if n % 50 == 0:
                print(f"Bilanço tarihleri: {n}/{len(symbols)}")
    df = pd.DataFrame(rows, columns=["symbol", "time"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return load_earnings(symbols) if not df.empty else {}


def earnings_reaction_days(earnings, trading_days):
    """Bilançonun fiyata yansıdığı ilk işlem günü.

    09:30'dan (New York) önce açıklanırsa aynı gün, sonra açıklanırsa bir sonraki
    işlem günü. Saati bilinmeyen (00:00) kayıtlarda iki gün de sayılır.
    """
    days = pd.DatetimeIndex(trading_days)
    out = {}
    for symbol, times in earnings.items():
        reaction = set()
        for t in times:
            if pd.isna(t):
                continue
            day = t.normalize()
            minutes = t.hour * 60 + t.minute
            if minutes == 0:
                candidates = [day, day + pd.Timedelta(days=1)]
            elif minutes >= 9 * 60 + 30:
                candidates = [day + pd.Timedelta(days=1)]
            else:
                candidates = [day]
            for c in candidates:
                pos = days.searchsorted(c)
                if pos < len(days):
                    reaction.add(days[pos])
        out[symbol] = reaction
    return out
