"""
S&P 500 evreninde tarayıp watchlist'te olmayan,
Stoch RSI %K veya %D <= 20 olan yeni hisse adaylarını bulur.
"""

import argparse
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

from stoch_rsi_bot import (
    INTERVAL,
    K_PERIOD,
    OVERSOLD_THRESHOLD,
    PERIOD,
    PREPOST,
    RSI_PERIOD,
    STOCH_PERIOD,
    D_PERIOD,
    WATCHLIST,
    calculate_stoch_rsi,
)

BATCH_SIZE = 40
MIN_BARS = RSI_PERIOD + STOCH_PERIOD + 5
MIN_PRICE = 5.0


def fetch_sp500_tickers():
    """Wikipedia'dan güncel S&P 500 sembol listesini çeker."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; trade-bot/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def build_universe(exclude_watchlist=True):
    """Tarama evrenini oluşturur; watchlist'teki hisseler hariç tutulur."""
    tickers = fetch_sp500_tickers()
    if exclude_watchlist:
        skip = {s for s in WATCHLIST if not s.endswith("-USD")}
        tickers = [t for t in tickers if t not in skip]
    return tickers


def _extract_ticker_df(raw, symbol, batch_len):
    """yf.download çıktısından tek sembol DataFrame'i ayıklar."""
    if batch_len == 1:
        return raw.dropna(how="all").copy()
    if symbol not in raw.columns.get_level_values(0):
        return pd.DataFrame()
    df = raw[symbol].dropna(how="all").copy()
    return df


def _analyze_symbol(df, symbol, threshold):
    """Tek sembol için Stoch RSI hesaplar; aşırı satımdaysa sonuç döner."""
    if df.empty or len(df) < MIN_BARS:
        return None

    df = calculate_stoch_rsi(
        df,
        rsi_period=RSI_PERIOD,
        stoch_period=STOCH_PERIOD,
        k_period=K_PERIOD,
        d_period=D_PERIOD,
    )

    row = df.iloc[-1]
    k, d = row["K"], row["D"]
    price = row["Close"]

    if pd.isna(k) or pd.isna(d) or price < MIN_PRICE:
        return None

    k_below = k <= threshold
    d_below = d <= threshold
    if not (k_below or d_below):
        return None

    return {
        "symbol": symbol,
        "k": k,
        "d": d,
        "price": price,
        "candle_time": df.index[-1],
        "k_below": k_below,
        "d_below": d_below,
    }


def discover_oversold(symbols, threshold=OVERSOLD_THRESHOLD, batch_size=BATCH_SIZE):
    """Sembol listesini batch halinde tarar, aşırı satım adaylarını döner."""
    results = []
    total = len(symbols)

    for i in range(0, total, batch_size):
        batch = symbols[i : i + batch_size]
        batch_num = i // batch_size + 1
        batch_total = (total + batch_size - 1) // batch_size
        print(f"Batch {batch_num}/{batch_total} taranıyor ({len(batch)} sembol)...")

        try:
            raw = yf.download(
                batch,
                period=PERIOD,
                interval=INTERVAL,
                prepost=PREPOST,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"  Batch indirme hatası: {e}")
            continue

        if raw.empty:
            continue

        for symbol in batch:
            try:
                df = _extract_ticker_df(raw, symbol, len(batch))
                hit = _analyze_symbol(df, symbol, threshold)
                if hit:
                    results.append(hit)
            except Exception as e:
                print(f"  Hata [{symbol}]: {e}")

    results.sort(key=lambda x: x["k"])
    return results


def print_results(results, scanned_count, threshold=OVERSOLD_THRESHOLD):
    print(f"\n--- Yeni Hisse Adayları ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    print(f"Taranan: {scanned_count} sembol | Eşik: K veya D <= {threshold} | {INTERVAL}")
    print(f"Watchlist hariç (mevcut: {', '.join(WATCHLIST)})\n")

    if not results:
        print("Yeni aşırı satım adayı bulunamadı.")
        return

    print(f"{'Sembol':<8} {'K':>7} {'D':>7} {'Fiyat':>12} {'Durum':<16} {'Mum'}")
    print("-" * 68)

    for r in results:
        flags = []
        if r["k_below"]:
            flags.append("K<20")
        if r["d_below"]:
            flags.append("D<20")
        status = ", ".join(flags)
        t = r["candle_time"].strftime("%Y-%m-%d %H:%M")
        print(f"{r['symbol']:<8} {r['k']:7.2f} {r['d']:7.2f} ${r['price']:>10,.2f} {status:<16} {t}")

    symbols = [r["symbol"] for r in results]
    print(f"\nToplam: {len(results)} yeni aday")
    print(f"\nWatchlist'e eklenebilir:\n{symbols}")
    return symbols


def notify_telegram(results, scanned_count, threshold=OVERSOLD_THRESHOLD):
    """Bulunan adayları Telegram'a özet olarak gönderir."""
    from stoch_rsi_bot import INTERVAL, send_telegram_message

    if not results:
        return

    lines = [
        f"🔍 <b>Yeni Hisse Adayları</b> ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        f"Taranan: {scanned_count} | Eşik: K/D ≤ {threshold} | {INTERVAL}",
        "",
    ]
    for r in results[:20]:
        flags = []
        if r["k_below"]:
            flags.append("K<20")
        if r["d_below"]:
            flags.append("D<20")
        lines.append(
            f"• <b>{r['symbol']}</b> K={r['k']:.1f} D={r['d']:.1f} ${r['price']:,.2f} ({', '.join(flags)})"
        )
    if len(results) > 20:
        lines.append(f"\n... ve {len(results) - 20} aday daha")

    send_telegram_message("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="S&P 500'de yeni aşırı satım hisseleri bul")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Taranacak maksimum sembol sayısı (test için, örn: 80)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=OVERSOLD_THRESHOLD,
        help=f"Stoch RSI eşiği (varsayılan: {OVERSOLD_THRESHOLD})",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Sonuçları Telegram'a gönder (CI için)",
    )
    args = parser.parse_args()

    print("S&P 500 sembol listesi alınıyor...")
    universe = build_universe(exclude_watchlist=True)

    if args.limit:
        universe = universe[: args.limit]

    print(f"Tarama başlıyor: {len(universe)} sembol")
    results = discover_oversold(universe, threshold=args.threshold)
    print_results(results, len(universe), threshold=args.threshold)

    if args.notify and results:
        notify_telegram(results, len(universe), threshold=args.threshold)


if __name__ == "__main__":
    main()
