"""
Stoch RSI %K veya %D değeri 20'nin altında olan hisseleri tarar.
Mevcut watchlist üzerinde çalışır; tek seferlik veya periyodik tarama yapılabilir.
"""

from datetime import datetime

import pandas as pd
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


def scan_oversold(symbols=None, threshold=OVERSOLD_THRESHOLD):
  """
  Watchlist'teki sembolleri tarar; K veya D <= threshold olanları döner.
  Sonuçlar K değerine göre artan sırada (en aşırı satılmış üstte).
  """
  symbols = symbols or WATCHLIST
  results = []

  for symbol in symbols:
    try:
      df = yf.Ticker(symbol).history(period=PERIOD, interval=INTERVAL, prepost=PREPOST)

      if df.empty or len(df) < (RSI_PERIOD + STOCH_PERIOD + 5):
        continue

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
      candle_time = df.index[-1]

      if pd.isna(k) or pd.isna(d):
        continue

      k_below = k <= threshold
      d_below = d <= threshold

      if k_below or d_below:
        results.append({
          "symbol": symbol,
          "k": k,
          "d": d,
          "price": price,
          "candle_time": candle_time,
          "k_below": k_below,
          "d_below": d_below,
        })

    except Exception as e:
      print(f"Hata [{symbol}]: {e}")

  results.sort(key=lambda x: x["k"])
  return results


def print_results(results, threshold=OVERSOLD_THRESHOLD):
  print(f"\n--- Aşırı Satım Taraması ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
  print(f"Eşik: Stoch RSI K veya D <= {threshold} | Zaman dilimi: {INTERVAL}\n")

  if not results:
    print("20 altında hisse bulunamadı.")
    return

  print(f"{'Sembol':<10} {'K':>7} {'D':>7} {'Fiyat':>12} {'Durum':<20} {'Mum'}")
  print("-" * 72)

  for r in results:
    flags = []
    if r["k_below"]:
      flags.append("K<20")
    if r["d_below"]:
      flags.append("D<20")
    status = ", ".join(flags)
    t = r["candle_time"].strftime("%Y-%m-%d %H:%M")
    print(f"{r['symbol']:<10} {r['k']:7.2f} {r['d']:7.2f} ${r['price']:>10,.2f} {status:<20} {t}")

  print(f"\nToplam: {len(results)} sembol")


if __name__ == "__main__":
  oversold = scan_oversold()
  print_results(oversold)
