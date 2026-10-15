"""
Momentum Swing kurulumları için indikatörler ve sinyaller.

Tüm hesaplar sütun=sembol, satır=tarih tablolarında vektörel yapılır.
Sinyal t günü kapanışında oluşur, giriş t+1 açılışında yapılır.
"""
import numpy as np
import pandas as pd

# Evren
MIN_PRICE = 10.0
MIN_DOLLAR_VOLUME = 20_000_000
RS_PERCENTILE = 0.80

# Kurulum A: kırılım
A_BREAKOUT_LOOKBACK = 20
A_NEAR_HIGH = 0.85           # 52 hafta zirvesinin en fazla %15 altı
A_VOLUME_MULT = 1.5
A_STOP_ATR = 1.5
A_TARGET_R = 2.0
A_CHECK_DAY = 5
A_CHECK_R = 0.5
A_MAX_DAYS = 15

# Kurulum B: trend içi geri çekilme
B_RSI2_MAX = 10
B_IBS_MAX = 0.2
B_STOP_ATR = 2.5
B_EXIT_RSI2 = 70
B_MAX_DAYS = 7

# Kurulum C: bilanço sonrası boşluk
C_GAP_MIN = 0.05
C_VOLUME_MULT = 3.0
C_STOP_ATR = 1.5
C_MIN_DAYS = 3
C_MAX_DAYS = 10

EARNINGS_BLACKOUT_DAYS = 5


def wilder(series_df, period):
    return series_df.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(close, period):
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), period)
    loss = wilder(-delta.clip(upper=0), period)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def compute_indicators(px):
    o, h, l, c, v = px["Open"], px["High"], px["Low"], px["Close"], px["Volume"]
    prev_c = c.shift(1)
    tr = np.fmax(np.fmax(h - l, (h - prev_c).abs()), (l - prev_c).abs())

    ind = {
        "sma5": c.rolling(5).mean(),
        "sma50": c.rolling(50).mean(),
        "sma200": c.rolling(200).mean(),
        "ema10": c.ewm(span=10, adjust=False, min_periods=10).mean(),
        "atr14": wilder(tr, 14),
        "rsi2": rsi(c, 2),
        "ibs": ((c - l) / (h - l)).where(h > l, 0.5),
        "ret63": c / c.shift(63) - 1,
        "ret126": c / c.shift(126) - 1,
        "high252": h.rolling(252, min_periods=200).max(),
        "hh20_prev": h.rolling(A_BREAKOUT_LOOKBACK).max().shift(1),
        "vol50_prev": v.rolling(50).mean().shift(1),
        "dollar_vol20": (c * v).rolling(20).mean(),
    }
    liquid = (c >= MIN_PRICE) & (ind["dollar_vol20"] >= MIN_DOLLAR_VOLUME)
    # Göreli güç: sadece likit evren içinde yüzdelik sıralama
    rank63 = ind["ret63"].where(liquid).rank(axis=1, pct=True)
    rank126 = ind["ret126"].where(liquid).rank(axis=1, pct=True)
    ind["liquid"] = liquid
    ind["strong_rs"] = (rank63 >= RS_PERCENTILE) & (rank126 >= RS_PERCENTILE)
    return ind


def regime_multiplier(spy_close):
    """1.0 = tam risk, 0.5 = yarım risk, 0 = yeni giriş yok."""
    sma50 = spy_close.rolling(50).mean()
    sma200 = spy_close.rolling(200).mean()
    mult = pd.Series(0.0, index=spy_close.index)
    mult[spy_close > sma200] = 0.5
    mult[(spy_close > sma50) & (sma50 > sma200)] = 1.0
    return mult


def earnings_masks(reaction_days, index, columns):
    """(bilanço günü maskesi, önümüzdeki N gün içinde bilanço maskesi)."""
    on_day = pd.DataFrame(False, index=index, columns=columns)
    for symbol, days in reaction_days.items():
        if symbol in on_day.columns and days:
            on_day.loc[on_day.index.intersection(list(days)), symbol] = True
    # t kapanışında sinyal -> t+1 giriş. t+1..t+N arasında bilanço varsa girme.
    upcoming = pd.DataFrame(False, index=index, columns=columns)
    for k in range(1, EARNINGS_BLACKOUT_DAYS + 1):
        upcoming |= on_day.shift(-k, fill_value=False)
    return on_day, upcoming


def compute_signals(px, ind, reaction_days):
    o, h, l, c, v = px["Open"], px["High"], px["Low"], px["Close"], px["Volume"]
    on_earnings, upcoming = earnings_masks(reaction_days, c.index, c.columns)
    uptrend = (c > ind["sma50"]) & (ind["sma50"] > ind["sma200"])

    a = (
        ind["liquid"] & ind["strong_rs"] & uptrend
        & (c >= A_NEAR_HIGH * ind["high252"])
        & (c > ind["hh20_prev"])
        & (v > A_VOLUME_MULT * ind["vol50_prev"])
        & ~upcoming
    )
    b = (
        ind["liquid"] & ind["strong_rs"] & (c > ind["sma200"])
        & (ind["sma50"] > ind["sma50"].shift(5))
        & (ind["rsi2"] < B_RSI2_MAX)
        & (ind["ibs"] < B_IBS_MAX)
        & ~upcoming
    )
    gap = o / c.shift(1) - 1
    cond_c = (
        ind["liquid"]
        & (gap > C_GAP_MIN)
        & (v > C_VOLUME_MULT * ind["vol50_prev"])
        & (c >= (h + l) / 2)
    )
    return {
        "A": a.fillna(False),
        "B": b.fillna(False),
        "C": (cond_c & on_earnings).fillna(False),
    }
