"""
Nasdaq teknoloji momentum kurulumları: günlük %5+ yükselişleri hedefler.

M1  Momentum patlaması: %5+ yükseliş günü, artan hacim, günün tepesine yakın kapanış.
M2  Boşluk ve devam (episodic pivot): %5+ boşlukla açılış, 3x hacim, gün boyu tutunma.
M3  Patlama sonrası bayrak: son 3-10 günde %5+ patlama, sığ geri çekilme, yeniden yukarı dönüş.

Sinyal t kapanışında, giriş t+1 açılışında. Stop sinyal mumunun dibi (M3'te geri çekilmenin dibi).
"""
import numpy as np
import pandas as pd

from .engine import Backtester

MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 20_000_000
BURST_MIN = 0.05
MAX_STOP_DISTANCE = 0.20

# Yüksek betalı, Nasdaq'ta işlem gören teknoloji/büyüme hisseleri (+ izleme listesindeki HIMS).
# DİKKAT: Liste bugünden seçildi; batan/listeden çıkan şirketler Yahoo'da olmadığı için yok.
TECH_UNIVERSE = [
    # Mega/large cap
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "NFLX", "AMD", "ADBE", "CSCO",
    "INTC", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "ADI", "MU", "MRVL", "NXPI", "MCHP", "ON", "MPWR",
    "ASML", "ARM", "SMCI", "INTU", "ADSK", "CDNS", "SNPS", "PANW", "FTNT", "CRWD", "ZS", "DDOG",
    "TEAM", "MDB", "WDAY", "PYPL", "ABNB", "DASH", "TTD", "ROKU", "ZM", "DOCU", "OKTA", "PLTR", "APP",
    "MELI", "SHOP", "PDD", "JD", "BIDU", "NTES", "BILI", "EA", "TTWO", "WDC", "STX", "NTAP", "LOGI",
    "TER", "ENTG", "QRVO", "SWKS", "GFS", "ZBRA", "CHKP", "CYBR", "MNDY", "WIX", "GEN", "EXPE",
    # Yüksek beta / orta-küçük ölçek
    "COIN", "HOOD", "MSTR", "MARA", "RIOT", "CLSK", "CIFR", "IREN", "WULF", "CORZ", "APLD", "BTDR",
    "SOFI", "AFRM", "UPST", "LCID", "RIVN", "IONQ", "RGTI", "QBTS", "QUBT", "SOUN", "NBIS", "TEM",
    "HIMS", "CRDO", "ALAB", "LSCC", "AMKR", "ACLS", "ONTO", "FORM", "ENPH", "SEDG", "FSLR", "RUN",
    "PLUG", "CELH", "DUOL", "ASTS", "RKLB", "LUNR", "GTLB", "NTNX", "FROG", "CFLT", "PTON", "LYFT",
    "TRIP", "APPN", "SMTC", "POWI", "SITM", "MTSI", "RMBS", "AEHR", "PI", "INOD", "SERV", "KTOS",
    "AVAV", "POET", "LITE", "AAOI", "VIAV", "MBLY", "NVTS", "INDI", "GRAB", "PCTY", "FIVN", "OPEN",
]


def compute_momentum(px, ind):
    o, h, l, c, v = px["Open"], px["High"], px["Low"], px["Close"], px["Volume"]
    ret1 = c / c.shift(1) - 1
    gap = o / c.shift(1) - 1
    rvol = v / ind["vol50_prev"]
    liquid = (c >= MIN_PRICE) & (ind["dollar_vol20"] >= MIN_DOLLAR_VOLUME)
    uptrend = c > ind["sma50"]

    m1 = (liquid & uptrend & (ret1 >= BURST_MIN) & (rvol >= 1.5) & (v > v.shift(1))
          & (ind["ibs"] >= 0.7) & (ret1.shift(1) < 0.03))
    m2 = liquid & (gap >= BURST_MIN) & (rvol >= 3.0) & (c > o) & (ind["ibs"] >= 0.5)

    # M3: patlama günü işaretle, 10 güne kadar ileri taşı
    burst = liquid & (ret1 >= BURST_MIN) & (rvol >= 2.0)
    burst_low = l.where(burst).ffill(limit=10)
    day_no = pd.DataFrame(np.arange(len(c))[:, None].repeat(c.shape[1], 1), index=c.index, columns=c.columns)
    burst_day = day_no.where(burst).ffill(limit=10)
    age = day_no - burst_day
    pullback_low = l.rolling(3).min()
    recent_high = h.rolling(10).max()
    m3 = (liquid & uptrend & (age >= 3) & (age <= 10)
          & (pullback_low > burst_low)
          & (c.shift(1) < c.shift(2))          # dün geri çekilme günü
          & (c > h.shift(1))                   # bugün dünün tepesinin üstünde kapanış
          & (c >= 0.85 * recent_high))

    signals = {"M1": m1.fillna(False), "M2": m2.fillna(False), "M3": m3.fillna(False)}
    stops = {"M1": l, "M2": l, "M3": pullback_low}
    return signals, stops, rvol


class MomentumBacktester(Backtester):
    """Stop: sinyal mumunun dibi. Çıkış: 'fast' = önceki günün dibinin altında kapanış (max 10 gün),
    'trend' = 10 günlük EMA altında kapanış (max 15 gün). İkisinde de en az 2 gün tutulur
    (stop hariç), kısmi satış yok (fazladan komisyon olmasın)."""

    def __init__(self, px, ind, signals, stops, rank, regime, sectors, cfg, exit_mode="trend"):
        super().__init__(px, ind, signals, regime, sectors, cfg)
        self.stop_level = {k: stops[k].to_numpy() for k in cfg.setups}
        self.rank_key = rank.to_numpy()
        self.exit_mode = exit_mode

    def _initial_stop(self, setup, si, j, entry):
        stop = self.stop_level[setup][si, j]
        if np.isnan(stop) or stop >= entry or (entry - stop) / entry > MAX_STOP_DISTANCE:
            return None
        return stop

    def _target(self, setup, entry, rps):
        return None

    def _exit_signal(self, pos, i, j, c):
        if pos.days_held < 2:
            return None
        if self.exit_mode == "fast":
            if c < self.L[i - 1, j]:
                return "close_below_prev_low"
            if pos.days_held >= 10:
                return "time_stop"
        else:
            if c < self.ema10[i, j]:
                return "close_below_ema10"
            if pos.days_held >= 15:
                return "time_stop"
        return None


def event_study(px, ind, horizons=(1, 3, 5, 10, 15)):
    """%5+ yükseliş günlerinden sonra ertesi açılıştan N gün sonraki kapanışa getiri."""
    o, c, v = px["Open"], px["Close"], px["Volume"]
    ret1 = c / c.shift(1) - 1
    rvol = v / ind["vol50_prev"]
    liquid = (c >= MIN_PRICE) & (ind["dollar_vol20"] >= MIN_DOLLAR_VOLUME)
    entry = o.shift(-1)
    groups = {
        "Tüm %5+ günler": liquid & (ret1 >= 0.05),
        "%5-10 günler": liquid & (ret1 >= 0.05) & (ret1 < 0.10),
        "%10+ günler": liquid & (ret1 >= 0.10),
        "%5+ ve hacim ≥2x": liquid & (ret1 >= 0.05) & (rvol >= 2),
        "%5+ ve hacim <2x": liquid & (ret1 >= 0.05) & (rvol < 2),
        "%5+ ve 50 gün ort. üstü": liquid & (ret1 >= 0.05) & (c > ind["sma50"]),
        "%5+ ve 50 gün ort. altı": liquid & (ret1 >= 0.05) & (c <= ind["sma50"]),
        "Karşılaştırma: tüm günler": liquid,
    }
    out = {}
    for name, mask in groups.items():
        row = {"olay_sayisi": int(mask.to_numpy().sum())}
        for n in horizons:
            fwd = (c.shift(-n) / entry - 1).where(mask).stack().dropna()
            row[f"{n}g_ort_%"] = round(100 * fwd.mean(), 2)
            row[f"{n}g_medyan_%"] = round(100 * fwd.median(), 2)
            row[f"{n}g_pozitif_%"] = round(100 * (fwd > 0).mean(), 1)
        out[name] = row
    return out


def yearly_burst_counts(px, ind):
    c, v = px["Close"], px["Volume"]
    ret1 = c / c.shift(1) - 1
    liquid = (c >= MIN_PRICE) & (ind["dollar_vol20"] >= MIN_DOLLAR_VOLUME)
    s = (liquid & (ret1 >= 0.05)).sum(axis=1)
    return {str(k): int(x) for k, x in s.groupby(s.index.year).sum().items()}
