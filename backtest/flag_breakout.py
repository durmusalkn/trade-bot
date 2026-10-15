"""
Videodaki swing stratejisi: EMA 8/21/50 + bayrak formasyonunda düşen trend çizgisi kırılımı.

Videodaki kuralların günlük mumlarla test edilebilir karşılığı:

  Piyasa filtresi  QQQ kapanışı 21 ve 50 günlük EMA'nın üstünde (sadece alım).
  Tarayıcı         Fiyat >= $3, ADR(20) >= %2, 10 günlük ort. hacim >= 500 bin adet,
                   kapanış 21 ve 50 günlük EMA'nın üstünde. (Piyasa değeri >= $300M geçmiş veride
                   yok; uygulanmadı.)
  Formasyon        Son 20 günün zirvesi 3-20 gün önce; o zirveden %3-25 geri çekilme;
                   son 3 günün ort. hacmi 50 günlük ortalamanın altında (hacim kuruyor).
  Trend çizgisi    Zirveden başlayıp sonraki tepelerin hepsinin üstünde kalan en dik olmayan düşen çizgi.
  Giriş            'gun_ici': ertesi gün fiyat çizgiyi geçerse çizgi seviyesinden alım (bekleyen stop emri).
                   'kapanis': kapanış çizginin üstünde + hacim önceki günden yüksek + gün pozitif ->
                   ertesi açılışta alım.
  Stop             Girişten önceki günün en düşüğü (%0.5-%15 arası değilse işlem yok).
  Kâr al           %25 formasyon zirvesinde (zirve girişin %2'sinden yakınsa giriş +%5), stop başa başa çekilir;
                   %25 ilk hedefin %6.5 üstünde; kalan %50, kapanış EMA8 (ya da EMA21) altına inince
                   ertesi açılışta. En fazla 40 işlem günü.
"""
import numpy as np
import pandas as pd

from .engine import Backtester

MIN_PRICE = 3.0
MIN_ADR = 0.02
MIN_AVG_VOLUME = 500_000
TOP_LOOKBACK = 20
MIN_TOP_AGE, MAX_TOP_AGE = 3, 20
MIN_DEPTH, MAX_DEPTH = 0.03, 0.25
STOP_MIN, STOP_MAX = 0.005, 0.15
T2_STEP = 0.065
MAX_DAYS = 40


def market_filter(qqq_close):
    ema21 = qqq_close.ewm(span=21, adjust=False).mean()
    ema50 = qqq_close.ewm(span=50, adjust=False).mean()
    return ((qqq_close > ema21) & (qqq_close > ema50)).astype(float)


def compute_flag(px, ind):
    o, h, l, c, v = px["Open"], px["High"], px["Low"], px["Close"], px["Volume"]
    ema = lambda span: c.ewm(span=span, adjust=False, min_periods=span).mean()
    ema8, ema21, ema50 = ema(8), ema(21), ema(50)
    adr = (h / l - 1).rolling(20).mean()
    screener = ((c >= MIN_PRICE) & (adr >= MIN_ADR) & (v.rolling(10).mean() >= MIN_AVG_VOLUME)
                & (c > ema21) & (c > ema50))

    # Son 20 günün zirvesi kaç gün önce?
    roll_max = h.rolling(TOP_LOOKBACK).max()
    top_age = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for k in range(TOP_LOOKBACK):
        top_age = top_age.where(top_age.notna() | (h.shift(k) != roll_max), k)
    top_age_np = top_age.to_numpy(copy=True)
    h_np = h.to_numpy()
    n, m = h_np.shape

    # Düşen trend çizgisi: zirveden sonraki tepeleri kapsayan en yüksek eğim
    # slope_close: t gününün tepesi hariç (kapanış kırılımı t'de), slope_next: t dahil (t+1 için emir)
    slope_close = np.full((n, m), np.nan)
    slope_next = np.full((n, m), np.nan)
    for k in range(MIN_TOP_AGE, MAX_TOP_AGE + 1):
        rows = top_age_np == k
        if not rows.any():
            continue
        hp = np.roll(h_np, k, axis=0)
        best_close = np.full((n, m), -np.inf)
        for d in range(1, k):
            best_close = np.fmax(best_close, (np.roll(h_np, k - d, axis=0) - hp) / d)
        best_next = np.fmax(best_close, (h_np - hp) / k)
        slope_close[rows] = best_close[rows]
        slope_next[rows] = best_next[rows]
    top_age_np[:MAX_TOP_AGE + 1] = np.nan  # np.roll başa sarar: ilk satırları kullanma

    hp = np.full((n, m), np.nan)
    valid = ~np.isnan(top_age_np)
    rr, cc = np.nonzero(valid)
    hp[rr, cc] = h_np[rr - top_age_np[rr, cc].astype(int), cc]
    line_today = hp + slope_close * top_age_np
    line_tomorrow = hp + slope_next * (top_age_np + 1)

    # Zirveden bu yana en düşük (bugün dahil): k gün için son k mumun en düşüğü
    low_since_top = np.full((n, m), np.nan)
    l_np = l.to_numpy()
    lows = l_np.copy()
    for k in range(2, MAX_TOP_AGE + 1):
        lows = np.fmin(lows, np.roll(l_np, k - 1, axis=0))
        rows = top_age_np == k
        if k >= MIN_TOP_AGE and rows.any():
            low_since_top[rows] = lows[rows]
    depth = 1 - low_since_top / hp

    mk = lambda a: pd.DataFrame(a, index=c.index, columns=c.columns)
    pattern = (mk(top_age_np).ge(MIN_TOP_AGE) & mk((depth >= MIN_DEPTH) & (depth <= MAX_DEPTH))
               & (v.rolling(3).mean() < ind["vol50_prev"]))
    base = (screener & pattern).fillna(False)

    # Gün içi: bugün çizginin altında kapanış, yarın çizgi seviyesine bekleyen alım emri
    intraday = base & (c < mk(line_tomorrow)) & mk(~np.isnan(line_tomorrow))
    # Kapanış: çizginin üstünde ilk kapanış, hacim önceki günden yüksek, gün pozitif
    line_t = mk(line_today)
    breakout = (c > line_t) & (c.shift(1) <= line_t.shift(1)) & (v > v.shift(1)) & (c > c.shift(1))
    close_sig = (screener & breakout & mk(top_age_np).ge(MIN_TOP_AGE)
                 & mk((depth >= MIN_DEPTH) & (depth <= MAX_DEPTH))).fillna(False)

    signals = {"gun_ici": intraday.fillna(False), "kapanis": close_sig}
    extra = {
        "limit_gun_ici": line_tomorrow,
        "pattern_top": hp,
        "low": l_np,
        "ema8": ema8.to_numpy(),
        "ema21": ema21.to_numpy(),
    }
    return signals, extra


class FlagBacktester(Backtester):
    no_same_day_target = True
    order_slots_factor = 20

    def __init__(self, px, ind, signals, extra, rank, regime, cfg, trail="ema8"):
        super().__init__(px, ind, signals, regime, {}, cfg)
        self.x = extra
        self.rank_key = rank.to_numpy()
        self.trail = self.x[trail]

    def _entry_fill(self, setup, si, i, j):
        if setup == "gun_ici":
            line = self.x["limit_gun_ici"][si, j]
            if np.isnan(line) or np.isnan(self.H[i, j]) or self.H[i, j] < line:
                return None
            return max(self.O[i, j], line)  # bekleyen alış-stop emri
        return self.O[i, j]

    def _initial_stop(self, setup, si, j, entry):
        stop = self.x["low"][si, j]  # girişten önceki günün en düşüğü
        if np.isnan(stop) or not STOP_MIN <= (entry - stop) / entry <= STOP_MAX:
            return None
        return stop

    def _target(self, setup, entry, rps):
        return None  # kademeli hedefler _manage_intraday içinde ayarlanır

    def _manage_intraday(self, pos, i, date):
        j = self.col[pos.symbol]
        if not hasattr(pos, "stage"):
            top = self.x["pattern_top"][self.dates.get_loc(pos.entry_date) - 1, j]
            t1 = top if top > pos.entry_price * 1.02 else pos.entry_price * 1.05
            pos.stage, pos.t1, pos.t2, pos.initial_shares = 0, t1, t1 * (1 + T2_STEP), pos.shares
        o, h, l = self.O[i, j], self.H[i, j], self.L[i, j]
        if np.isnan(l):
            return
        if l <= pos.stop:
            fill = pos.stop if pos.entry_date == date else min(o, pos.stop)
            self._close(pos, fill, date, "stop" if pos.stage == 0 else "breakeven_stop")
            return
        if pos.entry_date == date:
            return
        for stage, level, name in ((0, pos.t1, "kar_al_1"), (1, pos.t2, "kar_al_2")):
            if pos.stage == stage and h >= level:
                self._sell(pos, pos.initial_shares * 0.25, max(o, level), date, name)
                pos.stage += 1
                pos.partial_done = True
                pos.stop = max(pos.stop, pos.entry_price)

    def _exit_signal(self, pos, i, j, c):
        if pos.days_held >= 2 and c < self.trail[i, j]:
            return "trend_bitti"
        if pos.days_held >= MAX_DAYS:
            return "time_stop"
        return None
