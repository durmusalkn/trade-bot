"""
ICT / Smart Money Concepts kurulumlarının kural tabanlı (günlük mum) karşılıkları.

ICT kavramları el ile yorumlanır; burada test edilebilir olmaları için şöyle tanımlandı:

  Swing tepe/dip   Fraktal: iki yanındaki ikişer mumdan yüksek tepe / düşük dip. 2 mum sonra kesinleşir.
  BOS              Kapanışın son kesinleşmiş swing tepenin ilk kez üstüne çıkması (yükselen mum).
  OB               BOS'tan önceki son 10 mum içindeki son düşüş mumu; bölge = o mumun tepe-dip aralığı.
  FVG              3 mumluk boşluk: bugünün dibi > iki gün önceki mumun tepesi (en az 0.3 ATR).
  Likidite süpürme Fiyatın son swing dibinin altına inip aynı gün tekrar üstünde kapanması.

Kurulumlar (sadece uzun, fiyat 200 günlük ortalamanın üstünde, QQQ filtresi):
  OB     BOS sonrası 20 gün içinde fiyat OB bölgesine geri gelince bölgenin tepesinden limit alım.
         Stop: OB dibi - 0.1 ATR.
  FVG    FVG oluştuktan sonra 10 gün içinde fiyat boşluğa geri gelince boşluğun tepesinden limit alım.
         Stop: boşluğun dibi - 0.1 ATR.
  SWEEP  Süpürme gününün ertesi açılışında alım. Stop: süpürme günü dibi - 0.1 ATR.
Çıkış (hepsi): 2R kâr al (tamamı), stop ya da 15 işlem günü.
"""
import numpy as np
import pandas as pd

from .engine import Backtester

MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 20_000_000
TARGET_R = 2.0
MAX_DAYS = 15
OB_VALID_DAYS = 20
FVG_VALID_DAYS = 10
STOP_BUFFER_ATR = 0.1


def _arm_limit_orders(event, top, bottom, stop, close, low, bias, valid_days):
    """Bölge olaylarından günlük bekleyen limit emirleri üretir.

    t kapanışında emir aktifse order_limit[t] = bölge tepesi (t+1 günü geçerli).
    Bölgeye dokunulunca, bölge dibinin altında kapanınca ya da süre dolunca emir düşer.
    Yeni olay eskisinin yerini alır.
    """
    n, m = close.shape
    order_limit = np.full((n, m), np.nan)
    order_stop = np.full((n, m), np.nan)
    for j in range(m):
        ev = np.flatnonzero(event[:, j])
        if ev.size == 0:
            continue
        active = False
        z_top = z_bot = z_stop = 0.0
        t0 = -1
        nxt = 0
        for t in range(ev[0], n):
            if active and (low[t, j] <= z_top or close[t, j] < z_bot or t - t0 > valid_days):
                active = False
            if nxt < ev.size and ev[nxt] == t:
                if top[t, j] < close[t, j]:
                    active, z_top, z_bot, z_stop, t0 = True, top[t, j], bottom[t, j], stop[t, j], t
                nxt += 1
            if not active:
                if nxt >= ev.size:
                    break
                continue
            if close[t, j] > z_top and bias[t, j]:
                order_limit[t, j] = z_top
                order_stop[t, j] = z_stop
    return order_limit, order_stop


def compute_ict(px, ind):
    o, h, l, c = px["Open"], px["High"], px["Low"], px["Close"]
    atr = ind["atr14"]
    liquid = (c >= MIN_PRICE) & (ind["dollar_vol20"] >= MIN_DOLLAR_VOLUME)
    bias = (liquid & (c > ind["sma200"])).fillna(False)

    swing_high = (h > h.shift(1)) & (h > h.shift(2)) & (h >= h.shift(-1)) & (h >= h.shift(-2))
    swing_low = (l < l.shift(1)) & (l < l.shift(2)) & (l <= l.shift(-1)) & (l <= l.shift(-2))
    last_sh = h.where(swing_high).shift(2).ffill()      # 2 mum sonra kesinleşir: ileriye bakış yok
    last_sl = l.where(swing_low).shift(2).ffill()

    # BOS + order block
    bos = (c > last_sh) & (c.shift(1) <= last_sh.shift(1)) & (c > o)
    day_no = pd.DataFrame(np.arange(len(c))[:, None].repeat(c.shape[1], 1), index=c.index, columns=c.columns)
    bear_idx = day_no.where(c < o).ffill(limit=10)
    idx = bear_idx.fillna(0).astype(int).to_numpy()
    h_np, l_np = h.to_numpy(), l.to_numpy()
    ob_top = np.take_along_axis(h_np, idx, axis=0)
    ob_bot = np.take_along_axis(l_np, idx, axis=0)
    ob_event = (bos & bear_idx.notna()).to_numpy()
    ob_stop = ob_bot - STOP_BUFFER_ATR * atr.to_numpy()

    # Fair value gap
    fvg_top = l_np
    fvg_bot = h.shift(2).to_numpy()
    fvg_event = ((l > h.shift(2)) & ((l - h.shift(2)) >= 0.3 * atr)).to_numpy()
    fvg_stop = fvg_bot - STOP_BUFFER_ATR * atr.to_numpy()

    close_np, bias_np = c.to_numpy(), bias.to_numpy()
    ob_limit, ob_stop_arr = _arm_limit_orders(ob_event, ob_top, ob_bot, ob_stop, close_np, l_np, bias_np,
                                              OB_VALID_DAYS)
    fvg_limit, fvg_stop_arr = _arm_limit_orders(fvg_event, fvg_top, fvg_bot, fvg_stop, close_np, l_np, bias_np,
                                                FVG_VALID_DAYS)

    # Likidite süpürme (turtle soup)
    sweep = bias & (l < last_sl) & (c > last_sl) & (ind["ibs"] >= 0.5)
    sweep_stop = l - STOP_BUFFER_ATR * atr

    mk = lambda a: pd.DataFrame(a, index=c.index, columns=c.columns)
    signals = {
        "OB": mk(~np.isnan(ob_limit)),
        "FVG": mk(~np.isnan(fvg_limit)),
        "SWEEP": sweep.fillna(False),
    }
    limits = {"OB": ob_limit, "FVG": fvg_limit}
    stops = {"OB": ob_stop_arr, "FVG": fvg_stop_arr, "SWEEP": sweep_stop.to_numpy()}
    return signals, limits, stops


class ICTBacktester(Backtester):
    target_full_exit = True
    no_same_day_target = True
    order_slots_factor = 20   # boş pozisyon başına 20'ye kadar limit emir; ilk dolan alınır

    def __init__(self, px, ind, signals, limits, stops, rank, regime, cfg):
        super().__init__(px, ind, signals, regime, {}, cfg)
        self.limits = limits
        self.stops = stops
        self.rank_key = rank.to_numpy()

    def _entry_fill(self, setup, si, i, j):
        if setup in self.limits:
            limit = self.limits[setup][si, j]
            if np.isnan(limit) or np.isnan(self.L[i, j]) or self.L[i, j] > limit:
                return None
            return min(self.O[i, j], limit)
        return self.O[i, j]

    def _initial_stop(self, setup, si, j, entry):
        stop = self.stops[setup][si, j]
        # %0.5'ten dar stop spread/kayma içinde kalır, uygulanamaz; %20'den genişi aşırı risk
        if np.isnan(stop) or not 0.005 <= (entry - stop) / entry <= 0.20:
            return None
        return stop

    def _target(self, setup, entry, rps):
        return entry + TARGET_R * rps

    def _exit_signal(self, pos, i, j, c):
        return "time_stop" if pos.days_held >= MAX_DAYS else None
