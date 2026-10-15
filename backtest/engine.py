"""
Günlük bar üzerinde portföy simülasyonu.

Zamanlama (günlük tarayıcıyla aynı):
- t kapanışı: giriş/çıkış sinyalleri hesaplanır.
- t+1 açılışı: bekleyen çıkışlar, sonra yeni girişler açılış fiyatından yapılır.
- t+1 seans içi: stop ve kâr al emirleri bekleyen emir gibi çalışır. Aynı gün ikisi
  de tetiklenirse kötümser varsayımla önce stop sayılır.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import strategy as st


@dataclass
class Config:
    setups: tuple = ("A", "B", "C")
    start: str = "2010-01-01"
    end: str | None = None
    initial_capital: float = 500.0
    commission: float = 3.0          # emir başına sabit $
    slippage: float = 0.0005         # her işlemde fiyat aleyhine %0.05
    sizing: str = "small"            # "small": sermaye payı, "risk": işlem başına risk
    max_positions: int = 2
    position_fraction: float = 0.5   # small: pozisyon başına özsermaye payı
    risk_per_trade: float = 0.01     # risk: işlem başına özsermaye riski
    max_position_fraction: float = 0.20
    max_per_sector: int = 2


@dataclass
class Position:
    symbol: str
    setup: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    stop: float
    risk_per_share: float
    target: float | None
    days_held: int = 0
    partial_done: bool = False
    cost: float = 0.0                # giriş tutarı + komisyon
    proceeds: float = 0.0            # satışlardan net gelir
    commissions: float = 0.0
    exit_reason: str | None = None
    pending_exit: str | None = None
    fills: list = field(default_factory=list)


class Backtester:
    def __init__(self, px, ind, signals, regime, sectors, cfg: Config):
        self.cfg = cfg
        self.dates = px["Close"].index
        self.symbols = list(px["Close"].columns)
        self.col = {s: i for i, s in enumerate(self.symbols)}
        self.O = px["Open"].to_numpy()
        self.H = px["High"].to_numpy()
        self.L = px["Low"].to_numpy()
        self.C = px["Close"].to_numpy()
        self.atr = ind["atr14"].to_numpy()
        self.ema10 = ind["ema10"].to_numpy()
        self.sma5 = ind["sma5"].to_numpy()
        self.rsi2 = ind["rsi2"].to_numpy()
        self.rank_key = ind["ret126"].to_numpy()
        self.sig = {k: signals[k].to_numpy() for k in cfg.setups}
        self.regime = regime.reindex(self.dates).fillna(0).to_numpy()
        self.sectors = sectors

    # ---------- emir yardımcıları ----------
    def _buy_price(self, price):
        return price * (1 + self.cfg.slippage)

    def _sell_price(self, price):
        return price * (1 - self.cfg.slippage)

    def _sell(self, pos, shares, price, date, reason):
        fill = self._sell_price(price)
        gross = shares * fill
        self.cash += gross - self.cfg.commission
        pos.proceeds += gross - self.cfg.commission
        pos.commissions += self.cfg.commission
        pos.shares -= shares
        pos.fills.append((date, "SELL", shares, fill, reason))

    def _close(self, pos, price, date, reason):
        self._sell(pos, pos.shares, price, date, reason)
        pos.exit_reason = reason
        pnl = pos.proceeds - pos.cost
        self.trades.append({
            "symbol": pos.symbol,
            "setup": pos.setup,
            "sector": self.sectors.get(pos.symbol, "Unknown"),
            "entry_date": pos.entry_date.date(),
            "exit_date": date.date(),
            "days_held": pos.days_held,
            "entry_price": round(pos.entry_price, 4),
            "exit_price": round(self._sell_price(price), 4),
            "initial_stop": round(pos.entry_price - pos.risk_per_share, 4),
            "stop_distance_pct": round(100 * pos.risk_per_share / pos.entry_price, 2),
            "invested": round(pos.cost, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(100 * pnl / pos.cost, 2),
            "r_multiple": round(pnl / (pos.risk_per_share * sum(f[2] for f in pos.fills if f[1] == "BUY")), 2),
            "commissions": pos.commissions,
            "partial": pos.partial_done,
            "exit_reason": reason,
        })
        self.positions.remove(pos)

    # ---------- ana döngü ----------
    def run(self):
        cfg = self.cfg
        start = self.dates.searchsorted(pd.Timestamp(cfg.start))
        end = len(self.dates) if cfg.end is None else self.dates.searchsorted(pd.Timestamp(cfg.end), side="right")
        self.cash = cfg.initial_capital
        self.positions = []
        self.trades = []
        pending_entries = []  # (symbol, setup, signal_index)
        equity_curve = []
        skipped_small = 0

        for i in range(start, end):
            date = self.dates[i]

            # 1) Açılış: bekleyen çıkışlar
            for pos in list(self.positions):
                if pos.pending_exit:
                    j = self.col[pos.symbol]
                    price = self.O[i, j]
                    if np.isnan(price):
                        price = self.C[i - 1, j]
                    self._close(pos, price, date, pos.pending_exit)

            # 2) Açılış: yeni girişler
            for symbol, setup, si in pending_entries:
                j = self.col[symbol]
                open_px = self.O[i, j]
                if np.isnan(open_px):
                    continue
                if len(self.positions) >= cfg.max_positions:
                    break
                entry = self._buy_price(open_px)
                stop = self._initial_stop(setup, si, j, entry)
                if stop is None or not stop < entry:
                    continue
                rps = entry - stop
                equity = self._equity(i - 1)
                regime = self.regime[si]
                if cfg.sizing == "risk":
                    value = (equity * cfg.risk_per_trade * regime / rps) * entry
                    value = min(value, equity * cfg.max_position_fraction)
                else:
                    value = equity * cfg.position_fraction * regime
                value = min(value, self.cash - cfg.commission)
                # Komisyonu bile karşılamayan pozisyon açılmaz
                if value <= 2 * cfg.commission or value <= 1:
                    skipped_small += 1
                    continue
                shares = value / entry
                self.cash -= value + cfg.commission
                target = self._target(setup, entry, rps)
                pos = Position(symbol, setup, date, entry, shares, stop, rps, target,
                               cost=value + cfg.commission, commissions=cfg.commission)
                pos.fills.append((date, "BUY", shares, entry, setup))
                self.positions.append(pos)
            pending_entries = []

            # 3) Seans içi: stop ve kâr al
            for pos in list(self.positions):
                j = self.col[pos.symbol]
                o, h, l = self.O[i, j], self.H[i, j], self.L[i, j]
                if np.isnan(l):
                    continue
                if l <= pos.stop:
                    fill = pos.stop if pos.entry_date == date else min(o, pos.stop)
                    self._close(pos, fill, date, "stop")
                    continue
                if pos.target is not None and not pos.partial_done and h >= pos.target:
                    fill = pos.target if pos.entry_date == date else max(o, pos.target)
                    self._sell(pos, pos.shares / 2, fill, date, "target_2R")
                    pos.partial_done = True
                    pos.stop = max(pos.stop, pos.entry_price)

            # 4) Kapanış: gün sayacı ve çıkış sinyalleri
            for pos in self.positions:
                j = self.col[pos.symbol]
                c = self.C[i, j]
                if np.isnan(c):
                    continue
                pos.days_held += 1
                pos.pending_exit = self._exit_signal(pos, i, j, c)

            equity_curve.append((date, self._equity(i)))

            # 5) Kapanış: yeni giriş sinyalleri (ertesi açılışta işlenir)
            if self.regime[i] <= 0 or i + 1 >= end:
                continue
            held = {p.symbol for p in self.positions}
            free = cfg.max_positions - sum(1 for p in self.positions if not p.pending_exit)
            if free <= 0:
                continue
            sector_count = {}
            for p in self.positions:
                if not p.pending_exit:
                    sec = self.sectors.get(p.symbol, "Unknown")
                    sector_count[sec] = sector_count.get(sec, 0) + 1
            candidates = []
            for setup in cfg.setups:
                for j in np.flatnonzero(self.sig[setup][i]):
                    key = self.rank_key[i, j]
                    candidates.append((-(key if not np.isnan(key) else -9), setup, self.symbols[j]))
            candidates.sort()
            chosen = set()
            for _, setup, symbol in candidates:
                if symbol in held or symbol in chosen:
                    continue
                sec = self.sectors.get(symbol, "Unknown")
                if sector_count.get(sec, 0) >= cfg.max_per_sector:
                    continue
                pending_entries.append((symbol, setup, i))
                chosen.add(symbol)
                sector_count[sec] = sector_count.get(sec, 0) + 1
                if len(chosen) >= free:
                    break

        # Dönem sonunda açık pozisyonları son kapanıştan kapat
        last = end - 1
        for pos in list(self.positions):
            j = self.col[pos.symbol]
            self._close(pos, self.C[last, j], self.dates[last], "end_of_test")
        if equity_curve:
            equity_curve[-1] = (self.dates[last], self.cash)

        eq = pd.Series(dict(equity_curve), name="equity")
        trades = pd.DataFrame(self.trades)
        return eq, trades, {"skipped_small": skipped_small}

    def _initial_stop(self, setup, si, j, entry):
        atr = self.atr[si, j]
        if np.isnan(atr) or atr <= 0:
            return None
        stop_mult = {"A": st.A_STOP_ATR, "B": st.B_STOP_ATR, "C": st.C_STOP_ATR}[setup]
        return entry - stop_mult * atr

    def _target(self, setup, entry, rps):
        return entry + st.A_TARGET_R * rps if setup == "A" else None

    def _exit_signal(self, pos, i, j, c):
        if pos.setup == "A":
            if c < self.ema10[i, j]:
                return "close_below_ema10"
            if pos.days_held == st.A_CHECK_DAY and not pos.partial_done and \
                    c < pos.entry_price + st.A_CHECK_R * pos.risk_per_share:
                return "no_progress_day5"
            if pos.days_held >= st.A_MAX_DAYS:
                return "time_stop"
        elif pos.setup == "B":
            if c > self.sma5[i, j] or self.rsi2[i, j] > st.B_EXIT_RSI2:
                return "mean_reverted"
            if pos.days_held >= st.B_MAX_DAYS:
                return "time_stop"
        elif pos.setup == "C":
            if pos.days_held >= st.C_MIN_DAYS and c < self.ema10[i, j]:
                return "close_below_ema10"
            if pos.days_held >= st.C_MAX_DAYS:
                return "time_stop"
        return None

    def _equity(self, i):
        value = self.cash
        for pos in self.positions:
            j = self.col[pos.symbol]
            c = self.C[i, j]
            if np.isnan(c):
                c = pos.entry_price
            value += pos.shares * c
        return value
