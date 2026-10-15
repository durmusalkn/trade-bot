"""
Momentum Swing backtest'i: Kurulum A, B, C ve birleşik portföy.

Kullanım:
    python -m backtest.run_backtest              # Yahoo verisiyle
    python -m backtest.run_backtest --synthetic  # sahte veriyle hızlı kontrol
"""
import argparse
import json
import os
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data
from . import strategy as st
from .engine import Backtester, Config

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PERIODS = {"2010": "2010-01-01", "2020": "2020-01-01"}


def synthetic_data(n_symbols=80, n_days=4500, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-06-02", periods=n_days)
    symbols = [f"S{k:03d}" for k in range(n_symbols)] + [data.BENCHMARK]
    drift = rng.normal(0.0004, 0.0004, len(symbols))
    vol = rng.uniform(0.01, 0.03, len(symbols))
    rets = rng.normal(drift, vol, (n_days, len(symbols)))
    close = 50 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.006, close.shape))
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.01, (n_days - 1, len(symbols))))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.008, close.shape)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.008, close.shape)))
    volume = rng.lognormal(14, 0.5, close.shape)
    mk = lambda a: pd.DataFrame(a, index=dates, columns=symbols)
    px = {"Open": mk(open_), "High": mk(high), "Low": mk(low), "Close": mk(close), "Volume": mk(volume)}
    sectors = {s: f"Sector{k % 8}" for k, s in enumerate(symbols)}
    earnings = {s: [d + pd.Timedelta(hours=16) for d in dates[rng.integers(0, 60)::63]] for s in symbols[:60]}
    return px, sectors, earnings


def metrics(eq, trades, initial):
    if eq.empty:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    final = float(eq.iloc[-1])
    dd = eq / eq.cummax() - 1
    monthly = eq.resample("ME").last().pct_change()
    monthly.iloc[0] = eq.resample("ME").last().iloc[0] / initial - 1
    out = {
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "years": round(years, 2),
        "initial": initial,
        "final": round(final, 2),
        "total_return_pct": round(100 * (final / initial - 1), 1),
        "cagr_pct": round(100 * ((final / initial) ** (1 / years) - 1), 1) if final > 0 else -100.0,
        "max_drawdown_pct": round(100 * dd.min(), 1),
        "avg_month_pct": round(100 * monthly.mean(), 2),
        "median_month_pct": round(100 * monthly.median(), 2),
        "months_ge_5pct": round(100 * (monthly >= 0.05).mean(), 1),
        "months_negative": round(100 * (monthly < 0).mean(), 1),
        "best_month_pct": round(100 * monthly.max(), 1),
        "worst_month_pct": round(100 * monthly.min(), 1),
        "trades": int(len(trades)),
    }
    if len(trades):
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        out.update({
            "win_rate_pct": round(100 * len(wins) / len(trades), 1),
            "avg_win_usd": round(wins["pnl"].mean(), 2) if len(wins) else 0.0,
            "avg_loss_usd": round(losses["pnl"].mean(), 2) if len(losses) else 0.0,
            "avg_win_pct": round(wins["return_pct"].mean(), 2) if len(wins) else 0.0,
            "avg_loss_pct": round(losses["return_pct"].mean(), 2) if len(losses) else 0.0,
            "profit_factor": round(wins["pnl"].sum() / -losses["pnl"].sum(), 2) if losses["pnl"].sum() < 0 else None,
            "avg_r": round(trades["r_multiple"].mean(), 2),
            "avg_days_held": round(trades["days_held"].mean(), 1),
            "trades_per_month": round(len(trades) / (years * 12), 1),
            "total_commissions": round(trades["commissions"].sum(), 2),
            "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
        })
    yearly = eq.resample("YE").last()
    out["year_end_equity"] = {str(d.year): round(float(v), 2) for d, v in yearly.items()}
    return out


def buy_and_hold(close, start, initial, commission):
    s = close.loc[start:].dropna()
    shares = (initial - commission) / s.iloc[0]
    eq = s * shares
    eq.iloc[-1] -= commission
    return eq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--capital", type=float, default=500.0)
    parser.add_argument("--commission", type=float, default=3.0)
    args = parser.parse_args()

    if args.synthetic:
        px, sectors, earnings = synthetic_data()
    else:
        sectors = data.load_universe()
        px = data.load_prices(list(sectors))
        earnings = data.load_earnings([s for s in px["Close"].columns if s != data.BENCHMARK])

    spy_close = px["Close"][data.BENCHMARK]
    stocks = {k: v.drop(columns=[data.BENCHMARK]) for k, v in px.items()}
    print(f"Evren: {stocks['Close'].shape[1]} hisse, {stocks['Close'].index[0].date()} - {stocks['Close'].index[-1].date()}")

    reaction = data.earnings_reaction_days(earnings, stocks["Close"].index)
    ind = st.compute_indicators(stocks)
    signals = st.compute_signals(stocks, ind, reaction)
    regime = st.regime_multiplier(spy_close)

    variants = []
    for setups in [("A",), ("B",), ("C",), ("A", "B", "C")]:
        name = "+".join(setups)
        variants.append((f"{name} | 2 pozisyon x %50 | ${args.commission:g} komisyon",
                         dict(setups=setups, commission=args.commission)))
        variants.append((f"{name} | 2 pozisyon x %50 | komisyonsuz (referans)",
                         dict(setups=setups, commission=0.0)))
    variants.append((f"A+B+C | plan: %1 risk, 6 pozisyon | ${args.commission:g} komisyon",
                     dict(setups=("A", "B", "C"), commission=args.commission, sizing="risk",
                          max_positions=6)))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        "assumptions": {
            "initial_capital": args.capital,
            "commission_per_order": args.commission,
            "slippage_per_side_pct": 0.05,
            "fractional_shares": True,
            "universe": "Bugünkü S&P 500 + Nasdaq-100 üyeleri (survivorship bias: iyimser)",
            "data_source": "synthetic" if args.synthetic else "Yahoo Finance (yfinance), split-düzeltilmiş, temettüsüz",
            "earnings_coverage": {
                "symbols_with_dates": sum(1 for s in stocks["Close"].columns if reaction.get(s)),
                "symbols_total": int(stocks["Close"].shape[1]),
                "first_earnings_date": str(min((min(d) for d in reaction.values() if d), default=None)),
            },
            "signal_counts": {k: int(v.loc["2010":].to_numpy().sum()) for k, v in signals.items()},
        },
        "runs": {},
    }
    equity_curves = {}

    for period, start in PERIODS.items():
        for label, overrides in variants:
            cfg = Config(start=start, initial_capital=args.capital, **overrides)
            eq, trades, extra = Backtester(stocks, ind, signals, regime, sectors, cfg).run()
            key = f"{period} | {label}"
            m = metrics(eq, trades, args.capital)
            m.update(extra)
            m["config"] = {k: v for k, v in asdict(cfg).items() if k != "end"}
            summary["runs"][key] = m
            equity_curves[key] = eq
            slug = f"{period}_{'+'.join(cfg.setups)}_{cfg.sizing}_c{cfg.commission:g}".replace("+", "")
            trades.to_csv(os.path.join(RESULTS_DIR, f"trades_{slug}.csv"), index=False)
            print(f"{key}: ${m.get('final')} | işlem={m.get('trades')} | isabet=%{m.get('win_rate_pct')} | "
                  f"maxDD=%{m.get('max_drawdown_pct')}")

        bh = buy_and_hold(spy_close, start, args.capital, args.commission)
        key = f"{period} | SPY al-tut | ${args.commission:g} komisyon"
        summary["runs"][key] = metrics(bh, pd.DataFrame(), args.capital)
        equity_curves[key] = bh

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    pd.DataFrame(equity_curves).to_csv(os.path.join(RESULTS_DIR, "equity_curves.csv"))

    for period in PERIODS:
        fig, ax = plt.subplots(figsize=(11, 6))
        for key, eq in equity_curves.items():
            if not key.startswith(period) or "referans" in key:
                continue
            ax.plot(eq.index, eq.values, lw=1.4, label=key.split(" | ", 1)[1])
        ax.axhline(args.capital, color="grey", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"${args.capital:g} başlangıç, ${args.commission:g}/emir komisyon ({period} başlangıçlı)")
        ax.set_ylabel("Özsermaye ($, log)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"equity_{period}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
