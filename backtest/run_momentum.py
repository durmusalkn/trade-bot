"""
Nasdaq teknoloji momentum backtest'i (M1, M2, M3).

Kullanım:
    python -m backtest.run_momentum              # Yahoo verisiyle
    python -m backtest.run_momentum --synthetic  # sahte veriyle hızlı kontrol
"""
import argparse
import json
import os
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import data
from . import momentum as mo
from . import strategy as st
from .engine import Config
from .run_backtest import buy_and_hold, metrics, synthetic_data

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_momentum")
PERIODS = {"2015": "2015-01-01", "2020": "2020-01-01"}
TECH_BENCHMARK = "QQQ"
SLIPPAGE = 0.001  # momentum hisselerinde açılışta daha geniş spread: her işlemde %0.1


def qqq_regime(qqq_close):
    """QQQ 50 günlük ortalamanın üstündeyse 1 (giriş serbest), değilse 0."""
    return (qqq_close > qqq_close.rolling(50).mean()).astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--capital", type=float, default=500.0)
    parser.add_argument("--commission", type=float, default=3.0)
    parser.add_argument("--stop-pct", type=float, default=None,
                        help="Sabit yüzde stop (örn. 0.03). Verilmezse sinyal mumunun dibi kullanılır.")
    parser.add_argument("--take-profit", type=float, default=None,
                        help="Sabit yüzde kâr al (örn. 0.06). Verilirse çıkış sadece stop/kâr al/15 gün olur.")
    args = parser.parse_args()
    results_dir = (RESULTS_DIR + (f"_stop{100 * args.stop_pct:g}" if args.stop_pct else "")
                   + (f"_tp{100 * args.take_profit:g}" if args.take_profit else ""))
    exit_modes = ("bracket",) if args.take_profit else ("trend", "fast")

    if args.synthetic:
        px, sectors, _ = synthetic_data()
        px = {k: v.rename(columns={data.BENCHMARK: TECH_BENCHMARK}) for k, v in px.items()}
    else:
        px = data.load_prices(mo.TECH_UNIVERSE, cache_name="prices_tech.parquet",
                              benchmarks=(TECH_BENCHMARK,))
    sectors = {}  # sektör sınırı yok: evren zaten tek sektör

    qqq_close = px["Close"][TECH_BENCHMARK]
    stocks = {k: v.drop(columns=[TECH_BENCHMARK]) for k, v in px.items()}
    missing = sorted(set(mo.TECH_UNIVERSE) - set(stocks["Close"].columns))
    print(f"Evren: {stocks['Close'].shape[1]} hisse; veri yok: {missing}")

    ind = st.compute_indicators(stocks)
    signals, stops, rvol = mo.compute_momentum(stocks, ind)
    regime_on = qqq_regime(qqq_close)
    regime_all = pd.Series(1.0, index=qqq_close.index)

    variants = []
    for exit_mode in exit_modes:
        for setups in [("M1",), ("M2",), ("M3",), ("M1", "M2", "M3")]:
            name = "+".join(setups)
            base = dict(setups=setups, slippage=SLIPPAGE, max_per_sector=99)
            variants.append((f"{name} | çıkış={exit_mode} | 1 pozisyon x %100 | ${args.commission:g} komisyon",
                             exit_mode, regime_on, dict(base, commission=args.commission, max_positions=1,
                                                        position_fraction=1.0)))
            variants.append((f"{name} | çıkış={exit_mode} | 2 pozisyon x %50 | ${args.commission:g} komisyon",
                             exit_mode, regime_on, dict(base, commission=args.commission)))
            variants.append((f"{name} | çıkış={exit_mode} | 2 pozisyon x %50 | komisyonsuz (referans)",
                             exit_mode, regime_on, dict(base, commission=0.0)))
    variants.append((f"M1+M2+M3 | çıkış={exit_modes[0]} | 1 pozisyon x %100 | ${args.commission:g} komisyon | QQQ filtresi yok",
                     exit_modes[0], regime_all, dict(setups=("M1", "M2", "M3"), slippage=SLIPPAGE, max_per_sector=99,
                                               commission=args.commission, max_positions=1,
                                               position_fraction=1.0)))

    os.makedirs(results_dir, exist_ok=True)
    summary = {
        "assumptions": {
            "initial_capital": args.capital,
            "commission_per_order": args.commission,
            "slippage_per_side_pct": 100 * SLIPPAGE,
            "fractional_shares": True,
            "universe": f"{stocks['Close'].shape[1]} Nasdaq teknoloji/yüksek beta hissesi, bugünden seçilmiş "
                        "(survivorship + seçim yanlılığı: sonuçlar iyimser)",
            "missing_symbols": missing,
            "regime": "Yeni giriş için QQQ > 50 günlük ortalama",
            "stop": (f"Sabit %{100 * args.stop_pct:g} (giriş fiyatının altı)" if args.stop_pct
                     else "Sinyal mumunun dibi (M3: geri çekilmenin dibi), en fazla %20"),
            "exit": (f"Kâr al %{100 * args.take_profit:g} (tamamı), stop, ya da 15 işlem günü sonunda"
                     if args.take_profit else "trend: EMA10 altı kapanış / fast: önceki gün dibi altı kapanış"),
            "signal_counts_since_2015": {k: int(v.loc["2015":].to_numpy().sum()) for k, v in signals.items()},
            "data_source": "synthetic" if args.synthetic else "Yahoo Finance (yfinance)",
        },
        "burst_days_per_year": mo.yearly_burst_counts(stocks, ind),
        "event_study_since_2015": mo.event_study({k: v.loc["2014-06":] for k, v in stocks.items()},
                                                 {k: v.loc["2014-06":] for k, v in ind.items()}),
        "runs": {},
    }
    equity_curves = {}

    for period, start in PERIODS.items():
        for label, exit_mode, regime, overrides in variants:
            cfg = Config(start=start, **{"initial_capital": args.capital, **overrides})
            bt = mo.MomentumBacktester(stocks, ind, signals, stops, rvol, regime, sectors, cfg, exit_mode,
                                       stop_pct=args.stop_pct, take_profit=args.take_profit)
            eq, trades, extra = bt.run()
            key = f"{period} | {label}"
            m = metrics(eq, trades, cfg.initial_capital)
            m.update(extra)
            m["config"] = {k: v for k, v in asdict(cfg).items() if k != "end"}
            m["config"]["exit_mode"] = exit_mode
            summary["runs"][key] = m
            equity_curves[key] = eq
            slug = (f"{period}_{''.join(cfg.setups)}_{exit_mode}_{cfg.max_positions}pos_c{cfg.commission:g}"
                    f"{'_noregime' if regime is regime_all else ''}")
            trades.to_csv(os.path.join(results_dir, f"trades_{slug}.csv"), index=False)
            print(f"{key}: ${m.get('final')} | işlem={m.get('trades')} | isabet=%{m.get('win_rate_pct')} | "
                  f"maxDD=%{m.get('max_drawdown_pct')}")

        bh = buy_and_hold(qqq_close, start, args.capital, args.commission)
        key = f"{period} | QQQ al-tut | ${args.commission:g} komisyon"
        summary["runs"][key] = metrics(bh, pd.DataFrame(), args.capital)
        equity_curves[key] = bh

    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    pd.DataFrame(equity_curves).to_csv(os.path.join(results_dir, "equity_curves.csv"))

    for period in PERIODS:
        fig, ax = plt.subplots(figsize=(11, 6))
        for key, eq in equity_curves.items():
            if not key.startswith(period) or "referans" in key or "filtresi yok" in key:
                continue
            if "1 pozisyon" not in key and "QQQ" not in key:
                continue
            ax.plot(eq.index, eq.values, lw=1.3, label=key.split(" | ", 1)[1].replace("$", r"\$"))
        ax.axhline(args.capital, color="grey", lw=0.8, ls="--")
        ax.set_yscale("log")
        stop_label = f", %{100 * args.stop_pct:g} stop" if args.stop_pct else ""
        if args.take_profit:
            stop_label += f", %{100 * args.take_profit:g} kâr al"
        ax.set_title(f"Nasdaq tech momentum: \\${args.capital:g} başlangıç, 1 pozisyon, "
                     f"\\${args.commission:g}/emir{stop_label} ({period} başlangıçlı)")
        ax.set_ylabel("Özsermaye ($, log)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"equity_{period}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
