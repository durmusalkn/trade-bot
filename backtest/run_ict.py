"""
ICT / Smart Money Concepts backtest'i: order block, fair value gap, likidite süpürme.

Kullanım:
    python -m backtest.run_ict              # Yahoo verisiyle (Nasdaq evreni)
    python -m backtest.run_ict --synthetic  # sahte veriyle hızlı kontrol
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data
from . import ict
from . import momentum as mo
from . import strategy as st
from .engine import Config
from .run_backtest import buy_and_hold, metrics, synthetic_data
from .run_m3_robustness import PRICE_START, UNIVERSE_LABELS, log_growth_without_top
from .run_momentum import SLIPPAGE, TECH_BENCHMARK, qqq_regime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_ict")
PERIODS = {"2015": "2015-01-01", "2020": "2020-01-01"}
SETUPS = [("OB",), ("FVG",), ("SWEEP",), ("OB", "FVG", "SWEEP")]


def signal_stats(trades, cost_pct):
    """Her sinyal tek başına: maliyet sonrası işlem başı getiri ve R."""
    if trades.empty:
        return {}
    r = trades["return_pct"] - cost_pct
    net_r = r / trades["stop_distance_pct"]
    years = pd.to_datetime(trades["entry_date"]).dt.year
    return {
        "islem_sayisi": int(len(r)),
        "ort_net_getiri_%": round(r.mean(), 2),
        "medyan_net_getiri_%": round(r.median(), 2),
        "ort_net_R": round(net_r.mean(), 3),
        "kazanan_%": round(100 * (r > 0).mean(), 1),
        "hedefe_ulasan_%": round(100 * (trades["exit_reason"] == "take_profit").mean(), 1),
        "stopla_kapanan_%": round(100 * (trades["exit_reason"] == "stop").mean(), 1),
        "sure_dolan_%": round(100 * (trades["exit_reason"] == "time_stop").mean(), 1),
        "ort_stop_mesafesi_%": round(trades["stop_distance_pct"].mean(), 2),
        "ort_gun": round(trades["days_held"].mean(), 1),
        "yillik_ort_net_%": {str(y): round(v, 2) for y, v in r.groupby(years).mean().items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--capital", type=float, default=500.0)
    parser.add_argument("--commission", type=float, default=1.5, help="Emir başına komisyon ($)")
    parser.add_argument("--mc-runs", type=int, default=30)
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)

    if args.synthetic:
        px, _, _ = synthetic_data()
        px = {k: v.rename(columns={data.BENCHMARK: TECH_BENCHMARK}) for k, v in px.items()}
        cols = [c for c in px["Close"].columns if c != TECH_BENCHMARK]
        universes = {"secilmis": cols[:30], "nasdaq_tech": cols[:50], "nasdaq_tumu": cols}
    else:
        nasdaq = data.load_nasdaq_universe()
        symbols = sorted(set(nasdaq) | set(mo.TECH_UNIVERSE))
        px = data.load_prices(symbols, cache_name="prices_nasdaq.parquet", benchmarks=(TECH_BENCHMARK,),
                              start=PRICE_START)
        available = set(px["Close"].columns)
        universes = {
            "secilmis": [s for s in mo.TECH_UNIVERSE if s in available],
            "nasdaq_tech": sorted(s for s, sec in nasdaq.items() if sec == "Technology" and s in available),
            "nasdaq_tumu": sorted(s for s in nasdaq if s in available and s != TECH_BENCHMARK),
        }

    qqq_close = px["Close"][TECH_BENCHMARK]
    regime = qqq_regime(qqq_close)
    cost_pct = 100 * 2 * args.commission / args.capital
    summary = {
        "assumptions": {
            "initial_capital": args.capital,
            "commission_round_trip": 2 * args.commission,
            "slippage_per_side_pct": 100 * SLIPPAGE,
            "rules": ict.__doc__.strip(),
            "position": "1 pozisyon x %100 sermaye; aynı anda 20'ye kadar bekleyen limit emir, ilk dolan alınır",
            "survivorship": "Evrenler bugün işlem gören hisselerden oluşur; batan şirketler yok (iyimser)",
            "universe_sizes": {k: len(v) for k, v in universes.items()},
            "signal_cost_pct": cost_pct,
        },
        "universes": {},
    }
    curves = {}

    for uname, cols in universes.items():
        print(f"\n=== {UNIVERSE_LABELS[uname]}: {len(cols)} hisse ===")
        upx = {k: v[cols] for k, v in px.items()}
        ind = st.compute_indicators(upx)
        signals, limits, stops = ict.compute_ict(upx, ind)
        rng0 = np.random.default_rng(0)
        base_rank = pd.DataFrame(rng0.random(upx["Close"].shape), index=upx["Close"].index, columns=cols)
        ures = {"emir_gunu_sayisi_2015+": {k: int(v.loc["2015":].to_numpy().sum()) for k, v in signals.items()},
                "periods": {}}

        for period, start in PERIODS.items():
            pres = {}

            def run(setups, over, rank=base_rank):
                cfg = Config(start=start, setups=setups, slippage=SLIPPAGE, max_per_sector=99,
                             **{"initial_capital": args.capital, "commission": args.commission,
                                "max_positions": 1, "position_fraction": 1.0, **over})
                return ict.ICTBacktester(upx, ind, signals, limits, stops, rank, regime, cfg).run()[:2]

            for setups in SETUPS:
                name = "+".join(setups)
                eq, trades = run(setups, {})
                m = metrics(eq, trades, args.capital)
                m["en_iyi_5_islem_haric_carpan"] = log_growth_without_top(trades)
                _, all_trades = run(setups, dict(initial_capital=1e9, commission=0.0, max_positions=100_000,
                                                 position_fraction=1e-6))
                pres[name] = {"500_dolar_1_pozisyon": m, "tum_sinyaller": signal_stats(all_trades, cost_pct)}
                curves[f"{period} | {UNIVERSE_LABELS[uname]} | {name}"] = eq
                trades.to_csv(os.path.join(out, f"trades_{period}_{uname}_{name.replace('+', '')}.csv"),
                              index=False)
                s = pres[name]["tum_sinyaller"]
                print(f"{period} {name}: ${m.get('final')} | işlem={m.get('trades')} | isabet=%{m.get('win_rate_pct')}"
                      f" || tüm sinyaller: n={s.get('islem_sayisi')} ort net %{s.get('ort_net_getiri_%')}"
                      f" R={s.get('ort_net_R')} hedef %{s.get('hedefe_ulasan_%')}")

            rng = np.random.default_rng(42)
            finals = []
            for _ in range(args.mc_runs):
                rank = pd.DataFrame(rng.random(upx["Close"].shape), index=upx["Close"].index, columns=cols)
                eq, _ = run(("OB", "FVG", "SWEEP"), {}, rank=rank)
                finals.append(float(eq.iloc[-1]))
            finals = np.array(finals)
            qqq_final = float(buy_and_hold(qqq_close, start, args.capital, args.commission).iloc[-1])
            pres["sans_analizi_OB+FVG+SWEEP"] = {
                "tekrar": int(len(finals)),
                "son_deger_p10": round(float(np.percentile(finals, 10)), 2),
                "son_deger_medyan": round(float(np.median(finals)), 2),
                "son_deger_p90": round(float(np.percentile(finals, 90)), 2),
                "en_iyi": round(float(finals.max()), 2),
                "500_altinda_bitiren_%": round(100 * float((finals < args.capital).mean()), 1),
                "qqq_yu_gecen_%": round(100 * float((finals > qqq_final).mean()), 1),
                "qqq_son_deger": round(qqq_final, 2),
            }
            print(f"{period} şans analizi: medyan ${np.median(finals):.0f}, en iyi ${finals.max():.0f}")
            ures["periods"][period] = pres
        summary["universes"][uname] = ures

    for period, start in PERIODS.items():
        bh = buy_and_hold(qqq_close, start, args.capital, args.commission)
        summary.setdefault("benchmarks", {})[period] = metrics(bh, pd.DataFrame(), args.capital)
        curves[f"{period} | QQQ al-tut"] = bh

    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    pd.DataFrame(curves).to_csv(os.path.join(out, "equity_curves.csv"))

    for period in PERIODS:
        fig, ax = plt.subplots(figsize=(11, 6))
        for key, eq in curves.items():
            if key.startswith(period) and ("OB+FVG+SWEEP" in key or "QQQ" in key):
                ax.plot(eq.index, eq.values, lw=1.3, label=key.split(" | ", 1)[1].replace("$", r"\$"))
        ax.axhline(args.capital, color="grey", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"ICT (OB+FVG+SWEEP), \\${args.capital:g}, 1 pozisyon, "
                     f"\\${2 * args.commission:g} toplam komisyon ({period})")
        ax.set_ylabel("Özsermaye ($, log)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"ict_{period}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
