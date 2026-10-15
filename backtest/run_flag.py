"""
Videodaki swing stratejisinin backtest'i (EMA 8/21/50 + bayrak trend çizgisi kırılımı).

Kullanım:
    python -m backtest.run_flag              # Yahoo verisiyle (Nasdaq evreni)
    python -m backtest.run_flag --synthetic  # sahte veriyle hızlı kontrol
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
from . import flag_breakout as fb
from . import momentum as mo
from . import strategy as st
from .engine import Config
from .run_backtest import buy_and_hold, metrics, synthetic_data
from .run_m3_robustness import PRICE_START, UNIVERSE_LABELS, log_growth_without_top
from .run_momentum import SLIPPAGE, TECH_BENCHMARK

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_flag")
PERIODS = {"2015": "2015-01-01", "2020": "2020-01-01"}
VIDEO_STOCKS = ["SOFI", "HIMS", "HOOD", "APP", "MU", "TSLA", "GRAB", "OPEN"]
LABELS = dict(UNIVERSE_LABELS, video_hisseleri="Videoda geçen 8 hisse")
VARIANTS = [("gun_ici", "ema8"), ("gun_ici", "ema21"), ("kapanis", "ema8"), ("kapanis", "ema21")]
MAIN = ("gun_ici", "ema8")


def signal_stats(trades, cost_pct):
    if trades.empty:
        return {}
    r = trades["return_pct"] - cost_pct
    years = pd.to_datetime(trades["entry_date"]).dt.year
    top = r.nlargest(max(1, len(r) // 20))
    return {
        "islem_sayisi": int(len(r)),
        "ort_net_getiri_%": round(r.mean(), 2),
        "medyan_net_getiri_%": round(r.median(), 2),
        "kazanan_%": round(100 * (r > 0).mean(), 1),
        "ilk_hedefe_ulasan_%": round(100 * trades["partial"].mean(), 1),
        "ilk_gun_stop_%": round(100 * ((trades["exit_reason"] == "stop") & (trades["days_held"] == 0)).mean(), 1),
        "stopla_kapanan_%": round(100 * (trades["exit_reason"] == "stop").mean(), 1),
        "ort_stop_mesafesi_%": round(trades["stop_distance_pct"].mean(), 2),
        "ort_gun": round(trades["days_held"].mean(), 1),
        "en_iyi_%5_haric_ort_%": round(r.drop(top.index).mean(), 2),
        "cikis_nedenleri": trades["exit_reason"].value_counts().to_dict(),
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
        universes = {"video_hisseleri": cols[:8], "secilmis": cols[:30], "nasdaq_tech": cols[:50],
                     "nasdaq_tumu": cols}
    else:
        nasdaq = data.load_nasdaq_universe()
        symbols = sorted(set(nasdaq) | set(mo.TECH_UNIVERSE))
        px = data.load_prices(symbols, cache_name="prices_nasdaq.parquet", benchmarks=(TECH_BENCHMARK,),
                              start=PRICE_START)
        available = set(px["Close"].columns)
        universes = {
            "video_hisseleri": [s for s in VIDEO_STOCKS if s in available],
            "secilmis": [s for s in mo.TECH_UNIVERSE if s in available],
            "nasdaq_tech": sorted(s for s, sec in nasdaq.items() if sec == "Technology" and s in available),
            "nasdaq_tumu": sorted(s for s in nasdaq if s in available and s != TECH_BENCHMARK),
        }

    qqq_close = px["Close"][TECH_BENCHMARK]
    regime = fb.market_filter(qqq_close)
    cost_pct = 100 * 2 * args.commission / args.capital
    summary = {
        "assumptions": {
            "initial_capital": args.capital,
            "commission_per_order": args.commission,
            "note_commission": "Kademeli kâr almada her kısmi satış ayrı emir: bir işlem 2-4 emir ($3-$6)",
            "slippage_per_side_pct": 100 * SLIPPAGE,
            "rules": fb.__doc__.strip(),
            "position": "1 pozisyon x %100 sermaye; gün içi girişte 20'ye kadar bekleyen emir, ilk dolan alınır",
            "survivorship": "Evrenler bugün işlem gören hisselerden oluşur; batan şirketler yok (iyimser)",
            "universe_sizes": {k: len(v) for k, v in universes.items()},
            "signal_cost_pct": cost_pct,
        },
        "universes": {},
    }
    curves = {}

    for uname, cols in universes.items():
        print(f"\n=== {LABELS[uname]}: {len(cols)} hisse ===")
        upx = {k: v[cols] for k, v in px.items()}
        ind = st.compute_indicators(upx)
        signals, extra = fb.compute_flag(upx, ind)
        shape = upx["Close"].shape
        base_rank = pd.DataFrame(np.random.default_rng(0).random(shape), index=upx["Close"].index, columns=cols)
        ures = {"sinyal_gunu_2015+": {k: int(v.loc["2015":].to_numpy().sum()) for k, v in signals.items()},
                "periods": {}}

        for period, start in PERIODS.items():
            pres = {}

            def run(entry, trail, over, rank=base_rank):
                cfg = Config(start=start, setups=(entry,), slippage=SLIPPAGE, max_per_sector=99,
                             **{"initial_capital": args.capital, "commission": args.commission,
                                "max_positions": 1, "position_fraction": 1.0, **over})
                return fb.FlagBacktester(upx, ind, signals, extra, rank, regime, cfg, trail).run()[:2]

            for entry, trail in VARIANTS:
                name = f"giriş={entry} | çıkış={trail}"
                eq, trades = run(entry, trail, {})
                m = metrics(eq, trades, args.capital)
                m["en_iyi_5_islem_haric_carpan"] = log_growth_without_top(trades)
                if not trades.empty:
                    m["en_iyi_5_islem"] = trades.nlargest(5, "return_pct")[
                        ["symbol", "entry_date", "return_pct"]].astype(str).values.tolist()
                _, all_trades = run(entry, trail, dict(initial_capital=1e9, commission=0.0,
                                                       max_positions=100_000, position_fraction=1e-6))
                pres[name] = {"500_dolar_1_pozisyon": m, "tum_sinyaller": signal_stats(all_trades, cost_pct)}
                curves[f"{period} | {LABELS[uname]} | {name}"] = eq
                trades.to_csv(os.path.join(out, f"trades_{period}_{uname}_{entry}_{trail}.csv"), index=False)
                s = pres[name]["tum_sinyaller"]
                print(f"{period} {name}: ${m.get('final')} | işlem={m.get('trades')} | isabet=%{m.get('win_rate_pct')}"
                      f" || tüm sinyaller: n={s.get('islem_sayisi')} ort net %{s.get('ort_net_getiri_%')}"
                      f" medyan %{s.get('medyan_net_getiri_%')} kazanan %{s.get('kazanan_%')}")

            rng = np.random.default_rng(42)
            finals = []
            for _ in range(args.mc_runs):
                rank = pd.DataFrame(rng.random(shape), index=upx["Close"].index, columns=cols)
                eq, _ = run(*MAIN, {}, rank=rank)
                finals.append(float(eq.iloc[-1]))
            finals = np.array(finals)
            qqq_final = float(buy_and_hold(qqq_close, start, args.capital, args.commission).iloc[-1])
            pres["sans_analizi_gun_ici_ema8"] = {
                "tekrar": int(len(finals)),
                "son_deger_p10": round(float(np.percentile(finals, 10)), 2),
                "son_deger_medyan": round(float(np.median(finals)), 2),
                "son_deger_p90": round(float(np.percentile(finals, 90)), 2),
                "en_iyi": round(float(finals.max()), 2),
                "500_altinda_bitiren_%": round(100 * float((finals < args.capital).mean()), 1),
                "qqq_yu_gecen_%": round(100 * float((finals > qqq_final).mean()), 1),
                "qqq_son_deger": round(qqq_final, 2),
            }
            print(f"{period} şans analizi: medyan ${np.median(finals):.0f}, p90 ${np.percentile(finals, 90):.0f}")
            ures["periods"][period] = pres
        summary["universes"][uname] = ures

    for period, start in PERIODS.items():
        bh = buy_and_hold(qqq_close, start, args.capital, args.commission)
        summary.setdefault("benchmarks", {})[period] = metrics(bh, pd.DataFrame(), args.capital)
        curves[f"{period} | QQQ al-tut"] = bh

    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    pd.DataFrame(curves).to_csv(os.path.join(out, "equity_curves.csv"))

    main_name = f"giriş={MAIN[0]} | çıkış={MAIN[1]}"
    for period in PERIODS:
        fig, ax = plt.subplots(figsize=(11, 6))
        for key, eq in curves.items():
            if key.startswith(period) and (main_name in key or "QQQ" in key):
                ax.plot(eq.index, eq.values, lw=1.3, label=key.split(" | ", 1)[1].replace("$", r"\$"))
        ax.axhline(args.capital, color="grey", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"Video stratejisi (gün içi kırılım, EMA8 çıkış), \\${args.capital:g}, "
                     f"\\${2 * args.commission:g} alım+satım komisyonu ({period})")
        ax.set_ylabel("Özsermaye ($, log)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"flag_{period}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
