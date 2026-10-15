"""
M3 (patlama sonrası bayrak) sağlamlık testi: geniş hisse evreni, şans analizi, tek işlem bağımlılığı.

Evrenler:
  secilmis      Önceki testlerdeki 143 hisselik elle seçilmiş liste (karşılaştırma)
  nasdaq_tech   Bugün Nasdaq'ta işlem gören tüm Teknoloji hisseleri (~510)
  nasdaq_tumu   Bugün Nasdaq'ta işlem gören tüm adi hisseler (~3.260, tüm sektörler)

Kullanım:
    python -m backtest.run_m3_robustness              # Yahoo verisiyle
    python -m backtest.run_m3_robustness --synthetic  # sahte veriyle hızlı kontrol
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
from . import momentum as mo
from . import strategy as st
from .engine import Config
from .run_backtest import buy_and_hold, metrics, synthetic_data
from .run_momentum import SLIPPAGE, TECH_BENCHMARK, qqq_regime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_m3_robustness")
PERIODS = {"2015": "2015-01-01", "2020": "2020-01-01"}
PRICE_START = "2014-01-01"
UNIVERSE_LABELS = {
    "secilmis": "Elle seçilmiş 143 hisse",
    "nasdaq_tech": "Nasdaq Teknoloji (tümü)",
    "nasdaq_tumu": "Nasdaq tüm hisseler",
}


def log_growth_without_top(trades, k=5):
    """En iyi k işlem çıkarılınca sermaye çarpanı (işlemler sırayla tüm sermayeyle yapılmış gibi)."""
    if trades.empty:
        return None
    lg = np.log1p(trades["return_pct"] / 100)
    return round(float(np.exp(lg.sum() - lg.nlargest(k).sum())), 2)


def all_signal_stats(trades, cost_pct):
    """Her sinyalin tek başına işlem görmesi: maliyet sonrası işlem başı getiri istatistikleri."""
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
        "en_iyi_%5_haric_ort_%": round(r.drop(top.index).mean(), 2),
        "stopla_kapanan_%": round(100 * (trades["exit_reason"] == "stop").mean(), 1),
        "ort_stop_mesafesi_%": round(trades["stop_distance_pct"].mean(), 2),
        "yillik_ort_net_%": {str(y): round(v, 2) for y, v in r.groupby(years).mean().items()},
        "yillik_islem": {str(y): int(v) for y, v in r.groupby(years).size().items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--capital", type=float, default=500.0)
    parser.add_argument("--commission", type=float, default=1.5, help="Emir başına komisyon ($)")
    parser.add_argument("--mc-runs", type=int, default=50, help="Rastgele sıralama ile tekrar sayısı")
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
    trade_cost_1pos = 100 * 2 * args.commission / args.capital

    summary = {
        "assumptions": {
            "initial_capital": args.capital,
            "commission_per_order": args.commission,
            "commission_round_trip": 2 * args.commission,
            "slippage_per_side_pct": 100 * SLIPPAGE,
            "setup": "M3: son 3-10 günde %5+ ve 2x hacimli patlama, sığ geri çekilme, dünün tepesi üstünde kapanış",
            "stop": "Geri çekilmenin dibi (son 3 günün en düşüğü), en fazla %20",
            "exit": "10 günlük EMA altında kapanış (en az 2 gün sonra) veya 15 işlem günü",
            "regime": "Yeni giriş için QQQ > 50 günlük ortalama",
            "survivorship": "Tüm evrenler bugün işlem gören hisselerden oluşur; batan şirketler yok (iyimser)",
            "universe_sizes": {k: len(v) for k, v in universes.items()},
            "all_signals_cost_pct": trade_cost_1pos,
        },
        "universes": {},
    }
    curves = {}

    for uname, cols in universes.items():
        print(f"\n=== {UNIVERSE_LABELS[uname]}: {len(cols)} hisse ===")
        upx = {k: v[cols] for k, v in px.items()}
        ind = st.compute_indicators(upx)
        signals, stops, rvol = mo.compute_momentum(upx, ind)
        signals = {"M3": signals["M3"]}
        ures = {"signal_count_since_2015": int(signals["M3"].loc["2015":].to_numpy().sum()), "periods": {}}

        for period, start in PERIODS.items():
            pres = {}

            def run(cfg_over, rank=rvol, label=None):
                cfg = Config(start=start, setups=("M3",), slippage=SLIPPAGE, max_per_sector=99,
                             **{"initial_capital": args.capital, "commission": args.commission, **cfg_over})
                bt = mo.MomentumBacktester(upx, ind, signals, stops, rank, regime, {}, cfg, "trend")
                return bt.run()[:2]

            # 1) $500, tek pozisyon ve iki pozisyon
            for key, over in [("1 pozisyon x %100", dict(max_positions=1, position_fraction=1.0)),
                              ("2 pozisyon x %50", dict(max_positions=2, position_fraction=0.5))]:
                eq, trades = run(over)
                m = metrics(eq, trades, args.capital)
                m["en_iyi_5_islem_haric_carpan"] = log_growth_without_top(trades)
                if not trades.empty:
                    m["en_iyi_5_islem"] = trades.nlargest(5, "return_pct")[
                        ["symbol", "entry_date", "return_pct"]].astype(str).values.tolist()
                pres[key] = m
                curves[f"{period} | {UNIVERSE_LABELS[uname]} | {key}"] = eq
                trades.to_csv(os.path.join(out, f"trades_{period}_{uname}_{over['max_positions']}pos.csv"),
                              index=False)
                print(f"{period} {key}: ${m.get('final')} | işlem={m.get('trades')} | isabet=%{m.get('win_rate_pct')}"
                      f" | maxDD=%{m.get('max_drawdown_pct')} | top5 hariç x{m['en_iyi_5_islem_haric_carpan']}")

            # 2) Şans analizi: aynı gün birden fazla sinyal varsa hangisinin alınacağı rastgele
            rng = np.random.default_rng(42)
            finals = []
            for _ in range(args.mc_runs):
                rank = pd.DataFrame(rng.random(rvol.shape), index=rvol.index, columns=rvol.columns)
                eq, _ = run(dict(max_positions=1, position_fraction=1.0), rank=rank)
                finals.append(float(eq.iloc[-1]))
            finals = np.array(finals)
            qqq_final = float(buy_and_hold(qqq_close, start, args.capital, args.commission).iloc[-1])
            pres["sans_analizi_1pozisyon"] = {
                "tekrar": int(len(finals)),
                "son_deger_p10": round(float(np.percentile(finals, 10)), 2),
                "son_deger_medyan": round(float(np.median(finals)), 2),
                "son_deger_p90": round(float(np.percentile(finals, 90)), 2),
                "en_kotu": round(float(finals.min()), 2),
                "en_iyi": round(float(finals.max()), 2),
                "500_altinda_bitiren_%": round(100 * float((finals < args.capital).mean()), 1),
                "50_altina_inen_son_deger_%": round(100 * float((finals < 50).mean()), 1),
                "qqq_yu_gecen_%": round(100 * float((finals > qqq_final).mean()), 1),
                "qqq_son_deger": round(qqq_final, 2),
                "tum_son_degerler": [round(x, 2) for x in sorted(finals)],
            }
            print(f"{period} şans analizi: medyan ${np.median(finals):.0f}, p10 ${np.percentile(finals, 10):.0f}, "
                  f"p90 ${np.percentile(finals, 90):.0f}, QQQ'yu geçen %{100 * (finals > qqq_final).mean():.0f}")

            # 3) Tüm sinyaller: her sinyal ayrı ayrı işlem görür (sermaye kısıtı yok)
            eq, trades = run(dict(initial_capital=1e9, commission=0.0, max_positions=100_000,
                                  position_fraction=1e-6))
            pres["tum_sinyaller"] = all_signal_stats(trades, trade_cost_1pos)
            print(f"{period} tüm sinyaller: {pres['tum_sinyaller'].get('islem_sayisi')} işlem, "
                  f"ort net %{pres['tum_sinyaller'].get('ort_net_getiri_%')}, "
                  f"medyan %{pres['tum_sinyaller'].get('medyan_net_getiri_%')}")
            ures["periods"][period] = pres
        summary["universes"][uname] = ures

    for period, start in PERIODS.items():
        summary.setdefault("benchmarks", {})[period] = metrics(
            buy_and_hold(qqq_close, start, args.capital, args.commission), pd.DataFrame(), args.capital)
        curves[f"{period} | QQQ al-tut"] = buy_and_hold(qqq_close, start, args.capital, args.commission)

    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    pd.DataFrame(curves).to_csv(os.path.join(out, "equity_curves.csv"))

    for period in PERIODS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        ax = axes[0]
        for key, eq in curves.items():
            if key.startswith(period) and ("1 pozisyon" in key or "QQQ" in key):
                ax.plot(eq.index, eq.values, lw=1.3, label=key.split(" | ", 1)[1].replace("$", r"\$"))
        ax.axhline(args.capital, color="grey", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"M3, \\${args.capital:g}, 1 pozisyon, \\${2 * args.commission:g} toplam komisyon ({period})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1]
        for uname in universes:
            vals = summary["universes"][uname]["periods"][period]["sans_analizi_1pozisyon"]["tum_son_degerler"]
            ax.hist(np.log10(np.maximum(vals, 1)), bins=20, alpha=0.5, label=UNIVERSE_LABELS[uname])
        ax.axvline(np.log10(args.capital), color="grey", ls="--", lw=1, label=f"Başlangıç \\${args.capital:g}")
        qf = summary["universes"][next(iter(universes))]["periods"][period]["sans_analizi_1pozisyon"]["qqq_son_deger"]
        ax.axvline(np.log10(qf), color="black", lw=1.2, label="QQQ al-tut")
        ax.set_xlabel("Son değer (log10 $)")
        ax.set_title(f"Şans analizi: {args.mc_runs} rastgele sinyal sıralaması ({period})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"m3_{period}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
