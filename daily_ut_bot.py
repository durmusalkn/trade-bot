"""
Günlük UT Bot taraması.

After-hours (yazın 03:00 TSİ) bitsin diye Salı–Cumartesi 05:00 TSİ çalışır.
Yahoo'nun 1d bar'ı resmi seans kapanışıdır; 05:00 saati karar/emir için AH sonrası pencere sağlar.
Saatlik pre/post sinyalleri hourly_ut_bot.py'dedir.
"""
import argparse
import os

os.environ.setdefault("ALERT_STATE_FILE", "state/alerts-1d.json")

import pandas as pd

import hourly_ut_bot as bot


def apply_daily_settings():
    """Saatlik varsayılanları günlük taramaya çevirir."""
    bot.INTERVAL = "1d"
    bot.PERIOD = "3y"
    bot.PREPOST = True
    bot.TIMEFRAME_LABEL = "Günlük"
    bot.TIMEFRAME_SHORT = "1D"
    bot.IS_DAILY = True
    # Cumartesi 05:00 Cuma mumunu yakalar; bir gün kaçarsa yaş filtresi eskiyi atar.
    bot.LOOKBACK_BARS = 2
    bot.MAX_ALERT_AGE = pd.Timedelta(hours=16)
    bot.CHART_BARS = 120
    bot.BAR_DURATION = bot.get_bar_duration()
    bot.ALERT_STATE_FILE = os.environ.get("ALERT_STATE_FILE", "state/alerts-1d.json")


if __name__ == "__main__":
    apply_daily_settings()
    parser = argparse.ArgumentParser(description="Günlük UT Bot Telegram Tarayıcısı")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Tek tarama yap ve çık (GitHub Actions için)"
    )
    args = parser.parse_args()
    bot.run(once=args.once)
