import argparse
import io
import json
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import schedule
import yfinance as yf

# ==========================================
# AYARLAR VE YAPILANDIRMA
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ALERT_STATE_FILE = os.environ.get("ALERT_STATE_FILE", "state/alerts.json")

# Takip edilecek hisseler ve varlıklar
WATCHLIST = [
    "NVDA",      # NVIDIA
    "NBIS",      # Nebius Group
    "HIMS",      # Hims & Hers
    "BTC-USD",   # Bitcoin
    "TEM",       # Tempus AI
    "XAIR",      # Beyond Air
    "CIFR",      # Cipher Mining
    "MRVL"       # Marvell Technology
]

INTERVAL = "15m"      # 15 dakikalık periyot
PERIOD = "60d"        # İndikatör hesaplamasının TradingView ile tam oturması için 60 günlük geçmiş
PREPOST = True        # TradingView'de mavi alan (seans öncesi/sonrası) açık olduğu için True

# UT BOT ALERTS PARAMETRELERİ (TradingView Inputs sekmenizle birebir aynı)
UT_KEY_VALUE = 1      # Key Value (Hassasiyet)
UT_ATR_PERIOD = 10    # ATR Period
UT_USE_HA = True      # Signals from Heikin Ashi Candles

# Mükerrer bildirimleri engellemek için kaydedilen sinyal listesi
last_alert_timestamps = {}


def load_alert_state():
    """Önceki bildirim geçmişini dosyadan yükler (GitHub Actions cache için)."""
    global last_alert_timestamps
    if not ALERT_STATE_FILE or not os.path.exists(ALERT_STATE_FILE):
        return
    try:
        with open(ALERT_STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        last_alert_timestamps = {k: str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError) as e:
        print(f"Uyarı: Durum dosyası okunamadı: {e}")


def save_alert_state():
    """Bildirim geçmişini dosyaya kaydeder."""
    if not ALERT_STATE_FILE:
        return
    try:
        os.makedirs(os.path.dirname(ALERT_STATE_FILE) or ".", exist_ok=True)
        with open(ALERT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(last_alert_timestamps, f, indent=2)
    except OSError as e:
        print(f"Uyarı: Durum dosyası yazılamadı: {e}")


def _telegram_configured():
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


# ==========================================
# HEIKIN ASHI VE UT BOT ALGORİTMASI
# ==========================================
def calculate_heikin_ashi(df):
    """Standart OHLC verisini TradingView uyumlu Heikin Ashi mumlarına çevirir."""
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4.0
    ha_open = np.zeros(len(df))

    if len(df) > 0:
        ha_open[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2.0
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha_df = pd.DataFrame(index=df.index)
    ha_df['Open'] = ha_open
    ha_df['High'] = np.maximum.reduce([df['High'].values, ha_open, ha_close.values])
    ha_df['Low'] = np.minimum.reduce([df['Low'].values, ha_open, ha_close.values])
    ha_df['Close'] = ha_close
    return ha_df


def calculate_ut_bot(df, key_value=1, atr_period=10, use_ha=True):
    """
    Pine Script'teki UT Bot Alerts formülünü bar-by-bar durum makinesiyle hesaplar.
    """
    # 1. Kaynak seçimi
    if use_ha:
        ha_df = calculate_heikin_ashi(df)
        src = ha_df['Close']
        df['HA_Close'] = ha_df['Close']
    else:
        src = df['Close']
        df['HA_Close'] = df['Close']

    # 2. ATR (Wilder's Smoothing / RMA)
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()
    n_loss = key_value * atr

    n = len(df)
    src_vals = src.values
    loss_vals = n_loss.values

    trailing_stop = np.zeros(n)
    pos = np.zeros(n)

    # 3. Pine Script Trailing Stop Döngüsü
    for i in range(1, n):
        prev_stop = trailing_stop[i - 1]
        cur_src = src_vals[i]
        prev_src = src_vals[i - 1]
        loss = loss_vals[i]

        if np.isnan(loss):
            continue

        if cur_src > prev_stop and prev_src > prev_stop:
            cur_stop = max(prev_stop, cur_src - loss)
        elif cur_src < prev_stop and prev_src < prev_stop:
            cur_stop = min(prev_stop, cur_src + loss)
        elif cur_src > prev_stop:
            cur_stop = cur_src - loss
        else:
            cur_stop = cur_src + loss

        trailing_stop[i] = cur_stop

        # Pozisyon tespiti
        if prev_src < prev_stop and cur_src > prev_stop:
            pos[i] = 1
        elif prev_src > prev_stop and cur_src < prev_stop:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]

    df['ATR'] = atr
    df['UT_Src'] = src
    df['UT_Stop'] = trailing_stop
    df['UT_Pos'] = pos

    # 4. Kesişim Sinyalleri (Crossover)
    buy_signals = np.zeros(n, dtype=bool)
    sell_signals = np.zeros(n, dtype=bool)

    for i in range(1, n):
        prev_src = src_vals[i - 1]
        cur_src = src_vals[i]
        prev_stop = trailing_stop[i - 1]
        cur_stop = trailing_stop[i]

        above = (prev_src <= prev_stop) and (cur_src > cur_stop)
        below = (prev_stop <= prev_src) and (cur_stop > cur_src)

        buy_signals[i] = (cur_src > cur_stop) and above
        sell_signals[i] = (cur_src < cur_stop) and below

    df['UT_Buy'] = buy_signals
    df['UT_Sell'] = sell_signals
    return df


# ==========================================
# GRAFİK ÇİZİMİ
# ==========================================
def generate_chart_image(df, symbol, signal_type):
    """
    Fiyat, Heikin Ashi Kapanışı ve Trailing Stop çizgisini içeren görsel oluşturur.
    """
    plot_df = df.tail(60)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': (3, 1)},
        facecolor='#131722'
    )

    # Üst Panel: Fiyat ve UT Stop
    ax1.set_facecolor('#1e222d')
    ax1.plot(plot_df.index, plot_df['Close'], color='#787b86', label='Spot Fiyat', lw=1, alpha=0.5)
    ax1.plot(plot_df.index, plot_df['HA_Close'], color='#2962ff', label='HA Close', lw=1.5)
    ax1.plot(plot_df.index, plot_df['UT_Stop'], color='#f23645', label='UT Trailing Stop', lw=1.5, linestyle='--')

    # Al / Sat İşaretçileri
    buys = plot_df[plot_df['UT_Buy']]
    sells = plot_df[plot_df['UT_Sell']]
    if not buys.empty:
        ax1.scatter(buys.index, buys['UT_Stop'], color='#089981', marker='^', s=90, label='UT Buy', zorder=5)
    if not sells.empty:
        ax1.scatter(sells.index, sells['UT_Stop'], color='#f23645', marker='v', s=90, label='UT Sell', zorder=5)

    badge_color = '#089981' if signal_type == "BUY" else '#f23645'
    ax1.set_title(f"{symbol} - 15M UT Bot Alerts ({signal_type})", color=badge_color, fontsize=12, pad=10, weight='bold')
    ax1.grid(True, color='#2a2e39', linestyle='--', alpha=0.5)
    ax1.tick_params(colors='#d1d4dc')
    ax1.legend(loc='upper left', facecolor='#1e222d', edgecolor='#2a2e39', labelcolor='white')

    # Alt Panel: ATR
    ax2.set_facecolor('#1e222d')
    ax2.plot(plot_df.index, plot_df['ATR'], color='#ff9800', label=f'ATR ({UT_ATR_PERIOD})', lw=1.2)
    ax2.grid(True, color='#2a2e39', linestyle='--', alpha=0.5)
    ax2.tick_params(colors='#d1d4dc')
    ax2.legend(loc='upper left', facecolor='#1e222d', edgecolor='#2a2e39', labelcolor='white')

    plt.xticks(rotation=30, ha='right', color='#d1d4dc')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf


# ==========================================
# TELEGRAM GÖNDERİCİ
# ==========================================
def send_telegram_alert(symbol, signal_type, last_price, ha_price, stop_price, candle_time, chart_buf):
    """Telegram'a fotoğraf ve detaylı sinyal metnini iletir."""
    if not _telegram_configured():
        print(f"Telegram yapılandırılmamış, sinyal gönderilemedi: {symbol} ({signal_type})")
        return

    # Mum zamanını kesin olarak Türkiye Saati'ne (UTC+3) çevir
    try:
        if candle_time.tzinfo is not None:
            local_time = candle_time.tz_convert("Europe/Istanbul")
        else:
            local_time = candle_time.tz_localize("UTC").tz_convert("Europe/Istanbul")
        candle_str = local_time.strftime('%Y-%m-%d %H:%M TSİ')
    except Exception:
        candle_str = str(candle_time)

    is_long = signal_type == "BUY"
    header = "🟢 <b>UT BOT: LONG (AL) SİNYALİ</b> 🟢" if is_long else "🔴 <b>UT BOT: SHORT (SAT) SİNYALİ</b> 🔴"
    direction_desc = "Fiyat Trailing Stop çizgisini YUKARI kırdı." if is_long else "Fiyat Trailing Stop çizgisini AŞAĞI kırdı."

    caption = (
        f"{header}\n\n"
        f"📌 <b>Hisse:</b> #{symbol}\n"
        f"⏱ <b>Periyot:</b> 15 Dakika (Heikin Ashi)\n"
        f"💵 <b>Piyasa Fiyatı:</b> ${last_price:,.2f}\n"
        f"🕯 <b>HA Kapanış:</b> ${ha_price:,.2f}\n"
        f"🛡 <b>Trailing Stop:</b> ${stop_price:,.2f}\n"
        f"🕒 <b>Mum Zamanı:</b> {candle_str}\n\n"
        f"⚙️ <b>Ayarlar:</b> Key={UT_KEY_VALUE} | ATR={UT_ATR_PERIOD}\n"
        f"ℹ️ <i>{direction_desc}</i>"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", chart_buf, "image/png")}
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": caption,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, data=data, files=files, timeout=15)
        if response.status_code == 200:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {signal_type} sinyali başarıyla iletildi: {symbol}")
        else:
            print(f"Telegram Hatası ({symbol}): {response.text}")
    except Exception as e:
        print(f"Telegram bağlantı hatası ({symbol}): {e}")


# ==========================================
# TARAMA MOTORU
# ==========================================
def scan_symbols():
    print(f"\n--- UT Bot Taraması Başlatıldı ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")

    for symbol in WATCHLIST:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=PERIOD, interval=INTERVAL, prepost=PREPOST)

            if df.empty or len(df) < (UT_ATR_PERIOD + 30):
                continue

            # UT Bot değerlerini hesapla
            df = calculate_ut_bot(df, key_value=UT_KEY_VALUE, atr_period=UT_ATR_PERIOD, use_ha=UT_USE_HA)

            # Kapanmış son 4 mumu geriye dönük kontrol et (-5'ten -1'e)
            recent_closed_bars = df.iloc[-5:-1]

            has_signal = False
            for candle_time, row in recent_closed_bars.iterrows():
                is_buy = bool(row['UT_Buy'])
                is_sell = bool(row['UT_Sell'])

                if is_buy or is_sell:
                    signal_type = "BUY" if is_buy else "SELL"
                    # Mum zamanını yerel saate göre formatlayıp tekil anahtar yap
                    try:
                        if candle_time.tzinfo is not None:
                            loc_t = candle_time.tz_convert("Europe/Istanbul")
                        else:
                            loc_t = candle_time.tz_localize("UTC").tz_convert("Europe/Istanbul")
                        time_key = loc_t.strftime('%Y%m%d_%H%M')
                    except Exception:
                        time_key = str(candle_time)

                    alert_key = f"{symbol}_{signal_type}_{time_key}"

                    if alert_key not in last_alert_timestamps:
                        last_alert_timestamps[alert_key] = str(candle_time)
                        has_signal = True

                        print(f"🚨 YENİ SİNYAL: {symbol} {signal_type} (Mum: {candle_time})")
                        chart_img = generate_chart_image(df, symbol, signal_type)
                        send_telegram_alert(
                            symbol=symbol,
                            signal_type=signal_type,
                            last_price=row['Close'],
                            ha_price=row['HA_Close'],
                            stop_price=row['UT_Stop'],
                            candle_time=candle_time,
                            chart_buf=chart_img
                        )

            if not has_signal:
                last_closed = df.iloc[-2]
                pos_str = "LONG" if last_closed['UT_Pos'] == 1 else ("SHORT" if last_closed['UT_Pos'] == -1 else "NÖTR")
                print(f"• {symbol:7s} -> Son Durum: {pos_str:5s} | Fiyat: {last_closed['Close']:7.2f} | Stop: {last_closed['UT_Stop']:7.2f}")

        except Exception as e:
            print(f"Hata [{symbol}]: {e}")

    save_alert_state()


# ==========================================
# GİRİŞ NOKTASI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="15M UT Bot Telegram Tarayıcısı")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Tek tarama yap ve çık (GitHub Actions için)"
    )
    args = parser.parse_args()

    print("🤖 UT Bot Alerts (15M Heikin Ashi) Tarayıcısı Başlatıldı.")
    print(f"Takip Edilen Varlıklar: {', '.join(WATCHLIST)}")

    load_alert_state()
    scan_symbols()

    if args.once:
        print("Tek tarama tamamlandı.")
    else:
        schedule.every(1).minutes.do(scan_symbols)
        while True:
            schedule.run_pending()
            time.sleep(1)
