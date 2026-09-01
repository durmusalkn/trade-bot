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
ALERT_STATE_FILE = os.environ.get("ALERT_STATE_FILE", "")

# Görseldeki hisseler ve varlıklar
WATCHLIST = [
    "NVDA",      # NVIDIA
    "NBIS",      # Nebius Group
    "HIMS",      # Hims & Hers
    "BTC-USD",   # Bitcoin (yfinance formatı)
    "TEM",       # Tempus AI
    "XAIR",      # Beyond Air
    "CIFR",      # Cipher Mining
    "MRVL"       # Marvell Technology
]

INTERVAL = "15m"      # 15 dakikalık periyot
PERIOD = "5d"         # Veri aralığı
PREPOST = True        # Seans öncesi/sonrası (after-hours) mumlarını dahil et
RSI_PERIOD = 14
STOCH_PERIOD = 14
K_PERIOD = 3
D_PERIOD = 3
OVERSOLD_THRESHOLD = 20  # 20'nin altı aşırı satım bölgesi

# Mükerrer bildirimleri engellemek için son tetiklenen mum zamanları
last_alert_timestamps = {}


def load_alert_state():
    """Önceki bildirim zamanlarını dosyadan yükler (CI/cache için)."""
    global last_alert_timestamps
    if not ALERT_STATE_FILE or not os.path.exists(ALERT_STATE_FILE):
        return
    try:
        with open(ALERT_STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        last_alert_timestamps = {k: pd.Timestamp(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError) as e:
        print(f"Uyarı: durum dosyası okunamadı: {e}")


def save_alert_state():
    """Bildirim zamanlarını dosyaya kaydeder."""
    if not ALERT_STATE_FILE:
        return
    try:
        os.makedirs(os.path.dirname(ALERT_STATE_FILE) or ".", exist_ok=True)
        data = {k: str(v) for k, v in last_alert_timestamps.items()}
        with open(ALERT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        print(f"Uyarı: durum dosyası yazılamadı: {e}")


def _telegram_configured():
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


# ==========================================
# İNDİKATÖR HESAPLAMALARI
# ==========================================
def calculate_stoch_rsi(df, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """
    RSI ve ardından Stokastik RSI (K ve D) değerlerini hesaplar.
    """
    close = df['Close']
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's Smoothing RSI
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Stochastic RSI
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()

    rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
    stoch_rsi = (rsi - rsi_min) / rsi_range * 100

    # %K ve %D (SMA)
    k = stoch_rsi.rolling(window=k_period).mean()
    d = k.rolling(window=d_period).mean()

    df['RSI'] = rsi
    df['Stoch_RSI'] = stoch_rsi
    df['K'] = k
    df['D'] = d
    return df


# ==========================================
# GRAFİK OLUŞTURMA
# ==========================================
def generate_chart_image(df, symbol):
    """
    Fiyat grafiği ve alt panelde Stoch RSI indikatörünü çizip bellekten görsel döner.
    """
    plot_df = df.tail(50)  # Son 50 mumu çiz

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, 
        gridspec_kw={'height_ratios': [3, 1]}, 
        facecolor='#131722'
    )

    # Fiyat Paneli
    ax1.set_facecolor('#1e222d')
    ax1.plot(plot_df.index, plot_df['Close'], color='#2962ff', label='Fiyat (Kapanış)', lw=1.5)
    ax1.set_title(f"{symbol} - 15 Dakikalık Grafik & Stoch RSI Kesişimi", color='white', fontsize=12, pad=10)
    ax1.grid(True, color='#2a2e39', linestyle='--', alpha=0.5)
    ax1.tick_params(colors='#d1d4dc')
    ax1.legend(loc='upper left', facecolor='#1e222d', edgecolor='#2a2e39', labelcolor='white')

    # Stoch RSI Paneli
    ax2.set_facecolor('#1e222d')
    ax2.plot(plot_df.index, plot_df['K'], color='#2196f3', label='%K', lw=1.2)
    ax2.plot(plot_df.index, plot_df['D'], color='#ff6d00', label='%D', lw=1.2)
    ax2.axhline(80, color='#e91e63', linestyle=':', alpha=0.7)
    ax2.axhline(20, color='#4caf50', linestyle=':', alpha=0.7)
    ax2.fill_between(plot_df.index, 20, 80, color='#787b86', alpha=0.08)
    ax2.set_ylim(-5, 105)
    ax2.grid(True, color='#2a2e39', linestyle='--', alpha=0.5)
    ax2.tick_params(colors='#d1d4dc')
    ax2.legend(loc='upper left', facecolor='#1e222d', edgecolor='#2a2e39', labelcolor='white')

    plt.xticks(rotation=30, ha='right', color='#d1d4dc')
    plt.tight_layout()

    # Görseli RAM'de (BytesIO) kaydet
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf


# ==========================================
# TELEGRAM BİLDİRİM FONKSİYONU
# ==========================================
def send_telegram_message(text, parse_mode="HTML"):
    """Telegram'a düz metin/HTML mesaj gönderir."""
    if not _telegram_configured():
        print("Telegram yapılandırılmamış (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": parse_mode}

    try:
        response = requests.post(url, data=data, timeout=15)
        if response.status_code == 200:
            return True
        print(f"Telegram Hatası: {response.text}")
    except Exception as e:
        print(f"Telegram bağlantı hatası: {e}")
    return False


def send_telegram_alert(symbol, last_price, prev_k, prev_d, curr_k, curr_d, candle_time, chart_buf):
    """
    Telegram üzerinden hem grafik resmini hem de sinyal açıklamasını gönderir.
    """
    if not _telegram_configured():
        print(f"Telegram yapılandırılmamış, sinyal atlandı: {symbol}")
        return

    caption = (
        f"🚨 <b>STOKASTİK RSI AL SİNYALİ</b> 🚨\n\n"
        f"📌 <b>Sembol:</b> #{symbol}\n"
        f"⏱ <b>Zaman Dilimi:</b> 15 Dakika\n"
        f"💵 <b>Son Fiyat:</b> ${last_price:,.2f}\n"
        f"🕒 <b>Mum Zamanı:</b> {candle_time.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"📊 <b>Stokastik RSI Değerleri (20 Altı Kesişim):</b>\n"
        f"• Önceki Mum: K={prev_k:.2f} | D={prev_d:.2f}\n"
        f"• Güncel Mum: <b>K={curr_k:.2f}</b> ↗️ <b>D={curr_d:.2f}</b>\n\n"
        f"✅ <i>K değeri 20 seviyesi altındayken D değerini yukarı kesti.</i>"
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
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sinyal başarıyla gönderildi: {symbol}")
        else:
            print(f"Telegram Hatası ({symbol}): {response.text}")
    except Exception as e:
        print(f"Telegram bağlantı hatası ({symbol}): {e}")


# ==========================================
# TARAMA VE KONTROL MOTORU
# ==========================================
def scan_symbols():
    print(f"\n--- Tarama Başlatıldı ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    
    for symbol in WATCHLIST:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=PERIOD, interval=INTERVAL, prepost=PREPOST)

            if df.empty or len(df) < (RSI_PERIOD + STOCH_PERIOD + 5):
                continue

            # İndikatörleri ekle
            df = calculate_stoch_rsi(
                df, 
                rsi_period=RSI_PERIOD, 
                stoch_period=STOCH_PERIOD, 
                k_period=K_PERIOD, 
                d_period=D_PERIOD
            )

            # Son 2 mum verisi
            curr_row = df.iloc[-1]
            prev_row = df.iloc[-2]

            curr_time = df.index[-1]
            last_price = curr_row['Close']

            prev_k, prev_d = prev_row['K'], prev_row['D']
            curr_k, curr_d = curr_row['K'], curr_row['D']

            # KESİŞİM ŞARTI:
            # 1. Önceki mumda K <= D
            # 2. Güncel mumda K > D (Yukarı kesişim)
            # 3. Kesişimin 20'nin altında veya 20 bölgesinde gerçekleşmesi (curr_k <= 20 veya prev_k <= 20)
            is_crossover = (prev_k <= prev_d) and (curr_k > curr_d)
            is_in_oversold = (curr_k <= OVERSOLD_THRESHOLD) or (prev_k <= OVERSOLD_THRESHOLD)

            if is_crossover and is_in_oversold:
                # Aynı 15 dakikalık mumda tekrar bildirim atmaması kontrolü
                if str(last_alert_timestamps.get(symbol)) != str(curr_time):
                    last_alert_timestamps[symbol] = curr_time
                    
                    # Grafik oluştur ve gönder
                    chart_img = generate_chart_image(df, symbol)
                    send_telegram_alert(
                        symbol, last_price, 
                        prev_k, prev_d, 
                        curr_k, curr_d, 
                        curr_time, chart_img
                    )
            else:
                print(f"• {symbol:7s} -> K: {curr_k:5.2f}, D: {curr_d:5.2f} (Kesişim yok / >20)")

        except Exception as e:
            print(f"Hata [{symbol}]: {e}")

    save_alert_state()


# ==========================================
# ÇALIŞTIRMA DÖNGÜSÜ
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="15M Stoch RSI Telegram Tarayıcısı")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Tek tarama yap ve çık (GitHub Actions için)",
    )
    args = parser.parse_args()

    print("🤖 15M Stoch RSI Telegram Tarayıcısı Başlatıldı.")
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