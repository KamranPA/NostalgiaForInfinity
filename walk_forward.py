import os
import subprocess
from datetime import datetime, timedelta

# ==========================================
# تنظیمات اصلی Walk-Forward Analysis
# ==========================================
STRATEGY_NAME = "NostalgiaForInfinityX7"
CONFIG_FILE = "config.json"  # یا config_telegram.json
DATA_EXCHANGE = "kraken"     # تغییر به kraken برای دور زدن تحریم IP سرور گیت‌هاب
PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAMES = ["5m", "1h"]

# تنظیمات پنجره‌های زمانی (۱۲۰ روز کل، پنجره‌های ۳۰ روزه)
TOTAL_DAYS = 120
WINDOW_DAYS = 30
STEP_DAYS = 30

def run_command(command):
    """اجرای دستورات ترمینال و نمایش خروجی"""
    print(f"\n[EXEC] {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with return code {result.returncode}")
        return False
    return True

def generate_timeranges():
    """تولید بازه‌های زمانی برای بک‌تست پیش‌رونده"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=TOTAL_DAYS)
    
    timeranges = []
    current_start = start_date
    
    while current_start + timedelta(days=WINDOW_DAYS) <= end_date:
        current_end = current_start + timedelta(days=WINDOW_DAYS)
        
        # فرمت مورد نیاز Freqtrade: YYYYMMDD-YYYYMMDD
        start_str = current_start.strftime("%Y%m%d")
        end_str = current_end.strftime("%Y%m%d")
        
        timeranges.append(f"{start_str}-{end_str}")
        current_start += timedelta(days=STEP_DAYS)
        
    return timeranges

def main():
    print("=" * 60)
    print("Starting Walk-Forward Analysis")
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Data Source Exchange: {DATA_EXCHANGE}")
    print("=" * 60)
    
    timeranges = generate_timeranges()
    
    for idx, timerange in enumerate(timeranges, start=1):
        print(f"\n>>> Running Window {idx}/{len(timeranges)}: {timerange} <<<")
        
        # ۱. دانلود داده‌های تاریخی مربوط به بازه مشخص‌شده
        download_cmd = [
            "freqtrade", "download-data",
            "--exchange", DATA_EXCHANGE,
            "--pairs", *PAIRS,
            "--timeframes", *TIMEFRAMES,
            "--timerange", timerange
        ]
        
        if not run_command(download_cmd):
            print(f"[SKIP] Failed to download data for window {timerange}. Skipping...")
            continue
            
        # ۲. اجرای بک‌تست روی داده‌های دانلود شده
        backtest_cmd = [
            "freqtrade", "backtesting",
            "--config", CONFIG_FILE,
            "--strategy", STRATEGY_NAME,
            "--timerange", timerange
        ]
        
        if not run_command(backtest_cmd):
            print(f"[SKIP] Backtest failed for window {timerange}.")
            continue

    print("\n" + "=" * 60)
    print("Walk-Forward Analysis Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
