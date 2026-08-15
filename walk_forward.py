import os
import glob
import subprocess

STRATEGY_NAME = "NostalgiaForInfinityX"
CONFIG_FILE = "configs/exampleconfig.json"
DATA_EXCHANGE = "okx"

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = "5m"

# بازه زمانی دانلود دیتا عقب‌تر از تاریخ شروع بک‌تست است تا Warm-up به خوبی انجام شود
DOWNLOAD_TIMERANGE = "20251201-20260814"
BACKTEST_TIMERANGE = "20260101-20260814"

def main():
    print("=" * 60)
    print("FINAL WFA RUNNER - NOSTALGIA FOR INFINITY")
    print(f"Current Directory: {os.getcwd()}")
    print("=" * 60)
    
    # ۱. دانلود داده‌های تاریخی با بازه کافی
    print("\n[INFO] Downloading data...")
    download_cmd = [
        "freqtrade", "download-data",
        "--exchange", DATA_EXCHANGE,
        "-p", *PAIRS,
        "-t", TIMEFRAMES,
        "--timerange", DOWNLOAD_TIMERANGE
    ]
    subprocess.run(download_cmd, check=True)
    
    # ۲. اجرای بک‌تست (پارامتر startup_candle_count از داخل کانفیگ خوانده می‌شود)
    print(f"\n[INFO] Running backtest...")
    backtest_cmd = [
        "freqtrade", "backtesting",
        "--config", CONFIG_FILE,
        "--strategy", STRATEGY_NAME,
        "--timerange", BACKTEST_TIMERANGE,
        "--export", "trades",
        "--export-directory", "user_data/backtest_results"
    ]
    
    result = subprocess.run(backtest_cmd, capture_output=True, text=True)
    
    print("\n--- STDOUT ---")
    print(result.stdout[-3000:])
    
    print("\n--- STDERR ---")
    print(result.stderr[-3000:])
    
    print("\n[DEBUG] Scanning entire user_data directory for results:")
    all_files = glob.glob("user_data/**/*", recursive=True)
    for f in all_files:
        if os.path.isfile(f):
            print(f" - {f}")

if __name__ == "__main__":
    main()
