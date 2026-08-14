import os
import glob
import subprocess

STRATEGY_NAME = "NostalgiaForInfinityX"
CONFIG_FILE = "config_telegram.json"
DATA_EXCHANGE = "okx"

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = "5m"
TIMERANGE = "20260101-20260814"

def main():
    print("=" * 60)
    print("FINAL CORRECTED BACKTEST RUNNER")
    print(f"Current Directory: {os.getcwd()}")
    print("=" * 60)
    
    # ۱. دانلود دیتا با آرگومان‌های درست
    print("\n[INFO] Downloading data...")
    download_cmd = [
        "freqtrade", "download-data",
        "--exchange", DATA_EXCHANGE,
        "-p", *PAIRS,
        "-t", TIMEFRAMES,
        "--timerange", TIMERANGE
    ]
    subprocess.run(download_cmd, check=True)
    
    # ۲. اجرای بک‌تست بدون آرگومان‌های نامعتبر
    print(f"\n[INFO] Running backtest...")
    backtest_cmd = [
        "freqtrade", "backtesting",
        "--config", CONFIG_FILE,
        "--strategy", STRATEGY_NAME,
        "--timerange", TIMERANGE,
        "--export", "trades",
        "--export-directory", "user_data/backtest_results"
    ]
    
    result = subprocess.run(backtest_cmd, capture_output=True, text=True)
    
    print("\n--- STDOUT ---")
    print(result.stdout[-2000:])
    
    print("\n--- STDERR ---")
    print(result.stderr[-2000:])
    
    print("\n[DEBUG] Scanning entire user_data directory for results:")
    all_files = glob.glob("user_data/**/*", recursive=True)
    for f in all_files:
        if os.path.isfile(f):
            print(f" - {f}")

if __name__ == "__main__":
    main()
