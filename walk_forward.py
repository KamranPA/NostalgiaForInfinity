import os
import subprocess

STRATEGY_NAME = "NostalgiaForInfinityX"
CONFIG_FILE = "configs/exampleconfig.json"
DATA_EXCHANGE = "okx"

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

# دانلود هر دو تایم‌فریم برای رفع خطای محاسباتی داخلی استراتژی
TIMEFRAMES_TO_DOWNLOAD = ["5m", "15m"]

DOWNLOAD_TIMERANGE = "20240101-20260814"
BACKTEST_TIMERANGE = "20260101-20260814"

def main():
    print("=" * 60)
    print("FIXED WFA RUNNER - NOSTALGIA FOR INFINITY")
    print("=" * 60)
    
    for tf in TIMEFRAMES_TO_DOWNLOAD:
        print(f"\n[INFO] Downloading {tf} data for all pairs...")
        download_cmd = [
            "freqtrade", "download-data",
            "--exchange", DATA_EXCHANGE,
            "-p", *PAIRS,
            "-t", tf,
            "--timerange", DOWNLOAD_TIMERANGE
        ]
        subprocess.run(download_cmd, check=True)
    
    print(f"\n[INFO] Running backtest on 15m timeframe...")
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

if __name__ == "__main__":
    main()
