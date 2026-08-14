import os
import glob
import subprocess

STRATEGY_NAME = "NostalgiaForInfinityX"
CONFIG_FILE = "config_telegram.json"
DATA_EXCHANGE = "okx"

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = ["5m"]
TIMERANGE = "20260101-20260814"

DATA_DIR = os.path.join("user_data", "data", DATA_EXCHANGE)

def run_command(command):
    print(f"\n[EXEC] {' '.join(command)}")
    subprocess.run(command, capture_output=False, text=True)

def main():
    print("=" * 60)
    print("DEEP RECURSIVE SCANNER")
    print(f"Current Directory: {os.getcwd()}")
    print("=" * 60)
    
    # دانلود دیتا
    download_cmd = [
        "freqtrade", "download-data",
        "--exchange", DATA_EXCHANGE,
        "-p", *PAIRS,
        "-t", *TIMEFRAMES,
        "--timerange", TIMERANGE
    ]
    run_command(download_cmd)
    
    # اجرای بک‌تست با درخواست صریح ذخیره در پوشه استاندارد
    print(f"\n[INFO] Running backtest...")
    backtest_cmd = [
        "freqtrade", "backtesting",
        "--config", CONFIG_FILE,
        "--strategy", STRATEGY_NAME,
        "--data-format-exchange", DATA_EXCHANGE,
        "--timerange", TIMERANGE,
        "--export", "trades",
        "--export-directory", "user_data/backtest_results"
    ]
    run_command(backtest_cmd)
    
    # اسکن تمام فایل‌های ایجاد شده در کل پوشه user_data برای پیدا کردن خروجی
    print("\n[DEBUG] Scanning entire user_data directory for any new files:")
    all_files = glob.glob("user_data/**/*", recursive=True)
    for f in all_files:
        if os.path.isfile(f):
            print(f" - {f}")

if __name__ == "__main__":
    main()
