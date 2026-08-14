import os
import json
import glob
import subprocess

# تغییر استراتژی به DefaultStrategy برای تضمین تولید معامله و فایل خروجی
STRATEGY_NAME = "DefaultStrategy"
CONFIG_FILE = "config_telegram.json"
DATA_EXCHANGE = "okx"

# افزایش لیست ارزها برای بالا بردن شانس تولید سیگنال
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = ["5m"]
TIMERANGE = "20260101-20260814"

RESULTS_DIR = os.path.join("user_data", "backtest_results")
DATA_DIR = os.path.join("user_data", "data", DATA_EXCHANGE)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_command(command):
    print(f"\n[EXEC] {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    return result.returncode == 0

def main():
    ensure_dir(RESULTS_DIR)
    ensure_dir(DATA_DIR)
    
    print("=" * 60)
    print("ULTIMATE DEBUG SANITY CHECK (DefaultStrategy)")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Data Directory Path: {os.path.abspath(DATA_DIR)}")
    print("=" * 60)
    
    print("\n[INFO] Downloading explicit dataset...")
    download_cmd = [
        "freqtrade", "download-data",
        "--exchange", DATA_EXCHANGE,
        "-p", *PAIRS,
        "-t", *TIMEFRAMES,
        "--timerange", TIMERANGE
    ]
    run_command(download_cmd)
    
    downloaded_files = glob.glob(os.path.join(DATA_DIR, "*"))
    print(f"\n[DEBUG] Files found in data directory: {downloaded_files}")
    
    print(f"\n[INFO] Running direct backtest for {TIMERANGE}...")
    backtest_cmd = [
        "freqtrade", "backtesting",
        "--config", CONFIG_FILE,
        "--strategy", STRATEGY_NAME,
        "--data-format-exchange", DATA_EXCHANGE,
        "--timerange", TIMERANGE,
        "--export", "trades"
    ]
    
    run_command(backtest_cmd)
    
    pattern = os.path.join(RESULTS_DIR, "*.json")
    list_of_files = glob.glob(pattern)
    print(f"\n[DEBUG] Result JSON files found: {list_of_files}")
    
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"\n[INFO] Reading result file: {latest_file}")
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
                print(json.dumps(data, indent=2)[:1000])
        except Exception as e:
            print(f"[ERROR] Could not read JSON: {e}")
    else:
        print("\n[ERROR] No backtest result JSON file was generated!")

if __name__ == "__main__":
    main()
