import os
import glob
import subprocess

STRATEGY_NAME = "NostalgiaForInfinityX"
CONFIG_FILE = "config_telegram.json"
DATA_EXCHANGE = "okx"

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = ["5m"]
TIMERANGE = "20260101-20260814"

def main():
    print("=" * 60)
    print("STRICT ERROR CAPTURE SCANNER")
    print(f"Current Directory: {os.getcwd()}")
    print("=" * 60)
    
    # اجرای بک‌تست با چاپ کامل خروجی استاندارد و خطاهای احتمالی ترمینال
    print(f"\n[INFO] Running backtest and capturing raw output...")
    backtest_cmd = [
        "freqtrade", "backtesting",
        "--config", CONFIG_FILE,
        "--strategy", STRATEGY_NAME,
        "--data-format-exchange", DATA_EXCHANGE,
        "--timerange", TIMERANGE,
        "--export", "trades",
        "--export-directory", "user_data/backtest_results"
    ]
    
    result = subprocess.run(backtest_cmd, capture_output=True, text=True)
    
    print("\n--- STDOUT ---")
    print(result.stdout[-2000:]) # چاپ ۲۰۰۲ کاراکتر آخر خروجی استاندارد
    
    print("\n--- STDERR ---")
    print(result.stderr[-2000:]) # چاپ خطاهای احتمالی پایتون یا فریدتید
    
    print("\n[DEBUG] Scanning entire user_data directory:")
    all_files = glob.glob("user_data/**/*", recursive=True)
    for f in all_files:
        if os.path.isfile(f):
            print(f" - {f}")

if __name__ == "__main__":
    main()
