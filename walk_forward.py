import os
import json
import glob
import subprocess
from datetime import datetime, timedelta, timezone

# ==========================================
# تنظیمات اصلی Walk-Forward Analysis
# ==========================================
STRATEGY_NAME = "NostalgiaForInfinityX7"
CONFIG_FILE = "config_telegram.json"
DATA_EXCHANGE = "okx"
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAMES = ["5m", "1h"]

TOTAL_DAYS = 120
WINDOW_DAYS = 30
STEP_DAYS = 30

RESULTS_DIR = os.path.join("user_data", "backtest_results")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_command(command):
    print(f"\n[EXEC] {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    return result.returncode == 0

def generate_timeranges():
    """تولید بازه‌های زمانی واقعی ۳۰ روزه بر اساس تاریخ فعلی"""
    now = datetime.now(timezone.utc)
    end_date = now.replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=TOTAL_DAYS)
    
    timeranges = []
    current_start = start_date
    
    while current_start + timedelta(days=WINDOW_DAYS) <= end_date:
        current_end = current_start + timedelta(days=WINDOW_DAYS)
        start_str = current_start.strftime("%Y%m%d")
        end_str = current_end.strftime("%Y%m%d")
        timeranges.append((f"{start_str}-{end_str}", current_start, current_end))
        current_start += timedelta(days=STEP_DAYS)
        
    return timeranges

def parse_latest_backtest_result():
    pattern1 = os.path.join(RESULTS_DIR, ".backtest-result-*.json")
    pattern2 = os.path.join(RESULTS_DIR, "backtest-result-*.json")
    
    list_of_files = glob.glob(pattern1) + glob.glob(pattern2)
    if not list_of_files:
        return None
    
    latest_file = max(list_of_files, key=os.path.getctime)
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
            strategy_data = data['strategy'][STRATEGY_NAME]
            
            trades = strategy_data.get('total_trades', 0)
            profit_pct = strategy_data.get('profit_total_pct', 0.0) * 100
            profit_abs = strategy_data.get('profit_total_abs', 0.0)
            wins = strategy_data.get('wins', 0)
            win_rate = (wins / trades * 100) if trades > 0 else 0
            
            return {
                "trades": trades,
                "profit_pct": profit_pct,
                "profit_abs": profit_abs,
                "win_rate": win_rate
            }
    except Exception as e:
        print(f"[WARNING] Could not parse results: {e}")
        return None

def main():
    ensure_dir(RESULTS_DIR)
    
    print("=" * 60)
    print("Starting Walk-Forward Analysis")
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Data Source Exchange: {DATA_EXCHANGE}")
    print("=" * 60)
    
    timeranges = generate_timeranges()
    summary_results = []
    
    # برای محاسبه اندیکاتورهای سنگین استراتژی، داده دانلود را از ۶۰ روز قبل‌تر دانلود می‌کنیم
    download_start = (timeranges[0][1] - timedelta(days=60)).strftime("%Y%m%d")
    download_end = timeranges[-1][2].strftime("%Y%m%d")
    full_download_timerange = f"{download_start}-{download_end}"
    
    print(f"\n[INFO] Downloading full dataset for indicators ({full_download_timerange})...")
    download_cmd = [
        "freqtrade", "download-data",
        "--exchange", DATA_EXCHANGE,
        "-p", *PAIRS,
        "-t", *TIMEFRAMES,
        "--timerange", full_download_timerange
    ]
    run_command(download_cmd)
    
    for idx, (timerange, _, _) in enumerate(timeranges, start=1):
        print(f"\n>>> Running Backtest Window {idx}/{len(timeranges)}: {timerange} <<<")
        
        backtest_cmd = [
            "freqtrade", "backtesting",
            "--config", CONFIG_FILE,
            "--strategy", STRATEGY_NAME,
            "--data-format-exchange", DATA_EXCHANGE,
            "--export", "trades",
            "--timerange", timerange
        ]
        
        if run_command(backtest_cmd):
            res = parse_latest_backtest_result()
            if res:
                res['window'] = f"W{idx} ({timerange})"
                summary_results.append(res)
            else:
                summary_results.append({
                    'window': f"W{idx} ({timerange})",
                    'trades': 0,
                    'win_rate': 0.0,
                    'profit_pct': 0.0
                })
        else:
            print(f"[SKIP] Backtest failed for window {timerange}.")

    # نمایش جدول نهایی
    print("\n" + "=" * 70)
    print("                WALK-FORWARD SUMMARY RESULTS                ")
    print("=" * 70)
    print(f"{'Window':<25} | {'Trades':<8} | {'Win Rate':<10} | {'Profit (%)':<12}")
    print("-" * 70)
    
    total_profit = 0
    total_trades = 0
    
    for r in summary_results:
        print(f"{r['window']:<25} | {r['trades']:<8} | {r['win_rate']:<9.1f}% | {r['profit_pct']:<11.2f}%")
        total_profit += r['profit_pct']
        total_trades += r['trades']
        
    print("-" * 70)
    print(f"{'TOTAL / AVG':<25} | {total_trades:<8} | {'-':<10} | {total_profit:<11.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
