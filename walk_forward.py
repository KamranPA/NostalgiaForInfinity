import os
import json
import glob
import subprocess
from datetime import datetime, timedelta

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

def run_command(command):
    """اجرای دستورات ترمینال"""
    print(f"\n[EXEC] {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    return result.returncode == 0

def generate_timeranges():
    """تولید بازه‌های زمانی ۳۰ روزه"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=TOTAL_DAYS)
    
    timeranges = []
    current_start = start_date
    
    while current_start + timedelta(days=WINDOW_DAYS) <= end_date:
        current_end = current_start + timedelta(days=WINDOW_DAYS)
        start_str = current_start.strftime("%Y%m%d")
        end_str = current_end.strftime("%Y%m%d")
        timeranges.append(f"{start_str}-{end_str}")
        current_start += timedelta(days=STEP_DAYS)
        
    return timeranges

def parse_latest_backtest_result():
    """استخراج نتایج آخرین بک‌تست انجام شده"""
    results_dir = os.path.join("user_data", "backtest_results")
    list_of_files = glob.glob(os.path.join(results_dir, ".backtest-result-*.json"))
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
            win_rate = (strategy_data.get('wins', 0) / trades * 100) if trades > 0 else 0
            
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
    print("=" * 60)
    print("Starting Walk-Forward Analysis")
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Data Source Exchange: {DATA_EXCHANGE}")
    print("=" * 60)
    
    timeranges = generate_timeranges()
    summary_results = []
    
    for idx, timerange in enumerate(timeranges, start=1):
        print(f"\n>>> Running Window {idx}/{len(timeranges)}: {timerange} <<<")
        
        # ۱. دانلود داده
        download_cmd = [
            "freqtrade", "download-data",
            "--exchange", DATA_EXCHANGE,
            "-p", *PAIRS,
            "-t", *TIMEFRAMES,
            "--timerange", timerange
        ]
        
        if not run_command(download_cmd):
            print(f"[SKIP] Failed to download data for window {timerange}.")
            continue
            
        # ۲. اجرای بک‌تست
        backtest_cmd = [
            "freqtrade", "backtesting",
            "--config", CONFIG_FILE,
            "--strategy", STRATEGY_NAME,
            "--data-format-exchange", DATA_EXCHANGE,
            "--timerange", timerange
        ]
        
        if run_command(backtest_cmd):
            res = parse_latest_backtest_result()
            if res:
                res['window'] = f"W{idx} ({timerange})"
                summary_results.append(res)
        else:
            print(f"[SKIP] Backtest failed for window {timerange}.")

    # ۳. نمایش جدول نهایی نتایج Walk-Forward
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
