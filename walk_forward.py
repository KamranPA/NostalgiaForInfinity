import os
import subprocess
from datetime import datetime, timedelta

# ==================== تنظیمات ====================
STRATEGY = "NostalgiaForInfinityX7"
CONFIG = "config_telegram.json"
TIMEFRAME = "5m"
TOTAL_DAYS = 180       # بازه زمانی کل (مثلا ۱۸۰ روز گذشته)
WINDOW_DAYS = 30       # طول هر پنجره تست (۳۰ روزه)
STEP_DAYS = 30         # میزان حرکت به جلو در هر گام (۳۰ روز)
# =================================================

def run_command(command):
    """اجرای دستورات سیستم‌عامل"""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr

def main():
    print("=== شروع فرآیند Walk-Forward Analysis ===")
    
    # تعیین بازه‌های زمانی
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=TOTAL_DAYS)
    
    timeranges = []
    current_start = start_date
    
    while current_start + timedelta(days=WINDOW_DAYS) <= end_date:
        current_end = current_start + timedelta(days=WINDOW_DAYS)
        # فرمت تایم‌رنج استاندارد فرک‌ترد: YYYYMMDD-YYYYMMDD
        str_start = current_start.strftime("%Y%m%d")
        str_end = current_end.strftime("%Y%m%d")
        timeranges.append((str_start, str_end))
        current_start += timedelta(days=STEP_DAYS)

    print(f"تعداد پنجره‌های زمان یافت‌شده: {len(timeranges)}")
    
    # ۱. دانلود داده‌های تاریخچه
    first_date = timeranges[0][0]
    last_date = timeranges[-1][1]
    download_cmd = f"freqtrade download-data --config {CONFIG} --timeframe {TIMEFRAME} --timerange {first_date}-{last_date}"
    print(f"\n[۱/۲] در حال دانلود داده‌ها از تاریخ {first_date} تا {last_date}...")
    run_command(download_cmd)

    # ۲. اجرای بک‌تست چرخشی برای هر پنجره
    print("\n[۲/۲] اجرای Backtest بر روی پنجره‌های زمانی:")
    print("=" * 60)
    
    for idx, (start, end) in enumerate(timeranges, 1):
        timerange_str = f"{start}-{end}"
        print(f"\n---> پنجره شماره {idx}: از {start} تا {end}")
        
        bt_cmd = f"freqtrade backtesting --config {CONFIG} --strategy {STRATEGY} --timerange {timerange_str}"
        stdout, stderr = run_command(bt_cmd)
        
        # استخراج جدول خلاصه نتایج از لاگ
        if "STRATEGY SUMMARY" in stdout:
            lines = stdout.split("\n")
            recording = False
            for line in lines:
                if "BACKTESTING SUMMARY REPORT" in line or "STRATEGY SUMMARY" in line:
                    recording = True
                if recording:
                    print(line)
                if "=======================" in line and recording and line != "=======================":
                    break
        else:
            print("خطا در اجرای بک‌تست یا نبود داده در این بازه!")
            if stderr:
                print(f"جزییات خطا: {stderr[:200]}")

    print("\n=== فرآیند Walk-Forward با موفقیت به پایان رسید ===")

if __name__ == "__main__":
    main()
