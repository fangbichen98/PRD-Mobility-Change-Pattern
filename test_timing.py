#!/usr/bin/env python3
"""测试时间统计功能的简单示例"""
import time
from datetime import datetime

def format_time(seconds):
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def main():
    """演示时间统计功能"""
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("=" * 80)
    print("时间统计功能测试")
    print("=" * 80)
    print(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n执行任务中...")
    
    # 模拟一些工作
    time.sleep(3)
    
    end_time = time.time()
    end_datetime = datetime.now()
    total_time = end_time - start_time
    
    print(f"结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 80)
    print("时间统计")
    print("=" * 80)
    print(f"总耗时: {format_time(total_time)} ({total_time:.2f} 秒)")
    print(f"       ({total_time/60:.2f} 分钟, {total_time/3600:.2f} 小时)")
    print("=" * 80)

if __name__ == "__main__":
    main()
