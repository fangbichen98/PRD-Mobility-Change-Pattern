#!/usr/bin/env python3
"""
独立的数据预处理脚本
用于提前生成缓存，避免训练时重复预处理

使用方法:
    # 生成缓存（全部数据）
    python preprocess_data.py --label-path data/labels_1w.csv

    # 使用小样本快速测试
    python preprocess_data.py --label-path data/labels_1w.csv --samples-per-class 10

    # 强制重新生成缓存
    python preprocess_data.py --force
"""
import argparse
import logging
import os
import sys
import time
from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='预处理双年度数据并生成缓存',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成缓存（全部数据）
  python preprocess_data.py --label-path data/labels_2w.csv

  # 使用小样本快速测试
  python preprocess_data.py --label-path data/labels_1w.csv --samples-per-class 10

  # 强制重新生成缓存
  python preprocess_data.py --force

  # 指定缓存目录
  python preprocess_data.py --cache-dir data/my_cache
        """
    )
    parser.add_argument(
        '--label-path',
        default='data/labels_1w.csv',
        help='标签文件路径 (默认: data/labels_1w.csv)'
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=None,
        help='每类样本数（None=全部数据）'
    )
    parser.add_argument(
        '--cache-dir',
        default='data/cache',
        help='缓存目录 (默认: data/cache)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成缓存（忽略现有缓存）'
    )

    args = parser.parse_args()

    # 打印配置信息
    logger.info("")
    logger.info("=" * 80)
    logger.info("数据预处理脚本")
    logger.info("=" * 80)
    logger.info(f"标签文件: {args.label_path}")
    logger.info(f"每类样本数: {args.samples_per_class if args.samples_per_class else '全部'}")
    logger.info(f"缓存目录: {args.cache_dir}")
    logger.info(f"强制重新生成: {'是' if args.force else '否'}")
    logger.info("")

    # 检查标签文件是否存在
    if not os.path.exists(args.label_path):
        logger.error(f"错误: 标签文件不存在: {args.label_path}")
        sys.exit(1)

    # 检查数据文件是否存在
    data_files = ['data/2021_week.csv', 'data/2024_week.csv']
    for data_file in data_files:
        if not os.path.exists(data_file):
            logger.error(f"错误: 数据文件不存在: {data_file}")
            sys.exit(1)

    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)

    # 开始预处理
    logger.info("开始预处理...")
    logger.info("")

    start_time = time.time()

    try:
        # 准备数据
        data = prepare_dual_year_experiment_data(
            label_path=args.label_path,
            samples_per_class=args.samples_per_class,
            use_cache=not args.force,  # force=True时不使用缓存
            cache_dir=args.cache_dir
        )

        elapsed_time = time.time() - start_time

        # 打印结果摘要
        logger.info("")
        logger.info("=" * 80)
        logger.info("预处理完成！")
        logger.info("=" * 80)
        logger.info(f"总网格数: {len(data['labels'])}")

        # 获取特征形状
        sample_features = list(data['change_features'].values())[0]
        logger.info(f"特征形状: {sample_features.shape}")

        # 动态图信息
        logger.info(f"动态图 2021: {len(data['graphs_2021'])} 个时间快照")
        logger.info(f"动态图 2024: {len(data['graphs_2024'])} 个时间快照")

        # 类别分布
        from collections import Counter
        label_counts = Counter(data['labels'].values())
        logger.info(f"类别分布: {dict(sorted(label_counts.items()))}")

        logger.info(f"处理时间: {elapsed_time:.2f} 秒")
        logger.info("")
        logger.info("后续训练时将自动使用缓存，无需重新预处理")
        logger.info("")

        # 提示如何使用
        logger.info("使用方法:")
        logger.info("  python train_improved.py")
        logger.info("")

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("预处理失败！")
        logger.error("=" * 80)
        logger.error(f"错误信息: {str(e)}")
        logger.error("")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
