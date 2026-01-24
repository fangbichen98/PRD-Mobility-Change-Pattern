"""椭圆特征提取模块 - 精简版

This module extracts ellipse-based geometric features for direction classification.
Ellipse features capture the spatial distribution of mobility flows, which is the
true definition of direction labels (Balanced/Aggregation/Diffusion).
"""
import json
import numpy as np
import math
from typing import Dict, Optional


def load_ellipse_data(json_path: str) -> Dict:
    """
    加载椭圆数据

    Args:
        json_path: Path to ellipses.json file

    Returns:
        Dictionary containing ellipse data for all years
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_ellipse_features_simple(ellipse_data: Dict, grid_id: int, year: int) -> Optional[Dict]:
    """
    提取单个grid单年的精简椭圆特征

    Args:
        ellipse_data: 椭圆数据字典
        grid_id: 网格ID
        year: 年份 (2021 or 2024)

    Returns:
        dict with keys: eccentricity, log_area
        or None if not found
    """
    year_data = ellipse_data['years'].get(str(year), [])

    # 查找该grid的椭圆数据
    for item in year_data:
        if item['grid_id'] == grid_id:
            a = item['axes']['a']  # 长轴(米)
            b = item['axes']['b']  # 短轴(米)

            # 计算2个关键特征
            eccentricity = a / (b + 1e-6)  # 离心率
            area = math.pi * a * b  # 面积(平方米)
            log_area = math.log1p(area)  # 对数面积

            return {
                'eccentricity': eccentricity,
                'log_area': log_area
            }

    return None


def compute_ellipse_features_dual_year(grid_id: int, ellipse_data: Dict) -> Optional[Dict]:
    """
    计算两年的椭圆特征 (精简版)

    Args:
        grid_id: 网格ID
        ellipse_data: 椭圆数据字典

    Returns:
        dict with 4 features or None
        {
            'eccentricity_2021': float,
            'log_area_2021': float,
            'eccentricity_2024': float,
            'log_area_2024': float
        }
    """
    feat_2021 = extract_ellipse_features_simple(ellipse_data, grid_id, 2021)
    feat_2024 = extract_ellipse_features_simple(ellipse_data, grid_id, 2024)

    if feat_2021 is None or feat_2024 is None:
        return None

    return {
        'eccentricity_2021': feat_2021['eccentricity'],
        'log_area_2021': feat_2021['log_area'],
        'eccentricity_2024': feat_2024['eccentricity'],
        'log_area_2024': feat_2024['log_area'],
    }
