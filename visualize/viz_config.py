from pathlib import Path


EXPERIMENT_DIR = Path('outputs/multiscale_temporal_20260323_192012_label_sgh_phase31_transformer_gine_flowdistdir_spc250_seed202_e300p30')
MODEL_PREDICTIONS_DIR = EXPERIMENT_DIR / 'model_predictions'
METRICS_DIR = EXPERIMENT_DIR / 'metrics'

FONT_FAMILY = 'DejaVu Sans'
DEFAULT_DPI = 300
EXPORT_DPI = 600

CLASS_COLORS = {
    1: '#1E90FF',
    2: '#4682B4',
    3: '#87CEEB',
    4: '#FA8072',
    5: '#DC143C',
    6: '#FF6347',
    7: '#32CD32',
    8: '#228B22',
    9: '#90EE90',
}

CLASS_NAMES = {
    1: 'Stable Static',
    2: 'Stable Aggregation',
    3: 'Stable Diffusion',
    4: 'Growth Static',
    5: 'Growth Aggregation',
    6: 'Growth Diffusion',
    7: 'Decline Static',
    8: 'Decline Aggregation',
    9: 'Decline Diffusion',
}

CLASS_NAMES_MULTILINE = {
    1: 'Stable\nStatic',
    2: 'Stable\nAggregation',
    3: 'Stable\nDiffusion',
    4: 'Growth\nStatic',
    5: 'Growth\nAggregation',
    6: 'Growth\nDiffusion',
    7: 'Decline\nStatic',
    8: 'Decline\nAggregation',
    9: 'Decline\nDiffusion',
}

CITY_NAME_EN = {
    '深圳市': 'Shenzhen',
    '东莞市': 'Dongguan',
    '惠州市': 'Huizhou',
}

AREA_NAME_EN = {
    '福田区': 'Futian District',
    '罗湖区': 'Luohu District',
    '南山区': 'Nanshan District',
    '盐田区': 'Yantian District',
    '宝安区': 'Baoan District',
    '龙岗区': 'Longgang District',
    '龙华区': 'Longhua District',
    '坪山区': 'Pingshan District',
    '光明区': 'Guangming District',
    '惠城区': 'Huicheng District',
    '惠阳区': 'Huiyang District',
    '惠东县': 'Huidong County',
    '博罗县': 'Boluo County',
    '龙门县': 'Longmen County',
    '东莞市': 'Dongguan',
}

PERFORMANCE_METRIC_COLORS = {
    'precision': '#3B6EA8',
    'recall': '#F28E2B',
    'f1_score': '#59A14F',
}

CONFUSION_MATRIX_LABELS = [f'L{i}' for i in range(1, 10)]