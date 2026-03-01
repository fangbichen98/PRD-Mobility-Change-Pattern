"""
Model architectures for the PRD mobility pattern classification project.
"""

from .multi_scale_temporal import (
    MultiScaleTemporalBranch,
    SimplifiedMultiScaleTemporal
)
from .gated_fusion import (
    GatedFeatureFusion,
    AdaptiveGatedFusion,
    DynamicWeightFusion
)

__all__ = [
    'MultiScaleTemporalBranch',
    'SimplifiedMultiScaleTemporal',
    'GatedFeatureFusion',
    'AdaptiveGatedFusion',
    'DynamicWeightFusion'
]
