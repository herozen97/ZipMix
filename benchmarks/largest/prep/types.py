"""Shared datatypes for LargeST prep."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class ScenarioSeries:
    data: np.ndarray
    obs_mask: np.ndarray
    timestamps: pd.DatetimeIndex
    node_ids: np.ndarray
    segment_starts: list[int]
    segment_years: list[int]
    gap_after_segment: list[bool]


@dataclass
class WindowIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    stats: dict[str, int]
