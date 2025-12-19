"""An interface and local implementation for saving training run metrics."""

import math
from abc import ABC, abstractmethod
from typing import TypedDict

from .config_types import JSON


class SaveMetricDict(TypedDict):
    """Training run metric."""

    global_step: int
    """Global step at which this metric was measured"""
    tag: str
    """Metric name"""
    value: JSON
    """Metric value"""


class MetricSaver(ABC):
    """Interface for class that saves metrics."""

    @abstractmethod
    def save_metrics(self, metrics: list[SaveMetricDict]):
        """Save multiple metrics.

        Parameters
        ----------
        metrics : list[Metric]
            List of metrics to save.

        """
        raise NotImplementedError()


class MemoryMetricSaver(MetricSaver):
    """Saves metrics in memory to this object."""

    metrics: list[SaveMetricDict]
    """Stored metrics"""

    def __init__(self):
        self.metrics = []

    def save_metrics(self, metrics: list[SaveMetricDict]):
        """Save a list of metrics in memory.

        Parameters
        ----------
        metrics : list[SaveMetricDict]
            A list of dictionaries containing metrics to be saved. Each dictionary
            should contain the keys ``"global_step"``, ``"tag"``, and ``"value"``.

        Example
        -------
        .. code-block:: python

            metrics = [
                {"global_step": 1, "tag": "accuracy", "value": 0.95},
                {"global_step": 1, "tag": "loss", "value": 0.05},
            ]
            saver = MemoryMetricWriter()
            saver.save_metrics(metrics=metrics)
            print(saver.metrics)

        """
        self.metrics.extend(_encode_metrics(metrics))


def _encode_metrics(metrics: list[SaveMetricDict]) -> list[SaveMetricDict]:
    return [
        {
            "global_step": m["global_step"],
            "tag": m["tag"],
            "value": _encode_nan_and_inf_as_none(m["value"]),
        }
        for m in metrics
    ]


def _encode_nan_and_inf_as_none(o: JSON) -> JSON:
    if isinstance(o, dict):
        return {key: _encode_nan_and_inf_as_none(value) for key, value in o.items()}
    elif isinstance(o, list):
        return [_encode_nan_and_inf_as_none(element) for element in o]
    elif isinstance(o, float):
        if not math.isfinite(o):
            return None
        return o
    else:
        return o
