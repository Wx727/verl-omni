# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Metric aggregation rules for diffusion V1 parameter-sync cycles."""

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch

_DAPO_FILTERED_REWARD_COUNTS_KEY = "_dapo_filtered_reward_counts"


class DiffusionMetricsAggregator:
    """Aggregate metrics emitted by multiple local actor updates.

    This is kept local because verl-omni's pinned verl revision predates the V1
    ``MetricsAggregator``. The reduction rules follow the newer upstream
    implementation, with diffusion rollout-failure counters added as sums.
    """

    def __init__(self):
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        self.metric_weights: dict[str, list[int]] = defaultdict(list)
        self.dict_metrics: dict[str, Counter] = defaultdict(Counter)
        self.step_count = 0
        self.aggregation_rules = {
            "sum": [
                "training/off_policy/evicted_samples",
                "validation/off_policy/evicted_samples",
                "training/filter_groups/evicted_samples",
                "validation/filter_groups/evicted_samples",
                "training/filter_groups/discarded_surplus_samples",
                "validation/filter_groups/discarded_surplus_samples",
                "training/rollout_failure/evicted_samples",
                "validation/rollout_failure/evicted_samples",
            ],
            "last": [
                "training/global_step",
                "training/rollout_probs_diff_valid",
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int = 0) -> None:
        """Record metrics from one local actor update."""
        self.step_count += 1
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if key == _DAPO_FILTERED_REWARD_COUNTS_KEY and isinstance(value, dict):
                self.dict_metrics[key].update(value)
                continue
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
                self.metric_weights[key].append(self._get_metric_weight(key, metrics, sample_count))
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                self.metric_values[key].append(float(value.item()))
                self.metric_weights[key].append(self._get_metric_weight(key, metrics, sample_count))

    def _get_metric_weight(self, metric_name: str, metrics: dict[str, Any], sample_count: int) -> int:
        if metric_name.endswith("/off_policy/evicted_samples_staleness/mean"):
            prefix = metric_name.rsplit("_staleness/mean", 1)[0]
            evicted_samples = metrics.get(prefix, sample_count)
            if isinstance(evicted_samples, torch.Tensor):
                return int(evicted_samples.item()) if evicted_samples.numel() == 1 else sample_count
            if isinstance(evicted_samples, int | float | np.number):
                return int(evicted_samples)
        return sample_count

    def _get_aggregation_type(self, metric_name: str) -> str:
        if "/rollout_failure/" in metric_name and (
            metric_name.endswith(("/evicted_groups", "/evicted_trajectories", "/refilled_prompts", "/refill_rounds"))
            or ("/reason/" in metric_name and metric_name.endswith("_groups"))
        ):
            return "sum"
        for aggregation_type, metric_names in self.aggregation_rules.items():
            if metric_name in metric_names:
                return aggregation_type

        metric_lower = metric_name.lower()
        if metric_lower.endswith(("/lr", "_lr")) or metric_lower == "lr":
            return "last"
        if "timing_s/" in metric_lower or "timing_per_token_ms/" in metric_lower:
            return "time_sum"
        if any(keyword in metric_lower for keyword in ("max", "maximum")):
            return "max"
        if any(keyword in metric_lower for keyword in ("min", "minimum")):
            return "min"
        if any(keyword in metric_lower for keyword in ("sum", "total")):
            return "sum"
        return "weighted_avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        aggregation_type = self._get_aggregation_type(metric_name)
        if aggregation_type == "last":
            return values[-1]
        if aggregation_type == "weighted_avg":
            weights = self.metric_weights[metric_name]
            if len(values) != len(weights) or sum(weights) == 0:
                return sum(values) / len(values)
            return sum(value * weight for value, weight in zip(values, weights, strict=False)) / sum(weights)
        if aggregation_type in ("sum", "time_sum"):
            return sum(values)
        if aggregation_type == "max":
            return max(values)
        if aggregation_type == "min":
            return min(values)
        return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """Return one reduced value per metric for the parameter-sync cycle."""
        if self.step_count == 0:
            return {}

        aggregated = {
            name: self._aggregate_single_metric(name, values) for name, values in self.metric_values.items() if values
        }
        for name, counter in self.dict_metrics.items():
            if counter:
                aggregated[name] = dict(counter)
        if {"global_seqlen/minmax_diff", "global_seqlen/max", "global_seqlen/min"}.issubset(aggregated):
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]
        return aggregated

    def reset(self) -> None:
        """Clear all metrics accumulated for the current cycle."""
        self.metric_values.clear()
        self.metric_weights.clear()
        self.dict_metrics.clear()
        self.step_count = 0
