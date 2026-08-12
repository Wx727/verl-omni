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

from verl.trainer.ppo.v1.utils import MetricsAggregator


class DiffusionMetricsAggregator(MetricsAggregator):
    """Aggregate diffusion metrics emitted by multiple local actor updates."""

    _ROLLOUT_FAILURE_SUM_SUFFIXES = (
        "/evicted_groups",
        "/evicted_trajectories",
        "/refilled_prompts",
        "/refill_rounds",
    )

    def _get_aggregation_type(self, metric_name: str) -> str:
        if "/rollout_failure/" in metric_name and (
            metric_name.endswith(self._ROLLOUT_FAILURE_SUM_SUFFIXES)
            or ("/reason/" in metric_name and metric_name.endswith("_groups"))
        ):
            return "sum"
        return super()._get_aggregation_type(metric_name)
