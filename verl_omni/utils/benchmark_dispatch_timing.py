"""Opt-in controller-side timing for VeRL's nD DataProto dispatch path."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Iterator


@dataclass
class _DispatchTimingState:
    timings_ns: dict[str, int] = field(default_factory=dict)
    pipeline_start_ns: int | None = None
    dispatch_end_ns: int | None = None


_ACTIVE_STATE: ContextVar[_DispatchTimingState | None] = ContextVar(
    "verl_omni_benchmark_dispatch_timing",
    default=None,
)


@contextmanager
def capture_actor_dispatch_timing(enabled: bool) -> Iterator[dict[str, int]]:
    """Capture one synchronous WorkerGroup dispatch/execute/collect pipeline."""
    if not enabled:
        yield {}
        return

    state = _DispatchTimingState()
    token = _ACTIVE_STATE.set(state)
    try:
        yield state.timings_ns
    finally:
        _ACTIVE_STATE.reset(token)


def _timed_dispatch_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    state = _ACTIVE_STATE.get()
    if state is None:
        from verl.single_controller.base.decorator import dispatch_lazy_compute_data_proto

        return dispatch_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs)

    from verl.single_controller.base.decorator import _split_args_kwargs_data_proto
    from verl.single_controller.base.worker_group import WorkerGroup
    from verl.utils.ray_utils import parallel_put

    assert isinstance(worker_group, WorkerGroup)
    pipeline_start_ns = time.perf_counter_ns()
    state.pipeline_start_ns = pipeline_start_ns

    query_start_ns = time.perf_counter_ns()
    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
    assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size
    state.timings_ns["actor_dispatch_query"] = time.perf_counter_ns() - query_start_ns

    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    dp_size = max(dp_rank_mapping) + 1

    split_start_ns = time.perf_counter_ns()
    split_args, split_kwargs = _split_args_kwargs_data_proto(dp_size, *args, **kwargs)
    state.timings_ns["actor_dispatch_split"] = time.perf_counter_ns() - split_start_ns

    put_start_ns = time.perf_counter_ns()
    max_workers = max(1, min(len(split_args[0]), os.cpu_count()))
    put_args = [parallel_put(arg, max_workers=max_workers) for arg in split_args]
    put_kwargs = {key: parallel_put(value, max_workers=max_workers) for key, value in split_kwargs.items()}
    state.timings_ns["actor_dispatch_parallel_put"] = time.perf_counter_ns() - put_start_ns

    remap_start_ns = time.perf_counter_ns()
    all_args = []
    for arg in put_args:
        assert isinstance(arg, tuple | list) and len(arg) == dp_size
        transformed_args = []
        for worker_idx in range(worker_group.world_size):
            transformed_args.append(arg[dp_rank_mapping[worker_idx]])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for key, value in put_kwargs.items():
        assert isinstance(value, tuple | list) and len(value) == dp_size
        transformed_value = []
        for worker_idx in range(worker_group.world_size):
            transformed_value.append(value[dp_rank_mapping[worker_idx]])
        all_kwargs[key] = transformed_value
    state.timings_ns["actor_dispatch_remap"] = time.perf_counter_ns() - remap_start_ns

    dispatch_end_ns = time.perf_counter_ns()
    state.dispatch_end_ns = dispatch_end_ns
    state.timings_ns["actor_dispatch_total"] = dispatch_end_ns - pipeline_start_ns
    return all_args, all_kwargs


def _timed_collect_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    state = _ACTIVE_STATE.get()
    if state is None or state.pipeline_start_ns is None or state.dispatch_end_ns is None:
        from verl.single_controller.base.decorator import collect_lazy_compute_data_proto

        return collect_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs)

    from verl.protocol import BatchData
    from verl.single_controller.base.decorator import _concat_data_proto_or_future, collect_nd_compute
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    collect_start_ns = time.perf_counter_ns()
    state.timings_ns["actor_execute_wait"] = collect_start_ns - state.dispatch_end_ns

    query_start_ns = time.perf_counter_ns()
    assert mesh_name in worker_group._dispatch_info
    if mesh_name not in worker_group._collect_info:
        worker_group._collect_info[mesh_name] = worker_group._query_collect_info(mesh_name)
    assert len(worker_group._collect_info[mesh_name]) == worker_group.world_size
    state.timings_ns["actor_collect_query"] = time.perf_counter_ns() - query_start_ns

    select_start_ns = time.perf_counter_ns()
    collect_mask = worker_group._collect_info[mesh_name]
    output = collect_nd_compute(collect_mask, worker_group, *args, **kwargs)
    state.timings_ns["actor_collect_select"] = time.perf_counter_ns() - select_start_ns

    concat_start_ns = time.perf_counter_ns()
    assert BatchData(output).is_concatable(), (
        f"expecting concatable output, but got element type {type(output[0]) if output else 'empty'}"
    )
    collected = _concat_data_proto_or_future(output)
    state.timings_ns["actor_collect_concat"] = time.perf_counter_ns() - concat_start_ns

    collect_end_ns = time.perf_counter_ns()
    state.timings_ns["actor_collect_total"] = collect_end_ns - collect_start_ns
    state.timings_ns["actor_dispatch_pipeline_total"] = collect_end_ns - state.pipeline_start_ns
    return collected


def make_benchmark_nd_compute_dataproto_dispatch_fn(mesh_name):
    """Return VeRL-compatible dispatch functions with opt-in detailed timing."""
    return {
        "dispatch_fn": partial(_timed_dispatch_lazy_compute_data_proto, mesh_name),
        "collect_fn": partial(_timed_collect_lazy_compute_data_proto, mesh_name),
    }


__all__ = ["capture_actor_dispatch_timing", "make_benchmark_nd_compute_dataproto_dispatch_fn"]
