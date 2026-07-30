"""Opt-in driver wall-clock records for cross-framework speed benchmarks."""

from __future__ import annotations

import atexit
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

_COMMON_PHASES = (
    "data_prepare",
    "rollout_wake",
    "weight_sync",
    "model_offload",
    "rollout_generate",
    "rollout_sleep",
    "model_onload",
    "reward",
    "advantage",
    "policy_anchor",
    "actor_update",
    "bookkeeping",
)


class BenchmarkTiming:
    """Collect low-overhead driver timings and write them once at shutdown."""

    def __init__(self, framework: str, output_path: Optional[str]) -> None:
        self.framework = framework
        self.output_path = Path(output_path).expanduser() if output_path else None
        self.enabled = self.output_path is not None
        self._records: list[Dict[str, Any]] = []
        self._step: Optional[int] = None
        self._phases_ns: Dict[str, int] = {}
        self._open_phases_ns: Dict[str, int] = {}
        self._last_step_end_ns: Optional[int] = None
        self._closed = False
        if self.enabled:
            atexit.register(self.close)

    @classmethod
    def from_env(cls, framework: str) -> "BenchmarkTiming":
        return cls(framework, os.environ.get("BENCH_TIMING_JSONL"))

    def start_step(self, step: int) -> None:
        if not self.enabled:
            return
        if self._step is not None:
            raise RuntimeError(f"benchmark timing step {self._step} is still active")
        self._step = int(step)
        self._phases_ns = {}
        self._open_phases_ns = {}

    def cancel_step(self) -> None:
        if not self.enabled:
            return
        self._step = None
        self._phases_ns = {}
        self._open_phases_ns = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add_ns(name, time.perf_counter_ns() - start_ns)

    def begin(self, name: str) -> None:
        if not self.enabled:
            return
        if name in self._open_phases_ns:
            raise RuntimeError(f"benchmark timing phase {name!r} is already active")
        self._open_phases_ns[name] = time.perf_counter_ns()

    def end(self, name: str) -> None:
        if not self.enabled:
            return
        start_ns = self._open_phases_ns.pop(name)
        self.add_ns(name, time.perf_counter_ns() - start_ns)

    def add_ns(self, name: str, elapsed_ns: int) -> None:
        if not self.enabled:
            return
        if self._step is None:
            raise RuntimeError(f"benchmark timing phase {name!r} recorded outside a step")
        self._phases_ns[name] = self._phases_ns.get(name, 0) + int(elapsed_ns)

    def set_zero(self, name: str) -> None:
        if self.enabled and self._step is not None:
            self._phases_ns.setdefault(name, 0)

    def iter_batches(self, iterable, step_getter):
        """Start each cycle before the dataloader fetch and time that fetch."""
        iterator = iter(iterable)
        while True:
            self.start_step(step_getter())
            try:
                with self.measure("data_prepare"):
                    item = next(iterator)
            except StopIteration:
                self.cancel_step()
                return
            yield item

    def end_step(self, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        if self._step is None:
            raise RuntimeError("benchmark timing has no active step")
        if self._open_phases_ns:
            raise RuntimeError(f"benchmark timing phases still active: {sorted(self._open_phases_ns)}")

        step_end_ns = time.perf_counter_ns()
        cycle_ns = None if self._last_step_end_ns is None else step_end_ns - self._last_step_end_ns
        phases_s = {f"{name}_s": elapsed_ns / 1e9 for name, elapsed_ns in self._phases_ns.items()}
        for name in _COMMON_PHASES:
            phases_s.setdefault(f"{name}_s", None)

        rollout_parts = ("rollout_wake_s", "rollout_generate_s", "rollout_sleep_s")
        rollout_total_s = (
            sum(float(phases_s[name]) for name in rollout_parts)
            if all(phases_s[name] is not None for name in rollout_parts)
            else None
        )
        measured_s = sum(elapsed_ns for elapsed_ns in self._phases_ns.values()) / 1e9
        cycle_s = cycle_ns / 1e9 if cycle_ns is not None else None
        residual_s = cycle_s - measured_s if cycle_s is not None else None

        self._records.append(
            {
                "framework": self.framework,
                "step": self._step,
                "step_end_ns": step_end_ns,
                "cycle_s": cycle_s,
                **phases_s,
                "rollout_total_s": rollout_total_s,
                "measured_s": measured_s,
                "residual_s": residual_s,
                "metadata": dict(metadata or {}),
            }
        )
        self._last_step_end_ns = step_end_ns
        self._step = None
        self._phases_ns = {}

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as output:
            for record in self._records:
                output.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["BenchmarkTiming"]
