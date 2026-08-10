"""Low-overhead JSONL timing records for opt-in performance diagnostics."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

_RUN_ID = os.environ.get("BENCH_TIMING_RUN_ID") or uuid.uuid4().hex


def benchmark_timing_enabled() -> bool:
    """Return whether benchmark JSONL output is enabled for this process."""
    return bool(os.environ.get("BENCH_TIMING_JSONL"))


def append_benchmark_timing_record(
    *,
    framework: str,
    step: int,
    timing_raw_s: dict[str, Any],
    details_ns: dict[str, int],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one flattened timing record to ``BENCH_TIMING_JSONL``.

    Existing trainer timers are copied as ``<name>_s``. Nested diagnostics are
    accepted in nanoseconds and converted to the same unit. The function is
    called only by the controller process, so one append is one complete line
    and no cross-process locking is required.
    """
    output_path = os.environ.get("BENCH_TIMING_JSONL")
    if not output_path:
        return

    timings_s = {f"{name}_s": float(value) for name, value in timing_raw_s.items() if isinstance(value, int | float)}
    details_s = {f"{name}_s": int(value) / 1e9 for name, value in details_ns.items()}
    overlap = set(timings_s).intersection(details_s)
    if overlap:
        raise RuntimeError(f"benchmark timing names collide: {sorted(overlap)}")

    record = {
        "framework": framework,
        "run_id": _RUN_ID,
        "step": int(step),
        **timings_s,
        **details_s,
        "metadata": dict(metadata or {}),
    }
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["append_benchmark_timing_record", "benchmark_timing_enabled"]
