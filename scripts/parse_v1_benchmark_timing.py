#!/usr/bin/env python3
"""Summarize opt-in diffusion V1 benchmark timing JSONL records."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ORDERED_FIELDS = (
    "step_s",
    "feed_s",
    "gen_s",
    "tq_to_dataproto_s",
    "balance_batch_s",
    "old_log_prob_s",
    "adv_s",
    "update_actor_s",
    "actor_driver_prepare_s",
    "actor_prepare_to_tensordict_s",
    "actor_prepare_remove_padding_s",
    "actor_prepare_assign_metadata_s",
    "actor_prepare_misc_s",
    "actor_driver_rpc_s",
    "actor_dispatch_pipeline_total_s",
    "actor_dispatch_total_s",
    "actor_dispatch_query_s",
    "actor_dispatch_split_s",
    "actor_dispatch_parallel_put_s",
    "actor_dispatch_remap_s",
    "actor_execute_wait_s",
    "actor_worker_updates_s",
    "actor_worker_to_cpu_s",
    "actor_worker_total_s",
    "actor_worker_timing_collect_s",
    "actor_execute_wait_overhead_s",
    "actor_collect_total_s",
    "actor_collect_query_s",
    "actor_collect_select_s",
    "actor_collect_concat_s",
    "actor_rpc_overhead_s",
    "actor_rpc_wrapper_residual_s",
    "actor_driver_postprocess_s",
    "tq_writeback_s",
    "metrics_tq_to_dataproto_s",
)


def _numeric_values(records: list[dict], key: str) -> list[float]:
    return [float(record[key]) for record in records if isinstance(record.get(key), int | float)]


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int(0.9 * (len(ordered) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing", type=Path)
    parser.add_argument("--skip", type=int, default=1, help="warmup records to skip")
    parser.add_argument("--run-id", default=None, help="run identifier to select (default: latest)")
    args = parser.parse_args()

    all_records = [json.loads(line) for line in args.timing.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all_records:
        raise SystemExit("timing file contains no records")
    run_id = args.run_id or all_records[-1].get("run_id")
    selected_records = [record for record in all_records if record.get("run_id") == run_id]
    records = selected_records[args.skip :]
    if not records:
        raise SystemExit(f"no records remain after --skip {args.skip}")

    step_values = _numeric_values(records, "step_s")
    print(
        f"run_id={run_id} records={len(records)} median_step_s={statistics.median(step_values):.6f} "
        f"mean_step_s={statistics.mean(step_values):.6f} p90_step_s={_p90(step_values):.6f}"
    )
    metadata = records[0].get("metadata", {})
    samples_per_step = metadata.get("samples_per_step")
    n_gpus = metadata.get("n_gpus")
    if isinstance(samples_per_step, int | float) and isinstance(n_gpus, int | float):
        median_step = statistics.median(step_values)
        print(
            f"samples_per_s={samples_per_step / median_step:.6f} "
            f"samples_per_gpu_h={3600 * samples_per_step / median_step / n_gpus:.2f}"
        )

    available = set().union(*(record.keys() for record in records))
    fields = [field for field in ORDERED_FIELDS if field in available]
    fields.extend(sorted(field for field in available if field.endswith("_s") and field not in fields))

    print("\n| timing | median s | mean s | p90 s |")
    print("|---|---:|---:|---:|")
    for field in fields:
        values = _numeric_values(records, field)
        if values:
            print(f"| {field} | {statistics.median(values):.6f} | {statistics.mean(values):.6f} | {_p90(values):.6f} |")


if __name__ == "__main__":
    main()
