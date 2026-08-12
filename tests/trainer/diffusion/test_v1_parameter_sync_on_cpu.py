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
"""CPU regression tests for diffusion V1 parameter-sync cycles."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta
from verl import DataProto

from verl_omni.trainer.diffusion.v1.metrics import DiffusionMetricsAggregator
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1
from verl_omni.trainer.diffusion.v1.trainer_separate_async import (
    PolicyGradientDiffusionTrainerV1SeparateAsync,
)
from verl_omni.workers import detach_actor_worker as detach_actor_worker_module
from verl_omni.workers.detach_actor_worker import DiffusionDetachActorWorker


def _separate_async_config(*, train_batch_size=8, ppo_mini_batch_size=2, parameter_sync_step=4):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": train_batch_size},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": ppo_mini_batch_size},
                "rollout": {
                    "nnodes": 1,
                    "n_gpus_per_node": 1,
                    "checkpoint_engine": {"backend": "nccl"},
                },
            },
            "trainer": {
                "v1": {
                    "separate_async": {"parameter_sync_step": parameter_sync_step},
                }
            },
        }
    )


def test_separate_async_requires_n_local_mini_batches(monkeypatch):
    monkeypatch.setattr(PolicyGradientDiffusionTrainerV1, "__init__", lambda self, config: None)
    config = _separate_async_config()
    PolicyGradientDiffusionTrainerV1SeparateAsync(config)

    config.data.train_batch_size = 7
    with pytest.raises(AssertionError, match=r"parameter_sync_step \* ppo_mini_batch_size"):
        PolicyGradientDiffusionTrainerV1SeparateAsync(config)


def test_separate_async_rejects_non_positive_parameter_sync_step(monkeypatch):
    monkeypatch.setattr(PolicyGradientDiffusionTrainerV1, "__init__", lambda self, config: None)
    config = _separate_async_config(parameter_sync_step=0)
    with pytest.raises(AssertionError, match="must be positive"):
        PolicyGradientDiffusionTrainerV1SeparateAsync(config)


def test_base_step_samples_one_mini_batch_per_local_update():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 8}})
    trainer.parameter_sync_step = 4
    sample_sizes = []
    local_steps = []

    trainer._add_batch_to_generate = lambda: None

    def step_once(iter_metrics, timing_raw, sample_batch_size):
        del timing_raw
        sample_sizes.append(sample_batch_size)
        local_steps.append(trainer.local_trigger_step)
        iter_metrics["actor/loss/mean"] = float(trainer.local_trigger_step)
        iter_metrics["training/rollout_failure/refilled_prompts"] = 1
        return KVBatchMeta(
            partition_id="train",
            keys=[f"sample-{trainer.local_trigger_step}"],
            tags=[{"is_padding": False}],
        )

    trainer._step_once = step_once
    metrics = {}
    batch = trainer.step(metrics, {})

    assert sample_sizes == [2, 2, 2, 2]
    assert local_steps == [0, 1, 2, 3]
    assert batch.keys == ["sample-0", "sample-1", "sample-2", "sample-3"]
    assert metrics["actor/loss/mean"] == pytest.approx(1.5)
    assert metrics["training/rollout_failure/refilled_prompts"] == 4


def test_separate_async_refuses_to_pad_undersized_local_batch():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = "separate_async"
    trainer.actor_rollout_wg = SimpleNamespace()
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 2},
                "rollout": {"n": 2},
            }
        }
    )
    data = DataProto.from_dict(tensors={"value": torch.zeros(2)})

    with pytest.raises(ValueError, match="refusing to pad copied trajectories"):
        trainer._balance_batch(data, {})


def test_metrics_aggregator_sums_rollout_failure_counts():
    aggregator = DiffusionMetricsAggregator()
    aggregator.add_step_metrics(
        {
            "training/rollout_failure/evicted_groups": 1,
            "training/rollout_failure/evicted_trajectories": 2,
            "training/rollout_failure/refilled_prompts": 1,
            "training/rollout_failure/refill_rounds": 1,
            "training/rollout_failure/reason/timeout_groups": 1,
            "actor/loss/mean": 2.0,
            "actor/lr": 1e-4,
        },
        sample_count=1,
    )
    aggregator.add_step_metrics(
        {
            "training/rollout_failure/evicted_groups": 3,
            "training/rollout_failure/evicted_trajectories": 4,
            "training/rollout_failure/refilled_prompts": 3,
            "training/rollout_failure/refill_rounds": 2,
            "training/rollout_failure/reason/timeout_groups": 2,
            "actor/loss/mean": 4.0,
            "actor/lr": 2e-4,
        },
        sample_count=3,
    )

    metrics = aggregator.get_aggregated_metrics()
    assert metrics["training/rollout_failure/evicted_groups"] == 4
    assert metrics["training/rollout_failure/evicted_trajectories"] == 6
    assert metrics["training/rollout_failure/refilled_prompts"] == 4
    assert metrics["training/rollout_failure/refill_rounds"] == 3
    assert metrics["training/rollout_failure/reason/timeout_groups"] == 3
    assert metrics["actor/loss/mean"] == pytest.approx(3.5)
    assert metrics["actor/lr"] == pytest.approx(2e-4)


class _FakeSnapshotWorkerGroup:
    def __init__(self):
        self.current = "W0"
        self.snapshots = {}
        self.calls = []

    def save_model_to_cpu(self, snapshot_id):
        self.calls.append(("save", snapshot_id, self.current))
        self.snapshots[snapshot_id] = self.current

    def restore_model_from_cpu(self, snapshot_id):
        self.calls.append(("restore", snapshot_id, self.snapshots[snapshot_id]))
        self.current = self.snapshots[snapshot_id]

    def clear_cpu_model(self, snapshot_id):
        self.calls.append(("clear", snapshot_id))
        self.snapshots.pop(snapshot_id, None)


def test_old_policy_is_stable_across_local_updates(monkeypatch):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.parameter_sync_step = 3
    trainer.actor_rollout_wg = _FakeSnapshotWorkerGroup()

    def compute_old_log_prob(self, data):
        del data
        return DataProto(meta_info={"policy_version": self.actor_rollout_wg.current})

    monkeypatch.setattr(PolicyGradientDiffusionTrainerV1, "_compute_old_log_prob", compute_old_log_prob)

    versions = []
    for local_step, current_version in enumerate(("W0", "W1", "W2")):
        trainer.local_trigger_step = local_step
        trainer.actor_rollout_wg.current = current_version
        result = trainer._compute_old_log_prob(DataProto())
        versions.append(result.meta_info["policy_version"])
        assert trainer.actor_rollout_wg.current == current_version

    assert versions == ["W0", "W0", "W0"]
    assert trainer.actor_rollout_wg.snapshots == {}


def test_old_policy_restore_runs_when_inference_fails(monkeypatch):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.parameter_sync_step = 3
    trainer.local_trigger_step = 1
    trainer.actor_rollout_wg = _FakeSnapshotWorkerGroup()
    trainer.actor_rollout_wg.snapshots[0] = "W0"
    trainer.actor_rollout_wg.current = "W1"

    def fail_compute(self, data):
        del self, data
        raise RuntimeError("inference failed")

    monkeypatch.setattr(PolicyGradientDiffusionTrainerV1, "_compute_old_log_prob", fail_compute)

    with pytest.raises(RuntimeError, match="inference failed"):
        trainer._compute_old_log_prob(DataProto())

    assert trainer.actor_rollout_wg.current == "W1"
    assert trainer.actor_rollout_wg.snapshots == {0: "W0"}


def test_failed_current_snapshot_is_cleared(monkeypatch):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.parameter_sync_step = 3
    trainer.local_trigger_step = 1
    trainer.actor_rollout_wg = _FakeSnapshotWorkerGroup()
    trainer.actor_rollout_wg.snapshots[0] = "W0"
    trainer.actor_rollout_wg.current = "W1"

    def fail_partial_save(snapshot_id):
        trainer.actor_rollout_wg.snapshots[snapshot_id] = trainer.actor_rollout_wg.current
        raise RuntimeError("snapshot failed")

    trainer.actor_rollout_wg.save_model_to_cpu = fail_partial_save
    monkeypatch.setattr(
        PolicyGradientDiffusionTrainerV1,
        "_compute_old_log_prob",
        lambda self, data: pytest.fail("inference ran after failed snapshot"),
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        trainer._compute_old_log_prob(DataProto())

    assert trainer.actor_rollout_wg.current == "W1"
    assert trainer.actor_rollout_wg.snapshots == {0: "W0"}


def test_parameter_sync_cycle_releases_base_snapshot_after_failure(monkeypatch):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.parameter_sync_step = 3
    trainer.actor_rollout_wg = _FakeSnapshotWorkerGroup()
    trainer.actor_rollout_wg.snapshots[0] = "W0"

    def fail_step(self, metrics, timing_raw):
        del self, metrics, timing_raw
        raise RuntimeError("actor update failed")

    monkeypatch.setattr(PolicyGradientDiffusionTrainerV1, "step", fail_step)

    with pytest.raises(RuntimeError, match="actor update failed"):
        trainer.step({}, {})

    assert trainer.actor_rollout_wg.snapshots == {}


def test_on_step_end_syncs_every_outer_step():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    sync_steps = []
    trainer.global_steps = 1
    trainer.timing_raw = {}
    trainer.standalone_checkpoint_manager = SimpleNamespace(
        update_weights=lambda global_steps: sync_steps.append(global_steps)
    )
    trainer.sync_compatible = False
    trainer._standalone_paused = False

    trainer.on_step_end()
    trainer.global_steps = 2
    trainer.on_step_end()

    assert sync_steps == [1, 2]


def test_on_step_end_resumes_only_after_successful_sync():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    events = []
    trainer.global_steps = 1
    trainer.timing_raw = {}
    trainer.standalone_checkpoint_manager = SimpleNamespace(
        update_weights=lambda global_steps: events.append(("sync", global_steps))
    )
    trainer.sync_compatible = True
    trainer._standalone_paused = True
    trainer._resume_standalone_generation = lambda: events.append(("resume",))

    trainer.on_step_end()

    assert events == [("sync", 1), ("resume",)]


def test_on_step_end_does_not_resume_after_failed_sync():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.global_steps = 1
    trainer.timing_raw = {}

    def fail_sync(global_steps):
        del global_steps
        raise RuntimeError("sync failed")

    trainer.standalone_checkpoint_manager = SimpleNamespace(update_weights=fail_sync)
    trainer.sync_compatible = True
    trainer._standalone_paused = True
    trainer._resume_standalone_generation = lambda: pytest.fail("generation resumed without synced weights")

    with pytest.raises(RuntimeError, match="sync failed"):
        trainer.on_step_end()


@pytest.mark.parametrize(
    ("strategy", "save_handler_name", "restore_handler_name"),
    [
        ("fsdp", "fsdp1_sharded_save_to_cpu", "fsdp1_sharded_load_from_cpu"),
        ("fsdp2", "fsdp2_sharded_save_to_cpu", "fsdp2_sharded_load_from_cpu"),
        ("veomni", "fsdp2_sharded_save_to_cpu", "fsdp2_sharded_load_from_cpu"),
    ],
)
def test_snapshot_worker_selects_sharded_strategy_handlers(strategy, save_handler_name, restore_handler_name):
    worker = object.__new__(DiffusionDetachActorWorker)
    worker.config = OmegaConf.create({"actor": {"strategy": strategy}})
    worker._strategy_handlers = None

    save_handler, restore_handler = worker._get_strategy_handlers()

    assert save_handler.__name__ == save_handler_name
    assert restore_handler.__name__ == restore_handler_name


def test_snapshot_worker_materializes_and_reoffloads_parameters(monkeypatch):
    worker = object.__new__(DiffusionDetachActorWorker)
    worker.config = OmegaConf.create({"actor": {"strategy": "fsdp"}})
    module = torch.nn.Linear(1, 1, bias=False)
    device_transitions = []

    class _FakeEngine:
        is_param_offload_enabled = True

        def __init__(self):
            self.module = module

        def to(self, device, *, model, optimizer, grad):
            device_transitions.append((device, model, optimizer, grad))

    worker.actor = SimpleNamespace(engine=_FakeEngine())
    worker._strategy_handlers = (
        lambda actor_module: actor_module.weight.detach().clone(),
        lambda actor_module, state: actor_module.weight.data.copy_(state),
    )
    worker.cpu_saved_models = {}
    monkeypatch.setattr(detach_actor_worker_module, "get_device_name", lambda: "cuda")

    worker.save_model_to_cpu(0)
    worker.restore_model_from_cpu(0)

    assert device_transitions == [
        ("cuda", True, False, False),
        ("cpu", True, False, False),
        ("cuda", True, False, False),
        ("cpu", True, False, False),
    ]


def test_snapshot_worker_reoffloads_after_materialization_failure(monkeypatch):
    worker = object.__new__(DiffusionDetachActorWorker)
    worker.config = OmegaConf.create({"actor": {"strategy": "fsdp"}})
    device_transitions = []

    class _FailingEngine:
        is_param_offload_enabled = True
        module = torch.nn.Linear(1, 1, bias=False)

        def to(self, device, *, model, optimizer, grad):
            del model, optimizer, grad
            device_transitions.append(device)
            if device == "cuda":
                raise RuntimeError("materialization failed")

    worker.actor = SimpleNamespace(engine=_FailingEngine())
    worker._strategy_handlers = (lambda module: module, lambda module, state: None)
    worker.cpu_saved_models = {}
    monkeypatch.setattr(detach_actor_worker_module, "get_device_name", lambda: "cuda")

    with pytest.raises(RuntimeError, match="materialization failed"):
        worker.save_model_to_cpu(0)

    assert device_transitions == ["cuda", "cpu"]
    assert worker.cpu_saved_models == {}


def test_fsdp1_snapshot_round_trip_includes_lora_parameters(monkeypatch):
    worker = object.__new__(DiffusionDetachActorWorker)
    worker.config = OmegaConf.create({"actor": {"strategy": "fsdp"}})
    worker._strategy_handlers = None
    worker.cpu_saved_models = {}

    class _LoRAModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
            self.lora_A = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
            self.lora_B = torch.nn.Parameter(torch.tensor([4.0, 5.0]))

    module = _LoRAModule()
    worker.actor = SimpleNamespace(
        engine=SimpleNamespace(module=module, is_param_offload_enabled=False),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    original = {name: param.detach().clone() for name, param in module.named_parameters()}
    worker.save_model_to_cpu(0)
    with torch.no_grad():
        module.lora_A.add_(10)
        module.lora_B.sub_(10)
    worker.restore_model_from_cpu(0)
    worker.clear_cpu_model(0)

    for name, param in module.named_parameters():
        assert torch.equal(param, original[name])
    assert worker.cpu_saved_models == {}
