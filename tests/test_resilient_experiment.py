from pathlib import Path

import pytest

from prooftag_qr.resilient_experiment import (
    ArtifactPromotionError,
    ContractMismatchError,
    ResilientTaskStore,
    atomic_write_text,
    classify_failure,
    promote_attempt,
)


def test_task_store_retries_transient_but_blocks_oom(tmp_path: Path):
    store = ResilientTaskStore(tmp_path / "state.sqlite")
    store.register_run(
        run_id="run",
        plan={"schema": "v1"},
        source_commit="a" * 40,
    )
    store.register_tasks(
        run_id="run",
        tasks=[
            {
                "task_id": "generation",
                "kind": "generation",
                "spec": {"seed": 1},
                "priority": 1,
                "max_attempts": 2,
            },
            {
                "task_id": "training",
                "kind": "training",
                "spec": {"batch_size": 32},
                "priority": 2,
                "max_attempts": 3,
            },
        ],
    )

    first = store.claim_next(run_id="run", worker_id="w1")
    assert first is not None and first.task_id == "generation"
    decision = store.fail(first, TimeoutError("temporary timeout"), retry_delay_seconds=0)
    assert decision.kind == "transient"
    assert decision.retryable is True

    retry = store.claim_next(run_id="run", worker_id="w2")
    assert retry is not None and retry.task_id == "generation"
    assert retry.attempt_no == 2
    store.complete(retry, result={"image": "final.png"})

    training = store.claim_next(run_id="run", worker_id="w1")
    assert training is not None and training.task_id == "training"
    oom = store.fail(training, RuntimeError("CUDA out of memory"))
    assert oom.kind == "resource"
    assert oom.retryable is False
    assert oom.operator_action_required is True

    rows = {row["task_id"]: row for row in store.task_rows("run")}
    assert rows["generation"]["status"] == "succeeded"
    assert rows["training"]["status"] == "blocked"


def test_stale_lease_is_recovered_without_deleting_attempt(tmp_path: Path):
    store = ResilientTaskStore(tmp_path / "state.sqlite")
    store.register_run(run_id="run", plan={"schema": "v1"}, source_commit="b" * 40)
    store.register_tasks(
        run_id="run",
        tasks=[
            {
                "task_id": "task",
                "kind": "generation",
                "spec": {"seed": 2},
                "max_attempts": 2,
            }
        ],
    )
    claimed = store.claim_next(run_id="run", worker_id="dead", lease_seconds=0)
    assert claimed is not None
    assert store.recover_stale(run_id="run", force=True) == ["task"]
    retried = store.claim_next(run_id="run", worker_id="alive")
    assert retried is not None and retried.attempt_no == 2


def test_contract_hash_prevents_silent_task_mutation(tmp_path: Path):
    store = ResilientTaskStore(tmp_path / "state.sqlite")
    store.register_run(run_id="run", plan={"schema": "v1"}, source_commit="c" * 40)
    store.register_tasks(
        run_id="run",
        tasks=[{"task_id": "same", "spec": {"gamma": 500}}],
    )
    with pytest.raises(ContractMismatchError):
        store.register_tasks(
            run_id="run",
            tasks=[{"task_id": "same", "spec": {"gamma": 1000}}],
        )


def test_promotion_is_atomic_and_missing_files_are_rejected(tmp_path: Path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    atomic_write_text(attempt / "checkpoint.json", "{}\n")
    with pytest.raises(ArtifactPromotionError):
        promote_attempt(
            attempt_dir=attempt,
            final_dir=tmp_path / "final",
            required_files=("checkpoint.json", "result.json"),
            metadata={"kind": "training"},
        )
    assert attempt.is_dir()
    assert not (tmp_path / "final").exists()

    atomic_write_text(attempt / "result.json", "{}\n")
    manifest = promote_attempt(
        attempt_dir=attempt,
        final_dir=tmp_path / "final",
        required_files=("checkpoint.json", "result.json"),
        metadata={"kind": "training"},
    )
    assert not attempt.exists()
    assert (tmp_path / "final/PROMOTION_MANIFEST.json").is_file()
    assert len(manifest["manifest_hash"]) == 64


def test_failure_classification_is_fail_closed():
    assert classify_failure("CUDA out of memory").kind == "resource"
    assert classify_failure("checksum mismatch").kind == "deterministic"
    assert classify_failure("HTTP 503").retryable is True
