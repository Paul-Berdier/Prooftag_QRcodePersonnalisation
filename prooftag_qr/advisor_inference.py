from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .parameter_advisor import E026ParameterAdvisor, RecipeCandidate
from .qr import generate_diffqrcoder_qr
from .schemas import LabCampaignCreate, LabMethod, LabPrompt

TERMINAL_CAMPAIGN_STATUSES = {
    "completed",
    "completed_with_errors",
    "cancelled",
    "interrupted",
}
SUCCESSFUL_CAMPAIGN_STATUSES = {"completed", "completed_with_errors"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_identifier(value: Any, *, maximum: int = 70) -> str:
    result = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")
    return (result or "method")[:maximum]


def _normalized_prompt(value: str) -> str:
    return " ".join(value.split()).casefold()


def _finite(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "oui"}:
        return 1.0
    if lowered in {"false", "no", "non"}:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_finite(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = _finite(row.get(name))
        if value is not None:
            return value
    return None


@dataclass(frozen=True, slots=True)
class AdvisorInferencePlan:
    plan_id: str
    payload: str = field(repr=False)
    campaigns: tuple[dict[str, Any], ...]
    predictions: tuple[dict[str, Any], ...]
    public: dict[str, Any]


def _runtime_method(
    candidate: RecipeCandidate,
    *,
    runtime_id: str,
    display_name: str,
) -> dict[str, Any]:
    configuration = deepcopy(candidate.configuration)
    if "legacy_requested_parameters" in configuration:
        raise ValueError(
            f"candidate {candidate.method_id} is legacy-only and cannot be generated"
        )
    configuration.update(
        {
            "id": runtime_id,
            "name": display_name[:200],
            "enabled": True,
        }
    )
    return LabMethod.model_validate(configuration).model_dump(mode="json")


def build_advisor_inference_plan(
    *,
    advisor: E026ParameterAdvisor,
    candidates: Sequence[RecipeCandidate],
    prompts: Sequence[Mapping[str, Any]],
    payload: str,
    advisor_sha256: str,
    prompt_embedding_provider: Callable[[str], Sequence[float]] | None = None,
    seen_prompt_texts: Sequence[str] = (),
    seeds: Sequence[int] = (413_001, 523_001, 631_001),
    top_k: int = 3,
    baseline_method_id: str | None = "diffqrcoder_stage1",
    scan_probability_threshold: float = 0.80,
    error_correction: str = "M",
    qr_context: Mapping[str, Any] | None = None,
) -> AdvisorInferencePlan:
    """Create deterministic per-prompt campaigns from E026 recommendations.

    The clear payload is kept only in the returned in-memory plan. The public plan persisted
    by the runner contains its hash and length, never the payload itself.
    """

    if not candidates:
        raise ValueError("at least one historical recipe candidate is required")
    if not prompts:
        raise ValueError("at least one unseen inference prompt is required")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not advisor_sha256:
        raise ValueError("advisor_sha256 is required for a traceable plan")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("inference seeds must be non-empty and unique")

    context = {
        "qr_version": 3,
        "qr_mask_pattern": 4,
        "qr_module_size": 20,
        "qr_padding_px": 78,
        **dict(qr_context or {}),
    }
    generate_diffqrcoder_qr(
        payload,
        error_correction=error_correction,
        version=int(context["qr_version"]),
        mask_pattern=int(context["qr_mask_pattern"]),
        module_size=int(context["qr_module_size"]),
        border=4,
    )

    validated_prompts = [LabPrompt.model_validate(dict(prompt)) for prompt in prompts]
    if len({item.id for item in validated_prompts}) != len(validated_prompts):
        raise ValueError("inference prompt ids must be unique")
    seen = {_normalized_prompt(value) for value in seen_prompt_texts}
    overlap = [item.id for item in validated_prompts if _normalized_prompt(item.text) in seen]
    if overlap:
        raise ValueError(f"inference prompts already occur in training data: {overlap}")

    baseline = None
    if baseline_method_id:
        matching = [item for item in candidates if item.method_id == baseline_method_id]
        if not matching:
            raise ValueError(f"baseline candidate is absent: {baseline_method_id}")
        baseline = max(matching, key=lambda item: item.observations)

    drafts: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for prompt in validated_prompts:
        embedding = prompt_embedding_provider(prompt.text) if prompt_embedding_provider else None
        ranked = advisor.recommend(
            prompt=prompt.text,
            prompt_embedding=embedding,
            payload_length=len(payload),
            error_correction=error_correction,
            qr_context=context,
            candidates=candidates,
            scan_probability_threshold=scan_probability_threshold,
            limit=len(candidates),
        )
        selected = [item for item in ranked if item.scan_safe][:top_k]
        if not selected:
            raise RuntimeError(
                f"no scan-safe recipe for {prompt.id} at threshold "
                f"{scan_probability_threshold:.2f}"
            )

        methods = []
        selected_signatures = set()
        for recommendation in selected:
            candidate = recommendation.candidate
            selected_signatures.add(candidate.signature)
            runtime_id = (
                f"e026i_r{recommendation.rank:02d}_"
                f"{_safe_identifier(candidate.method_id)}"
            )[:100]
            methods.append(
                _runtime_method(
                    candidate,
                    runtime_id=runtime_id,
                    display_name=(
                        f"E026 advisor rank {recommendation.rank} | {candidate.method_id}"
                    ),
                )
            )
            prediction_rows.append(
                {
                    "prompt_id": prompt.id,
                    "prompt_text": prompt.text,
                    "plan_method_id": runtime_id,
                    "source_method_id": candidate.method_id,
                    "role": "advisor_recommendation",
                    "advisor_rank": recommendation.rank,
                    "candidate_signature": candidate.signature,
                    "candidate_observations": candidate.observations,
                    "scan_safe": recommendation.scan_safe,
                    "predicted_qr_success": recommendation.predicted_qr_success,
                    "predicted_qr_success_lower_bound": (
                        recommendation.qr_success_lower_bound
                    ),
                    "predicted_qr_success_uncertainty": (
                        recommendation.qr_success_uncertainty
                    ),
                    "predicted_qr_tolerance": recommendation.predicted_qr_tolerance,
                    "predicted_clip_aesthetic": recommendation.predicted_clip_aesthetic,
                    "predicted_clip_score": recommendation.predicted_clip_score,
                    "predicted_hpsv2_1": recommendation.predicted_hpsv2_1,
                    "predicted_saturation_risk": (
                        recommendation.predicted_saturation_risk
                    ),
                    "predicted_duration_ms": recommendation.predicted_duration_ms,
                }
            )

        if baseline is not None and baseline.signature not in selected_signatures:
            baseline_prediction = next(
                item for item in ranked if item.candidate.signature == baseline.signature
            )
            runtime_id = f"e026i_baseline_{_safe_identifier(baseline.method_id)}"[:100]
            methods.append(
                _runtime_method(
                    baseline,
                    runtime_id=runtime_id,
                    display_name=f"E026 fixed baseline | {baseline.method_id}",
                )
            )
            prediction_rows.append(
                {
                    "prompt_id": prompt.id,
                    "prompt_text": prompt.text,
                    "plan_method_id": runtime_id,
                    "source_method_id": baseline.method_id,
                    "role": "fixed_baseline",
                    "advisor_rank": baseline_prediction.rank,
                    "candidate_signature": baseline.signature,
                    "candidate_observations": baseline.observations,
                    "scan_safe": baseline_prediction.scan_safe,
                    "predicted_qr_success": baseline_prediction.predicted_qr_success,
                    "predicted_qr_success_lower_bound": (
                        baseline_prediction.qr_success_lower_bound
                    ),
                    "predicted_qr_success_uncertainty": (
                        baseline_prediction.qr_success_uncertainty
                    ),
                    "predicted_qr_tolerance": baseline_prediction.predicted_qr_tolerance,
                    "predicted_clip_aesthetic": (
                        baseline_prediction.predicted_clip_aesthetic
                    ),
                    "predicted_clip_score": baseline_prediction.predicted_clip_score,
                    "predicted_hpsv2_1": baseline_prediction.predicted_hpsv2_1,
                    "predicted_saturation_risk": (
                        baseline_prediction.predicted_saturation_risk
                    ),
                    "predicted_duration_ms": baseline_prediction.predicted_duration_ms,
                }
            )

        drafts.append(
            {
                "prompt": prompt.model_dump(mode="json"),
                "methods": methods,
            }
        )

    plan_core = {
        "protocol": "e026i-v1",
        "advisor_sha256": advisor_sha256,
        "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "payload_length": len(payload),
        "error_correction": error_correction,
        "qr_context": context,
        "scan_probability_threshold": scan_probability_threshold,
        "top_k": top_k,
        "baseline_method_id": baseline_method_id,
        "seeds": list(normalized_seeds),
        "prompts": [item.model_dump(mode="json") for item in validated_prompts],
        "selection": [
            {
                "prompt_id": item["prompt"]["id"],
                "methods": [
                    {
                        "id": method["id"],
                        "source_signature": next(
                            row["candidate_signature"]
                            for row in prediction_rows
                            if row["prompt_id"] == item["prompt"]["id"]
                            and row["plan_method_id"] == method["id"]
                        ),
                    }
                    for method in item["methods"]
                ],
            }
            for item in drafts
        ],
        "predictions": prediction_rows,
    }
    plan_id = hashlib.sha256(_canonical_json(plan_core).encode("utf-8")).hexdigest()[:16]
    campaigns = []
    public_campaigns = []
    for index, draft in enumerate(drafts, start=1):
        base_name = f"E026I {plan_id} {index:02d} {draft['prompt']['id']}"
        request = {
            "name": base_name,
            "payload": payload,
            "error_correction": error_correction,
            "prompts": [draft["prompt"]],
            "seeds": list(normalized_seeds),
            "methods": draft["methods"],
            "max_attempts": 1,
        }
        validated = LabCampaignCreate.model_validate(request).model_dump(mode="json")
        campaigns.append(validated)
        public_campaigns.append(
            {
                "name": base_name,
                "prompt": draft["prompt"],
                "seeds": list(normalized_seeds),
                "methods": draft["methods"],
                "trials": len(normalized_seeds) * len(draft["methods"]),
            }
        )
    public = {
        **plan_core,
        "plan_id": plan_id,
        "campaign_count": len(campaigns),
        "trial_count": sum(item["trials"] for item in public_campaigns),
        "campaigns": public_campaigns,
    }
    return AdvisorInferencePlan(
        plan_id=plan_id,
        payload=payload,
        campaigns=tuple(campaigns),
        predictions=tuple(prediction_rows),
        public=public,
    )


class AdvisorInferenceRunner:
    """Run advisor-selected laboratory campaigns with power-cut-safe checkpoints."""

    def __init__(
        self,
        *,
        plan: AdvisorInferencePlan,
        api_url: str,
        output_root: Path,
        poll_seconds: float = 15.0,
        maximum_campaign_attempts: int = 2,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.plan = plan
        self.api_url = api_url.rstrip("/")
        self.poll_seconds = poll_seconds
        self.maximum_campaign_attempts = maximum_campaign_attempts
        self.progress_callback = progress_callback
        self.output_dir = Path(output_root) / plan.plan_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir = self.output_dir / "exports"
        self.exports_dir.mkdir(exist_ok=True)
        self.state_path = self.output_dir / "state.json"
        self.plan_path = self.output_dir / "plan-redacted.json"
        self.predictions_path = self.output_dir / "advisor-predictions.jsonl"
        self._write_or_verify_plan()
        self.state = self._load_state()

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def _write_or_verify_plan(self) -> None:
        if self.plan_path.exists():
            saved = json.loads(self.plan_path.read_text(encoding="utf-8"))
            if saved != self.plan.public:
                raise RuntimeError("stored inference plan differs from the requested plan")
        else:
            self._atomic_json(self.plan_path, self.plan.public)
        prediction_text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in self.plan.predictions
        )
        if self.predictions_path.exists():
            if self.predictions_path.read_text(encoding="utf-8") != prediction_text:
                raise RuntimeError("stored advisor predictions differ from the plan")
        else:
            temporary = self.predictions_path.with_suffix(".jsonl.tmp")
            temporary.write_text(prediction_text, encoding="utf-8")
            temporary.replace(self.predictions_path)

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("plan_id") != self.plan.plan_id:
                raise RuntimeError("stored inference state belongs to another plan")
            return state
        state = {
            "version": 1,
            "plan_id": self.plan.plan_id,
            "status": "running",
            "completed_campaigns": [],
            "failed_campaigns": [],
            "attempts": {},
            "active_campaign": None,
            "history": [],
            "created_at": time.time(),
        }
        self._atomic_json(self.state_path, state)
        return state

    def _save_state(self) -> None:
        self.state["updated_at"] = time.time()
        self._atomic_json(self.state_path, self.state)

    def _notify(self, event: str, **values: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(
                {"event": event, "timestamp": time.time(), **values}
            )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        raw: bool = False,
    ) -> Any:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.api_url}{path}", data=data, headers=headers, method=method
        )
        last_error: Exception | None = None
        for retry in range(8):
            try:
                with urlopen(request, timeout=120) as response:
                    body = response.read()
                    return body if raw else json.loads(body.decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                time.sleep(min(60.0, 2.0 ** min(retry, 5)))
        raise RuntimeError(f"inference API unavailable after retries: {last_error}")

    def _acquire_lock(self):
        handle = (self.output_dir / "runner.lock").open("a+", encoding="utf-8")
        if os.name != "nt":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RuntimeError(
                    f"another inference runner owns plan {self.plan.plan_id}"
                ) from exc
        return handle

    @staticmethod
    def _release_lock(handle) -> None:
        if os.name != "nt":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    def _find_campaign(self, name: str) -> dict[str, Any] | None:
        campaigns = self._request("GET", "/v1/lab/campaigns?limit=500")
        matching = [item for item in campaigns if item.get("name") == name]
        return max(matching, key=lambda item: item.get("created_at", "")) if matching else None

    def _wait_for_foreign_campaigns(self, own_id: str | None = None) -> None:
        while True:
            campaigns = self._request("GET", "/v1/lab/campaigns?limit=100")
            active = [
                item
                for item in campaigns
                if item.get("status") in {"queued", "running"}
                and item.get("id") != own_id
            ]
            if not active:
                return
            self._notify(
                "waiting_foreign_campaign",
                campaigns=[item.get("id") for item in active],
            )
            time.sleep(self.poll_seconds)

    def _export(self, index: int, attempt: int, campaign_id: str) -> Path:
        path = self.exports_dir / (
            f"prompt-{index + 1:02d}-attempt-{attempt:02d}-{campaign_id}.csv"
        )
        if path.is_file() and path.stat().st_size > 0:
            return path
        body = self._request(
            "GET", f"/v1/lab/campaigns/{campaign_id}/results.csv", raw=True
        )
        temporary = path.with_suffix(".csv.tmp")
        temporary.write_bytes(body)
        temporary.replace(path)
        return path

    def _wait_campaign(self, index: int, attempt: int, campaign_id: str) -> dict[str, Any]:
        last_progress = None
        while True:
            campaign = self._request("GET", f"/v1/lab/campaigns/{campaign_id}")
            progress = (
                campaign.get("status"),
                campaign.get("completed_trials"),
                campaign.get("total_trials"),
                campaign.get("accepted_trials"),
            )
            if progress != last_progress:
                print(
                    f"inference prompt={index + 1:02d}/{len(self.plan.campaigns):02d} "
                    f"attempt={attempt} status={progress[0]} "
                    f"trials={progress[1]}/{progress[2]} accepted={progress[3]}"
                )
                last_progress = progress
            current_trial = next(
                (
                    item
                    for item in campaign.get("trials", [])
                    if item.get("status") == "running"
                ),
                None,
            )
            self._notify(
                "inference_progress",
                plan_id=self.plan.plan_id,
                prompt_number=index + 1,
                prompt_count=len(self.plan.campaigns),
                attempt=attempt,
                campaign_id=campaign_id,
                status=progress[0],
                completed_trials=progress[1],
                total_trials=progress[2],
                accepted_trials=progress[3],
                current_prompt_id=(current_trial or {}).get("prompt_id"),
                current_method_id=(current_trial or {}).get("method_id"),
                current_seed=(current_trial or {}).get("seed"),
            )
            if campaign.get("status") in TERMINAL_CAMPAIGN_STATUSES:
                export_path = self._export(index, attempt, campaign_id)
                campaign["export_path"] = str(export_path)
                return campaign
            time.sleep(self.poll_seconds)

    def run(self) -> dict[str, Any]:
        handle = self._acquire_lock()
        try:
            for index, base_request in enumerate(self.plan.campaigns):
                if index in self.state["completed_campaigns"]:
                    continue
                succeeded = False
                while (
                    int(self.state["attempts"].get(str(index), 0))
                    < self.maximum_campaign_attempts
                ):
                    active = self.state.get("active_campaign")
                    if active and int(active["index"]) == index:
                        attempt = int(active["attempt"])
                        campaign_id = str(active["campaign_id"])
                    else:
                        attempt = int(self.state["attempts"].get(str(index), 0)) + 1
                        request_payload = deepcopy(base_request)
                        request_payload["name"] = f"{base_request['name']} a{attempt:02d}"
                        existing = self._find_campaign(request_payload["name"])
                        if existing is not None:
                            campaign = existing
                        else:
                            self._wait_for_foreign_campaigns()
                            campaign = self._request(
                                "POST", "/v1/lab/campaigns", request_payload
                            )
                        campaign_id = str(campaign["id"])
                        self.state["attempts"][str(index)] = attempt
                        self.state["active_campaign"] = {
                            "index": index,
                            "attempt": attempt,
                            "campaign_id": campaign_id,
                            "name": request_payload["name"],
                        }
                        self._save_state()
                    campaign = self._wait_campaign(index, attempt, campaign_id)
                    status = str(campaign["status"])
                    self.state["history"].append(
                        {
                            "index": index,
                            "attempt": attempt,
                            "campaign_id": campaign_id,
                            "status": status,
                            "export_path": campaign.get("export_path"),
                        }
                    )
                    self.state["active_campaign"] = None
                    if status in SUCCESSFUL_CAMPAIGN_STATUSES:
                        self.state["completed_campaigns"].append(index)
                        succeeded = True
                    self._save_state()
                    if succeeded:
                        break
                if not succeeded and index not in self.state["failed_campaigns"]:
                    self.state["failed_campaigns"].append(index)
                    self._save_state()
            self.state["status"] = (
                "completed"
                if not self.state["failed_campaigns"]
                else "completed_with_errors"
            )
            self._save_state()
            return self.summary()
        finally:
            self._release_lock(handle)

    def summary(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan.plan_id,
            "status": self.state["status"],
            "campaigns": len(self.plan.campaigns),
            "completed_campaigns": len(self.state["completed_campaigns"]),
            "failed_campaigns": list(self.state["failed_campaigns"]),
            "exports": len(list(self.exports_dir.glob("*.csv"))),
            "state_path": str(self.state_path),
        }


def load_advisor_inference_results(output_dir: Path) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    predictions = {}
    prediction_path = output_dir / "advisor-predictions.jsonl"
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            predictions[(row["prompt_id"], row["plan_method_id"])] = row

    trials: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "exports").glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                row["_source_file"] = str(path)
                trials[str(row.get("trial_id"))] = row

    results = []
    for row in trials.values():
        key = (row.get("prompt_id"), row.get("method_id"))
        prediction = predictions.get(key)
        if prediction is None:
            continue
        results.append(
            {
                **prediction,
                "trial_id": row.get("trial_id"),
                "campaign_id": row.get("campaign_id"),
                "method_id": row.get("method_id"),
                "output_variant": row.get("selected_variant")
                or row.get("output_variant_requested"),
                "seed": _finite(row.get("seed")),
                "status": row.get("status"),
                "generation_run_id": row.get("generation_run_id"),
                "qr_success": _first_finite(
                    row, ("quality_qr_verify_any_exact", "exact_payload_match")
                ),
                "qr_tolerance": _first_finite(
                    row, ("quality_qr_verify_tolerance_score", "scan_pass_rate")
                ),
                "clip_aesthetic": _finite(row.get("quality_clip_aesthetic")),
                "clip_score": _finite(row.get("quality_clip_score")),
                "hpsv2_1": _finite(row.get("quality_hpsv2_1")),
                "saturation_risk": max(
                    value
                    for value in (
                        _finite(row.get("quality_high_saturation_pixel_ratio")),
                        _finite(row.get("quality_rgb_clipped_channel_ratio")),
                        0.0,
                    )
                    if value is not None
                ),
                "duration_ms": _first_finite(row, ("total_ms", "generation_ms")),
                "module_error_rate": _finite(row.get("module_error_rate")),
                "error": row.get("error"),
                "section": "advisor_inference",
                "predicted_qr_probability": prediction.get("predicted_qr_success"),
                "_source_file": row.get("_source_file"),
            }
        )
    return sorted(
        results,
        key=lambda item: (
            str(item["prompt_id"]),
            int(item.get("advisor_rank") or 999),
            str(item.get("role")),
            float(item.get("seed") or 0),
        ),
    )


def select_advisor_inference_winners(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    winners = []
    for prompt_id in sorted({str(item["prompt_id"]) for item in entries}):
        candidates = [dict(item) for item in entries if item["prompt_id"] == prompt_id]
        if not candidates:
            continue
        winners.append(
            max(
                candidates,
                key=lambda item: (
                    item.get("qr_success") or 0.0,
                    item.get("qr_tolerance") or 0.0,
                    item.get("hpsv2_1") or -math.inf,
                    item.get("clip_aesthetic") or -math.inf,
                    item.get("clip_score") or -math.inf,
                    -(item.get("saturation_risk") or 0.0),
                    item.get("role") == "advisor_recommendation",
                    -(item.get("advisor_rank") or 999),
                ),
            )
        )
    return winners
