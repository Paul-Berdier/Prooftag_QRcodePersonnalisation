from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .lab import laboratory_profiles
from .qr import generate_diffqrcoder_qr
from .schemas import LabCampaignCreate

TERMINAL_CAMPAIGN_STATUSES = {
    "completed",
    "completed_with_errors",
    "cancelled",
    "interrupted",
}


def build_week_prompts(count: int = 300) -> list[dict[str, str]]:
    """Build a deterministic, diverse prompt bank without calling an external model."""

    subjects = [
        "amber pear",
        "red paper boat",
        "porcelain cup",
        "brass key",
        "moon-shaped lamp",
        "kingfisher",
        "glass greenhouse",
        "clockwork fox",
        "turquoise peacock",
        "ceramic whale",
        "crystal droplet",
        "mechanical orrery",
        "embroidered moth",
        "floating library",
        "bonsai observatory",
        "coral archive",
        "paper city",
        "mycelium cube",
        "silver pomegranate",
        "opal jellyfish",
        "wooden automaton",
        "astronomical vase",
        "botanical cabinet",
        "miniature harbor",
        "origami cathedral",
        "bioluminescent violin",
        "mosaic garden",
        "recursive staircase",
        "lacquered beetle",
        "travelling apothecary",
    ]
    settings = [
        "on charcoal slate",
        "inside an indigo gallery",
        "in a tiled Mediterranean courtyard",
        "beneath a glass railway roof",
        "in a misty alpine valley",
        "inside an underwater museum",
        "at a rainy night market",
        "within an illuminated manuscript forest",
        "on burgundy velvet",
        "in a quiet Japanese workshop",
        "above a sea of clouds",
        "inside an Art Deco hotel",
        "in a moonlit Persian garden",
        "on a pale laboratory table",
        "within an impossible opera house",
        "inside a translucent ice cave",
        "in a sunlit botanical conservatory",
        "above a dense retro-futurist city",
    ]
    styles = [
        "minimalist studio photograph",
        "elegant editorial photograph",
        "cinematic wide-angle scene",
        "intricate Art Nouveau mosaic",
        "surreal macro photograph",
        "detailed gouache illustration",
        "restrained Japanese woodblock print",
        "luminous oil painting",
        "isometric paper-cut theatre",
        "museum-quality product photograph",
        "embroidered textile artwork",
        "retro-futurist architectural rendering",
    ]
    lightings = [
        "soft morning light",
        "dramatic museum lighting",
        "blue hour reflections",
        "warm window light",
        "volumetric rays",
        "gentle overcast light",
        "neon reflections in rain",
        "high-contrast stage lighting",
        "pearlescent twilight",
        "clean diffused studio light",
    ]
    details = [
        "simple composition and generous negative space",
        "balanced geometry and restrained colors",
        "fine botanical details and natural textures",
        "layered depth with a clear central subject",
        "ornamental borders and repeating organic shapes",
        "tiny figures and richly articulated architecture",
        "unexpected scale and physically impossible perspective",
        "translucent materials and complex reflections",
        "dense craftsmanship without letters or typography",
        "quiet atmosphere with a coherent limited palette",
    ]
    prompts = []
    for index in range(count):
        subject = subjects[index % len(subjects)]
        setting = settings[(index * 7 + index // len(subjects)) % len(settings)]
        style = styles[(index * 5 + index // 11) % len(styles)]
        lighting = lightings[(index * 3 + index // 13) % len(lightings)]
        detail = details[(index * 9 + index // 17) % len(details)]
        family = ("simple", "scene", "detailed", "atypical")[index % 4]
        prompts.append(
            {
                "id": f"e026w_{family}_{index + 1:03d}",
                "text": (f"A {subject} {setting}, {style}, {lighting}, {detail}."),
                "negative_prompt": "text, letters, watermark, logo, barcode",
            }
        )
    if len({item["text"] for item in prompts}) != count:
        raise RuntimeError("week prompt generator produced duplicates")
    return prompts


def _profile(profile_id: str, profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = deepcopy(profiles[profile_id])
    result["enabled"] = True
    result.pop("description", None)
    return result


def build_week_methods() -> list[dict[str, Any]]:
    profiles = {item["id"]: item for item in laboratory_profiles()}
    method_ids = [
        "diffqrcoder_stage1",
        "diffqrcoder_srpg",
        "diffqrcoder_paper_srpg",
        "diffqrcoder_srmpgd",
        "diffqrcoder_srmpgd_robust",
        "diffqrcoder_srpg_s035",
        "diffqrcoder_srpg_s050",
        "diffqrcoder_srpg_s080",
        "diffqrcoder_qart_srpg",
    ]
    methods = [_profile(profile_id, profiles) for profile_id in method_ids]

    def srpg_variant(
        identifier: str,
        name: str,
        *,
        generation: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = _profile("diffqrcoder_srpg", profiles)
        result["id"] = identifier
        result["name"] = name
        result["generation"].update(generation or {})
        result["tools"]["settings"].update(settings or {})
        return result

    methods.extend(
        [
            srpg_variant(
                "e026w_srpg_q250_pg1",
                "E026W SRPG QR250 PG1",
                generation={"steps": 30, "guidance_scale": 5.5, "controlnet_scale": 1.15},
                settings={
                    "srpg_qr_weight": 250.0,
                    "srpg_perceptual_weight": 1.0,
                    "diffqrcoder_stage2_strength": 0.50,
                },
            ),
            srpg_variant(
                "e026w_srpg_q750_pg1",
                "E026W SRPG QR750 PG1",
                generation={"steps": 50, "guidance_scale": 9.0, "controlnet_scale": 1.55},
                settings={
                    "srpg_qr_weight": 750.0,
                    "srpg_perceptual_weight": 1.0,
                    "diffqrcoder_stage2_strength": 0.80,
                },
            ),
            srpg_variant(
                "e026w_srpg_s32_c110",
                "E026W SRPG 32 steps control 1.10",
                settings={
                    "srpg_steps": 32,
                    "srpg_controlnet_scale": 1.10,
                    "srpg_qr_weight": 400.0,
                    "srpg_perceptual_weight": 3.0,
                    "diffqrcoder_stage2_strength": 0.50,
                },
            ),
            srpg_variant(
                "e026w_srpg_s50_c150",
                "E026W SRPG 50 steps control 1.50",
                settings={
                    "srpg_steps": 50,
                    "srpg_controlnet_scale": 1.50,
                    "srpg_qr_weight": 650.0,
                    "srpg_perceptual_weight": 1.5,
                    "diffqrcoder_stage2_strength": 0.80,
                },
            ),
        ]
    )

    def srmpgd_variant(
        identifier: str,
        gamma: float,
        iterations: int,
        lpips_weight: float,
    ) -> dict[str, Any]:
        result = _profile("diffqrcoder_srmpgd", profiles)
        result["id"] = identifier
        result["name"] = f"E026W SR-MPGD gamma {gamma:g} iter {iterations} LPIPS {lpips_weight:g}"
        settings = result["tools"]["settings"]
        settings["srmpgd_step_size"] = gamma
        settings["srmpgd_max_iterations"] = iterations
        settings["srmpgd_lpips_weight"] = lpips_weight
        return result

    methods.extend(
        [
            srmpgd_variant("e026w_srmpgd_g30_i4_l10", 30.0, 4, 0.10),
            srmpgd_variant("e026w_srmpgd_g300_i4_l10", 300.0, 4, 0.10),
            srmpgd_variant("e026w_srmpgd_g100_i8_l25", 100.0, 8, 0.25),
        ]
    )
    if len(methods) != 16 or len({item["id"] for item in methods}) != len(methods):
        raise RuntimeError("E026 week method catalogue must contain 16 unique recipes")
    return methods


def build_week_batches(
    payload: str,
    *,
    prompt_count: int = 300,
    prompts_per_batch: int = 10,
    seeds: tuple[int, ...] = (113_001, 223_001, 337_001),
) -> list[dict[str, Any]]:
    # Fail before touching Kubernetes if the real payload cannot use the exact
    # DiffQRCoder geometry exercised by every campaign in this plan.
    generate_diffqrcoder_qr(
        payload,
        error_correction="M",
        version=3,
        mask_pattern=4,
        module_size=20,
        border=4,
    )
    prompts = build_week_prompts(prompt_count)
    methods = build_week_methods()
    batches = []
    for offset in range(0, len(prompts), prompts_per_batch):
        selected = prompts[offset : offset + prompts_per_batch]
        batch_index = len(batches) + 1
        request = {
            "name": f"E026W unattended batch {batch_index:02d}",
            "payload": payload,
            "error_correction": "M",
            "prompts": selected,
            "seeds": list(seeds),
            "methods": methods,
            "max_attempts": 1,
        }
        # Validate the exact API contract now, before a week-long run starts.
        LabCampaignCreate.model_validate(request)
        batches.append(request)
    return batches


class WeekCampaignRunner:
    def __init__(
        self,
        *,
        api_url: str,
        payload: str,
        output_root: Path,
        duration_hours: float,
        minimum_free_gib: float,
        poll_seconds: float,
        maximum_campaign_attempts: int = 3,
        prompt_count: int = 300,
        prompts_per_batch: int = 10,
        seeds: tuple[int, ...] = (113_001, 223_001, 337_001),
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.payload = payload
        self.duration_hours = duration_hours
        self.minimum_free_gib = minimum_free_gib
        self.poll_seconds = poll_seconds
        self.maximum_campaign_attempts = maximum_campaign_attempts
        self.progress_callback = progress_callback
        self.batches = build_week_batches(
            payload,
            prompt_count=prompt_count,
            prompts_per_batch=prompts_per_batch,
            seeds=seeds,
        )
        plan_public = {
            "protocol": "e026w-v1",
            "prompt_count": sum(len(item["prompts"]) for item in self.batches),
            "prompt_bank_sha256": hashlib.sha256(
                json.dumps(
                    [prompt for batch in self.batches for prompt in batch["prompts"]],
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "batch_count": len(self.batches),
            "methods": self.batches[0]["methods"],
            "seeds": self.batches[0]["seeds"],
            "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "payload_length": len(payload),
        }
        self.plan_id = hashlib.sha256(
            json.dumps(plan_public, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        self.output_dir = output_root / self.plan_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir = self.output_dir / "exports"
        self.exports_dir.mkdir(exist_ok=True)
        self.state_path = self.output_dir / "state.json"
        self.plan_path = self.output_dir / "plan-redacted.json"
        if not self.plan_path.exists():
            self._atomic_json(self.plan_path, plan_public)
        self.state = self._load_state()
        self.stop_requested = False
        self.active_campaign_id: str | None = self.state.get("active_campaign_id")

    def _notify(self, event: str, **values: Any) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(
                {
                    "event": event,
                    "timestamp": datetime.now(UTC).isoformat(),
                    **values,
                }
            )
        except Exception as exc:  # pragma: no cover - notebook display is best effort
            print(f"Progress callback failed: {type(exc).__name__}: {exc}")

    def _acquire_runner_lock(self):
        lock_path = self.output_dir / "runner.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        if os.name != "nt":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RuntimeError(
                    f"another E026 runner already owns plan {self.plan_id}; "
                    "do not run the notebook and Kubernetes Job concurrently"
                ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "plan_id": self.plan_id,
                    "pid": os.getpid(),
                    "acquired_at": datetime.now(UTC).isoformat(),
                }
            )
        )
        handle.flush()
        return handle

    @staticmethod
    def _release_runner_lock(handle) -> None:
        if os.name != "nt":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("plan_id") != self.plan_id:
                raise RuntimeError("stored E026 plan does not match the current plan")
            return state
        started = datetime.now(UTC)
        state = {
            "version": 1,
            "plan_id": self.plan_id,
            "started_at": started.isoformat(),
            "deadline": (started + timedelta(hours=self.duration_hours)).isoformat(),
            "active_campaign_id": None,
            "completed_batches": [],
            "batch_attempts": {},
            "campaigns": [],
            "status": "running",
        }
        self._atomic_json(self.state_path, state)
        return state

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _save_state(self) -> None:
        self.state["active_campaign_id"] = self.active_campaign_id
        self.state["updated_at"] = datetime.now(UTC).isoformat()
        self._atomic_json(self.state_path, self.state)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        raw: bool = False,
        respect_stop: bool = True,
    ) -> Any:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(f"{self.api_url}{path}", data=data, headers=headers, method=method)
        last_error: Exception | None = None
        for retry in range(8):
            if self.stop_requested and respect_stop:
                raise InterruptedError("stop requested")
            try:
                with urlopen(request, timeout=120) as response:
                    body = response.read()
                    return body if raw else json.loads(body.decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                delay = min(60, 2 ** min(retry, 5))
                print(f"HTTP retry {retry + 1}/8 in {delay}s: {type(exc).__name__}: {exc}")
                time.sleep(delay)
        raise RuntimeError(f"API unavailable after retries: {last_error}")

    def _deadline_reached(self) -> bool:
        deadline = datetime.fromisoformat(self.state["deadline"])
        return datetime.now(UTC) >= deadline

    def _disk_guard_ok(self) -> bool:
        free_gib = shutil.disk_usage(self.output_dir).free / 2**30
        self.state["last_free_gib"] = round(free_gib, 3)
        self._save_state()
        if free_gib < self.minimum_free_gib:
            print(f"DISK GUARD: {free_gib:.2f} GiB free < {self.minimum_free_gib:.2f} GiB")
            self.state["status"] = "stopped_disk_guard"
            self._save_state()
            return False
        return True

    def _cancel_active(self) -> None:
        if not self.active_campaign_id:
            return
        try:
            self._request(
                "POST",
                f"/v1/lab/campaigns/{self.active_campaign_id}/cancel",
                respect_stop=False,
            )
        except Exception as exc:  # pragma: no cover - best effort during SIGTERM
            print(f"Unable to cancel active campaign: {exc}")

    def request_stop(self, *_: Any) -> None:
        if self.stop_requested:
            return
        print("Stop requested; cancelling the active campaign before exit.")
        self.stop_requested = True
        self._cancel_active()

    def _wait_for_foreign_campaigns(self) -> None:
        while not self.stop_requested and not self._deadline_reached():
            campaigns = self._request("GET", "/v1/lab/campaigns?limit=100")
            active = [
                item
                for item in campaigns
                if item["status"] in {"queued", "running"} and item["id"] != self.active_campaign_id
            ]
            if not active:
                return
            print(
                "Waiting for existing laboratory campaign(s):",
                [(item["id"], item["name"], item["status"]) for item in active],
            )
            self._notify(
                "waiting_foreign_campaign",
                batch_count=len(self.batches),
                completed_batches=len(self.state["completed_batches"]),
                foreign_campaigns=[
                    {"id": item["id"], "name": item["name"], "status": item["status"]}
                    for item in active
                ],
            )
            time.sleep(self.poll_seconds)

    def _export(self, batch_index: int, attempt: int, campaign_id: str) -> Path:
        body = self._request(
            "GET",
            f"/v1/lab/campaigns/{campaign_id}/results.csv",
            raw=True,
            respect_stop=False,
        )
        path = self.exports_dir / (
            f"batch-{batch_index + 1:02d}-attempt-{attempt:02d}-{campaign_id}.csv"
        )
        path.write_bytes(body)
        return path

    def _wait_campaign(self, batch_index: int, attempt: int, campaign_id: str) -> str:
        last_progress = None
        while not self.stop_requested and not self._deadline_reached():
            campaign = self._request("GET", f"/v1/lab/campaigns/{campaign_id}")
            current_trial = next(
                (item for item in campaign.get("trials", []) if item["status"] == "running"),
                None,
            )
            progress = (
                campaign["status"],
                campaign["completed_trials"],
                campaign["total_trials"],
                campaign["accepted_trials"],
            )
            if progress != last_progress:
                print(
                    f"batch={batch_index + 1:02d} attempt={attempt} "
                    f"status={progress[0]} trials={progress[1]}/{progress[2]} "
                    f"accepted={progress[3]}"
                )
                last_progress = progress
            # Emit a heartbeat even while one GPU trial is still running. The notebook
            # dashboard therefore proves that polling is alive instead of appearing frozen.
            self._notify(
                "campaign_progress",
                batch_index=batch_index,
                batch_number=batch_index + 1,
                batch_count=len(self.batches),
                attempt=attempt,
                campaign_id=campaign_id,
                status=progress[0],
                completed_trials=progress[1],
                total_trials=progress[2],
                accepted_trials=progress[3],
                completed_batches=len(self.state["completed_batches"]),
                current_prompt_id=current_trial["prompt_id"] if current_trial else None,
                current_method_id=current_trial["method_id"] if current_trial else None,
                current_seed=current_trial["seed"] if current_trial else None,
            )
            if campaign["status"] in TERMINAL_CAMPAIGN_STATUSES:
                exported = self._export(batch_index, attempt, campaign_id)
                print("Exported:", exported)
                self._notify(
                    "campaign_exported",
                    batch_index=batch_index,
                    batch_number=batch_index + 1,
                    batch_count=len(self.batches),
                    campaign_id=campaign_id,
                    status=campaign["status"],
                    export_path=str(exported),
                )
                return str(campaign["status"])
            time.sleep(self.poll_seconds)
        reason = "deadline" if self._deadline_reached() else "cancelled"
        self._cancel_active()
        # Preserve the completed rows of the interrupted batch. Only that batch may be
        # retried later; every prior export and completed batch remains untouched.
        for _ in range(30):
            try:
                campaign = self._request(
                    "GET",
                    f"/v1/lab/campaigns/{campaign_id}",
                    respect_stop=False,
                )
                if campaign["status"] in TERMINAL_CAMPAIGN_STATUSES:
                    exported = self._export(batch_index, attempt, campaign_id)
                    self._notify(
                        "campaign_partial_exported",
                        batch_index=batch_index,
                        batch_number=batch_index + 1,
                        batch_count=len(self.batches),
                        campaign_id=campaign_id,
                        status=campaign["status"],
                        completed_trials=campaign["completed_trials"],
                        total_trials=campaign["total_trials"],
                        accepted_trials=campaign["accepted_trials"],
                        export_path=str(exported),
                    )
                    break
            except Exception as exc:  # pragma: no cover - best effort during shutdown
                print(f"Unable to export partial campaign yet: {exc}")
            time.sleep(min(self.poll_seconds, 2.0))
        return reason

    def _resume_active_campaign(self) -> None:
        if not self.active_campaign_id:
            return
        matching = next(
            (
                item
                for item in reversed(self.state["campaigns"])
                if item["campaign_id"] == self.active_campaign_id
            ),
            None,
        )
        if matching is None:
            raise RuntimeError("active campaign is absent from the persisted campaign history")
        batch_index = int(matching["batch_index"])
        attempt = int(matching["attempt"])
        print(
            f"Resuming active campaign {self.active_campaign_id} "
            f"for batch {batch_index + 1:02d} attempt {attempt}"
        )
        try:
            status = self._wait_campaign(batch_index, attempt, self.active_campaign_id)
        except RuntimeError as exc:
            print(f"Unable to resume active campaign; it will be retried: {exc}")
            status = "interrupted"
        matching["terminal_status"] = status
        matching["finished_at"] = datetime.now(UTC).isoformat()
        if status in {"completed", "completed_with_errors"}:
            if batch_index not in self.state["completed_batches"]:
                self.state["completed_batches"].append(batch_index)
        self.active_campaign_id = None
        self._save_state()

    def run(self) -> int:
        runner_lock = self._acquire_runner_lock()
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        print(
            f"E026 week runner plan={self.plan_id} batches={len(self.batches)} "
            f"deadline={self.state['deadline']}"
        )
        self._notify(
            "runner_started",
            plan_id=self.plan_id,
            batch_count=len(self.batches),
            completed_batches=len(self.state["completed_batches"]),
            deadline=self.state["deadline"],
            state_path=str(self.state_path),
        )
        try:
            self._resume_active_campaign()
            for batch_index, request_payload in enumerate(self.batches):
                if batch_index in self.state["completed_batches"]:
                    continue
                if self.stop_requested or self._deadline_reached():
                    break
                if not self._disk_guard_ok():
                    return 0
                self._wait_for_foreign_campaigns()
                attempts = int(self.state["batch_attempts"].get(str(batch_index), 0))
                while attempts < self.maximum_campaign_attempts:
                    if self.stop_requested or self._deadline_reached():
                        break
                    attempts += 1
                    self.state["batch_attempts"][str(batch_index)] = attempts
                    campaign = self._request("POST", "/v1/lab/campaigns", request_payload)
                    self.active_campaign_id = str(campaign["id"])
                    self.state["campaigns"].append(
                        {
                            "batch_index": batch_index,
                            "attempt": attempts,
                            "campaign_id": self.active_campaign_id,
                            "submitted_at": datetime.now(UTC).isoformat(),
                        }
                    )
                    self._save_state()
                    status = self._wait_campaign(batch_index, attempts, self.active_campaign_id)
                    self.state["campaigns"][-1]["terminal_status"] = status
                    self.state["campaigns"][-1]["finished_at"] = datetime.now(UTC).isoformat()
                    self.active_campaign_id = None
                    self._save_state()
                    if status in {"completed", "completed_with_errors"}:
                        self.state["completed_batches"].append(batch_index)
                        self._save_state()
                        break
                    if status in {"deadline", "cancelled"}:
                        break
                    print(f"Retrying interrupted batch {batch_index + 1:02d}")
                if batch_index not in self.state["completed_batches"]:
                    self.state.setdefault("failed_batches", []).append(batch_index)
                    self._save_state()
            if self.stop_requested:
                self.state["status"] = "cancelled"
            elif self._deadline_reached():
                self.state["status"] = "deadline_reached"
            elif len(self.state["completed_batches"]) == len(self.batches):
                self.state["status"] = "completed"
            else:
                self.state["status"] = "completed_with_failed_batches"
            self._save_state()
            self._notify(
                "runner_finished",
                plan_id=self.plan_id,
                status=self.state["status"],
                batch_count=len(self.batches),
                completed_batches=len(self.state["completed_batches"]),
                failed_batches=len(self.state.get("failed_batches", [])),
                state_path=str(self.state_path),
            )
            return 0
        finally:
            if self.stop_requested:
                self._cancel_active()
            self._release_runner_lock(runner_lock)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bounded E026 week campaign")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = os.environ.get("E026_PAYLOAD_BASE", "").strip()
    if args.plan_only:
        payload = payload or "https://ptag.io/t/w"
        batches = build_week_batches(payload)
        print(
            json.dumps(
                {
                    "batches": len(batches),
                    "prompts": sum(len(item["prompts"]) for item in batches),
                    "methods": len(batches[0]["methods"]),
                    "seeds": len(batches[0]["seeds"]),
                    "trials": sum(
                        len(item["prompts"]) * len(item["methods"]) * len(item["seeds"])
                        for item in batches
                    ),
                },
                indent=2,
            )
        )
        return 0
    if not payload:
        raise SystemExit("E026_PAYLOAD_BASE is required")
    runner = WeekCampaignRunner(
        api_url=os.environ.get(
            "E026_API_URL", "http://prooftag-qr-svc.qr-core.svc.cluster.local:8080"
        ),
        payload=payload,
        output_root=Path(os.environ.get("E026_OUTPUT_ROOT", "/data/e026-week")),
        duration_hours=float(os.environ.get("E026_DURATION_HOURS", "162")),
        minimum_free_gib=float(os.environ.get("E026_MINIMUM_FREE_GIB", "8")),
        poll_seconds=float(os.environ.get("E026_POLL_SECONDS", "30")),
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
