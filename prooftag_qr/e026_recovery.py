from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

TERMINAL_STATUSES = {
    "completed",
    "completed_with_errors",
    "cancelled",
    "interrupted",
}


def _get(api_url: str, path: str, *, raw: bool = False) -> Any:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urlopen(f"{api_url.rstrip('/')}{path}", timeout=120) as response:
                body = response.read()
                return body if raw else json.loads(body.decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(min(10, 2**attempt))
    raise RuntimeError(f"E026 recovery API unavailable: {last_error}")


def _atomic_bytes(path: Path, body: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(path)


def _logical_key(row: dict[str, str]) -> str:
    configuration = row.get("method_configuration_json", "").strip()
    if not configuration:
        configuration = json.dumps(
            {
                "method_id": row.get("method_id"),
                "generation": row.get("method_generation_json"),
                "model": row.get("method_model_json"),
                "tools": row.get("method_tools_json"),
            },
            sort_keys=True,
        )
    value = {
        "payload_hash": row.get("payload_hash"),
        "prompt": row.get("prompt_text") or row.get("prompt_id"),
        "configuration": configuration,
        "seed": row.get("seed"),
        "error_correction": row.get("error_correction"),
    }
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _generated(row: dict[str, str]) -> bool:
    return bool(row.get("generation_run_id", "").strip()) and row.get("status") in {
        "accepted",
        "rejected",
    }


def recover_e026_exports(
    *,
    api_url: str,
    plan_dir: Path,
    campaign_prefix: str = "E026W unattended batch ",
) -> dict[str, Any]:
    """Recover every database-backed E026 CSV without restarting generation."""

    plan_path = plan_dir / "plan-redacted.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    payload_hash = str(plan["payload_sha256"])
    destination = plan_dir / "exports-recovered"
    destination.mkdir(parents=True, exist_ok=True)

    campaigns = _get(api_url, "/v1/lab/campaigns?limit=500")
    selected = [
        item
        for item in campaigns
        if item.get("payload_hash") == payload_hash
        and str(item.get("name", "")).startswith(campaign_prefix)
        and item.get("status") in TERMINAL_STATUSES
    ]

    rows: list[dict[str, str]] = []
    for campaign in selected:
        campaign_id = str(campaign["id"])
        path = destination / f"prooftag-lab-{campaign_id}.csv"
        body = _get(api_url, f"/v1/lab/campaigns/{campaign_id}/results.csv", raw=True)
        _atomic_bytes(path, body)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))

    generated_rows = [row for row in rows if _generated(row)]
    logical_counts = Counter(_logical_key(row) for row in generated_rows)
    summary = {
        "plan_id": plan_dir.name,
        "payload_hash": payload_hash,
        "database_campaigns": len(selected),
        "campaign_statuses": dict(Counter(str(item["status"]) for item in selected)),
        "export_files": len(list(destination.glob("*.csv"))),
        "raw_trial_rows": len(rows),
        "generated_rows": len(generated_rows),
        "unique_logical_rows": len(logical_counts),
        "duplicate_generated_rows": len(generated_rows) - len(logical_counts),
        "prompts": len({row.get("prompt_text") for row in generated_rows}),
        "methods": len({row.get("method_id") for row in generated_rows}),
        "seeds": len({row.get("seed") for row in generated_rows}),
        "destination": str(destination),
    }
    summary_path = plan_dir / "recovery-summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(summary_path)
    return summary
