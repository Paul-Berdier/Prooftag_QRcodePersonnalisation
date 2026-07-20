#!/usr/bin/env python3
# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CASES = (
    {
        "name": "botanical-short",
        "payload": "https://example.prooftag.test/t/botanical-42",
        "prompt": "premium botanical engraving, elegant packaging, detailed leaves and flowers",
        "seed": 42,
    },
    {
        "name": "wine-label",
        "payload": "https://example.prooftag.test/t/wine-7Y3K9Q",
        "prompt": "luxury wine label engraving, vineyard leaves, grapes, refined copper details",
        "seed": 137,
    },
    {
        "name": "geometric-packaging",
        "payload": "https://example.prooftag.test/t/geometry-2026",
        "prompt": "premium geometric packaging pattern, art deco, strong negative space, elegant",
        "seed": 2026,
    },
    {
        "name": "cosmetics-organic",
        "payload": "https://example.prooftag.test/t/cosmetics-31415",
        "prompt": "organic luxury cosmetics illustration, flowing petals, cream and sage palette",
        "seed": 31415,
    },
    {
        "name": "industrial-technical",
        "payload": "https://example.prooftag.test/t/industrial-271828",
        "prompt": "precise industrial technical illustration, brushed metal, circuits, high contrast",
        "seed": 271828,
    },
    {
        "name": "dense-payload",
        "payload": (
            "https://example.prooftag.test/verify/product/2026/07/20/lot/FR-9A71C2"
            "?serial=0001847295&channel=packaging&campaign=benchmark"
        ),
        "prompt": "dense ornamental security engraving, guilloche lines, premium certificate design",
        "seed": 9001,
    },
)

DEBUG_VARIANTS = (
    "raw",
    "incorrect_80",
    "incorrect_85",
    "uncertain_16",
    "uncertain_32",
    "uncertain_48",
    "uncertain_64",
    "tonal_90",
    "tonal_95",
)

GLOBAL_VARIANTS = frozenset(
    {
        "centers_45",
        "centers_60",
        "centers_72",
        "centers_85",
        "tonal_90",
        "tonal_95",
        "centers_90",
        "centers_95",
    }
)

SUMMARY_FIELDS = (
    "case",
    "run_id",
    "status",
    "selected_variant",
    "selected_attempt",
    "attempts",
    "first_attempt_accepted",
    "first_attempt_scan_pass_rate",
    "global_fallback_used",
    "qr_version",
    "scan_pass_rate",
    "module_error_rate",
    "generation_ms",
    "validation_ms",
    "total_ms",
    "brightness_mean",
    "contrast_std",
    "entropy_bits",
    "sharpness_laplacian",
    "clipped_pixel_ratio",
    "changed_pixel_ratio",
    "mean_absolute_change",
    "error",
)

GPU_FIELDS = (
    "captured_at",
    "index",
    "name",
    "utilization_gpu_percent",
    "memory_used_mib",
    "memory_total_mib",
    "temperature_c",
    "power_w",
)


class GPUSampler:
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 5)

    def _sample_until_stopped(self) -> None:
        while not self._stop.is_set():
            result = run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ]
            )
            if result["returncode"] == 0:
                for values in csv.reader(result["stdout"].splitlines()):
                    if len(values) != 7:
                        continue
                    try:
                        self.samples.append(
                            {
                                "captured_at": datetime.now(UTC).isoformat(),
                                "index": int(values[0].strip()),
                                "name": values[1].strip(),
                                "utilization_gpu_percent": float(values[2].strip()),
                                "memory_used_mib": float(values[3].strip()),
                                "memory_total_mib": float(values[4].strip()),
                                "temperature_c": float(values[5].strip()),
                                "power_w": float(values[6].strip()),
                            }
                        )
                    except ValueError:
                        continue
            self._stop.wait(self.interval_seconds)


def request_bytes(
    api_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    retries: int = 1,
    timeout: int = 600,
) -> bytes:
    url = f"{api_url.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 5))
    raise RuntimeError(f"API inaccessible: {last_error}")


def request_json(
    api_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    retries: int = 1,
) -> dict[str, Any]:
    return json.loads(
        request_bytes(
            api_url,
            path,
            method=method,
            payload=payload,
            retries=retries,
        )
    )


def wait_until_ready(api_url: str, timeout_seconds: int = 60) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            return request_json(api_url, "/readyz")
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"L'API n'est pas prête après {timeout_seconds} secondes: {api_url}")


def run_command(arguments: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(arguments, text=True, capture_output=True, check=False)
    except OSError as exc:
        return {"command": arguments, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "command": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def git_value(*arguments: str) -> str:
    result = run_command(["git", *arguments])
    return result["stdout"] if result["returncode"] == 0 else "unknown"


def parse_prometheus(content: str) -> list[dict[str, Any]]:
    samples = []
    pattern = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)(?:\s+\d+)?$")
    for line in content.splitlines():
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        labels = dict(re.findall(r'(\w+)="((?:\\.|[^"])*)"', match.group(2) or ""))
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        samples.append({"name": match.group(1), "labels": labels, "value": value})
    return samples


def variant_metrics(content: str) -> dict[str, dict[str, float]]:
    variants: dict[str, dict[str, float]] = {}
    names = {
        "prooftag_qr_repair_variant_scan_pass_rate": "scan_pass_rate",
        "prooftag_qr_repair_variant_module_error_rate": "module_error_rate",
    }
    for sample in parse_prometheus(content):
        variant = sample["labels"].get("variant")
        if not variant:
            continue
        values = variants.setdefault(variant, {})
        if sample["name"] in names:
            values[names[sample["name"]]] = sample["value"]
        elif sample["name"] == "prooftag_qr_repair_variant_image_quality":
            metric = sample["labels"].get("metric")
            if metric:
                values[metric] = sample["value"]
    return variants


def read_run_events(run_id: str) -> list[dict[str, Any]]:
    result = run_command(
        [
            "kubectl",
            "logs",
            "-n",
            "qr-core",
            "deployment/prooftag-qr",
            "-c",
            "api",
            "--since=15m",
        ]
    )
    events = []
    for line in result["stdout"].splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("run_id") == run_id:
            events.append(event)
    return events


def download_optional(api_url: str, path: str, destination: Path) -> bool:
    try:
        destination.write_bytes(request_bytes(api_url, path))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    return True


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def average(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if isinstance(row.get(field), int | float)]
    return statistics.fmean(values) if values else None


def format_number(value: Any, digits: int = 3) -> str:
    if not isinstance(value, int | float):
        return "—"
    return f"{value:.{digits}f}"


def format_percent(value: Any) -> str:
    if not isinstance(value, int | float):
        return "—"
    return f"{value * 100:.1f}%"


def render_report(
    run_name: str,
    summary: dict[str, Any],
    previous: dict[str, Any] | None,
    comparisons: list[dict[str, Any]],
) -> str:
    rows = summary["results"]
    previous_name = previous.get("run_name") if previous else None
    max_duration = max((row.get("total_ms") or 0 for row in rows), default=1) or 1

    chart_rows = []
    for row in rows:
        scan = row.get("scan_pass_rate") or 0
        changed = row.get("changed_pixel_ratio") or 0
        duration = row.get("total_ms") or 0
        chart_rows.append(
            f"""
            <div class="chart-row">
              <strong>{html.escape(row["case"])}</strong>
              <span>Lecture {format_percent(scan)}</span>
              <div class="track"><i class="scan" style="width:{scan * 100:.2f}%"></i></div>
              <span>Pixels modifiés {format_percent(changed)}</span>
              <div class="track"><i class="change" style="width:{changed * 100:.2f}%"></i></div>
              <span>Temps {duration / 1000:.2f}s</span>
              <div class="track"><i class="time" style="width:{duration / max_duration * 100:.2f}%"></i></div>
            </div>
            """
        )

    comparison_rows = []
    for row in comparisons:
        comparison_rows.append(
            "<tr>"
            f"<td>{html.escape(row['case'])}</td>"
            f"<td>{format_percent(row.get('scan_pass_rate'))}</td>"
            f"<td>{format_percent(row.get('scan_delta'))}</td>"
            f"<td>{format_percent(row.get('changed_pixel_ratio'))}</td>"
            f"<td>{format_percent(row.get('changed_delta'))}</td>"
            f"<td>{format_number(row.get('total_ms'), 0)} ms</td>"
            f"<td>{format_number(row.get('total_delta_ms'), 0)} ms</td>"
            "</tr>"
        )

    gallery = []
    for row in rows:
        case = html.escape(row["case"])
        raw_path = f"cases/{case}/raw.png"
        final_path = f"cases/{case}/final.png"
        gallery.append(
            f"""
            <article>
              <h3>{case}</h3>
              <div class="images">
                <figure><img src="{raw_path}" alt="Image brute {case}"><figcaption>Brute</figcaption></figure>
                <figure><img src="{final_path}" alt="Image finale {case}"><figcaption>Finale</figcaption></figure>
              </div>
              <dl>
                <dt>Résultat</dt><dd>{html.escape(str(row.get("status", "—")))}</dd>
                <dt>Profil</dt><dd>{html.escape(str(row.get("selected_variant") or "—"))}</dd>
                <dt>Tentatives</dt><dd>{html.escape(str(row.get("attempts") or "—"))}</dd>
                <dt>Accepté au premier essai</dt><dd>{"oui" if row.get("first_attempt_accepted") else "non"}</dd>
                <dt>Lecture</dt><dd>{format_percent(row.get("scan_pass_rate"))}</dd>
                <dt>Pixels modifiés</dt><dd>{format_percent(row.get("changed_pixel_ratio"))}</dd>
                <dt>Entropie</dt><dd>{format_number(row.get("entropy_bits"))}</dd>
                <dt>Temps total</dt><dd>{format_number((row.get("total_ms") or 0) / 1000, 2)} s</dd>
              </dl>
              <a href="cases/{case}/response.json">Réponse JSON</a>
            </article>
            """
        )

    previous_label = html.escape(previous_name) if previous_name else "aucun — première référence"
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark Prooftag QR — {html.escape(run_name)}</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f6f7f8; --surface:#fff; --text:#172026; --muted:#63717a; --line:#d9dfe3; --scan:#15803d; --change:#b45309; --time:#2563eb; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg:#101416; --surface:#182024; --text:#edf2f4; --muted:#a9b5bb; --line:#344047; --scan:#4ade80; --change:#fbbf24; --time:#60a5fa; }} }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui,sans-serif; }}
    main {{ max-width:1440px; margin:auto; padding:24px; }} h1,h2,h3 {{ font-weight:600; }} h1 {{ margin-bottom:4px; }} .muted {{ color:var(--muted); }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:22px 0; }}
    .stat,article {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px; }} .stat strong {{ display:block; font-size:1.7rem; }}
    .charts {{ display:grid; gap:14px; }} .chart-row {{ display:grid; grid-template-columns:minmax(150px,1.4fr) 145px 2fr; gap:7px 12px; align-items:center; }}
    .track {{ height:13px; background:var(--line); border-radius:7px; overflow:hidden; }} .track i {{ display:block; height:100%; }} .scan {{ background:var(--scan); }} .change {{ background:var(--change); }} .time {{ background:var(--time); }}
    .gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:16px; }} .images {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    figure {{ margin:0; }} img {{ display:block; width:100%; aspect-ratio:1; object-fit:contain; background:var(--bg); }} figcaption {{ color:var(--muted); text-align:center; }}
    dl {{ display:grid; grid-template-columns:1fr 1fr; gap:5px 10px; }} dt {{ color:var(--muted); }} dd {{ margin:0; text-align:right; }}
    .table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }}
    a {{ color:var(--time); }} section {{ margin-top:30px; }}
    @media (max-width:680px) {{ main {{ padding:14px; }} .chart-row {{ grid-template-columns:1fr; }} .gallery {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <h1>Benchmark Prooftag QR</h1>
  <p class="muted">Version {html.escape(summary["git_commit"])} · {html.escape(summary["created_at"])} · comparaison : {previous_label}</p>
  <div class="stats">
    <div class="stat"><span>Livraison finale</span><strong>{format_percent(summary["acceptance_rate"])}</strong><small>{summary["accepted_cases"]}/{summary["case_count"]} images</small></div>
    <div class="stat"><span>Premier essai</span><strong>{format_percent(summary.get("first_attempt_acceptance_rate"))}</strong></div>
    <div class="stat"><span>Tentatives moyennes</span><strong>{format_number(summary.get("mean_attempts"), 2)}</strong></div>
    <div class="stat"><span>Fallback global</span><strong>{summary.get("global_fallback_cases", 0)}</strong></div>
    <div class="stat"><span>Lecture moyenne</span><strong>{format_percent(summary["mean_scan_pass_rate"])}</strong></div>
    <div class="stat"><span>Pixels modifiés</span><strong>{format_percent(summary["mean_changed_pixel_ratio"])}</strong></div>
    <div class="stat"><span>Temps moyen</span><strong>{format_number((summary["mean_total_ms"] or 0) / 1000, 2)} s</strong></div>
    <div class="stat"><span>GPU maximum</span><strong>{format_number(summary.get("max_gpu_utilization_percent"), 0)}%</strong></div>
    <div class="stat"><span>VRAM maximum</span><strong>{format_number(summary.get("max_gpu_memory_used_mib"), 0)} MiB</strong></div>
  </div>
  <section><h2>Mesures par cas</h2><div class="charts">{"".join(chart_rows)}</div></section>
  <section><h2>Évolution depuis le benchmark précédent</h2><div class="table-wrap"><table>
    <thead><tr><th>Cas</th><th>Lecture</th><th>Δ lecture</th><th>Pixels modifiés</th><th>Δ pixels</th><th>Temps</th><th>Δ temps</th></tr></thead>
    <tbody>{"".join(comparison_rows) or '<tr><td colspan="7">Première référence : les deltas apparaîtront au prochain benchmark.</td></tr>'}</tbody>
  </table></div></section>
  <section><h2>Galerie brute / finale</h2><div class="gallery">{"".join(gallery)}</div></section>
</main></body></html>"""


def benchmark_case(
    api_url: str,
    case: dict[str, Any],
    case_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    request = {
        **case,
        "backend": "controlnet",
        "negative_prompt": "blurry, low quality, text, watermark, unreadable QR",
        "error_correction": "H",
        "steps": 12,
        "guidance_scale": 12,
        "controlnet_scale": 1.5,
        "strength": 0.9,
        "max_attempts": 3,
    }
    request.pop("name")
    response = request_json(
        api_url,
        "/v1/generations",
        method="POST",
        payload=request,
        retries=2,
    )
    (case_dir / "response.json").write_text(
        json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    run_id = response["id"]
    download_optional(api_url, f"/v1/generations/{run_id}/image", case_dir / "final.png")

    prometheus = request_bytes(api_url, "/metrics").decode("utf-8")
    (case_dir / "metrics.prom").write_text(prometheus, encoding="utf-8")
    metrics_by_variant = variant_metrics(prometheus)
    events = read_run_events(run_id)
    (case_dir / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )
    evaluated = [event for event in events if event.get("message") == "repair_variant_validated"]
    completed = next(
        (event for event in reversed(events) if event.get("message") == "generation_completed"),
        {},
    )
    selected_variant = completed.get("repair_variant")
    selected_attempt = completed.get("attempt")
    if not selected_variant:
        accepted = next((event for event in evaluated if event.get("status") == "accepted"), None)
        selected_variant = accepted.get("repair_variant") if accepted else None
        selected_attempt = accepted.get("attempt") if accepted else None

    artifact_names = set(DEBUG_VARIANTS)
    artifact_names.update(
        f"attempt_{event['attempt']}_{event['repair_variant']}"
        for event in evaluated
        if event.get("attempt") and event.get("repair_variant") in DEBUG_VARIANTS
    )
    available_variants = set()
    for artifact_name in sorted(artifact_names):
        destination = case_dir / f"{artifact_name}.png"
        if download_optional(
            api_url,
            f"/v1/generations/{run_id}/variants/{artifact_name}",
            destination,
        ):
            available_variants.add(artifact_name)

    variant_rows = []
    variant_failures = []
    for event in evaluated:
        variant = event["repair_variant"]
        attempt = event.get("attempt")
        artifact_name = f"attempt_{attempt}_{variant}" if attempt else variant
        event_metrics = event.get("quality_metrics") or metrics_by_variant.get(variant, {})
        failures = event.get("validation_failures") or []
        variant_rows.append(
            {
                "case": case["name"],
                "run_id": run_id,
                "attempt": attempt,
                "seed": event.get("seed"),
                "variant": variant,
                "status": event.get("status"),
                "exact_payload_match": event.get("exact_payload_match"),
                "artifact_available": artifact_name in available_variants,
                "failure_count": len(failures),
                **event_metrics,
                "scan_pass_rate": event.get("scan_pass_rate"),
                "module_error_rate": event.get("module_error_rate"),
            }
        )
        variant_failures.extend(
            {
                "case": case["name"],
                "run_id": run_id,
                "attempt": attempt,
                "seed": event.get("seed"),
                "variant": variant,
                **failure,
            }
            for failure in failures
        )
    selected_event = next(
        (
            event
            for event in reversed(evaluated)
            if event.get("repair_variant") == selected_variant
            and (selected_attempt is None or event.get("attempt") == selected_attempt)
        ),
        {},
    )
    selected_metrics = selected_event.get("quality_metrics") or metrics_by_variant.get(
        selected_variant, {}
    )
    quality = response.get("quality_metrics") or {}
    attempt_details = response.get("attempt_details") or []
    first_attempt = attempt_details[0] if attempt_details else {}
    row = {
        "case": case["name"],
        "run_id": run_id,
        "status": response.get("status"),
        "selected_variant": selected_variant,
        "selected_attempt": selected_attempt,
        "attempts": response.get("attempts"),
        "first_attempt_accepted": first_attempt.get("accepted"),
        "first_attempt_scan_pass_rate": first_attempt.get("scan_pass_rate"),
        "global_fallback_used": selected_variant in GLOBAL_VARIANTS,
        "qr_version": response.get("qr_version"),
        "scan_pass_rate": response.get("scan_pass_rate"),
        "module_error_rate": response.get("module_error_rate"),
        "generation_ms": response.get("generation_ms"),
        "validation_ms": response.get("validation_ms"),
        "total_ms": response.get("total_ms"),
        **quality,
        "changed_pixel_ratio": selected_metrics.get("changed_pixel_ratio"),
        "mean_absolute_change": selected_metrics.get("mean_absolute_change"),
        "error": response.get("error"),
    }
    validations = [
        {"case": case["name"], "run_id": run_id, **validation}
        for validation in response.get("validations", [])
    ]
    return row, variant_rows, validations, variant_failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark reproductible des QR Prooftag")
    parser.add_argument("--api-url", default="http://127.0.0.1:18080")
    parser.add_argument("--output-root", type=Path, default=Path("benchmark-results"))
    arguments = parser.parse_args()

    ready = wait_until_ready(arguments.api_url)
    root = arguments.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "summary.json").is_file()
    )
    previous = None
    if previous_dirs:
        previous = json.loads((previous_dirs[-1] / "summary.json").read_text(encoding="utf-8"))

    commit = git_value("rev-parse", "HEAD")
    created_at = datetime.now(UTC)
    run_name = f"{created_at:%Y%m%dT%H%M%SZ}-{commit[:8]}"
    run_dir = root / run_name
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True)

    environment = {
        "created_at": created_at.isoformat(),
        "api_url": arguments.api_url,
        "ready": ready,
        "health": request_json(arguments.api_url, "/healthz"),
        "runtime": request_json(arguments.api_url, "/v1/runtime"),
        "git_commit": commit,
        "git_branch": git_value("branch", "--show-current"),
        "git_status": git_value("status", "--porcelain"),
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
        "kubernetes_pods": run_command(["kubectl", "get", "pods", "-n", "qr-core", "-o", "wide"]),
        "kubernetes_image_id": run_command(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                "qr-core",
                "-l",
                "app=prooftag-qr",
                "-o",
                "jsonpath={.items[0].status.containerStatuses[?(@.name=='api')].imageID}",
            ]
        ),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "cases.json").write_text(
        json.dumps(CASES, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    variant_failures: list[dict[str, Any]] = []
    gpu_sampler = GPUSampler()
    gpu_sampler.start()
    try:
        for index, case in enumerate(CASES, start=1):
            print(f"[{index}/{len(CASES)}] {case['name']}", flush=True)
            case_dir = cases_dir / case["name"]
            case_dir.mkdir()
            try:
                row, variant_rows, validation_rows, failure_rows = benchmark_case(
                    arguments.api_url, case, case_dir
                )
            except Exception as exc:
                row = {"case": case["name"], "status": "error", "error": str(exc)}
                variant_rows = []
                validation_rows = []
                failure_rows = []
                (case_dir / "benchmark-error.txt").write_text(str(exc), encoding="utf-8")
            results.append(row)
            variants.extend(variant_rows)
            validations.extend(validation_rows)
            variant_failures.extend(failure_rows)
            print(
                f"    {row.get('status')} · lecture={format_percent(row.get('scan_pass_rate'))}"
                f" · profil={row.get('selected_variant') or '—'}",
                flush=True,
            )
    finally:
        gpu_sampler.stop()
    write_csv(run_dir / "gpu-samples.csv", gpu_sampler.samples, GPU_FIELDS)

    accepted_cases = sum(row.get("status") == "accepted" for row in results)
    first_attempt_accepted_cases = sum(row.get("first_attempt_accepted") is True for row in results)
    summary = {
        "run_name": run_name,
        "created_at": created_at.isoformat(),
        "git_commit": commit,
        "previous_run": previous.get("run_name") if previous else None,
        "case_count": len(results),
        "accepted_cases": accepted_cases,
        "acceptance_rate": accepted_cases / len(results),
        "first_attempt_accepted_cases": first_attempt_accepted_cases,
        "first_attempt_acceptance_rate": first_attempt_accepted_cases / len(results),
        "mean_attempts": average(results, "attempts"),
        "global_fallback_cases": sum(row.get("global_fallback_used") is True for row in results),
        "mean_scan_pass_rate": average(results, "scan_pass_rate"),
        "mean_changed_pixel_ratio": average(results, "changed_pixel_ratio"),
        "mean_total_ms": average(results, "total_ms"),
        "mean_entropy_bits": average(results, "entropy_bits"),
        "mean_clipped_pixel_ratio": average(results, "clipped_pixel_ratio"),
        "max_gpu_utilization_percent": max(
            (sample["utilization_gpu_percent"] for sample in gpu_sampler.samples),
            default=None,
        ),
        "max_gpu_memory_used_mib": max(
            (sample["memory_used_mib"] for sample in gpu_sampler.samples),
            default=None,
        ),
        "max_gpu_temperature_c": max(
            (sample["temperature_c"] for sample in gpu_sampler.samples),
            default=None,
        ),
        "mean_gpu_power_w": average(gpu_sampler.samples, "power_w"),
        "results": results,
    }
    previous_rows = {row["case"]: row for row in (previous.get("results", []) if previous else [])}
    comparisons = []
    for row in results:
        old = previous_rows.get(row["case"])
        if not old:
            continue
        comparisons.append(
            {
                "case": row["case"],
                "scan_pass_rate": row.get("scan_pass_rate"),
                "scan_delta": (row.get("scan_pass_rate") or 0) - (old.get("scan_pass_rate") or 0),
                "changed_pixel_ratio": row.get("changed_pixel_ratio"),
                "changed_delta": (row.get("changed_pixel_ratio") or 0)
                - (old.get("changed_pixel_ratio") or 0),
                "total_ms": row.get("total_ms"),
                "total_delta_ms": (row.get("total_ms") or 0) - (old.get("total_ms") or 0),
            }
        )

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(run_dir / "summary.csv", results, SUMMARY_FIELDS)
    variant_fields = (
        "case",
        "run_id",
        "attempt",
        "seed",
        "variant",
        "status",
        "exact_payload_match",
        "artifact_available",
        "failure_count",
        "scan_pass_rate",
        "module_error_rate",
        "changed_pixel_ratio",
        "mean_absolute_change",
        "entropy_bits",
        "clipped_pixel_ratio",
        "brightness_mean",
        "contrast_std",
        "sharpness_laplacian",
    )
    write_csv(run_dir / "variants.csv", variants, variant_fields)
    validation_fields = (
        "case",
        "run_id",
        "decoder",
        "scenario",
        "success",
        "exact_payload_match",
        "latency_ms",
    )
    write_csv(run_dir / "validations.csv", validations, validation_fields)
    failure_fields = (
        "case",
        "run_id",
        "attempt",
        "seed",
        "variant",
        "decoder",
        "scenario",
        "outcome",
    )
    write_csv(
        run_dir / "variant-failures.csv",
        variant_failures,
        failure_fields,
    )
    comparison_fields = (
        "case",
        "scan_pass_rate",
        "scan_delta",
        "changed_pixel_ratio",
        "changed_delta",
        "total_ms",
        "total_delta_ms",
    )
    write_csv(run_dir / "comparison.csv", comparisons, comparison_fields)
    (run_dir / "report.html").write_text(
        render_report(run_name, summary, previous, comparisons), encoding="utf-8"
    )
    archive = Path(
        shutil.make_archive(
            str(root / run_name),
            "gztar",
            root_dir=root,
            base_dir=run_name,
        )
    )

    print(f"BENCHMARK_DIR={run_dir}")
    print(f"BENCHMARK_ARCHIVE={archive}")
    print(f"BENCHMARK_REPORT={run_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
