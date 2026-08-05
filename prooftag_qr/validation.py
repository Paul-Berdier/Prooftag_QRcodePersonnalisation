from __future__ import annotations

import atexit
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .domain import ValidationRecord


def compare_validation_to_reference(
    records: list[ValidationRecord],
    reference_records: list[ValidationRecord],
) -> dict[str, Any]:
    """Score only decoder/scenario pairs that the binary control can pass.

    A robust transform that already defeats a pristine QR is not a meaningful
    failure of an artistic candidate. Raw and reference rates remain available
    for audit, while ``normalized_pass_rate`` is the production gate.
    """
    candidate_by_key = {
        (record.decoder, record.scenario): record.exact_payload_match
        for record in records
    }
    supported = [
        record
        for record in reference_records
        if record.exact_payload_match
    ]
    supported_values = [
        bool(candidate_by_key.get((record.decoder, record.scenario), False))
        for record in supported
    ]
    original_supported = [
        record for record in supported if record.scenario == "original"
    ]
    original_values = [
        bool(candidate_by_key.get((record.decoder, record.scenario), False))
        for record in original_supported
    ]
    grouped_decoders: dict[str, list[bool]] = defaultdict(list)
    grouped_scenarios: dict[str, list[bool]] = defaultdict(list)
    for reference, passed in zip(supported, supported_values, strict=True):
        grouped_decoders[reference.decoder].append(passed)
        grouped_scenarios[reference.scenario].append(passed)
    decoder_pass_rates = {
        name: sum(values) / len(values)
        for name, values in sorted(grouped_decoders.items())
    }
    scenario_pass_rates = {
        name: sum(values) / len(values)
        for name, values in sorted(grouped_scenarios.items())
    }
    raw_passed = sum(record.exact_payload_match for record in records)
    reference_passed = sum(
        record.exact_payload_match for record in reference_records
    )
    normalized_passed = sum(supported_values)
    normalized_total = len(supported_values)
    original_passed = sum(original_values)
    original_total = len(original_values)
    qr_verify_supported = [
        record for record in supported if record.decoder == "qr_verify"
    ]
    qr_verify_values = [
        bool(candidate_by_key.get((record.decoder, record.scenario), False))
        for record in qr_verify_supported
    ]
    qr_verify_direct = candidate_by_key.get(("qr_verify", "original"))
    return {
        "raw_passed": raw_passed,
        "raw_total": len(records),
        "raw_pass_rate": raw_passed / len(records) if records else 0.0,
        "reference_passed": reference_passed,
        "reference_total": len(reference_records),
        "reference_pass_rate": (
            reference_passed / len(reference_records)
            if reference_records
            else 0.0
        ),
        "normalized_passed": normalized_passed,
        "normalized_total": normalized_total,
        "normalized_pass_rate": (
            normalized_passed / normalized_total if normalized_total else 0.0
        ),
        "normalized_strict_all": (
            normalized_total > 0 and normalized_passed == normalized_total
        ),
        "original_passed": original_passed,
        "original_total": original_total,
        "original_strict_all": (
            original_total > 0 and original_passed == original_total
        ),
        "decoder_pass_rates": decoder_pass_rates,
        "scenario_pass_rates": scenario_pass_rates,
        "worst_decoder_pass_rate": min(decoder_pass_rates.values(), default=0.0),
        "worst_scenario_pass_rate": min(
            scenario_pass_rates.values(), default=0.0
        ),
        "qr_verify_mode": bool(qr_verify_supported),
        "qr_verify_any_exact": any(qr_verify_values),
        "qr_verify_direct_exact": (
            bool(qr_verify_direct) if qr_verify_direct is not None else None
        ),
        "qr_verify_exact_presets": sum(qr_verify_values),
        "qr_verify_supported_presets": len(qr_verify_values),
        "qr_verify_tolerance_score": (
            sum(qr_verify_values) / len(qr_verify_values)
            if qr_verify_values
            else 0.0
        ),
    }


def summarize_validation_records(records: list[ValidationRecord]) -> dict[str, Any]:
    """Expose the weak decoder and scenario instead of hiding them in a global mean."""
    grouped_decoders: dict[str, list[bool]] = defaultdict(list)
    grouped_scenarios: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        grouped_decoders[record.decoder].append(record.exact_payload_match)
        grouped_scenarios[record.scenario].append(record.exact_payload_match)
    decoder_pass_rates = {
        name: sum(values) / len(values) for name, values in sorted(grouped_decoders.items())
    }
    scenario_pass_rates = {
        name: sum(values) / len(values) for name, values in sorted(grouped_scenarios.items())
    }
    return {
        "decoder_pass_rates": decoder_pass_rates,
        "scenario_pass_rates": scenario_pass_rates,
        "worst_decoder_pass_rate": min(decoder_pass_rates.values(), default=0.0),
        "worst_scenario_pass_rate": min(scenario_pass_rates.values(), default=0.0),
    }


class Decoder(ABC):
    name: str

    @abstractmethod
    def decode(self, image: Image.Image) -> str:
        raise NotImplementedError


def decode_safely(decoder: Decoder, image: Image.Image) -> tuple[str, dict[str, str] | None]:
    """Treat a native decoder failure as an unreadable sample, while preserving diagnostics."""
    try:
        return decoder.decode(image), None
    except Exception as exc:
        return "", {
            "type": type(exc).__name__,
            "message": str(exc)[:500],
        }


class OpenCVDecoder(Decoder):
    name = "opencv"

    def __init__(self) -> None:
        self.detector = cv2.QRCodeDetector()

    def decode(self, image: Image.Image) -> str:
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        value, _, _ = self.detector.detectAndDecode(bgr)
        return value or ""


class PyzbarDecoder(Decoder):
    name = "zbar"

    def __init__(self) -> None:
        from pyzbar.pyzbar import decode

        self._decode = decode

    def decode(self, image: Image.Image) -> str:
        results = self._decode(image)
        return results[0].data.decode("utf-8") if results else ""


class ZXingCPPDecoder(Decoder):
    name = "zxingcpp"

    def __init__(self) -> None:
        import zxingcpp

        self._zxingcpp = zxingcpp

    def decode(self, image: Image.Image) -> str:
        results = self._zxingcpp.read_barcodes(np.asarray(image.convert("RGB")))
        return results[0].text if results else ""


class WeChatQRCodeDecoder(Decoder):
    """CNN detector plus super-resolution decoder shipped by OpenCV contrib.

    The four model files are deliberately required and pinned by the Docker
    image.  Falling back to an unconfigured detector would make benchmark
    revisions impossible to compare.
    """

    name = "wechat_qrcode"
    model_filenames = (
        "detect.prototxt",
        "detect.caffemodel",
        "sr.prototxt",
        "sr.caffemodel",
    )

    def __init__(self, models_dir: str | Path | None = None) -> None:
        root = Path(
            models_dir
            or os.environ.get("PROOFTAG_QR_WECHAT_MODELS_DIR", "/opt/wechat-qrcode")
        )
        paths = [root / name for name in self.model_filenames]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "WeChat QR model files are missing: " + ", ".join(missing)
            )
        constructor = getattr(cv2, "wechat_qrcode_WeChatQRCode", None)
        if constructor is None:
            namespace = getattr(cv2, "wechat_qrcode", None)
            constructor = getattr(namespace, "WeChatQRCode", None)
        if constructor is None:
            raise ImportError(
                "OpenCV was built without wechat_qrcode; install "
                "opencv-contrib-python-headless"
            )
        self.detector = constructor(*(str(path) for path in paths))

    def decode(self, image: Image.Image) -> str:
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        values, _ = self.detector.detectAndDecode(bgr)
        if isinstance(values, str):
            return values
        return next((str(value) for value in values if value), "")


class QRVerifyDecoder(Decoder):
    """Deterministic adapter for antfu/qr-verify's WeChat WASM scanner.

    The upstream CLI is intended for interactive file sorting, accepts any
    decoded text and shuffles its high-tolerance presets.  The bridge keeps its
    300 px input geometry and all 37 presets, but returns every decoded value so
    the Python validator can enforce the exact Prooftag payload deterministically.
    """

    name = "qr_verify"

    def __init__(
        self,
        bridge_path: str | Path | None = None,
        node_executable: str | None = None,
    ) -> None:
        default_bridge = Path(__file__).resolve().parent.parent / "qr_verify_bridge" / "bridge.mjs"
        self.bridge_path = Path(
            bridge_path
            or os.environ.get("PROOFTAG_QR_QR_VERIFY_BRIDGE", default_bridge)
        )
        self.node_executable = (
            node_executable
            or os.environ.get("PROOFTAG_QR_NODE_EXECUTABLE")
            or shutil.which("node")
        )
        if not self.node_executable:
            raise FileNotFoundError("Node.js is required by qr-verify")
        if not self.bridge_path.is_file():
            raise FileNotFoundError(f"qr-verify bridge is missing: {self.bridge_path}")
        self.timeout_seconds = float(
            os.environ.get("PROOFTAG_QR_QR_VERIFY_TIMEOUT_SECONDS", "120")
        )
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._registered_at_exit = False

    def _start(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._process = subprocess.Popen(
            [self.node_executable, str(self.bridge_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if not self._registered_at_exit:
            atexit.register(self.close)
            self._registered_at_exit = True
        return self._process

    def decode_presets(self, image: Image.Image) -> list[dict[str, Any]]:
        stream = io.BytesIO()
        image.convert("RGB").save(stream, format="PNG", optimize=False)
        request_id = str(uuid.uuid4())
        request = {
            "id": request_id,
            "image_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
        }
        with self._lock:
            process = self._start()
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("qr-verify bridge has no standard streams")
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
            response_lines: list[str] = []
            reader = threading.Thread(
                target=lambda: response_lines.append(process.stdout.readline()),
                daemon=True,
            )
            reader.start()
            reader.join(timeout=self.timeout_seconds)
            if reader.is_alive():
                self.close()
                reader.join(timeout=2)
                raise TimeoutError(
                    "qr-verify bridge exceeded "
                    f"{self.timeout_seconds:.0f} seconds"
                )
            line = response_lines[0] if response_lines else ""
            if not line:
                error = process.stderr.read() if process.stderr else ""
                self.close()
                raise RuntimeError(f"qr-verify bridge stopped unexpectedly: {error[-1000:]}")
        response = json.loads(line)
        if response.get("id") != request_id:
            raise RuntimeError("qr-verify bridge response id mismatch")
        if not response.get("ok"):
            raise RuntimeError(f"qr-verify bridge failed: {response.get('error')}")
        attempts = response.get("attempts")
        if response.get("engine") != "qr-verify@0.2.0" or not isinstance(attempts, list):
            raise RuntimeError("qr-verify bridge returned an invalid protocol response")
        if len(attempts) != 37:
            raise RuntimeError(f"qr-verify returned {len(attempts)} presets instead of 37")
        return attempts

    def decode(self, image: Image.Image) -> str:
        attempts = self.decode_presets(image)
        return next((str(item.get("text") or "") for item in attempts if item.get("text")), "")

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            process.kill()


def available_decoders() -> list[Decoder]:
    # E024 deliberately has one validation authority. There is no silent
    # fallback to another decoder when qr-verify is unavailable.
    return [QRVerifyDecoder()]


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    parameters: dict

    def apply(self, image: Image.Image) -> Image.Image:
        if self.name == "original":
            return image.copy()
        if self.name.startswith("jpeg"):
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=self.parameters["quality"])
            stream.seek(0)
            return Image.open(stream).convert("RGB")
        if self.name == "blur_3":
            return image.filter(ImageFilter.GaussianBlur(radius=1.2))
        if self.name == "brightness_low":
            return ImageEnhance.Brightness(image).enhance(0.72)
        if self.name == "brightness_high":
            return ImageEnhance.Brightness(image).enhance(1.25)
        if self.name == "contrast_low":
            return ImageEnhance.Contrast(image).enhance(0.72)
        if self.name == "downscale_75":
            reduced = image.resize(
                (round(image.width * 0.75), round(image.height * 0.75)),
                Image.Resampling.LANCZOS,
            )
            return reduced.resize(image.size, Image.Resampling.LANCZOS)
        if self.name == "noise_gaussian":
            source = np.asarray(image.convert("RGB"), dtype=np.float32)
            noise = np.random.default_rng(2026).normal(0, self.parameters["sigma"], source.shape)
            return Image.fromarray(np.clip(source + noise, 0, 255).astype(np.uint8))
        if self.name == "rotation_3":
            return image.rotate(
                self.parameters["degrees"],
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(255, 255, 255),
            )
        if self.name in {"print_dot_gain", "print_dot_loss"}:
            source = np.asarray(image.convert("L"), dtype=np.uint8)
            dark = (source < 128).astype(np.uint8) * 255
            kernel = np.ones((self.parameters["kernel"], self.parameters["kernel"]), np.uint8)
            if self.name == "print_dot_gain":
                changed = cv2.dilate(dark, kernel, iterations=1)
            else:
                changed = cv2.erode(dark, kernel, iterations=1)
            return Image.fromarray(255 - changed).convert("RGB")
        if self.name == "perspective_mild":
            source = np.float32(
                [
                    [0, 0],
                    [image.width - 1, 0],
                    [image.width - 1, image.height - 1],
                    [0, image.height - 1],
                ]
            )
            delta = image.width * 0.035
            target = np.float32(
                [
                    [delta, delta],
                    [image.width - 1 - delta, 0],
                    [image.width - 1, image.height - 1 - delta],
                    [0, image.height - 1],
                ]
            )
            matrix = cv2.getPerspectiveTransform(source, target)
            warped = cv2.warpPerspective(
                np.asarray(image.convert("RGB")),
                matrix,
                image.size,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            return Image.fromarray(warped)
        raise ValueError(f"Unknown validation scenario: {self.name}")


DEFAULT_SCENARIOS = [
    Scenario("original", {}),
    Scenario("jpeg_90", {"quality": 90}),
    Scenario("jpeg_70", {"quality": 70}),
    Scenario("blur_3", {"radius": 1.2}),
    Scenario("brightness_low", {"factor": 0.72}),
    Scenario("brightness_high", {"factor": 1.25}),
    Scenario("contrast_low", {"factor": 0.72}),
    Scenario("downscale_75", {"factor": 0.75}),
    Scenario("noise_gaussian", {"sigma": 6}),
    Scenario("rotation_3", {"degrees": 3}),
    Scenario("print_dot_gain", {"kernel": 3}),
    Scenario("print_dot_loss", {"kernel": 3}),
    Scenario("perspective_mild", {"ratio": 0.035}),
]


class QRValidator:
    def __init__(self, decoders: list[Decoder] | None = None):
        self.decoders = decoders or available_decoders()
        self.qr_verify_only = bool(self.decoders) and all(
            isinstance(decoder, QRVerifyDecoder) for decoder in self.decoders
        )

    def validate(
        self,
        image: Image.Image,
        expected_payload: str,
        *,
        matcher: Callable[[str, str], bool] | None = None,
        match_mode: str = "exact",
    ) -> list[ValidationRecord]:
        expected_hash = hashlib.sha256(expected_payload.encode()).hexdigest()
        comparator = matcher or (lambda decoded, expected: decoded == expected)
        if self.qr_verify_only:
            decoder = self.decoders[0]
            if not isinstance(decoder, QRVerifyDecoder):
                raise AssertionError("qr_verify_only requires QRVerifyDecoder")
            records = []
            for attempt in decoder.decode_presets(image):
                decoded = str(attempt.get("text") or "")
                exact = comparator(decoded, expected_payload)
                preset = str(attempt.get("preset") or "unknown")
                records.append(
                    ValidationRecord(
                        decoder=decoder.name,
                        scenario=("original" if preset == "original" else f"qr_verify_{preset}"),
                        success=bool(decoded),
                        exact_payload_match=exact,
                        latency_ms=float(attempt.get("latency_ms") or 0.0),
                        decoded_hash=(
                            hashlib.sha256(decoded.encode()).hexdigest() if decoded else None
                        ),
                        parameters={
                            "engine": "qr-verify@0.2.0",
                            "preset": preset,
                            "contrast": attempt.get("contrast"),
                            "brightness": attempt.get("brightness"),
                            "blur": attempt.get("blur"),
                            "expected_hash": expected_hash,
                            "match_mode": match_mode,
                            "decoder_error": attempt.get("error"),
                        },
                    )
                )
            return records
        records: list[ValidationRecord] = []
        for scenario in DEFAULT_SCENARIOS:
            transformed = scenario.apply(image)
            for decoder in self.decoders:
                started = time.perf_counter()
                decoded, decoder_error = decode_safely(decoder, transformed)
                elapsed_ms = (time.perf_counter() - started) * 1000
                exact = comparator(decoded, expected_payload)
                records.append(
                    ValidationRecord(
                        decoder=decoder.name,
                        scenario=scenario.name,
                        success=bool(decoded),
                        exact_payload_match=exact,
                        latency_ms=elapsed_ms,
                        decoded_hash=(
                            hashlib.sha256(decoded.encode()).hexdigest() if decoded else None
                        ),
                        parameters={
                            **scenario.parameters,
                            "expected_hash": expected_hash,
                            "match_mode": match_mode,
                            "decoder_error": decoder_error,
                        },
                    )
                )
        return records

    def validate_phone_proxy(
        self,
        image: Image.Image,
        expected_payload: str,
        *,
        matcher: Callable[[str, str], bool] | None = None,
        match_mode: str = "exact",
    ) -> list[ValidationRecord]:
        """Approximate a phone scanner's internal image enhancement pipeline.

        This is deliberately reported as a separate calibration metric. It must
        not be used as the production acceptance gate until it has been
        calibrated against repeated scans on real devices.
        """
        expected_hash = hashlib.sha256(expected_payload.encode()).hexdigest()
        comparator = matcher or (lambda decoded, expected: decoded == expected)
        variants = _phone_proxy_variants(image)
        records: list[ValidationRecord] = []
        for decoder in self.decoders:
            started = time.perf_counter()
            selected_name: str | None = None
            selected_decoded = ""
            first_decoded = ""
            attempts: list[dict[str, Any]] = []
            for name, transformed in variants:
                decoded, decoder_error = decode_safely(decoder, transformed)
                exact = comparator(decoded, expected_payload)
                attempts.append(
                    {
                        "preprocessor": name,
                        "decoded": bool(decoded),
                        "exact": exact,
                        "decoder_error": decoder_error,
                    }
                )
                if decoded and not first_decoded:
                    first_decoded = decoded
                if exact:
                    selected_name = name
                    selected_decoded = decoded
                    break
            decoded = selected_decoded or first_decoded
            exact = bool(selected_name)
            records.append(
                ValidationRecord(
                    decoder=decoder.name,
                    scenario="phone_proxy_original",
                    success=bool(decoded),
                    exact_payload_match=exact,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    decoded_hash=(
                        hashlib.sha256(decoded.encode()).hexdigest() if decoded else None
                    ),
                    parameters={
                        "expected_hash": expected_hash,
                        "match_mode": match_mode,
                        "selected_preprocessor": selected_name,
                        "attempts": attempts,
                    },
                )
            )
        return records


def _phone_proxy_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
    """Return deterministic, non-generative views commonly tried by scanners."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    doubled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(doubled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        doubled,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(doubled)
    blurred = cv2.GaussianBlur(doubled, (0, 0), 1.2)
    unsharp = cv2.addWeighted(doubled, 1.8, blurred, -0.8, 0)

    def as_rgb(array: np.ndarray) -> Image.Image:
        return Image.fromarray(array).convert("RGB")

    return [
        ("raw", image.convert("RGB")),
        ("grayscale_x2", as_rgb(doubled)),
        ("clahe_x2", as_rgb(clahe)),
        ("unsharp_x2", as_rgb(unsharp)),
        ("otsu_x2", as_rgb(otsu)),
        ("adaptive_x2", as_rgb(adaptive)),
    ]
