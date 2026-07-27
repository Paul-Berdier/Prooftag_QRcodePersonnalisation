from __future__ import annotations

import hashlib
import io
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .domain import ValidationRecord


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


def available_decoders() -> list[Decoder]:
    decoders: list[Decoder] = [OpenCVDecoder()]
    try:
        decoders.append(PyzbarDecoder())
    except (ImportError, OSError):
        pass
    try:
        decoders.append(ZXingCPPDecoder())
    except (ImportError, OSError):
        pass
    return decoders


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
