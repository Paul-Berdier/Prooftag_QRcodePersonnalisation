from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from .qr import QRBlueprint

ModelFamily = Literal["sd15", "sdxl"]
ConditioningProfile = Literal["binary", "gray_quiet_zone", "nacholmo_extremes_25"]


@dataclass(frozen=True, slots=True)
class ControlNetProfile:
    name: str
    model_id: str
    family: ModelFamily
    subfolder: str = ""
    conditioning_profile: ConditioningProfile = "binary"
    license: str = "unknown"
    status: str = "candidate"


CONTROLNET_PROFILES = {
    "dion_sd15": ControlNetProfile(
        name="dion_sd15",
        model_id="DionTimmer/controlnet_qrcode-control_v1p_sd15",
        family="sd15",
        license="openrail++",
        status="current_baseline",
    ),
    "monster_sd15_v1": ControlNetProfile(
        name="monster_sd15_v1",
        model_id="monster-labs/control_v1p_sd15_qrcode_monster",
        family="sd15",
        license="openrail++",
        status="diffqrcoder_reference",
    ),
    "monster_sd15_v2": ControlNetProfile(
        name="monster_sd15_v2",
        model_id="monster-labs/control_v1p_sd15_qrcode_monster",
        family="sd15",
        subfolder="v2",
        conditioning_profile="gray_quiet_zone",
        license="openrail++",
        status="primary_candidate",
    ),
    "nacholmo_sd15_v2": ControlNetProfile(
        name="nacholmo_sd15_v2",
        model_id="Nacholmo/controlnet-qr-pattern-v2",
        family="sd15",
        conditioning_profile="nacholmo_extremes_25",
        license="creativeml-openrail-m",
        status="primary_candidate",
    ),
    "monster_sdxl_v1": ControlNetProfile(
        name="monster_sdxl_v1",
        model_id="monster-labs/control_v1p_sdxl_qrcode_monster",
        family="sdxl",
        conditioning_profile="gray_quiet_zone",
        license="openrail++",
        status="deferred_vram_and_pipeline",
    ),
    "nacholmo_sdxl": ControlNetProfile(
        name="nacholmo_sdxl",
        model_id="Nacholmo/controlnet-qr-pattern-sdxl",
        family="sdxl",
        license="creativeml-openrail-m",
        status="work_in_progress_deferred",
    ),
}


def benchmark_profiles(*, family: ModelFamily = "sd15") -> tuple[ControlNetProfile, ...]:
    return tuple(profile for profile in CONTROLNET_PROFILES.values() if profile.family == family)


def control_image_for_profile(
    blueprint: QRBlueprint,
    profile: ConditioningProfile,
) -> Image.Image:
    """Build the condition image expected by a ControlNet profile.

    QR Code Monster recommends a neutral gray background around the code. Nacholmo v2 was trained
    with only the darkest and lightest quartiles conditioned; because its exact preprocessing code
    is not published, ``nacholmo_extremes_25`` is an explicit approximation: neutral mid-gray
    elsewhere and a circular extreme-tone center covering about half of each data-module area.
    """
    if profile == "binary":
        return blueprint.image.copy()
    if profile == "nacholmo_extremes_25":
        height = blueprint.image.height
        width = blueprint.image.width
        array = np.full((height, width, 3), 128, dtype=np.uint8)
        module_count = blueprint.matrix.shape[0]
        border = blueprint.border
        # With the standard four-module quiet zone, a radius of 0.47 module leaves about half the
        # whole image at an extreme tone. A balanced QR core is therefore close to 25% black,
        # 25% white and 50% unconditioned gray, matching the only quantitative description
        # published by the model author without inventing hidden code.
        for row in range(border, module_count - border):
            y0 = round(row * height / module_count)
            y1 = max(y0 + 1, round((row + 1) * height / module_count))
            for col in range(border, module_count - border):
                x0 = round(col * width / module_count)
                x1 = max(x0 + 1, round((col + 1) * width / module_count))
                yy, xx = np.ogrid[y0:y1, x0:x1]
                center_x = (x0 + x1 - 1) / 2
                center_y = (y0 + y1 - 1) / 2
                radius_x = max(0.5, (x1 - x0) * 0.47)
                radius_y = max(0.5, (y1 - y0) * 0.47)
                mask = (
                    ((xx - center_x) / radius_x) ** 2
                    + ((yy - center_y) / radius_y) ** 2
                    <= 1.0
                )
                target = 0 if blueprint.matrix[row, col] else 255
                region = array[y0:y1, x0:x1]
                region[mask] = target
        return Image.fromarray(array)
    if profile != "gray_quiet_zone":
        raise ValueError(f"unsupported ControlNet conditioning profile: {profile}")
    array = np.asarray(blueprint.image.convert("RGB")).copy()
    module_count = blueprint.matrix.shape[0]
    border = blueprint.border
    core_start = round(border * array.shape[0] / module_count)
    core_end = round((module_count - border) * array.shape[0] / module_count)
    array[:core_start, :] = 128
    array[core_end:, :] = 128
    array[:, :core_start] = 128
    array[:, core_end:] = 128
    return Image.fromarray(array)
