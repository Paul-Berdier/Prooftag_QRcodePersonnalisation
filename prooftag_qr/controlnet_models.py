from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from .qr import QRBlueprint

ModelFamily = Literal["sd15", "sdxl"]
ConditioningProfile = Literal["binary", "gray_quiet_zone"]


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

    QR Code Monster recommends a neutral gray background around the code. We limit gray to the
    quiet zone so the actual QR modules remain exact and comparable across models.
    """
    if profile == "binary":
        return blueprint.image.copy()
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
