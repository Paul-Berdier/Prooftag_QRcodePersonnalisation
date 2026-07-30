from __future__ import annotations

import gc
import hashlib
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from . import metrics
from .artifacts import ArtifactStore
from .backends import build_backends
from .config import Settings
from .domain import RunRecord
from .lab_repository import LabRepository
from .quality_scoring import CLIPQualityScorer
from .repository import RunRepository
from .schemas import GenerationRequest, LabCampaignCreate, LabMethod
from .service import GenerationService
from .validation import QRValidator

logger = logging.getLogger(__name__)

DIFFQRCODER_BASE_MODEL = (
    "https://huggingface.co/fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned/"
    "blob/main/cetusMix_Whalefall2_fp16.safetensors"
)
DIFFQRCODER_MODEL_SETTINGS = {
    "base_model_id": DIFFQRCODER_BASE_MODEL,
    "controlnet_model_id": "monster-labs/control_v1p_sd15_qrcode_monster",
    "controlnet_model_subfolder": "v2",
    "controlnet_conditioning_profile": "binary",
    "controlnet_pipeline_mode": "text2img",
    "diffqrcoder_upstream_enabled": True,
    "diffqrcoder_revision": "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00",
    "diffqrcoder_qr_version": 3,
    "diffqrcoder_qr_mask_pattern": 4,
    "diffqrcoder_qr_module_size": 20,
    "diffqrcoder_qr_padding_px": 78,
}

GENERATION_KEYS = {
    "steps",
    "guidance_scale",
    "controlnet_scale",
    "strength",
}
MODEL_SETTING_KEYS = {
    "base_model_id",
    "controlnet_model_id",
    "controlnet_model_subfolder",
    "controlnet_conditioning_profile",
    "controlnet_pipeline_mode",
    "diffqrcoder_upstream_enabled",
    "diffqrcoder_revision",
    "diffqrcoder_qr_version",
    "diffqrcoder_qr_mask_pattern",
    "diffqrcoder_qr_module_size",
    "diffqrcoder_qr_padding_px",
}
TOOL_SETTING_KEYS = {
    "srpg_steps",
    "srpg_strength",
    "srpg_controlnet_scale",
    "srpg_qr_weight",
    "srpg_perceptual_weight",
    "srpg_functional_weight",
    "srpg_center_fraction",
    "srpg_dark_threshold",
    "srpg_light_threshold",
    "srpg_robust_blur_weight",
    "srpg_robust_blur_kernel",
    "srpg_robust_downscale_weight",
    "srpg_robust_downscale_factor",
    "srpg_robust_brightness_weight",
    "srpg_robust_brightness_low",
    "srpg_robust_brightness_high",
    "srpg_robust_contrast_weight",
    "srpg_robust_contrast_factor",
    "srpg_target_module_error_rate",
    "srpg_max_noise_delta_rms",
    "srpg_eta",
    "srpg_max_mean_absolute_change",
    "srpg_min_relative_module_improvement",
    "srpg_save_step_previews",
    "srpg_preview_interval",
    "srpg_seed_offset",
    "diffqrcoder_control_guidance_start",
    "diffqrcoder_control_guidance_end",
    "srpg_latent_fusion_enabled",
    "srpg_latent_fusion_channel",
    "srpg_latent_fusion_alpha",
    "srpg_latent_fusion_start",
    "srpg_latent_fusion_end",
    "srpg_quiet_zone_mode",
    "srpg_quiet_zone_minimum_luminance",
    "srpg_functional_pattern_tone_factor",
    "srmpgd_max_iterations",
    "srmpgd_step_size",
    "srmpgd_lpips_weight",
    "srmpgd_lpips_net",
    "srmpgd_crop_padding_px",
    "srmpgd_dark_threshold",
    "srmpgd_light_threshold",
    "srmpgd_center_fraction",
    "srmpgd_max_initial_module_error_rate",
    "guided_rediffusion_steps",
    "guided_rediffusion_strength",
    "guided_rediffusion_controlnet_scale",
    "guided_rediffusion_guide_center_scale",
    "guided_rediffusion_guide_confidence_margin",
    "guided_rediffusion_mask_dilation_px",
    "guided_rediffusion_mask_feather_px",
    "guided_rediffusion_max_mean_absolute_change",
    "guided_rediffusion_min_relative_module_improvement",
    "guided_rediffusion_seed_offset",
    "latent_refinement_iterations",
    "latent_refinement_learning_rate",
    "latent_refinement_qr_weight",
    "latent_refinement_preservation_weight",
    "latent_refinement_functional_weight",
    "latent_refinement_target_module_error_rate",
    "latent_refinement_max_latent_delta",
    "latent_refinement_max_mean_absolute_change",
    "latent_refinement_min_relative_module_improvement",
}


def _legacy_laboratory_profiles() -> list[dict[str, Any]]:
    """Editable starting points; none of them is presented as a production guarantee."""

    return [
        {
            "id": "qr_reference",
            "name": "QR témoin",
            "backend": "qr",
            "enabled": True,
            "output_variant": "raw",
            "reuse_stage1": False,
            "generation": {"steps": 1, "guidance_scale": 0, "controlnet_scale": 0, "strength": 1},
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {"settings": {}},
            "description": "Contrôle binaire sans diffusion.",
        },
        {
            "id": "controlnet_raw",
            "name": "DiffQRCoder — Stage 1 brut",
            "backend": "controlnet",
            "enabled": True,
            "output_variant": "raw",
            "reuse_stage1": True,
            "generation": {
                "steps": 40,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {"settings": {}},
            "description": "Cetus-Mix + QR Monster v2, première diffusion sans Stage 2.",
        },
        {
            "id": "srpg_late_2",
            "name": "DiffQRCoder SRPG tardif — 2 pas",
            "backend": "controlnet",
            "enabled": True,
            "output_variant": "srpg",
            "reuse_stage1": True,
            "generation": {
                "steps": 40,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {
                "srpg_enabled": True,
                "settings": {
                    "srpg_steps": 40,
                    "srpg_strength": 0.05,
                    "srpg_controlnet_scale": 1.35,
                    "srpg_qr_weight": 500.0,
                    "srpg_perceptual_weight": 3.0,
                    "srpg_functional_weight": 1.0,
                    "srpg_dark_threshold": 0.45,
                    "srpg_light_threshold": 0.65,
                    "srpg_max_noise_delta_rms": 0.50,
                    "srpg_max_mean_absolute_change": 0.12,
                    "srpg_min_relative_module_improvement": 0.0,
                    "srpg_save_step_previews": True,
                    "srpg_preview_interval": 1,
                    "srpg_quiet_zone_mode": "adaptive_light",
                    "srpg_quiet_zone_minimum_luminance": 0.90,
                    "srpg_functional_pattern_tone_factor": 0.0,
                },
            },
            "description": (
                "Raffinement équilibré : 40 pas DDIM configurés, strength 0,05, "
                "donc 2 pas tardifs réellement exécutés, avec marge claire adaptée "
                "à la palette mais sans renforcement fonctionnel."
            ),
        },
        {
            "id": "srpg_late_2_functional",
            "name": "SRPG tardif — repères fonctionnels intégrés",
            "backend": "controlnet",
            "enabled": True,
            "output_variant": "srpg",
            "reuse_stage1": True,
            "generation": {
                "steps": 40,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {
                "srpg_enabled": True,
                "settings": {
                    "srpg_steps": 40,
                    "srpg_strength": 0.05,
                    "srpg_controlnet_scale": 1.35,
                    "srpg_qr_weight": 500.0,
                    "srpg_perceptual_weight": 3.0,
                    "srpg_functional_weight": 1.0,
                    "srpg_dark_threshold": 0.45,
                    "srpg_light_threshold": 0.65,
                    "srpg_max_noise_delta_rms": 0.50,
                    "srpg_max_mean_absolute_change": 0.20,
                    "srpg_min_relative_module_improvement": 0.0,
                    "srpg_save_step_previews": True,
                    "srpg_preview_interval": 1,
                    "srpg_quiet_zone_mode": "adaptive_light",
                    "srpg_quiet_zone_minimum_luminance": 0.90,
                    "srpg_functional_pattern_tone_factor": 0.12,
                },
            },
            "description": (
                "Même latent que le profil 2 pas. La marge prend une teinte claire de "
                "l'image et seuls les finders, timings, formats et alignements sont "
                "tonifiés. Aucun module de données n'est projeté."
            ),
        },
        {
            "id": "srpg_late_2_functional_srmpgd",
            "name": "SRPG tardif fonctionnel + SR-MPGD",
            "backend": "controlnet",
            "enabled": True,
            "output_variant": "srmpgd",
            "reuse_stage1": True,
            "generation": {
                "steps": 40,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {
                "srpg_enabled": True,
                "srmpgd_enabled": True,
                "settings": {
                    "srpg_steps": 40,
                    "srpg_strength": 0.05,
                    "srpg_controlnet_scale": 1.35,
                    "srpg_qr_weight": 500.0,
                    "srpg_perceptual_weight": 3.0,
                    "srpg_functional_weight": 1.0,
                    "srpg_dark_threshold": 0.45,
                    "srpg_light_threshold": 0.65,
                    "srpg_max_noise_delta_rms": 0.50,
                    "srpg_max_mean_absolute_change": 0.20,
                    "srpg_min_relative_module_improvement": 0.0,
                    "srpg_save_step_previews": True,
                    "srpg_preview_interval": 1,
                    "srpg_quiet_zone_mode": "adaptive_light",
                    "srpg_quiet_zone_minimum_luminance": 0.90,
                    "srpg_functional_pattern_tone_factor": 0.12,
                    "srmpgd_max_iterations": 20,
                    "srmpgd_step_size": 1000.0,
                    "srmpgd_lpips_weight": 0.01,
                    "srmpgd_lpips_net": "vgg",
                    "srmpgd_crop_padding_px": -1,
                    "srmpgd_dark_threshold": 0.5,
                    "srmpgd_light_threshold": 0.5,
                    "srmpgd_max_initial_module_error_rate": 0.10,
                },
            },
            "description": (
                "Même présentation fonctionnelle, puis SR-MPGD sur le latent propre. "
                "Ce profil mesure si la finition latente apporte encore quelque chose "
                "une fois les motifs de détection rendus fiables."
            ),
        },
        {
            "id": "srpg_late_4",
            "name": "DiffQRCoder SRPG tardif — 4 pas",
            "backend": "controlnet",
            "enabled": False,
            "output_variant": "srpg",
            "reuse_stage1": True,
            "generation": {
                "steps": 40,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {
                "srpg_enabled": True,
                "settings": {
                    "srpg_steps": 40,
                    "srpg_strength": 0.10,
                    "srpg_controlnet_scale": 1.35,
                    "srpg_qr_weight": 500.0,
                    "srpg_perceptual_weight": 3.0,
                    "srpg_functional_weight": 1.0,
                    "srpg_dark_threshold": 0.45,
                    "srpg_light_threshold": 0.65,
                    "srpg_max_noise_delta_rms": 0.75,
                    "srpg_max_mean_absolute_change": 0.18,
                    "srpg_min_relative_module_improvement": 0.0,
                    "srpg_save_step_previews": True,
                    "srpg_preview_interval": 1,
                },
            },
            "description": (
                "Raffinement robuste : 40 pas DDIM configurés, strength 0,10, "
                "donc 4 pas tardifs réellement exécutés."
            ),
        },
        {
            "id": "srpg_late_4_srmpgd",
            "name": "Ablation tardive — 4 pas SRPG + SR-MPGD",
            "backend": "controlnet",
            "enabled": False,
            "output_variant": "srmpgd",
            "reuse_stage1": True,
            "generation": {
                "steps": 40,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {
                "srpg_enabled": True,
                "srmpgd_enabled": True,
                "settings": {
                    "srpg_steps": 40,
                    "srpg_strength": 0.10,
                    "srpg_controlnet_scale": 1.35,
                    "srpg_qr_weight": 500.0,
                    "srpg_perceptual_weight": 3.0,
                    "srpg_functional_weight": 1.0,
                    "srpg_dark_threshold": 0.45,
                    "srpg_light_threshold": 0.65,
                    "srpg_max_noise_delta_rms": 0.75,
                    "srpg_max_mean_absolute_change": 0.18,
                    "srpg_min_relative_module_improvement": 0.0,
                    "srpg_save_step_previews": True,
                    "srpg_preview_interval": 1,
                    "srmpgd_max_iterations": 20,
                    "srmpgd_step_size": 1000.0,
                    "srmpgd_lpips_weight": 0.01,
                    "srmpgd_lpips_net": "vgg",
                    "srmpgd_crop_padding_px": -1,
                    "srmpgd_dark_threshold": 0.5,
                    "srmpgd_light_threshold": 0.5,
                    "srmpgd_max_initial_module_error_rate": 0.10,
                },
            },
            "description": (
                "Ancienne ablation conservée pour audit. Elle est désactivée : quatre pas "
                "tardifs ne reproduisent pas le Stage 2 à 40 pas du papier et ne doivent "
                "pas servir de point de départ à SR-MPGD."
            ),
        },
        {
            "id": "srpg_full_restart",
            "name": "DiffQRCoder public — Stage 2 SRPG complet",
            "backend": "controlnet",
            "enabled": False,
            "output_variant": "srpg",
            "reuse_stage1": True,
            "generation": {
                "steps": 40,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "model": {
                **DIFFQRCODER_MODEL_SETTINGS,
                "controlnet_conditioning_profile": "binary",
            },
            "tools": {
                "srpg_enabled": True,
                "settings": {
                    "srpg_steps": 40,
                    "srpg_strength": 1.0,
                    "srpg_controlnet_scale": 1.35,
                    "srpg_qr_weight": 500.0,
                    "srpg_perceptual_weight": 3.0,
                    "srpg_functional_weight": 1.0,
                    "srpg_dark_threshold": 0.5,
                    "srpg_light_threshold": 0.5,
                    "srpg_max_noise_delta_rms": 100.0,
                    "srpg_max_mean_absolute_change": 1.0,
                    "srpg_min_relative_module_improvement": 0.0,
                    "srpg_save_step_previews": True,
                    "srpg_preview_interval": 5,
                },
            },
            "description": (
                "Chemin exécutable public : rebruitage complet, 40 pas DDIM, SRL λ1=500, "
                "LPIPS λ2=3, QR binaire, quiet zone exclue des pertes puis restaurée en "
                "clair avant validation. Le transformateur QArt Reed-Solomon décrit dans "
                "le papier reste absent du dépôt public."
            ),
        },
        {
            "id": "srpg_full_restart_srmpgd",
            "name": "DiffQRCoder public — SRPG complet + SR-MPGD",
            "backend": "controlnet",
            "enabled": False,
            "output_variant": "srmpgd",
            "reuse_stage1": True,
            "generation": {
                "steps": 40,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "model": {
                **DIFFQRCODER_MODEL_SETTINGS,
                "controlnet_conditioning_profile": "binary",
            },
            "tools": {
                "srpg_enabled": True,
                "srmpgd_enabled": True,
                "settings": {
                    "srpg_steps": 40,
                    "srpg_strength": 1.0,
                    "srpg_controlnet_scale": 1.35,
                    "srpg_qr_weight": 500.0,
                    "srpg_perceptual_weight": 3.0,
                    "srpg_functional_weight": 1.0,
                    "srpg_dark_threshold": 0.5,
                    "srpg_light_threshold": 0.5,
                    "srpg_max_noise_delta_rms": 100.0,
                    "srpg_max_mean_absolute_change": 1.0,
                    "srpg_min_relative_module_improvement": 0.0,
                    "srpg_save_step_previews": True,
                    "srpg_preview_interval": 5,
                    "srmpgd_max_iterations": 20,
                    "srmpgd_step_size": 1000.0,
                    "srmpgd_lpips_weight": 0.01,
                    "srmpgd_lpips_net": "vgg",
                    "srmpgd_crop_padding_px": -1,
                    "srmpgd_dark_threshold": 0.5,
                    "srmpgd_light_threshold": 0.5,
                    "srmpgd_max_initial_module_error_rate": 0.10,
                },
            },
            "description": (
                "Même Stage 2 complet que le profil précédent, puis équations 12-14 sur "
                "son latent propre. SR-MPGD n'est appliqué que si le MER initial est au "
                "plus 10 %. La quiet zone est restaurée avant chaque test de décodeur."
            ),
        },
        {
            "id": "srpg_freeqr",
            "name": "SRPG + fusion latente",
            "backend": "controlnet",
            "enabled": False,
            "output_variant": "srpg",
            "reuse_stage1": True,
            "generation": {
                "steps": 40,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {
                "srpg_enabled": True,
                "settings": {
                    "srpg_steps": 40,
                    "srpg_strength": 1.0,
                    "srpg_controlnet_scale": 1.35,
                    "srpg_qr_weight": 500.0,
                    "srpg_perceptual_weight": 3.0,
                    "srpg_functional_weight": 1.0,
                    "srpg_dark_threshold": 0.45,
                    "srpg_light_threshold": 0.65,
                    "srpg_latent_fusion_enabled": True,
                    "srpg_latent_fusion_channel": 1,
                    "srpg_latent_fusion_alpha": 0.15,
                    "srpg_latent_fusion_start": 0.0,
                    "srpg_latent_fusion_end": 1.0,
                    "srpg_max_mean_absolute_change": 0.40,
                    "srpg_min_relative_module_improvement": 0.0,
                    "srpg_save_step_previews": True,
                    "srpg_preview_interval": 5,
                },
            },
            "description": "Ablation FreeQR inspirée d’E014B, désactivée par défaut.",
        },
    ]


TOOL_SETTING_KEYS = {
    "srpg_steps",
    "srpg_controlnet_scale",
    "srpg_qr_weight",
    "srpg_perceptual_weight",
    "srpg_eta",
    "srpg_seed_offset",
    "srpg_save_step_previews",
    "srpg_preview_interval",
    "srmpgd_max_iterations",
    "srmpgd_step_size",
    "srmpgd_lpips_weight",
    "srmpgd_lpips_net",
    "srmpgd_crop_padding_px",
    "srmpgd_dark_threshold",
    "srmpgd_light_threshold",
    "srmpgd_center_fraction",
    "srmpgd_max_initial_module_error_rate",
    "diffqrcoder_control_guidance_start",
    "diffqrcoder_control_guidance_end",
    "diffqrcoder_stage2_initialization",
    "diffqrcoder_stage2_strength",
    "diffqrcoder_stage2_target_mode",
    "diffqrcoder_qart_executable",
    "diffqrcoder_qart_thresholds",
    "diffqrcoder_guard_max_changed_pixel_ratio",
    "diffqrcoder_guard_max_mean_absolute_change",
    "diffqrcoder_guard_max_clipped_pixel_ratio_increase",
    "diffqrcoder_guard_max_rgb_clipped_channel_ratio_increase",
    "diffqrcoder_guard_max_saturation_mean_increase",
    "diffqrcoder_guard_max_high_saturation_ratio_increase",
}


def laboratory_profiles() -> list[dict[str, Any]]:
    """Pinned models with the Stage-2 algorithm reconstructed from the paper."""

    generation = {
        "steps": 40,
        "guidance_scale": 7.5,
        "controlnet_scale": 1.35,
        "strength": 1.0,
    }
    stage2 = {
        "srpg_steps": 40,
        "srpg_controlnet_scale": 1.35,
        "srpg_qr_weight": 500.0,
        "srpg_perceptual_weight": 2.0,
        "srpg_eta": 0.0,
        "srpg_seed_offset": 2_000_003,
        "srpg_save_step_previews": True,
        "srpg_preview_interval": 5,
        "diffqrcoder_control_guidance_start": 0.0,
        "diffqrcoder_control_guidance_end": 1.0,
        "diffqrcoder_stage2_initialization": "paper_stage1_noise",
        "diffqrcoder_stage2_strength": 1.0,
        "diffqrcoder_stage2_target_mode": "qart_url_fragment",
        "diffqrcoder_qart_thresholds": [96, 112, 128, 144, 160],
        "diffqrcoder_guard_max_changed_pixel_ratio": 0.995,
        "diffqrcoder_guard_max_mean_absolute_change": 0.35,
        "diffqrcoder_guard_max_clipped_pixel_ratio_increase": 0.05,
        "diffqrcoder_guard_max_rgb_clipped_channel_ratio_increase": 0.02,
        "diffqrcoder_guard_max_saturation_mean_increase": 0.08,
        "diffqrcoder_guard_max_high_saturation_ratio_increase": 0.05,
    }
    return [
        {
            "id": "qr_reference",
            "name": "QR témoin",
            "backend": "qr",
            "enabled": True,
            "output_variant": "raw",
            "reuse_stage1": False,
            "generation": {
                "steps": 1,
                "guidance_scale": 0,
                "controlnet_scale": 0,
                "strength": 1,
            },
            "model": {},
            "tools": {"settings": {}},
            "description": "QR binaire exact, sans diffusion. Contrôle des décodeurs.",
        },
        {
            "id": "diffqrcoder_stage1",
            "name": "DiffQRCoder — Stage 1",
            "backend": "controlnet",
            "enabled": True,
            "output_variant": "raw",
            "reuse_stage1": True,
            "generation": generation.copy(),
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {"settings": {}},
            "description": "Cetus-Mix Whalefall + QR Monster v2, sortie Stage 1 brute.",
        },
        {
            "id": "diffqrcoder_srpg",
            "name": "DiffQRCoder — Stage 2 SRPG",
            "backend": "controlnet",
            "enabled": True,
            "output_variant": "srpg",
            "reuse_stage1": True,
            "generation": generation.copy(),
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {
                "srpg_enabled": True,
                "settings": stage2.copy(),
            },
            "description": (
                "Chaîne du papier : Stage 1 bruité, vraie cible QArt "
                "Reed-Solomon puis SRPG. QArt conserve l'URL avant le fragment, "
                "mais le payload n'est pas identique byte à byte."
            ),
        },
        {
            "id": "diffqrcoder_srmpgd",
            "name": "DiffQRCoder — Stage 2 + SR-MPGD",
            "backend": "controlnet",
            "enabled": False,
            "output_variant": "srmpgd",
            "reuse_stage1": True,
            "generation": generation.copy(),
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {
                "srpg_enabled": True,
                "srmpgd_enabled": True,
                "settings": {
                    **stage2,
                    "srmpgd_max_iterations": 20,
                    "srmpgd_step_size": 1000.0,
                    "srmpgd_lpips_weight": 0.01,
                    "srmpgd_lpips_net": "vgg",
                    "srmpgd_crop_padding_px": 78,
                    "srmpgd_dark_threshold": 0.45,
                    "srmpgd_light_threshold": 0.65,
                    "srmpgd_center_fraction": 1 / 3,
                    "srmpgd_max_initial_module_error_rate": 0.35,
                },
            },
            "description": (
                "Stage 2 QArt puis Eq. 13-14 : SRL + 0,01 LPIPS, gamma 1000, "
                "validation à chaque itération et conservation du meilleur état."
            ),
        },
        {
            "id": "diffqrcoder_binary_srpg",
            "name": "DiffQRCoder — SRPG cible binaire (témoin)",
            "backend": "controlnet",
            "enabled": False,
            "output_variant": "srpg",
            "reuse_stage1": True,
            "generation": generation.copy(),
            "model": DIFFQRCODER_MODEL_SETTINGS.copy(),
            "tools": {
                "srpg_enabled": True,
                "settings": {
                    **stage2,
                    "diffqrcoder_stage2_target_mode": "binary_exact",
                },
            },
            "description": (
                "Ablation payload exact : cible QR binaire sans QArt. Le "
                "redémarrage complet peut reconstruire et dégrader tout le Stage 1."
            ),
        },
    ]


class LabService:
    def __init__(
        self,
        *,
        base_settings: Settings,
        run_repository: RunRepository,
        lab_repository: LabRepository,
        artifact_store: ArtifactStore,
        validator: QRValidator,
    ):
        self.base_settings = base_settings
        self.run_repository = run_repository
        self.lab_repository = lab_repository
        self.artifact_store = artifact_store
        self.validator = validator
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qr-lab")
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()
        self._quality_scorer: CLIPQualityScorer | None = None

    def start(self) -> None:
        self.lab_repository.mark_running_interrupted()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def create_campaign(self, request: LabCampaignCreate) -> dict:
        active_methods = [method for method in request.methods if method.enabled]
        for method in active_methods:
            self._settings_for_method(method)
            self._target_variant_for_method(method)
        campaign_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        specification = request.model_dump(exclude={"payload"})
        total = len(request.prompts) * len(request.seeds) * len(active_methods)
        campaign = self.lab_repository.create_campaign(
            {
                "id": campaign_id,
                "created_at": now,
                "updated_at": now,
                "name": request.name,
                "status": "queued",
                "payload_hash": hashlib.sha256(request.payload.encode()).hexdigest(),
                "specification": specification,
                "total_trials": total,
                "completed_trials": 0,
                "accepted_trials": 0,
                "error": None,
            }
        )
        trial_plan = []
        for method in active_methods:
            for prompt in request.prompts:
                for seed in request.seeds:
                    trial_id = str(uuid.uuid4())
                    configuration = {
                        "prompt": prompt.model_dump(),
                        "method": method.model_dump(),
                        "error_correction": request.error_correction,
                        "max_attempts": request.max_attempts,
                    }
                    self.lab_repository.create_trial(
                        {
                            "id": trial_id,
                            "campaign_id": campaign_id,
                            "created_at": now,
                            "completed_at": None,
                            "prompt_id": prompt.id,
                            "method_id": method.id,
                            "seed": seed,
                            "status": "queued",
                            "generation_run_id": None,
                            "configuration": configuration,
                            "error": None,
                        }
                    )
                    trial_plan.append((trial_id, prompt, seed, method))
        self._executor.submit(self._run_campaign, campaign_id, request, trial_plan)
        return campaign

    def cancel(self, campaign_id: str) -> dict:
        campaign = self.lab_repository.get_campaign(campaign_id)
        if campaign is None:
            raise KeyError(campaign_id)
        with self._lock:
            self._cancelled.add(campaign_id)
        if campaign["status"] == "queued":
            return self.lab_repository.update_campaign(campaign_id, status="cancelled")
        return campaign

    def _is_cancelled(self, campaign_id: str) -> bool:
        with self._lock:
            return campaign_id in self._cancelled

    def _run_campaign(
        self,
        campaign_id: str,
        request: LabCampaignCreate,
        trial_plan: list[tuple],
    ) -> None:
        metrics.LAB_ACTIVE_CAMPAIGNS.inc()
        self.lab_repository.update_campaign(campaign_id, status="running")
        completed = 0
        accepted = 0
        errors = 0
        current_method_id = None
        generation_service = None
        shared_stage1: dict[str, tuple[Any, str]] = {}
        shared_stage2: dict[str, dict] = {}
        try:
            for trial_id, prompt, seed, method in trial_plan:
                if self._is_cancelled(campaign_id):
                    self.lab_repository.update_trial(
                        trial_id,
                        status="cancelled",
                        completed_at=datetime.now(UTC),
                    )
                    continue
                if method.id != current_method_id:
                    self._release_generation_service(generation_service)
                    generation_service = self._generation_service(method)
                    current_method_id = method.id
                self.lab_repository.update_trial(trial_id, status="running")
                started = time.perf_counter()
                try:
                    generation_request = self._generation_request(
                        request,
                        method,
                        prompt.text,
                        prompt.negative_prompt,
                        seed,
                    )
                    stage1_key = None
                    stage1_override = None
                    stage1_source_run_id = None
                    if method.backend == "controlnet" and method.reuse_stage1:
                        stage1_key = self._stage1_cache_key(
                            method,
                            prompt.text,
                            prompt.negative_prompt,
                            seed,
                            request.error_correction,
                        )
                        cached = shared_stage1.get(stage1_key)
                        if cached is not None:
                            stage1_override, stage1_source_run_id = cached
                    stage2_key = None
                    if (
                        method.backend == "controlnet"
                        and method.tools.srpg_enabled
                    ):
                        stage2_key = self._stage2_cache_key(
                            method,
                            prompt.text,
                            prompt.negative_prompt,
                            seed,
                            request.error_correction,
                            request.payload,
                        )
                        cached_stage2 = shared_stage2.get(stage2_key)
                        backend = generation_service.backends.get("controlnet")
                        if (
                            cached_stage2 is not None
                            and hasattr(backend, "import_stage2_state")
                        ):
                            backend.import_stage2_state(cached_stage2)
                    run = generation_service.generate(
                        generation_request,
                        raw_candidate_override=stage1_override,
                        target_variant=self._target_variant_for_method(method),
                        stage1_source_run_id=stage1_source_run_id,
                    )
                    if stage1_key is not None and stage1_key not in shared_stage1:
                        raw_candidate = generation_service.last_raw_candidate
                        if raw_candidate is not None:
                            shared_stage1[stage1_key] = (raw_candidate, run.id)
                    if stage2_key is not None and stage2_key not in shared_stage2:
                        backend = generation_service.backends.get("controlnet")
                        if hasattr(backend, "export_stage2_state"):
                            stage2_state = backend.export_stage2_state()
                            if stage2_state is not None:
                                shared_stage2[stage2_key] = stage2_state
                    self._record_method_diagnostics(run, method)
                    self._score_quality(run, prompt.text)
                    status = (
                        run.status
                        if run.status in {"accepted", "rejected", "error"}
                        else "error"
                    )
                    self.lab_repository.update_trial(
                        trial_id,
                        status=status,
                        generation_run_id=run.id,
                        completed_at=datetime.now(UTC),
                        error=run.error,
                    )
                    accepted += int(status == "accepted")
                    errors += int(status == "error")
                    metrics.LAB_TRIALS.labels(method.id, status).inc()
                except Exception as exc:
                    errors += 1
                    self.lab_repository.update_trial(
                        trial_id,
                        status="error",
                        completed_at=datetime.now(UTC),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    metrics.LAB_TRIALS.labels(method.id, "error").inc()
                    logger.exception(
                        "lab_trial_failed",
                        extra={"campaign_id": campaign_id, "trial_id": trial_id},
                    )
                finally:
                    completed += 1
                    metrics.LAB_TRIAL_DURATION.labels(method.id).observe(
                        time.perf_counter() - started
                    )
                    self.lab_repository.update_campaign(
                        campaign_id,
                        completed_trials=completed,
                        accepted_trials=accepted,
                    )
            cancelled = self._is_cancelled(campaign_id)
            status = (
                "cancelled"
                if cancelled
                else ("completed_with_errors" if errors else "completed")
            )
            self.lab_repository.update_campaign(campaign_id, status=status)
            metrics.LAB_CAMPAIGNS.labels(status).inc()
        except Exception as exc:
            self.lab_repository.update_campaign(
                campaign_id,
                status="completed_with_errors",
                error=f"{type(exc).__name__}: {exc}",
            )
            metrics.LAB_CAMPAIGNS.labels("completed_with_errors").inc()
            logger.exception("lab_campaign_failed", extra={"campaign_id": campaign_id})
        finally:
            self._release_generation_service(generation_service)
            shared_stage1.clear()
            shared_stage2.clear()
            metrics.LAB_ACTIVE_CAMPAIGNS.dec()
            with self._lock:
                self._cancelled.discard(campaign_id)

    def _score_quality(self, run: RunRecord, prompt: str) -> None:
        if (
            run.backend == "qr"
            or not self.base_settings.lab_clip_scoring_enabled
            or not run.image_path
        ):
            return
        started = time.perf_counter()
        try:
            if self._quality_scorer is None:
                self._quality_scorer = CLIPQualityScorer(
                    self.base_settings.model_cache_dir,
                    device="cpu",
                )
            image = self.artifact_store.load_image(run.image_path)
            run.quality_metrics.update(asdict(self._quality_scorer.score(image, prompt)))
            self.run_repository.save(run)
            metrics.LAB_QUALITY_SCORES.labels("success").inc()
        except Exception:
            metrics.LAB_QUALITY_SCORES.labels("error").inc()
            logger.exception("lab_quality_scoring_failed", extra={"run_id": run.id})
        finally:
            metrics.LAB_QUALITY_SCORE_DURATION.observe(time.perf_counter() - started)

    def _record_method_diagnostics(self, run: RunRecord, method: LabMethod) -> None:
        if method.tools.srpg_enabled:
            settings = self._settings_for_method(method)
            if settings.diffqrcoder_upstream_enabled:
                run.quality_metrics.update(
                    {
                        "diffqrcoder_stage2_steps_requested": float(settings.srpg_steps),
                        "diffqrcoder_controlnet_scale_requested": float(
                            settings.srpg_controlnet_scale
                        ),
                        "diffqrcoder_srg_requested": float(settings.srpg_qr_weight),
                        "diffqrcoder_pg_requested": float(
                            settings.srpg_perceptual_weight
                        ),
                        "diffqrcoder_eta_requested": float(settings.srpg_eta),
                        "diffqrcoder_stage2_strength_requested": float(
                            settings.diffqrcoder_stage2_strength
                        ),
                        "diffqrcoder_stage2_paper_initialization_requested": float(
                            settings.diffqrcoder_stage2_initialization
                            == "paper_stage1_noise"
                        ),
                        "diffqrcoder_stage2_control_target_exact_requested": float(
                            settings.diffqrcoder_stage2_target_mode
                            == "binary_exact"
                        ),
                        "diffqrcoder_stage2_control_target_qart_requested": float(
                            settings.diffqrcoder_stage2_target_mode
                            == "qart_url_fragment"
                        ),
                        "diffqrcoder_qr_version": float(
                            settings.diffqrcoder_qr_version
                        ),
                        "diffqrcoder_qr_mask_pattern": float(
                            settings.diffqrcoder_qr_mask_pattern
                        ),
                        "diffqrcoder_qr_module_size": float(
                            settings.diffqrcoder_qr_module_size
                        ),
                        "diffqrcoder_qr_padding_px": float(
                            settings.diffqrcoder_qr_padding_px
                        ),
                    }
                )
            else:
                run.quality_metrics.update(
                    {
                        "srpg_requested_steps": float(settings.srpg_steps),
                        "srpg_effective_steps": float(settings.srpg_effective_steps),
                        "srpg_restart_strength": float(settings.srpg_strength),
                        "srpg_controlnet_scale": float(settings.srpg_controlnet_scale),
                        "srpg_qr_weight": float(settings.srpg_qr_weight),
                        "srpg_perceptual_weight": float(settings.srpg_perceptual_weight),
                        "srpg_functional_weight": float(settings.srpg_functional_weight),
                        "srpg_max_noise_delta_rms": float(
                            settings.srpg_max_noise_delta_rms
                        ),
                    }
                )
            if method.tools.srmpgd_enabled:
                run.quality_metrics.update(
                    {
                        "srmpgd_requested_max_iterations": float(
                            settings.srmpgd_max_iterations
                        ),
                        "srmpgd_requested_step_size": float(
                            settings.srmpgd_step_size
                        ),
                        "srmpgd_requested_lpips_weight": float(
                            settings.srmpgd_lpips_weight
                        ),
                        "srmpgd_requested_max_initial_mer": float(
                            settings.srmpgd_max_initial_module_error_rate
                        ),
                    }
                )
        self.run_repository.save(run)

    def _generation_service(self, method: LabMethod) -> GenerationService:
        method_settings = self._settings_for_method(method)
        return GenerationService(
            settings=method_settings,
            repository=self.run_repository,
            artifact_store=self.artifact_store,
            backends=build_backends(method_settings),
            validator=self.validator,
        )

    def _settings_for_method(self, method: LabMethod) -> Settings:
        updates: dict[str, Any] = {
            "default_backend": method.backend,
            "save_debug_artifacts": True,
            "srpg_enabled": method.tools.srpg_enabled,
            "srmpgd_enabled": method.tools.srmpgd_enabled,
            "guided_rediffusion_enabled": method.tools.guided_rediffusion_enabled,
            "latent_refinement_enabled": method.tools.latent_refinement_enabled,
        }
        unknown_model = set(method.model) - MODEL_SETTING_KEYS
        unknown_tools = set(method.tools.settings) - TOOL_SETTING_KEYS
        if unknown_model:
            raise ValueError(f"unsupported model settings: {sorted(unknown_model)}")
        if unknown_tools:
            raise ValueError(f"unsupported tool settings: {sorted(unknown_tools)}")
        updates.update(method.model)
        updates.update(method.tools.settings)
        if method.tools.srpg_enabled and not updates.get("diffqrcoder_upstream_enabled"):
            updates["controlnet_pipeline_mode"] = "img2img"
        return Settings.model_validate({**self.base_settings.model_dump(), **updates})

    @staticmethod
    def _target_variant_for_method(method: LabMethod) -> str | None:
        target = method.output_variant
        if target == "auto":
            return None
        if target == "srpg" and not method.tools.srpg_enabled:
            raise ValueError("output_variant 'srpg' requires Stage 2 SRPG")
        if target == "srmpgd" and not method.tools.srmpgd_enabled:
            raise ValueError("output_variant 'srmpgd' requires paper SR-MPGD")
        if target == "guided" and not method.tools.guided_rediffusion_enabled:
            raise ValueError("output_variant 'guided' requires guided rediffusion")
        if target == "latent" and not method.tools.latent_refinement_enabled:
            raise ValueError("output_variant 'latent' requires latent refinement")
        if method.backend == "qr" and target != "raw":
            raise ValueError("the QR reference backend can only expose the raw variant")
        return target

    def _stage1_cache_key(
        self,
        method: LabMethod,
        prompt: str,
        negative_prompt: str,
        seed: int,
        error_correction: str,
    ) -> str:
        settings = self._settings_for_method(method)
        stage1_specification = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "error_correction": error_correction,
            "generation": method.generation,
            "base_model_id": settings.base_model_id,
            "controlnet_model_id": settings.controlnet_model_id,
            "controlnet_model_subfolder": settings.controlnet_model_subfolder,
            "controlnet_conditioning_profile": settings.controlnet_conditioning_profile,
            "controlnet_pipeline_mode": settings.controlnet_pipeline_mode,
            "diffqrcoder_upstream_enabled": settings.diffqrcoder_upstream_enabled,
            "diffqrcoder_revision": settings.diffqrcoder_revision,
            "diffqrcoder_qr_version": settings.diffqrcoder_qr_version,
            "diffqrcoder_qr_mask_pattern": settings.diffqrcoder_qr_mask_pattern,
            "diffqrcoder_qr_module_size": settings.diffqrcoder_qr_module_size,
            "diffqrcoder_qr_padding_px": settings.diffqrcoder_qr_padding_px,
        }
        encoded = json.dumps(
            stage1_specification,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _stage2_cache_key(
        self,
        method: LabMethod,
        prompt: str,
        negative_prompt: str,
        seed: int,
        error_correction: str,
        payload: str,
    ) -> str:
        settings = self._settings_for_method(method)
        stage2_settings = {
            key: value
            for key, value in method.tools.settings.items()
            if not key.startswith("srmpgd_")
        }
        specification = {
            "stage1": self._stage1_cache_key(
                method,
                prompt,
                negative_prompt,
                seed,
                error_correction,
            ),
            "payload_hash": hashlib.sha256(payload.encode()).hexdigest(),
            "guidance_scale": method.generation.get("guidance_scale"),
            "stage2_settings": stage2_settings,
            "target_mode": settings.diffqrcoder_stage2_target_mode,
        }
        return hashlib.sha256(
            json.dumps(
                specification,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    @staticmethod
    def _generation_request(
        campaign: LabCampaignCreate,
        method: LabMethod,
        prompt: str,
        negative_prompt: str,
        seed: int,
    ) -> GenerationRequest:
        unknown = set(method.generation) - GENERATION_KEYS
        if unknown:
            raise ValueError(f"unsupported generation settings: {sorted(unknown)}")
        values = {
            "payload": campaign.payload,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "backend": method.backend,
            "error_correction": campaign.error_correction,
            "seed": seed,
            "max_attempts": campaign.max_attempts,
            **method.generation,
        }
        return GenerationRequest.model_validate(values)

    @staticmethod
    def _release_generation_service(service: GenerationService | None) -> None:
        if service is None:
            return
        for backend in service.backends.values():
            if hasattr(backend, "_pipeline"):
                backend._pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            logger.debug("lab_cuda_cleanup_failed", exc_info=True)


def method_schema(settings: Settings | None = None) -> dict[str, Any]:
    profiles = laboratory_profiles()
    if settings is not None:
        resolved_model = {
            "base_model_id": settings.base_model_id,
            "controlnet_model_id": settings.controlnet_model_id,
            "controlnet_model_subfolder": settings.controlnet_model_subfolder,
            "controlnet_conditioning_profile": settings.controlnet_conditioning_profile,
            "controlnet_pipeline_mode": settings.controlnet_pipeline_mode,
            "diffqrcoder_upstream_enabled": settings.diffqrcoder_upstream_enabled,
            "diffqrcoder_revision": settings.diffqrcoder_revision,
            "diffqrcoder_qr_version": settings.diffqrcoder_qr_version,
            "diffqrcoder_qr_mask_pattern": settings.diffqrcoder_qr_mask_pattern,
            "diffqrcoder_qr_module_size": settings.diffqrcoder_qr_module_size,
            "diffqrcoder_qr_padding_px": settings.diffqrcoder_qr_padding_px,
        }
        for profile in profiles:
            if profile["backend"] == "controlnet":
                profile["model"] = {**resolved_model, **profile["model"]}
    return {
        "generation": sorted(GENERATION_KEYS),
        "model": sorted(MODEL_SETTING_KEYS),
        "tools": sorted(TOOL_SETTING_KEYS),
        "profiles": profiles,
        "notes": {
            "scope": (
                "Pinned DiffQRCoder + Cetus-Mix Whalefall + QR Monster v2 only. "
                "Stage 2 follows Algorithm 1 with a real Reed-Solomon QArt target; "
                "no deterministic final repair or alternative ControlNet."
            ),
            "upstream_revision": DIFFQRCODER_MODEL_SETTINGS["diffqrcoder_revision"],
            "upstream_compatibility_patch": (
                "PerceptualLoss uses torch.stack instead of torch.tensor so the public "
                "VGG loss remains connected to autograd. Stage 2 is initialized from the "
                "noised Stage-1 VAE latent and SR-MPGD uses separate gamma/LPIPS weights."
            ),
            "payload_storage": (
                "The clear payload is held only in worker memory and is never persisted."
            ),
            "qart_contract": (
                "Public QArt appends a URL fragment. Results are validated against "
                "the same canonical URL and are never labelled exact byte payload."
            ),
        },
    }
