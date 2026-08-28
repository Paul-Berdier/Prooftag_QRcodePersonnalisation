from __future__ import annotations

import torch

from prooftag_qr.e036_trust_region import (
    BRANCH_GLOBAL,
    BRANCH_LOCAL,
    BRANCH_STRICT,
    DEFAULT_POLICIES,
    E036Config,
    dilate_active_modules,
    expand_active_module_mask,
    project_latent_candidate,
)


def test_e036_gamma_is_frozen_to_1000() -> None:
    assert E036Config().gamma == 1000.0
    assert {policy.name for policy in DEFAULT_POLICIES} == {
        BRANCH_GLOBAL,
        BRANCH_STRICT,
        BRANCH_LOCAL,
    }


def test_project_latent_candidate_keeps_candidate_inside_ball() -> None:
    center = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    candidate = torch.ones_like(center)
    projected = project_latent_candidate(candidate, center, 0.025)
    rms = torch.sqrt(torch.mean((projected - center) ** 2)).item()
    assert abs(rms - 0.025) < 1e-6


def test_project_latent_candidate_does_not_modify_safe_candidate() -> None:
    center = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    candidate = torch.full_like(center, 0.01)
    projected = project_latent_candidate(candidate, center, 0.025)
    assert torch.equal(projected, candidate)


def test_module_dilation_and_pixel_expansion() -> None:
    active = torch.zeros((1, 9), dtype=torch.bool)
    active[0, 4] = True
    dilated = dilate_active_modules(active, radius=1)
    assert int(dilated.sum()) == 9

    # 3x3 modules, each 2x2 pixels => 6x6 canvas.
    module_ids = torch.arange(9).reshape(3, 3).repeat_interleave(2, 0).repeat_interleave(2, 1).reshape(-1)
    pixels = expand_active_module_mask(
        active,
        module_ids,
        height=6,
        width=6,
        dilation_modules=0,
    )
    assert pixels.shape == (1, 1, 6, 6)
    assert int(pixels.sum()) == 4
