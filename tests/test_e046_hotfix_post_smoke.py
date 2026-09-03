from pathlib import Path

from prooftag_qr.e046_campaign import _parent_visual_guard, _row_qr_fields


def test_original_exact_uses_conservative_original_field():
    fields = _row_qr_fields(
        {
            "conservative_exact_presets": 37,
            "direct_exact_all_repetitions": True,
        }
    )
    assert fields["wechat_exact_presets"] == 37
    assert fields["wechat_original_exact"] is True


def test_parent_guard_allows_stage2_relative_clip_drop():
    row = {
        "mean_absolute_change": 0.176,
        "clipped_pixel_ratio_increase": -0.0002,
        "rgb_clipped_channel_ratio_increase": -0.0002,
        "saturation_mean_increase": -0.045,
        "high_saturation_ratio_increase": 0.0,
        "clip_score": 0.702,
        "clip_aesthetic": 5.04,
        "hpsv2_1": 0.162,
    }
    stage1 = {
        "clip_score": 0.813,
        "clip_aesthetic": 5.226,
        "hpsv2_1": 0.192,
    }
    guard = _parent_visual_guard(
        row=row,
        stage1_quality=stage1,
        scene_qz_guard=None,
    )
    assert guard["passed"] is True
    assert guard["stage1_quality_deltas_diagnostic_only"]["clip_score"] < -0.10


def test_notebook_optional_pareto_columns_do_not_raise():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/build_e046_notebooks.py").read_text(encoding="utf-8")
    assert "pareto_df.reindex(columns=pareto_columns)" in source
    assert "best_df.reindex(columns=best_columns)" in source
