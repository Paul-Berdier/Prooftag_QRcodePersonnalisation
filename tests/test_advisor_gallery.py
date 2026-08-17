from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from prooftag_qr.advisor_gallery import (
    download_advisor_gallery,
    render_advisor_contact_sheet,
    select_advisor_gallery,
    write_gallery_index,
)


def _record(prompt: str, method: str, seed: int, success: int):
    return SimpleNamespace(
        trial_id=f"{prompt}-{method}-{seed}",
        prompt_id=prompt,
        prompt_text=f"Visual prompt for {prompt}",
        parameters={"id": method, "output_variant": "srpg"},
        metadata={
            "method_id": method,
            "seed": seed,
            "generation_run_id": f"run-{prompt}-{method}-{seed}",
        },
        targets={
            "qr_success": success,
            "qr_tolerance": 0.9 if success else 0.1,
            "clip_aesthetic": 6.0 + success,
            "clip_score": 0.7,
            "hpsv2_1": 0.2,
            "saturation_risk": 0.01,
        },
    )


def test_advisor_gallery_keeps_paired_comparisons_and_survives_downloads(tmp_path):
    methods = ("stage1", "srpg", "srmpgd", "q250")
    records = []
    for prompt_index, family in enumerate(("simple", "atypical")):
        prompt = f"e026w_{family}_{prompt_index:03d}"
        for method_index, method in enumerate(methods):
            records.append(_record(prompt, method, 113_001, int(method_index > 0)))
            records.append(_record(prompt, method, 223_001, int(method_index > 0)))
    predictions = [
        {"calibrated_probability": 0.9 if record.targets["qr_success"] else 0.4}
        for record in records
    ]

    selected = select_advisor_gallery(
        records,
        validation_predictions=predictions,
        comparison_method_ids=methods,
        comparison_prompt_count=2,
        preferred_seed=113_001,
        section_size=1,
    )
    comparison = [entry for entry in selected if entry["section"] == "comparison"]
    assert len(comparison) == 8
    assert {entry["seed"] for entry in comparison} == {113_001.0}

    buffer = BytesIO()
    Image.new("RGB", (64, 64), "navy").save(buffer, format="PNG")
    downloaded = download_advisor_gallery(
        selected,
        api_url="http://unused",
        output_dir=tmp_path / "gallery" / "images",
        fetcher=lambda run_id: buffer.getvalue(),
    )
    assert all(entry["local_image"] for entry in downloaded)
    sheet = render_advisor_contact_sheet(
        comparison,
        title="Paired comparison",
        output_path=tmp_path / "gallery" / "comparison.png",
    )
    assert sheet.is_file()

    write_gallery_index(downloaded, tmp_path / "gallery")
    assert (tmp_path / "gallery" / "gallery-index.csv").is_file()
    assert (tmp_path / "gallery" / "gallery-audit.json").is_file()
