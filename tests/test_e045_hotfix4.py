from pathlib import Path

from PIL import Image

from prooftag_qr.e045_foundation import FoundationConfig, _walk_relevant_files


def test_generic_artifact_policy_does_not_depend_on_raster_name(tmp_path: Path):
    data = tmp_path / "data"
    cfg = FoundationConfig(
        data_root=data,
        output_root=data / "e045-foundation-v1",
        source_commit="b" * 40,
        worker_id="pytest-hotfix4",
        max_files=100,
    )

    folder = data / "artifacts" / "nested" / "srmpgd" / "stage2"
    folder.mkdir(parents=True)

    for name in (
        "final.png",
        "winner.png",
        "stage2.png",
        "srmpgd-final.png",
        "ordinary-frame.png",
    ):
        Image.new("RGB", (8, 8), "white").save(folder / name)

    (folder / "results.json").write_text("[]", encoding="utf-8")

    stats = {}
    selected = list(_walk_relevant_files(cfg, selection_stats=stats))

    assert selected == [folder / "results.json"]
    assert stats["generic_artifact_images_deferred"] == 5
    assert stats["generic_artifact_priority_images"] == 0
