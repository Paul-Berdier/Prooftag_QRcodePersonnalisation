import importlib.util
from pathlib import Path

from prooftag_qr.config import Settings
from prooftag_qr.lab import method_schema
from prooftag_qr.schemas import LabCampaignCreate


def _module():
    path = Path(__file__).parents[1] / "scripts" / "e020-srmpgd-robust-probe.py"
    spec = importlib.util.spec_from_file_location("e020_srmpgd_robust_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e020_probe_is_minimal_paired_and_api_valid(tmp_path):
    module = _module()
    manifest = module.build_manifest(
        method_schema(Settings(data_dir=tmp_path)),
        payload="https://ptag.io/t/e020",
    )

    validated = LabCampaignCreate.model_validate(manifest)
    by_id = {method.id: method for method in validated.methods}
    official = by_id["diffqrcoder_srmpgd"]
    robust = by_id["diffqrcoder_srmpgd_robust"]

    assert tuple(by_id) == module.PROFILE_IDS
    assert len(validated.prompts) == 1
    assert validated.seeds == [51001]
    assert validated.max_attempts == 1
    assert official.tools.srmpgd_enabled is True
    assert robust.tools.srmpgd_enabled is True
    assert official.tools.settings.get("srmpgd_robust_blur_weight", 0.0) == 0.0
    assert robust.tools.settings["srmpgd_robust_blur_weight"] == 1.0
    assert official.tools.settings["srmpgd_step_size"] == robust.tools.settings[
        "srmpgd_step_size"
    ]
    assert official.tools.settings["srmpgd_max_iterations"] == robust.tools.settings[
        "srmpgd_max_iterations"
    ]
