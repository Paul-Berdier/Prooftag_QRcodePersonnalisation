import importlib.util
from pathlib import Path

from prooftag_qr.config import Settings
from prooftag_qr.lab import method_schema
from prooftag_qr.schemas import LabCampaignCreate


def _module():
    path = Path(__file__).parents[1] / "scripts" / "e019-srmpgd-grid.py"
    spec = importlib.util.spec_from_file_location("e019_srmpgd_grid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e019_factorial_batch_is_complete_paired_and_api_valid(tmp_path):
    module = _module()
    manifest = module.build_manifest(
        method_schema(Settings(data_dir=tmp_path)),
        payload="https://ptag.io/t/e019",
        gamma=100.0,
        seeds=[51001],
    )

    validated = LabCampaignCreate.model_validate(manifest)
    srmpgd = [method for method in validated.methods if method.tools.srmpgd_enabled]
    triplets = {
        (
            method.tools.settings["srmpgd_max_iterations"],
            method.tools.settings["srmpgd_step_size"],
            method.tools.settings["srmpgd_lpips_weight"],
        )
        for method in srmpgd
    }

    assert len(validated.methods) == 23
    assert len(srmpgd) == 20
    assert triplets == {
        (iterations, 100.0, lpips)
        for iterations in module.ITERATIONS
        for lpips in module.LPIPS_WEIGHTS
    }
    assert validated.max_attempts == 1
