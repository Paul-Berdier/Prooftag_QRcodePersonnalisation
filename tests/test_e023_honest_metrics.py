import importlib.util
from pathlib import Path

from prooftag_qr.config import Settings
from prooftag_qr.lab import method_schema
from prooftag_qr.schemas import LabCampaignCreate


def _module():
    path = Path(__file__).parents[1] / "scripts" / "e023-honest-software-metrics.py"
    spec = importlib.util.spec_from_file_location("e023_honest_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e023_reuses_the_exact_e022_comparison_matrix(tmp_path):
    module = _module()
    manifest = module.build_manifest(
        method_schema(Settings(data_dir=tmp_path)),
        payload="https://ptag.io/t/e022",
        prompt_family="all",
        seeds=[61001],
    )
    request = LabCampaignCreate.model_validate(manifest)

    assert len(request.prompts) == 10
    assert len(request.methods) == 2
    assert request.seeds == [61001]
    assert request.payload == "https://ptag.io/t/e022"
    assert {method.id for method in request.methods} == {
        "diffqrcoder_srpg",
        "diffqrcoder_paper_srpg",
    }
