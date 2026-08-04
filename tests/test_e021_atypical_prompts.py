import importlib.util
from pathlib import Path

from prooftag_qr.config import Settings
from prooftag_qr.lab import method_schema
from prooftag_qr.schemas import LabCampaignCreate


def _module():
    path = Path(__file__).parents[1] / "scripts" / "e021-atypical-prompts.py"
    spec = importlib.util.spec_from_file_location("e021_atypical_prompts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e021_prompt_sets_are_disjoint_and_complete():
    module = _module()
    core = module.selected_prompts("core")
    stress = module.selected_prompts("stress")
    all_prompts = module.selected_prompts("all")

    assert len(core) == 6
    assert len(stress) == 6
    assert len(all_prompts) == 12
    assert {item["id"] for item in core}.isdisjoint(
        {item["id"] for item in stress}
    )
    assert all(item["negative_prompt"] for item in all_prompts)


def test_e021_core_manifest_is_paired_minimal_and_api_valid(tmp_path):
    module = _module()
    manifest = module.build_manifest(
        method_schema(Settings(data_dir=tmp_path)),
        payload="https://ptag.io/t/e021",
        prompt_set="core",
        seeds=[51001],
    )

    validated = LabCampaignCreate.model_validate(manifest)
    assert tuple(method.id for method in validated.methods) == module.PROFILE_IDS
    assert len(validated.prompts) == 6
    assert len(validated.methods) * len(validated.prompts) == 24
    assert validated.max_attempts == 1


def test_e021_can_add_reference_and_two_seeds(tmp_path):
    module = _module()
    manifest = module.build_manifest(
        method_schema(Settings(data_dir=tmp_path)),
        payload="https://ptag.io/t/e021",
        prompt_set="stress",
        seeds=[51001, 52001],
        include_reference=True,
    )

    validated = LabCampaignCreate.model_validate(manifest)
    assert validated.methods[0].id == "qr_reference"
    assert len(validated.methods) == 5
    assert len(validated.prompts) * len(validated.seeds) * len(validated.methods) == 60
