import importlib.util
from pathlib import Path

from prooftag_qr.config import Settings
from prooftag_qr.lab import LabService, method_schema
from prooftag_qr.schemas import LabCampaignCreate, LabMethod


def _module():
    path = Path(__file__).parents[1] / "scripts" / "e022-paper-vs-prooftag.py"
    spec = importlib.util.spec_from_file_location("e022_paper_vs_prooftag", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e022_has_five_new_prompts_per_prespecified_family():
    module = _module()
    simple = module.selected_prompts("simple")
    atypical = module.selected_prompts("atypical")
    all_prompts = module.selected_prompts("all")
    assert len(simple) == 5
    assert len(atypical) == 5
    assert len(all_prompts) == 10
    assert {item["id"] for item in simple}.isdisjoint({item["id"] for item in atypical})
    assert all(item["negative_prompt"] for item in all_prompts)


def test_e022_manifest_contains_only_the_two_paired_stage2_recipes(tmp_path):
    module = _module()
    schema = method_schema(Settings(data_dir=tmp_path))
    manifest = module.build_manifest(
        schema, payload="https://ptag.io/t/e022", prompt_family="all", seeds=[61001]
    )
    validated = LabCampaignCreate.model_validate(manifest)
    assert tuple(method.id for method in validated.methods) == module.PROFILE_IDS
    assert len(validated.prompts) == 10
    assert len(validated.methods) * len(validated.prompts) == 20

    profiles = {item["id"]: item for item in schema["profiles"]}
    safe = LabMethod.model_validate(profiles["diffqrcoder_srpg"])
    paper = LabMethod.model_validate(profiles["diffqrcoder_paper_srpg"])
    assert safe.tools.settings["diffqrcoder_stage2_target_mode"] == "binary_exact"
    assert safe.tools.settings["diffqrcoder_stage2_strength"] == 0.65
    assert safe.tools.settings["srpg_perceptual_weight"] == 2.0
    assert paper.tools.settings["diffqrcoder_stage2_target_mode"] == "qart_url_fragment"
    assert paper.tools.settings["diffqrcoder_stage2_strength"] == 1.0
    assert paper.tools.settings["srpg_perceptual_weight"] == 3.0


def test_e022_recipes_share_the_exact_same_stage1_cache_key(tmp_path):
    module = _module()
    profiles = {
        item["id"]: LabMethod.model_validate(item)
        for item in method_schema(Settings(data_dir=tmp_path))["profiles"]
    }
    safe = profiles[module.PROFILE_IDS[0]]
    paper = profiles[module.PROFILE_IDS[1]]
    service = LabService.__new__(LabService)
    service.base_settings = Settings(data_dir=tmp_path)
    safe_key = service._stage1_cache_key(safe, "new prompt", "easynegative", 61001, "M")
    paper_key = service._stage1_cache_key(paper, "new prompt", "easynegative", 61001, "M")
    assert safe_key == paper_key
