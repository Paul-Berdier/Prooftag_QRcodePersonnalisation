from prooftag_qr.e045_registry import EXPERIMENTS, SUBEXPERIMENTS, by_id


def test_registry_covers_every_integer_experiment_e000_to_e044():
    assert [item["id"] for item in EXPERIMENTS] == [
        f"E{index:03d}" for index in range(45)
    ]
    assert len({item["id"] for item in EXPERIMENTS}) == 45


def test_known_invalid_and_holdout_policies_are_not_training_positives():
    assert by_id("E016")["training_policy"] == "quarantine"
    assert by_id("E031")["training_policy"] == "evaluation_only"
    assert by_id("E044")["training_policy"] == "training_candidate_software_only"


def test_subexperiments_remain_visible():
    ids = {item["id"] for item in SUBEXPERIMENTS}
    assert {"E014A", "E014B", "E014C", "E014D", "E014E", "E014F"} <= ids
    assert {"E026I", "E026J"} <= ids
