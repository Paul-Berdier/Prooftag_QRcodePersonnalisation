from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_large_gpu_jobs_are_single_gpu_sequential_and_persistent():
    for name, command in (
        ("e046-large-parent-job.yaml", "generate-parent"),
        ("e046-large-refinement-job.yaml", "generate-refinement"),
    ):
        document = yaml.safe_load(_read(f"deploy/k8s/{name}"))
        assert document["metadata"]["labels"]["prooftag.io/experiment"] == (
            "e046-large-advisor-dataset-v1"
        )
        spec = document["spec"]
        assert spec["backoffLimit"] == 0
        assert spec["completions"] == 1
        assert spec["parallelism"] == 1
        assert spec["ttlSecondsAfterFinished"] == 86400
        assert "prooftag.io/plan-id" in document["metadata"]["annotations"]
        assert "prooftag.io/candidate-id" in document["metadata"]["annotations"]
        pod = spec["template"]["spec"]
        assert pod["runtimeClassName"] == "nvidia"
        assert pod["restartPolicy"] == "Never"
        container = pod["containers"][0]
        assert container["imagePullPolicy"] == "Never"
        assert command in container["command"]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
        assert any(
            volume.get("persistentVolumeClaim", {}).get("claimName")
            == "prooftag-qr-data"
            for volume in pod["volumes"]
        )


def test_large_job_templates_render_substitutions_as_strings():
    replacements = {
        "__JOB_NAME__": "prooftag-qr-e046-large-p-123456789012",
        "__NAMESPACE__": "qr-core",
        "__IMAGE__": "docker.io/library/prooftag-qr@sha256:" + "a" * 64,
        "__OUTPUT_ROOT__": "/data/e046-large-advisor-dataset-v1",
        "__PLAN_ID__": "1234567890123456",
        "__CANDIDATE_ID__": "1234567890123456",
        "__RECIPE_ID__": "12345678",
        "__SOURCE_COMMIT__": "1" * 40,
        "__IMAGE_DIGEST__": "sha256:" + "a" * 64,
        "__OVERRIDE_SRL_TERMINAL__": "0",
    }
    for name in (
        "e046-large-parent-job.yaml",
        "e046-large-refinement-job.yaml",
    ):
        rendered = _read(f"deploy/k8s/{name}")
        for source, target in replacements.items():
            rendered = rendered.replace(source, target)
        assert "__" not in rendered
        document = yaml.safe_load(rendered)
        assert document["metadata"]["annotations"]["prooftag.io/plan-id"] == (
            "1234567890123456"
        )
        container = document["spec"]["template"]["spec"]["containers"][0]
        assert all(isinstance(argument, str) for argument in container["args"])
        assert "@sha256:" in container["image"]


def test_large_runner_is_safe_resumable_and_smoke_by_default():
    text = _read("scripts/run-e046-large-dataset.sh")
    assert 'action="${1:-plan}"' in text
    assert 'profile="${PROOFTAG_E046_LARGE_PROFILE:-smoke}"' in text
    assert (
        'override_srl_terminal="${PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL:-0}"'
        in text
    )
    assert '"$override_srl_terminal" != "0"' in text
    assert '"$override_srl_terminal" != "1"' in text
    assert "__OVERRIDE_SRL_TERMINAL__" in text
    assert (
        '"PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL=$override_srl_terminal"'
        in text
    )
    assert "job_has_terminal_srl_failure" in text
    assert "local upstream SRL port diverged from the pinned official class" in text
    assert "/data/e046-large-advisor-dataset-v1" in text
    assert "list-parents" in text
    assert "select-refinements" in text
    assert "list-refinements" in text
    assert "score-parents" in text
    assert "score-refinements" in text
    assert "GENERATION_COMPLETE" in text
    assert "SCORING_COMPLETE" in text
    assert "pause-after-current" in text
    assert "clear-pause" in text
    assert "cancel-current" in text
    assert "restore-runtime" in text
    assert "RUN-3072" in text
    assert 'plan["runtime_image_digest"]' in text
    assert '"$planned_commit" != "$source_commit"' in text
    assert 'if [[ "$action" == "plan" ]]; then' in text
    assert 'if [[ "$action" != "plan" && "$planned_profile" == "full"' in text
    assert "foreign_gpu_pods" in text
    assert "foreign_gpu_pods_except_api" in text
    ensure_api = text[text.index("ensure_api() {"):text.index("stop_gpu_services() {")]
    assert ensure_api.index("foreign_gpu_pods_except_api") < ensure_api.index(
        '--replicas=1'
    )
    restore = text[
        text.index("restore_runtime_if_idle() {"):text.index("active_exec_target() {")
    ]
    assert restore.index("foreign_gpu_pods_except_api") < restore.index(
        '--replicas=1'
    )
    assert "flock -n 9" in text
    assert '"$action" == "plan" || "$action" == "run"' in text
    assert "status.containerStatuses" in text
    assert ".imageID" in text
    assert 'current_kind="$(' in text
    assert "parallel" not in text.lower()
    assert "rm -rf" not in text
    assert "logs -f" not in text
    assert "kubectl delete namespace" not in text


def test_large_runner_keeps_old_e045_and_e046_roots_immutable():
    text = _read("scripts/run-e046-large-dataset.sh")
    assert "Ancien E045 resté intact" in text
    assert "Ancien E046 resté intact" in text
    assert "/data/e045-foundation-v1" in text
    assert "/data/e046-controlled-best-generator-v1" in text
    assert "dossiers historiques E045/E046 et leurs enfants sont intouchables" in text
    assert "PROOFTAG_E046_OUTPUT_ROOT" not in text


def test_large_refinement_job_receives_only_explicit_srl_terminal_override():
    refinement = yaml.safe_load(_read("deploy/k8s/e046-large-refinement-job.yaml"))
    parent = yaml.safe_load(_read("deploy/k8s/e046-large-parent-job.yaml"))

    def env_by_name(document):
        values = document["spec"]["template"]["spec"]["containers"][0]["env"]
        return {entry["name"]: entry.get("value") for entry in values}

    assert env_by_name(refinement)["PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL"] == (
        "__OVERRIDE_SRL_TERMINAL__"
    )
    assert "PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL" not in env_by_name(parent)


def test_generic_retry_cannot_override_terminal_srl_policy():
    text = _read("scripts/run-e046-large-dataset.sh")
    failed_branch = text.split(
        'elif [[ "${failed:-0}" -ge 1 ]]; then', 1
    )[1].split('elif [[ "${succeeded:-0}" -ge 1 ]]; then', 1)[0]

    terminal_check = (
        'if [[ "$kind" == "refinement" ]] && '
        'job_has_terminal_srl_failure "$job"; then'
    )
    assert terminal_check in failed_branch
    assert 'if [[ "$override_srl_terminal" != "1" ]]; then' in failed_branch
    assert "la relance générique est interdite" in failed_branch
    assert 'elif [[ "$retry_failed_jobs" != "1" ]]; then' in failed_branch
    assert failed_branch.index(terminal_check) < failed_branch.index("retry_failed_jobs")


def test_complete_plan_is_verified_without_rewriting_scoring_or_aggregate():
    text = _read("scripts/run-e046-large-dataset.sh")
    guard = 'if [[ "$action" != "plan" ]] && plan_is_complete; then'
    guard_start = text.index(guard)
    guard_end = text.index("\nfi", guard_start)
    guard_block = text[guard_start:guard_end]

    assert 'campaign_read_command verify --plan-id "$plan_id"' in guard_block
    assert "exit 0" in guard_block
    assert "clear-pause" not in guard_block
    assert guard_start < text.index("clear-pause", guard_start)
    assert guard_start < text.index("score-parents", guard_start)
    assert guard_start < text.index("aggregate", guard_start)


def test_large_deployer_checks_contract_without_launching_full():
    text = _read("scripts/deploy-e046-large-dataset.sh")
    assert "e046-large-advisor-dataset-v1" in text
    assert "e046-controlled-best-generator-v1" in text
    assert "bash scripts/deploy-app-image.sh" in text
    assert "bash scripts/deploy-notebook-image.sh" in text
    assert "notebooks/50_e046_large_advisor_dataset.ipynb" in text
    assert "bash -n scripts/run-e046-large-dataset.sh" in text
    assert "e046_large_campaign --help" in text
    assert "len(prompts) == 256" in text
    assert "CONFIG_COUNT == 12" in text
    assert "RUN-3072" in text
    assert "wait_for_pods_to_stop" in text
    assert "-l prooftag.io/experiment" in text
    assert "flock -n 9" in text
    assert "restore_previous_replicas" in text
    assert "previous_api_revision" in text
    assert "previous_api_template_sha" in text
    assert "rollback_api_if_changed" in text
    assert "rollback_notebook_if_changed" in text
    assert 'rollout undo deployment/"$api_deployment"' in text
    assert '--to-revision="$previous_api_revision"' in text
    assert "rollout status" in text
    assert "rollback_confirmed" in text
    assert 'if [[ "$rollback_confirmed" -eq 1 ]]' in text
    rollback = text.index("rollback_api_if_changed")
    restore = text.index("restore_previous_replicas", rollback)
    assert rollback < restore
    assert "nonterminal_experiment_pods" in text
    assert "foreign_gpu_pods_except" in text
    assert 'get pods -A -o json' in text
    assert "previous_api + previous_notebook + previous_vllm > 1" in text
    restore_body = text[
        text.index("restore_previous_replicas() {"):
        text.index("rollback_api_if_changed() {")
    ]
    assert restore_body.index('--replicas=0') < restore_body.index(
        'foreign_gpu_pods_except'
    ) < restore_body.index('--replicas="$previous_notebook"')
    assert '"$deployed_commit" == "$git_sha"' in text
    assert "pod_image_id" in text
    assert 'normalized_pod_image_id" =~ ^sha256:' in text
    assert 'deployed_build_digest" =~ ^sha256:' in text
    assert '"$normalized_pod_image_id" == "$deployed_build_digest"' in text
    assert 'normalized_pod_image_id" =~ @sha256:' in text
    assert "prooftag-build-commit.txt" in text
    assert "run-e046-large-dataset.sh run" in text
    assert "rm -rf" not in text
    assert "kubectl apply -f deploy/k8s/e046-large" not in text


def test_main_image_bakes_and_deployers_verify_an_independent_commit_attestation():
    dockerfile = _read("Dockerfile")
    notebook_dockerfile = _read("Dockerfile.notebook")
    app_deployer = _read("scripts/deploy-app-image.sh")
    notebook_deployer = _read("scripts/deploy-notebook-image.sh")
    large_deployer = _read("scripts/deploy-e046-large-dataset.sh")
    large_runner = _read("scripts/run-e046-large-dataset.sh")

    assert "ARG PROOFTAG_BUILD_COMMIT=development" in dockerfile
    assert 'LABEL org.opencontainers.image.revision="${PROOFTAG_BUILD_COMMIT}"' in dockerfile
    assert "/app/prooftag-build-commit.txt" in dockerfile
    assert "ARG PROOFTAG_BUILD_COMMIT=development" in notebook_dockerfile
    assert "/app/prooftag-build-commit.txt" in notebook_dockerfile
    assert '--build-arg "PROOFTAG_BUILD_COMMIT=${git_sha}"' in app_deployer
    assert "image_build_commit" in app_deployer
    assert "/app/prooftag-build-commit.txt" in app_deployer
    assert "docker run" not in app_deployer
    assert "docker create" in app_deployer
    assert "docker cp" in app_deployer
    assert '--build-arg "PROOFTAG_BUILD_COMMIT=${git_sha}"' in notebook_deployer
    assert "/app/prooftag-build-commit.txt" in notebook_deployer
    assert "/app/prooftag-build-commit.txt" in large_deployer
    assert "/app/prooftag-build-commit.txt" in large_runner
    assert 'embedded_commit" != "$deployed_commit"' in large_runner
    assert 'normalized_image_id" =~ ^sha256:' in large_runner
    assert '"$normalized_image_id" != "$build_image_digest"' in large_runner
    assert 'image="$image_tag"' in large_runner
    assert 'normalized_image_id" =~ @sha256:' in large_runner


def test_large_runtime_contract_uses_the_installed_qr_verify_bridge():
    campaign = _read("prooftag_qr/e046_large_campaign.py")
    dockerfile = _read("Dockerfile")
    notebook_dockerfile = _read("Dockerfile.notebook")

    assert 'os.environ.get("PROOFTAG_QR_QR_VERIFY_BRIDGE")' in campaign
    expected = "PROOFTAG_QR_QR_VERIFY_BRIDGE=/opt/prooftag-qr-verify/bridge.mjs"
    assert expected in dockerfile
    assert expected in notebook_dockerfile


def test_large_scripts_have_bash_syntax_when_bash_is_available():
    import os
    import shutil
    import subprocess

    import pytest

    if os.name == "nt":
        pytest.skip("Windows system bash is WSL and cannot read this workspace here")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    for name in (
        "scripts/run-e046-large-dataset.sh",
        "scripts/deploy-e046-large-dataset.sh",
    ):
        result = subprocess.run(
            [bash, "-n", str(ROOT / name)],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr
