from __future__ import annotations

import pytest

from prooftag_qr.runtime import runtime_deployment_identity

RUNTIME_ENVIRONMENT_KEYS = (
    "PROOFTAG_GIT_COMMIT",
    "PROOFTAG_RUNTIME_IMAGE",
    "PROOFTAG_RUNTIME_IMAGE_DIGEST",
)


def clear_runtime_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in RUNTIME_ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)


def test_runtime_identity_is_explicitly_unconfigured_outside_deployment(monkeypatch):
    clear_runtime_identity(monkeypatch)

    assert runtime_deployment_identity() == {
        "configured": False,
        "git_commit": None,
        "image": None,
        "image_digest": None,
    }


def test_runtime_identity_exposes_the_exact_deployed_commit_image_and_digest(monkeypatch):
    clear_runtime_identity(monkeypatch)
    monkeypatch.setenv("PROOFTAG_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("PROOFTAG_RUNTIME_IMAGE", "prooftag-qr:git-aaaaaaaaaaaa")
    monkeypatch.setenv("PROOFTAG_RUNTIME_IMAGE_DIGEST", f"sha256:{'b' * 64}")

    assert runtime_deployment_identity() == {
        "configured": True,
        "git_commit": "a" * 40,
        "image": "prooftag-qr:git-aaaaaaaaaaaa",
        "image_digest": f"sha256:{'b' * 64}",
    }


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("PROOFTAG_GIT_COMMIT", "short", "full lowercase Git SHA"),
        ("PROOFTAG_RUNTIME_IMAGE", "bad image", "image reference"),
        ("PROOFTAG_RUNTIME_IMAGE_DIGEST", "sha256:short", "sha256 digest"),
    ],
)
def test_runtime_identity_fails_closed_on_partial_or_invalid_values(
    monkeypatch,
    name,
    value,
    message,
):
    clear_runtime_identity(monkeypatch)
    monkeypatch.setenv("PROOFTAG_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("PROOFTAG_RUNTIME_IMAGE", "prooftag-qr:git-aaaaaaaaaaaa")
    monkeypatch.setenv("PROOFTAG_RUNTIME_IMAGE_DIGEST", f"sha256:{'b' * 64}")
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        runtime_deployment_identity()


def test_runtime_identity_rejects_an_incomplete_binding(monkeypatch):
    clear_runtime_identity(monkeypatch)
    monkeypatch.setenv("PROOFTAG_GIT_COMMIT", "a" * 40)

    with pytest.raises(RuntimeError, match="image reference"):
        runtime_deployment_identity()
