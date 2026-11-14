from __future__ import annotations

import os
from pathlib import Path

import pytest

from scs.config import SCSSettings


@pytest.fixture
def fresh_mac(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    user_home = tmp_path / "user"

    def home(_cls: type[Path]) -> Path:
        return user_home

    monkeypatch.setattr(Path, "home", classmethod(home))
    monkeypatch.setattr("scs.config.supports_local_models", lambda: True, raising=False)
    for key in tuple(os.environ):
        if key.startswith("SCS_") or key == "OPENAI_API_KEY":
            monkeypatch.delenv(key)
    return user_home


def test_fresh_supported_mac_selects_complete_local_profile(fresh_mac: Path) -> None:
    settings = SCSSettings()

    assert settings.embedding_provider == "local"
    assert settings.embedding_model.startswith("Qwen/Qwen3-Embedding-0.6B@")
    assert settings.embedding_dimension == 1024
    assert settings.reranking_provider == "local"
    assert settings.reranking_model is not None
    assert settings.reranking_model.startswith("Qwen/Qwen3-Reranker-0.6B@")
    assert settings.decision_provider == "local"
    assert settings.decision_model is not None
    assert settings.decision_model.startswith("aac6fef/laya-mlx@")
    assert settings.paths.model_cache == fresh_mac / ".scs" / "models"
    assert not settings.home.exists()


@pytest.mark.parametrize(
    "configured",
    [
        {"embedding_provider": "openai_compatible"},
        {"embedding_model": "existing/model", "embedding_dimension": 16},
        {"openai_compatible_base_url": "http://localhost:9999/v1"},
        {"openai_api_key": "generated-test-key"},
        {"decision_model": "disabled"},
        {"decision_model": "custom/model"},
        {"reranking_model": "custom/reranker"},
        {"embedding_batch_size": 8},
    ],
)
def test_existing_model_settings_keep_legacy_selection(
    fresh_mac: Path,
    configured: dict[str, object],
) -> None:
    del fresh_mac
    settings = SCSSettings.model_validate(configured)

    assert settings.embedding_provider != "local"
    assert settings.reranking_provider == "external"
    assert settings.decision_provider == "external"
    if "reranking_model" not in configured:
        assert settings.reranking_model is None
    if configured.get("decision_model") in (None, "disabled"):
        assert settings.decision_model is None


def test_existing_environment_and_toml_select_legacy_behavior(
    fresh_mac: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = fresh_mac / ".scs/config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('embedding_provider = "openai_compatible"\n')
    settings = SCSSettings()

    assert settings.embedding_provider == "openai_compatible"
    assert settings.embedding_dimension == 4096
    assert settings.decision_model is None
    assert settings.reranking_model is None

    config_path.unlink()
    monkeypatch.setenv("OPENAI_API_KEY", "generated-environment-key")
    settings = SCSSettings()

    assert settings.embedding_provider == "openai"
    assert settings.effective_openai_api_key == "generated-environment-key"
    assert settings.decision_model is None


def test_explicit_local_profile_supports_custom_external_embedding_role(
    fresh_mac: Path,
) -> None:
    del fresh_mac
    settings = SCSSettings(
        model_profile="local",
        embedding_provider="openai_compatible",
        embedding_model="custom/embedding",
        embedding_dimension=16,
        openai_compatible_base_url="http://localhost:9999/v1",
    )

    assert settings.embedding_provider == "openai_compatible"
    assert settings.embedding_model == "custom/embedding"
    assert settings.embedding_dimension == 16
    assert settings.reranking_provider == "local"
    assert settings.decision_provider == "local"


def test_explicit_local_role_does_not_enable_other_roles(fresh_mac: Path) -> None:
    del fresh_mac
    settings = SCSSettings(embedding_provider="local")

    assert settings.embedding_model.startswith("Qwen/Qwen3-Embedding-0.6B@")
    assert settings.embedding_dimension == 1024
    assert settings.reranking_model is None
    assert settings.decision_model is None


def test_explicit_disabled_optional_models_stay_disabled(fresh_mac: Path) -> None:
    del fresh_mac
    settings = SCSSettings(
        model_profile="local",
        decision_model="disabled",
        reranking_model=None,
    )

    assert settings.embedding_provider == "local"
    assert settings.decision_model is None
    assert settings.reranking_model is None


def test_external_profile_restores_legacy_defaults(fresh_mac: Path) -> None:
    del fresh_mac
    settings = SCSSettings(model_profile="external")

    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.reranking_model is None
    assert settings.decision_model is None


def test_unsupported_platform_retains_external_defaults(
    fresh_mac: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fresh_mac
    monkeypatch.setattr(
        "scs.config.supports_local_models", lambda: False, raising=False
    )

    assert SCSSettings().embedding_provider == "openai"
    with pytest.raises(ValueError, match="Apple Silicon"):
        SCSSettings(model_profile="local")


def test_local_roles_reject_wrong_identity_and_escaping_cache(fresh_mac: Path) -> None:
    with pytest.raises(ValueError, match="model"):
        SCSSettings(embedding_provider="local", embedding_model="unshipped/model")
    with pytest.raises(ValueError, match="dimension"):
        SCSSettings(embedding_provider="local", embedding_dimension=16)
    with pytest.raises(ValueError, match="model_cache"):
        SCSSettings(model_profile="local", model_cache=fresh_mac / "foreign")
