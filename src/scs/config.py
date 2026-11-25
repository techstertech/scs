"""Typed configuration for the independent SCS daemon."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from ipaddress import ip_address
import os
from pathlib import Path
import platform
import sys
import tomllib
from typing import ClassVar, Final, Literal, cast, override
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from scs.paths import SCSPaths, default_log_directory, default_runtime_directory

DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_OPENAI_EMBEDDING_DIMENSION = 3072
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LOCAL_EMBEDDING_MODEL = "Qwen3-Embedding-8B-4bit-DWQ"
DEFAULT_LOCAL_EMBEDDING_DIMENSION = 4096
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "http://127.0.0.1:10000/v1"
DEFAULT_DECISION_BASE_URL = "http://127.0.0.1:10000/v1"
DEFAULT_BUNDLED_EMBEDDING_DIMENSION: Final[int] = 1024

# Presence, including an explicit disable value, preserves existing intent.
# Paths alone do not select a provider, except the legacy model-cache override.
LEGACY_MODEL_SETTINGS: Final[frozenset[str]] = frozenset(
    {
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "embedding_batch_size",
        "embedding_concurrency",
        "model_cache",
        "openai_base_url",
        "openai_api_key",
        "OPENAI_API_KEY",
        "SCS_OPENAI_API_KEY",
        "openai_compatible_base_url",
        "openai_compatible_trusted_hosts",
        "reranking_model",
        "reranking_provider",
        "decision_provider",
        "decision_model",
        "decision_base_url",
        "decision_trusted_hosts",
        "decision_timeout_seconds",
        "decision_max_concurrency",
    }
)


def supports_local_models() -> bool:
    """Keep unsupported systems on the existing external-provider defaults."""

    if sys.platform != "darwin" or platform.machine().lower() != "arm64":
        return False
    release = platform.mac_ver()[0]
    return bool(release) and int(release.split(".", 1)[0]) >= 14


def _resolve_config_toml_path(home: str | Path | None = None) -> Path:
    """Check <scs_home>/config.toml when SCS_HOME/home is set, else ~/.scs/config.toml."""

    scs_home = home if home is not None else os.environ.get("SCS_HOME")
    if scs_home is not None and str(scs_home).strip():
        candidate = Path(scs_home).expanduser() / "config.toml"
        if candidate.is_file():
            return candidate
    return Path.home() / ".scs" / "config.toml"


class SCSSettings(BaseSettings):
    """Environment-backed SCS settings."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="SCS_", extra="ignore"
    )

    home: Path = Field(default_factory=lambda: Path.home() / ".scs")
    model_cache: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "scs" / "models"
    )
    runtime_dir: Path = Field(default_factory=default_runtime_directory)
    log_dir: Path = Field(default_factory=default_log_directory)
    model_profile: Literal["auto", "local", "external"] = "auto"
    embedding_provider: Literal["openai", "openai_compatible", "mlx", "local"] = (
        DEFAULT_EMBEDDING_PROVIDER
    )
    embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    embedding_dimension: int = Field(default=DEFAULT_OPENAI_EMBEDDING_DIMENSION, gt=0)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_concurrency: int = Field(default=1, ge=1, le=4)
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SCS_OPENAI_API_KEY", "OPENAI_API_KEY", "openai_api_key"
        ),
    )
    openai_compatible_base_url: str = DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    openai_compatible_trusted_hosts: list[str] = Field(default_factory=list)
    reranking_model: str | None = None
    reranking_provider: Literal["external", "local", "disabled"] = "external"
    decision_model: str | None = Field(default=None, min_length=1, max_length=256)
    decision_provider: Literal["external", "local", "disabled"] = "external"
    decision_base_url: str = DEFAULT_DECISION_BASE_URL
    decision_trusted_hosts: list[str] = Field(default_factory=list)
    decision_timeout_seconds: float = Field(default=0.5, gt=0, le=1.0)
    decision_max_concurrency: int = Field(default=1, ge=1, le=4)
    index_text_fallback: bool = True
    index_max_file_bytes: int = Field(default=1_048_576, ge=1)
    index_text_sample_bytes: int = Field(default=8_192, ge=1)
    index_large_dir_files: int = Field(default=10_000, ge=1)
    index_large_dir_bytes: int = Field(default=536_870_912, ge=1)
    auto_reindex_enabled: bool = True
    auto_reindex_active_seconds: float = Field(default=2.0, gt=0)
    auto_reindex_idle_seconds: float = Field(default=30.0, gt=0)
    auto_reindex_debounce_seconds: float = Field(default=0.5, ge=0)
    auto_reindex_git_timeout_seconds: float = Field(default=10.0, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _select_provider_defaults(cls, values: object) -> object:
        """Keep model and dimension defaults coherent with provider selection."""

        if not isinstance(values, Mapping):
            return values
        configured = dict(cast(Mapping[str, object], values))
        profile = configured.get("model_profile", "auto")
        local_defaults = profile == "local" or (
            profile == "auto"
            and not LEGACY_MODEL_SETTINGS.intersection(configured)
            and supports_local_models()
        )
        if local_defaults:
            configured.setdefault("embedding_provider", "local")
            for role in ("reranking", "decision"):
                configured.setdefault(
                    f"{role}_provider",
                    "external" if f"{role}_model" in configured else "local",
                )

        provider = configured.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
        if provider in {"openai_compatible", "mlx"}:
            configured.setdefault("embedding_model", DEFAULT_LOCAL_EMBEDDING_MODEL)
            configured.setdefault(
                "embedding_dimension", DEFAULT_LOCAL_EMBEDDING_DIMENSION
            )

        local_embedding = provider == "local"
        local_reranking = configured.get("reranking_provider") == "local"
        local_decision = configured.get("decision_provider") == "local"
        if local_embedding or local_reranking or local_decision:
            from scs.local_models.artifacts import ModelRole, load_manifest

            models = load_manifest()
            if local_embedding:
                configured.setdefault(
                    "embedding_model", models[ModelRole.EMBEDDING].identity
                )
                configured.setdefault(
                    "embedding_dimension", DEFAULT_BUNDLED_EMBEDDING_DIMENSION
                )
            if local_reranking:
                configured.setdefault(
                    "reranking_model", models[ModelRole.RERANKING].identity
                )
            if local_decision:
                configured.setdefault(
                    "decision_model", models[ModelRole.DECISION].identity
                )
            home = configured.get("home", Path.home() / ".scs")
            if isinstance(home, (str, Path)):
                configured.setdefault("model_cache", Path(home) / "models")
        return configured

    @model_validator(mode="after")
    def _validate_local_models(self) -> "SCSSettings":
        """Reject unsupported bundled identities without affecting custom services."""

        if self.model_profile == "external" and (
            self.embedding_provider == "local"
            or self.reranking_provider == "local"
            or self.decision_provider == "local"
        ):
            raise ValueError("model_profile='external' cannot enable local model roles")
        if not self.uses_local_models:
            return self
        if not supports_local_models():
            raise ValueError("local models require Apple Silicon and macOS 14 or later")

        from scs.local_models.artifacts import ModelRole, load_manifest

        models = load_manifest()
        for selected, identity, role in (
            (
                self.embedding_provider == "local",
                self.embedding_model,
                ModelRole.EMBEDDING,
            ),
            (
                self.reranking_provider == "local",
                self.reranking_model,
                ModelRole.RERANKING,
            ),
            (
                self.decision_provider == "local",
                self.decision_model,
                ModelRole.DECISION,
            ),
        ):
            if selected and identity is not None and identity != models[role].identity:
                raise ValueError(
                    f"local {role.value} model must use the shipped identity"
                )
        if (
            self.embedding_provider == "local"
            and self.embedding_dimension != DEFAULT_BUNDLED_EMBEDDING_DIMENSION
        ):
            raise ValueError("local embedding dimension must be 1024")
        if (
            not self.model_cache.expanduser()
            .resolve()
            .is_relative_to(self.home.expanduser().resolve())
        ):
            raise ValueError("local model_cache must remain inside SCS_HOME")
        return self

    @property
    def uses_local_models(self) -> bool:
        """Describe effective ownership, including mixed provider configurations."""

        return (
            self.embedding_provider == "local"
            or (self.reranking_provider == "local" and self.reranking_model is not None)
            or (self.decision_provider == "local" and self.decision_model is not None)
        )

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load the SCS-owned TOML file below explicit values and environment."""

        raw_init_kwargs = cast(object, getattr(init_settings, "init_kwargs", None))
        init_home: str | Path | None = None
        if isinstance(raw_init_kwargs, Mapping):
            candidate_home = cast(Mapping[str, object], raw_init_kwargs).get("home")
            if isinstance(candidate_home, (str, Path)):
                init_home = candidate_home

        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(
                settings_cls, _resolve_config_toml_path(init_home)
            ),
            dotenv_settings,
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _validate_index_limits(self) -> "SCSSettings":
        """Keep bounded text detection within the accepted file envelope."""

        if self.index_text_sample_bytes > self.index_max_file_bytes:
            raise ValueError("index text sample cannot exceed maximum file size")
        if self.auto_reindex_idle_seconds < self.auto_reindex_active_seconds:
            raise ValueError(
                "automatic reindex idle interval cannot be shorter than active interval"
            )
        config_path = _resolve_config_toml_path(self.home)
        if config_path.is_file():
            with config_path.open("rb") as config_file:
                config_values = tomllib.load(config_file)
            if "openai_api_key" in config_values:
                permissions = config_path.stat().st_mode & 0o777
                if permissions & 0o077:
                    raise ValueError(
                        "config.toml containing openai_api_key must have "
                        "owner-only permissions"
                    )
        return self

    @field_validator("decision_base_url")
    @classmethod
    def _validate_decision_base_url(cls, value: str) -> str:
        """Validate the decision transport before applying host trust policy."""

        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "decision_base_url must be an absolute credential-free http URL"
            )
        if parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1":
            raise ValueError("decision_base_url must end at /v1")
        return value.rstrip("/")

    @field_validator("openai_compatible_base_url")
    @classmethod
    def _validate_openai_compatible_base_url(cls, value: str) -> str:
        """Validate the local transport before applying the host trust policy."""

        parsed = urlsplit(value)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("openai_compatible_base_url must be an absolute http URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _validate_openai_compatible_host_trust(self) -> "SCSSettings":
        """Require explicit trust before sending entity text off the workstation."""

        for base_url, trusted, field in (
            (
                self.openai_compatible_base_url,
                self.openai_compatible_trusted_hosts,
                "openai_compatible_base_url",
            ),
            (self.decision_base_url, self.decision_trusted_hosts, "decision_base_url"),
        ):
            host = urlsplit(base_url).hostname
            assert host is not None
            host = host.lower()
            try:
                is_loopback = ip_address(host).is_loopback
            except ValueError:
                is_loopback = host == "localhost"
            if not is_loopback and host not in {item.lower() for item in trusted}:
                raise ValueError(
                    f"{field} must use a loopback host or an exact trusted host"
                )
        return self

    @field_validator("decision_model")
    @classmethod
    def _validate_decision_model(cls, value: str | None) -> str | None:
        """Select an external model while retaining the existing disable marker.

        External identities belong to the configured service. Bundled local
        identities are validated separately against the shipped manifest.
        """

        if value is None or value == "disabled":
            return None

        model = value.strip()
        if not model:
            raise ValueError("decision model cannot be blank")

        return model

    @field_validator("reranking_model")
    @classmethod
    def _validate_reranking_model(cls, value: str | None) -> str | None:
        """Treat model presence as explicit reranking activation."""

        if value is None:
            return None
        model = value.strip()
        if not model:
            raise ValueError("reranking model cannot be blank")
        return model

    @field_validator("openai_base_url")
    @classmethod
    def _validate_openai_base_url(cls, value: str) -> str:
        """Require TLS for source-bearing requests sent beyond the workstation."""

        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("openai_base_url must be an absolute https URL")
        return value.rstrip("/")

    @property
    def effective_openai_api_key(self) -> str | None:
        """Expose credentials only when the selected provider may receive them."""

        if self.embedding_provider != "openai" or self.openai_api_key is None:
            return None
        return self.openai_api_key.get_secret_value()

    @cached_property
    def paths(self) -> SCSPaths:
        """Return validated product paths without creating them."""

        return SCSPaths.resolve(
            self.home,
            model_cache=self.model_cache,
            runtime=self.runtime_dir,
            logs=self.log_dir,
        )
