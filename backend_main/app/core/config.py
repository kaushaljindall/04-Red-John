"""
Typed application settings, loaded from environment variables and `.env`.

Every knob the backend has lives here. Nothing else reads `os.environ`.
"""
from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class StorageBackend(StrEnum):
    MEMORY = "memory"
    MONGO = "mongo"


class BusBackend(StrEnum):
    MEMORY = "memory"
    REDIS = "redis"


class ExecutionMode(StrEnum):
    INLINE = "inline"   # missions run inside the API process (dev, single box)
    QUEUE = "queue"     # the API enqueues; `python -m app.worker` processes run them


def _split_csv(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(v).strip() for v in value if str(v).strip()]  # type: ignore[union-attr]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # List fields are Annotated with NoDecode: env values are comma-separated, not JSON.
    # Fields with env aliases (GROQ_EXTREME_POOL…) must still accept their own name in code.
        populate_by_name=True,
    )

    # ── Application ───────────────────────────────────────────────
    app_name: str = "DevPilot"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["http://localhost:3000"])

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = False

    # ── Storage / bus ─────────────────────────────────────────────
    storage_backend: StorageBackend = StorageBackend.MEMORY
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "devpilot"
    bus_backend: BusBackend = BusBackend.MEMORY
    redis_url: str = "redis://localhost:6379/0"

    # ── LLM ───────────────────────────────────────────────────────
    # Accepts Mycel's variable names so an existing .env keeps working.
    groq_api_keys: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        validation_alias=AliasChoices("GROQ_API_KEYS", "GROQ_EXTREME_POOL"),
    )
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_api_keys: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        validation_alias=AliasChoices("GEMINI_API_KEYS", "GEMINI_API_KEY"),
    )
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    llm_model_smart: str = "openai/gpt-oss-120b"
    llm_model_fast: str = "openai/gpt-oss-20b"
    llm_fallback_model: str = "gemini-3.8-flash"
    llm_max_concurrency: int = 4
    llm_request_timeout_s: float = 90.0
    llm_max_attempts: int = 12
    # Per-request limits. Groq's on_demand tier rejects any single request above 8,000 tokens (input plus
    # max_tokens), so agents fit their context to this. Raise it on a paid tier or another provider.
    llm_max_request_tokens: int = 7800
    llm_max_output_tokens: int = 2000
    llm_chars_per_token: float = 3.0       # conservative estimate for JSON-heavy prompts

    # ── Sandbox ───────────────────────────────────────────────────
    data_dir: Path = BACKEND_ROOT / "var"
    command_timeout_s: int = 180
    command_max_output_chars: int = 30_000
    max_file_read_bytes: int = 400_000
    # Target repos may only be opened from under these roots. Empty = any local path.
    allowed_repo_roots: Annotated[list[Path], NoDecode] = Field(default_factory=list)

    # ── Governance ────────────────────────────────────────────────
    approval_timeout_s: int = 300
    # Approval categories that skip the human (e.g. "install" on a trusted dev box). Empty in production.
    auto_approve_categories: Annotated[list[str], NoDecode] = Field(default_factory=list)
    mission_max_tool_calls: int = 400
    mission_max_llm_tokens: int = 2_000_000
    mission_max_wall_clock_s: int = 45 * 60
    mission_max_identical_failures: int = 3
    mission_max_consecutive_failures: int = 8

    # ── Hiring (Maya) ─────────────────────────────────────────────
    hiring_threshold: float = 0.5
    hiring_llm_mandates: bool = True       # let the LLM write each hire's one-line mandate
    escalate_after_failures: int = 2       # a task failing this often goes to a different specialist

    # ── Orchestration (Atlas) ─────────────────────────────────────
    orchestrator_max_parallel: int = 3     # tasks run at once when the plan allows it
    orchestrator_max_rounds: int = 60      # hard stop for one run of the decision loop
    repair_max_attempts: int = 4           # rounds of fix → re-check before a human is asked
    llm_retry_backoff_s: float = 5.0       # wait before retrying a task whose model call failed

    # ── Execution (Phase 8) ───────────────────────────────────────
    # The queue and leases use Redis when BUS_BACKEND=redis, otherwise they are in-process.
    execution_mode: ExecutionMode = ExecutionMode.INLINE
    worker_concurrency: int = 4            # missions one worker runs at once
    lease_ttl_s: float = 60.0              # a runner that stops renewing for this long loses the mission
    job_claim_after_s: float = 180.0       # a job silent this long is handed to another worker
    worker_max_deliveries: int = 5         # a mission that crashes its worker this often is failed

    @field_validator("groq_api_keys", "gemini_api_keys", "cors_origins", "auto_approve_categories", mode="before")
    @classmethod
    def _csv(cls, value: object) -> list[str]:
        return _split_csv(value)

    @field_validator("allowed_repo_roots", mode="before")
    @classmethod
    def _csv_paths(cls, value: object) -> list[Path]:
        return [Path(p).expanduser().resolve() for p in _split_csv(value)]

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


@lru_cache
def get_settings() -> Settings:
    return Settings()
