from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scidata_agent.agent.schemas import AgentState


CHECKPOINT_VERSION = 1


def build_run_fingerprint(
    research_question: str,
    files: list[str | Path],
    options: dict[str, Any],
) -> str:
    """Build a stable identity for a resumable pipeline invocation."""
    file_specs: list[dict[str, Any]] = []
    for value in files:
        path = Path(value).expanduser()
        spec: dict[str, Any] = {"path": str(path.resolve())}
        try:
            stat = path.stat()
            spec.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        except OSError:
            spec["missing"] = True
        file_specs.append(spec)
    payload = {
        "research_question": research_question,
        "files": file_specs,
        "options": options,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AgentCheckpointStore:
    """Persist stage results atomically under one task's output directory."""

    def __init__(self, task_dir: str | Path) -> None:
        self.task_dir = Path(task_dir).expanduser().resolve()
        self.path = self.task_dir / "agent_checkpoint.json"
        self.last_load_reason = "not_attempted"

    def load(
        self,
        *,
        fingerprint: str,
    ) -> tuple[AgentState, set[str]] | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.last_load_reason = "missing_or_invalid"
            return None
        if payload.get("version") != CHECKPOINT_VERSION:
            self.last_load_reason = "unsupported_version"
            return None
        if payload.get("fingerprint") != fingerprint:
            self.last_load_reason = "fingerprint_mismatch"
            return None
        state_payload = payload.get("state")
        completed = payload.get("completed_steps")
        if not isinstance(state_payload, dict) or not isinstance(completed, list):
            self.last_load_reason = "invalid_state"
            return None
        try:
            state = AgentState.model_validate(state_payload)
            completed_steps = {str(step) for step in completed if str(step).strip()}
        except Exception:
            self.last_load_reason = "state_validation_failed"
            return None
        self.last_load_reason = "loaded"
        return state, completed_steps

    def save(
        self,
        state: AgentState,
        *,
        fingerprint: str,
        completed_steps: set[str],
        last_error: str | None = None,
    ) -> None:
        payload = {
            "version": CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "task_id": state.task_id,
            "completed_steps": sorted(completed_steps),
            "last_error": last_error,
            "state": state.model_dump(mode="json"),
        }
        self.task_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".part")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

