from __future__ import annotations

"""Deterministic preflight for model-authored artifact action plans.

Policy remains the final safety boundary.  This module removes actions that
are already known to be impossible from the current catalog so one stale
artifact does not invalidate an otherwise useful LLM plan.
"""

from scidata_agent.agent.action_registry import (
    artifact_type_supported,
    get_action_capability,
    is_global_action,
)
from scidata_agent.agent.schemas import AgentState, ArtifactActionPlan


TERMINAL_ARTIFACT_STATUSES = frozenset({"skipped", "failed"})


def preflight_artifact_action_plan(
    plan: ArtifactActionPlan,
    state: AgentState,
) -> list[str]:
    """Remove artifact actions with deterministic, already-known failures.

    Returns concise audit reasons and appends the same summary to ``plan``.
    Global workflow actions are left to the regular policy layer because their
    validity depends on workflow readiness rather than an artifact route.
    """

    artifacts = {
        artifact.artifact_id: artifact
        for entry in state.source_catalog
        for artifact in entry.artifacts
    }
    retained = []
    dropped: list[str] = []
    for action in plan.actions:
        if action.action == "search_more":
            group_id = str(action.parameters.get("field_group_id") or "").strip().casefold()
            initial_group_search = action.parameters.get("initial_group_search") is True
            count = (
                int(state.runtime_group_search_more_counts.get(group_id, 0))
                if group_id
                else int(getattr(state, "runtime_search_more_count", 0))
            )
            if not initial_group_search and count >= int(getattr(state, "runtime_search_more_limit", 2)):
                dropped.append(
                    f"{action.action_id}: search_more limit is exhausted for "
                    f"{group_id or 'legacy_global'} "
                    f"({state.runtime_search_more_limit} maximum)"
                )
                continue
        if is_global_action(action.action):
            retained.append(action)
            continue
        artifact_id = str(action.artifact_id or "")
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            dropped.append(f"{action.action_id}: artifact is absent from the current catalog")
            continue
        if artifact.status in TERMINAL_ARTIFACT_STATUSES:
            dropped.append(
                f"{action.action_id}: artifact {artifact.artifact_id} is terminal ({artifact.status})"
            )
            continue
        if action.action in artifact.completed_operations:
            dropped.append(
                f"{action.action_id}: artifact operation {action.action} is already complete"
            )
            continue
        capability = get_action_capability(action.action)
        if capability.requires_local_path and not artifact.local_path:
            dropped.append(
                f"{action.action_id}: {action.action} requires a materialized local file"
            )
            continue
        if not artifact_type_supported(action.action, artifact.artifact_type):
            dropped.append(
                f"{action.action_id}: {action.action} cannot handle {artifact.artifact_type}"
            )
            continue
        if action.action == "download_artifact" and artifact.local_path:
            dropped.append(
                f"{action.action_id}: artifact {artifact.artifact_id} is already materialized"
            )
            continue
        retained.append(action)

    if dropped:
        plan.actions = retained
        plan.notes.append(
            f"Runtime preflight removed {len(dropped)} impossible artifact action(s): "
            + "; ".join(dropped[:4])
        )
    return dropped
