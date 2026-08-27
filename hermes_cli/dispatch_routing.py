"""Pure policy-only model-routing decisions for ephemeral dispatch workers.

This module contains no I/O, no DB access, and no subprocess spawning. It
takes an explicit metadata mapping and returns a :class:`RouteDecision`
describing which model/provider (if any) an ephemeral worker should be
pinned to. Callers own the metadata's provenance and the actual argv/DB
writes; this module only encodes the routing rule table.

Baseline: ``openai-codex/gpt-5.6-luna``. A task escalates to Terra when it
shows cross-file scope, uses at least three tools, needs multi-step
verification, or has an explicit Luna-insufficiency marker. A task escalates
to Sol only when it is high-value AND either Terra proved insufficient or
deep reasoning is explicitly required. Any explicit model/provider pin on
the task always bypasses this router entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

BASE_PROVIDER = "openai-codex"
LUNA_MODEL = "gpt-5.6-luna"
TERRA_MODEL = "gpt-5.6-terra"
SOL_MODEL = "gpt-5.6-sol"

ROUTE_VERSION = "1"

ALLOWED_TARGET_MODELS = (LUNA_MODEL, TERRA_MODEL, SOL_MODEL)

_RISK_LEVELS = ("low", "medium", "high")

_ROUTING_BOOL_KEYS = (
    "enabled",
    "explicit_pin",
    "cross_file",
    "multi_step_verification",
    "luna_insufficiency",
    "terra_insufficient",
    "deep_reasoning_required",
    "high_value",
    "external_send_requested",
)

ROUTING_METADATA_KEYS = frozenset(
    _ROUTING_BOOL_KEYS
    + (
        "provider",
        "model",
        "risk_level",
        "tool_count",
        "model_override",
        "provider_override",
        "task_model_override",
        "task_provider_override",
    )
)


def validate_routing_metadata(
    metadata: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Validate and copy the task-level routing metadata contract.

    The field is optional, so ``None`` remains ``None``. A supplied mapping
    may contain any subset of the allowlisted keys; omission is distinct
    from an enabled route and therefore remains fail-closed at decision
    time. Validation is deliberately strict at the persistence boundary:
    unknown keys, integer/bool confusion, and stringified booleans are
    rejected.
    """

    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("routing_metadata must be an object")

    unknown = set(metadata) - ROUTING_METADATA_KEYS
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"routing_metadata has unknown key(s): {names}")

    validated = dict(metadata)
    for key in _ROUTING_BOOL_KEYS:
        if key in validated and type(validated[key]) is not bool:
            raise ValueError(f"routing_metadata.{key} must be a boolean")

    if "risk_level" in validated:
        risk_level = validated["risk_level"]
        if type(risk_level) is not str or risk_level not in _RISK_LEVELS:
            raise ValueError(
                "routing_metadata.risk_level must be one of: low, medium, high"
            )

    if "tool_count" in validated:
        tool_count = validated["tool_count"]
        if type(tool_count) is not int or tool_count < 0:
            raise ValueError(
                "routing_metadata.tool_count must be a non-negative integer"
            )

    return validated


@dataclass(frozen=True)
class RouteDecision:
    """The complete, prompt-free result of one routing decision.

    ``selected_*`` is the effective route when the decision is safe to use.
    It is ``None`` for a missing/disabled/unsupported/bypassed metadata
    input; a valid Luna baseline is retained for a valid task that is
    deliberately not escalated. ``routed`` means an ephemeral child argv
    should receive a model/provider addition. ``bypass`` means the router
    must not add one.
    """

    selected_provider: str | None
    selected_model: str | None
    routed: bool
    rule: str
    reason_codes: tuple[str, ...]
    explicit_pin: bool
    bypass: bool
    route_version: str = ROUTE_VERSION

    @property
    def provider(self) -> str | None:
        """Short alias for callers that use provider/model terminology."""

        return self.selected_provider

    @property
    def model(self) -> str | None:
        """Short alias for callers that use provider/model terminology."""

        return self.selected_model

    def to_event_payload(self) -> dict[str, Any]:
        """Return the allowlisted payload for a ``model_routed`` event.

        No input metadata is copied into this result. In particular, task
        prompt/title/body values and secret-like values cannot cross this
        boundary through an event payload.
        """

        return {
            "selected_provider": self.selected_provider,
            "selected_model": self.selected_model,
            "routed": self.routed,
            "rule": self.rule,
            "reason_codes": list(self.reason_codes),
            "explicit_pin": self.explicit_pin,
            "bypass": self.bypass,
            "route_version": self.route_version,
        }


def _is_true(metadata: Mapping[str, object], key: str) -> bool:
    """Accept only a typed boolean true value, failing closed otherwise."""

    return metadata.get(key) is True


def _has_value(metadata: Mapping[str, object], key: str) -> bool:
    """Whether an override field contains an explicit value.

    ``None`` and an empty string are the unset representation used by the
    existing task model. Any other value is treated as a pin, including a
    malformed value, so malformed explicit pins cannot be auto-routed.
    """

    value = metadata.get(key)
    if value is None:
        return False
    return not (isinstance(value, str) and not value.strip())


def _has_explicit_pin(metadata: Mapping[str, object]) -> bool:
    """Detect only explicit pin metadata; never infer a pin from task text."""

    if _is_true(metadata, "explicit_pin"):
        return True
    for key in (
        "model_override",
        "provider_override",
        "task_model_override",
        "task_provider_override",
    ):
        if _has_value(metadata, key):
            return True
    return False


def _bypass(
    reason_codes: tuple[str, ...],
    *,
    explicit_pin: bool = False,
    selected_provider: str | None = None,
    selected_model: str | None = None,
) -> RouteDecision:
    return RouteDecision(
        selected_provider=selected_provider,
        selected_model=selected_model,
        routed=False,
        rule="bypass",
        reason_codes=reason_codes,
        explicit_pin=explicit_pin,
        bypass=True,
    )


def route_task_model(metadata: Mapping[str, object] | None) -> RouteDecision:
    """Choose a safe ephemeral Kanban worker model from explicit metadata.

    The baseline must be ``openai-codex/gpt-5.6-luna`` and ``enabled`` must
    be the typed boolean ``True``. Explicit task model/provider overrides
    always bypass this router. Only low-risk, non-external tasks may be
    escalated:

    * Terra: cross-file work, at least three tools, multi-step
      verification, or an explicit Luna insufficiency marker.
    * Sol: high-value work plus Terra insufficiency or deep-reasoning
      marker; this check has precedence over Terra.

    All other cases remain on Luna (or bypass entirely when the baseline is
    absent/invalid). No field other than the explicit metadata mapping is
    consulted.
    """

    if not isinstance(metadata, Mapping) or not metadata:
        return _bypass(("missing_metadata",))

    explicit_pin = _has_explicit_pin(metadata)
    if explicit_pin:
        return _bypass(("explicit_pin",), explicit_pin=True)

    if not _is_true(metadata, "enabled"):
        return _bypass(("disabled",))

    provider = metadata.get("provider")
    model = metadata.get("model")
    if provider != BASE_PROVIDER or model != LUNA_MODEL:
        return _bypass(("unsupported_baseline",))

    # A valid baseline is retained for all non-escalated decisions. It is
    # still ``bypass=True`` because the child argv must not be augmented.
    luna = {
        "selected_provider": BASE_PROVIDER,
        "selected_model": LUNA_MODEL,
    }
    risk_level = metadata.get("risk_level")
    if risk_level != "low":
        return _bypass(("risk_not_low",), **luna)
    if _is_true(metadata, "external_send_requested"):
        return _bypass(("external_send_requested",), **luna)

    terra_reasons: list[str] = []
    if _is_true(metadata, "cross_file"):
        terra_reasons.append("cross_file")
    tool_count = metadata.get("tool_count")
    if isinstance(tool_count, int) and not isinstance(tool_count, bool) and tool_count >= 3:
        terra_reasons.append("tool_count_gte_3")
    if _is_true(metadata, "multi_step_verification"):
        terra_reasons.append("multi_step_verification")
    if _is_true(metadata, "luna_insufficiency"):
        terra_reasons.append("luna_insufficiency")

    terra_insufficient = _is_true(metadata, "terra_insufficient")
    deep_reasoning = _is_true(metadata, "deep_reasoning_required")
    if _is_true(metadata, "high_value") and (terra_insufficient or deep_reasoning):
        sol_reasons = ["high_value"]
        if terra_insufficient:
            sol_reasons.append("terra_insufficient")
        if deep_reasoning:
            sol_reasons.append("deep_reasoning_required")
        return RouteDecision(
            selected_provider=BASE_PROVIDER,
            selected_model=SOL_MODEL,
            routed=True,
            rule="sol",
            reason_codes=tuple(sol_reasons),
            explicit_pin=False,
            bypass=False,
        )

    if terra_reasons:
        return RouteDecision(
            selected_provider=BASE_PROVIDER,
            selected_model=TERRA_MODEL,
            routed=True,
            rule="terra",
            reason_codes=tuple(terra_reasons),
            explicit_pin=False,
            bypass=False,
        )

    return _bypass(("luna_default",), **luna)


def derive_routing_metadata(
    *,
    workspace_kind: str,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> dict[str, object]:
    """Build a routing-metadata mapping from what ``create_task`` actually knows.

    This is intentionally narrow. At task-creation time, the only signal
    honestly available for the escalation rules is ``workspace_kind``: a
    ``"worktree"`` task touches a checked-out repo and can plausibly span
    multiple files, so it maps to ``cross_file=True``. A ``"scratch"`` task
    has no such implication.

    Every other escalation signal in :func:`route_task_model` --
    ``tool_count``, ``multi_step_verification``, ``high_value``,
    ``deep_reasoning_required``, ``luna_insufficiency`` -- has no honest
    source at this call site and is deliberately omitted rather than
    guessed. In particular, a task having ``parents`` (a dependency-chain
    position) is NOT evidence that the task itself needs multi-step
    verification, so it is not consulted here. These fields can be added
    later once a real caller-supplied hint (not a proxy) is available;
    until then, the router's fail-closed default (bypass to Luna) applies
    to all of them via simple absence.

    ``model_override`` / ``provider_override`` are passed through as the
    router's explicit-pin keys so a task-level pin (already the
    highest-priority path in ``create_task``) also bypasses this router
    the same way it bypasses the dispatcher's own model selection.
    """

    metadata: dict[str, object] = {
        "enabled": True,
        "provider": BASE_PROVIDER,
        "model": LUNA_MODEL,
        "risk_level": "low",
    }
    if workspace_kind == "worktree":
        metadata["cross_file"] = True
    if model_override:
        metadata["model_override"] = model_override
    if provider_override:
        metadata["provider_override"] = provider_override
    return metadata


__all__ = [
    "ALLOWED_TARGET_MODELS",
    "BASE_PROVIDER",
    "LUNA_MODEL",
    "ROUTE_VERSION",
    "ROUTING_METADATA_KEYS",
    "SOL_MODEL",
    "TERRA_MODEL",
    "RouteDecision",
    "derive_routing_metadata",
    "route_task_model",
    "validate_routing_metadata",
]
