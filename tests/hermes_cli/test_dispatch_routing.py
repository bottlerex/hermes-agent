from __future__ import annotations

import pytest

from hermes_cli.dispatch_routing import (
    BASE_PROVIDER,
    LUNA_MODEL,
    SOL_MODEL,
    TERRA_MODEL,
    derive_routing_metadata,
    route_task_model,
    validate_routing_metadata,
)

_BASELINE = {"enabled": True, "provider": BASE_PROVIDER, "model": LUNA_MODEL, "risk_level": "low"}


def _with(**overrides):
    merged = dict(_BASELINE)
    merged.update(overrides)
    return merged


def test_missing_metadata_bypasses():
    decision = route_task_model(None)
    assert decision.bypass is True
    assert decision.routed is False
    assert decision.selected_model is None
    assert decision.reason_codes == ("missing_metadata",)


def test_empty_metadata_bypasses():
    decision = route_task_model({})
    assert decision.reason_codes == ("missing_metadata",)


def test_disabled_bypasses():
    decision = route_task_model(_with(enabled=False))
    assert decision.bypass is True
    assert decision.reason_codes == ("disabled",)


def test_missing_enabled_key_bypasses():
    metadata = dict(_BASELINE)
    del metadata["enabled"]
    decision = route_task_model(metadata)
    assert decision.reason_codes == ("disabled",)


def test_unsupported_baseline_provider_bypasses():
    decision = route_task_model(_with(provider="anthropic"))
    assert decision.reason_codes == ("unsupported_baseline",)


def test_unsupported_baseline_model_bypasses():
    decision = route_task_model(_with(model="gpt-5.6-terra"))
    assert decision.reason_codes == ("unsupported_baseline",)


def test_explicit_pin_via_flag_bypasses_before_baseline_check():
    decision = route_task_model(_with(explicit_pin=True, provider="anthropic"))
    assert decision.bypass is True
    assert decision.explicit_pin is True
    assert decision.reason_codes == ("explicit_pin",)


@pytest.mark.parametrize(
    "key",
    ["model_override", "provider_override", "task_model_override", "task_provider_override"],
)
def test_explicit_override_value_bypasses(key):
    decision = route_task_model(_with(**{key: "some-model"}))
    assert decision.bypass is True
    assert decision.explicit_pin is True
    assert decision.reason_codes == ("explicit_pin",)


@pytest.mark.parametrize(
    "key",
    ["model_override", "provider_override", "task_model_override", "task_provider_override"],
)
def test_override_key_with_none_or_blank_does_not_count_as_pin(key):
    decision = route_task_model(_with(**{key: None}))
    assert decision.reason_codes != ("explicit_pin",)
    decision_blank = route_task_model(_with(**{key: "   "}))
    assert decision_blank.reason_codes != ("explicit_pin",)


def test_risk_not_low_bypasses_with_luna_baseline_retained():
    decision = route_task_model(_with(risk_level="medium"))
    assert decision.bypass is True
    assert decision.routed is False
    assert decision.selected_provider == BASE_PROVIDER
    assert decision.selected_model == LUNA_MODEL
    assert decision.reason_codes == ("risk_not_low",)


def test_external_send_requested_bypasses_with_luna_baseline_retained():
    decision = route_task_model(_with(external_send_requested=True))
    assert decision.bypass is True
    assert decision.selected_model == LUNA_MODEL
    assert decision.reason_codes == ("external_send_requested",)


def test_plain_low_risk_task_defaults_to_luna_bypass():
    decision = route_task_model(_with())
    assert decision.bypass is True
    assert decision.routed is False
    assert decision.selected_provider == BASE_PROVIDER
    assert decision.selected_model == LUNA_MODEL
    assert decision.reason_codes == ("luna_default",)


def test_cross_file_escalates_to_terra():
    decision = route_task_model(_with(cross_file=True))
    assert decision.routed is True
    assert decision.bypass is False
    assert decision.rule == "terra"
    assert decision.selected_provider == BASE_PROVIDER
    assert decision.selected_model == TERRA_MODEL
    assert "cross_file" in decision.reason_codes


def test_tool_count_gte_3_escalates_to_terra():
    decision = route_task_model(_with(tool_count=3))
    assert decision.selected_model == TERRA_MODEL
    assert "tool_count_gte_3" in decision.reason_codes


def test_tool_count_below_3_does_not_escalate():
    decision = route_task_model(_with(tool_count=2))
    assert decision.bypass is True
    assert decision.selected_model == LUNA_MODEL


def test_tool_count_bool_is_not_treated_as_int():
    decision = route_task_model(_with(tool_count=True))
    assert decision.selected_model == LUNA_MODEL


def test_multi_step_verification_escalates_to_terra():
    decision = route_task_model(_with(multi_step_verification=True))
    assert decision.selected_model == TERRA_MODEL
    assert "multi_step_verification" in decision.reason_codes


def test_luna_insufficiency_escalates_to_terra():
    decision = route_task_model(_with(luna_insufficiency=True))
    assert decision.selected_model == TERRA_MODEL
    assert "luna_insufficiency" in decision.reason_codes


def test_multiple_terra_reasons_all_recorded():
    decision = route_task_model(
        _with(cross_file=True, multi_step_verification=True, luna_insufficiency=True)
    )
    assert decision.rule == "terra"
    assert set(decision.reason_codes) == {
        "cross_file",
        "multi_step_verification",
        "luna_insufficiency",
    }


def test_high_value_alone_does_not_escalate_to_sol():
    decision = route_task_model(_with(high_value=True))
    assert decision.selected_model == LUNA_MODEL
    assert decision.bypass is True


def test_high_value_with_terra_insufficient_escalates_to_sol():
    decision = route_task_model(_with(high_value=True, terra_insufficient=True))
    assert decision.routed is True
    assert decision.rule == "sol"
    assert decision.selected_provider == BASE_PROVIDER
    assert decision.selected_model == SOL_MODEL
    assert "terra_insufficient" in decision.reason_codes


def test_high_value_with_deep_reasoning_required_escalates_to_sol():
    decision = route_task_model(_with(high_value=True, deep_reasoning_required=True))
    assert decision.selected_model == SOL_MODEL
    assert "deep_reasoning_required" in decision.reason_codes


def test_sol_takes_precedence_over_terra_reasons():
    decision = route_task_model(
        _with(high_value=True, deep_reasoning_required=True, cross_file=True)
    )
    assert decision.rule == "sol"
    assert decision.selected_model == SOL_MODEL
    assert "cross_file" not in decision.reason_codes


def test_risk_not_low_bypasses_even_with_escalation_markers():
    decision = route_task_model(_with(risk_level="high", cross_file=True))
    assert decision.bypass is True
    assert decision.reason_codes == ("risk_not_low",)


def test_to_event_payload_contains_no_raw_metadata_keys():
    decision = route_task_model(_with(cross_file=True))
    payload = decision.to_event_payload()
    assert set(payload.keys()) == {
        "selected_provider",
        "selected_model",
        "routed",
        "rule",
        "reason_codes",
        "explicit_pin",
        "bypass",
        "route_version",
    }
    assert payload["reason_codes"] == ["cross_file"]


def test_provider_and_model_properties_are_aliases():
    decision = route_task_model(_with(cross_file=True))
    assert decision.provider == decision.selected_provider
    assert decision.model == decision.selected_model


class TestValidateRoutingMetadata:
    def test_none_stays_none(self):
        assert validate_routing_metadata(None) is None

    def test_non_mapping_rejected(self):
        with pytest.raises(ValueError, match="must be an object"):
            validate_routing_metadata("not-a-mapping")  # type: ignore[arg-type]

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown key"):
            validate_routing_metadata({"totally_unknown_field": True})

    def test_bool_key_with_wrong_type_rejected(self):
        with pytest.raises(ValueError, match="must be a boolean"):
            validate_routing_metadata({"enabled": "true"})

    def test_bool_key_with_int_rejected(self):
        with pytest.raises(ValueError, match="must be a boolean"):
            validate_routing_metadata({"enabled": 1})

    def test_valid_risk_level_accepted(self):
        validated = validate_routing_metadata({"risk_level": "medium"})
        assert validated == {"risk_level": "medium"}

    def test_invalid_risk_level_rejected(self):
        with pytest.raises(ValueError, match="risk_level"):
            validate_routing_metadata({"risk_level": "critical"})

    def test_risk_level_wrong_type_rejected(self):
        with pytest.raises(ValueError, match="risk_level"):
            validate_routing_metadata({"risk_level": 1})

    def test_valid_tool_count_accepted(self):
        validated = validate_routing_metadata({"tool_count": 5})
        assert validated == {"tool_count": 5}

    def test_negative_tool_count_rejected(self):
        with pytest.raises(ValueError, match="tool_count"):
            validate_routing_metadata({"tool_count": -1})

    def test_tool_count_bool_rejected(self):
        with pytest.raises(ValueError, match="tool_count"):
            validate_routing_metadata({"tool_count": True})

    def test_tool_count_wrong_type_rejected(self):
        with pytest.raises(ValueError, match="tool_count"):
            validate_routing_metadata({"tool_count": "3"})

    def test_valid_full_metadata_returns_copy_not_same_object(self):
        source = dict(_BASELINE)
        validated = validate_routing_metadata(source)
        assert validated == source
        assert validated is not source


class TestDeriveRoutingMetadata:
    def test_scratch_workspace_does_not_set_cross_file(self):
        metadata = derive_routing_metadata(workspace_kind="scratch")
        assert "cross_file" not in metadata
        assert metadata["enabled"] is True
        assert metadata["provider"] == BASE_PROVIDER
        assert metadata["model"] == LUNA_MODEL
        assert metadata["risk_level"] == "low"

    def test_worktree_workspace_sets_cross_file(self):
        metadata = derive_routing_metadata(workspace_kind="worktree")
        assert metadata["cross_file"] is True

    def test_unrecognized_workspace_kind_does_not_set_cross_file(self):
        metadata = derive_routing_metadata(workspace_kind="something_new")
        assert "cross_file" not in metadata

    def test_never_produces_multi_step_verification(self):
        metadata = derive_routing_metadata(workspace_kind="worktree")
        assert "multi_step_verification" not in metadata

    def test_never_produces_tool_count_high_value_deep_reasoning_luna_insufficiency(self):
        metadata = derive_routing_metadata(workspace_kind="worktree")
        for key in (
            "tool_count",
            "high_value",
            "deep_reasoning_required",
            "luna_insufficiency",
            "terra_insufficient",
            "external_send_requested",
        ):
            assert key not in metadata

    def test_model_override_present_is_passed_through_as_explicit_pin_key(self):
        metadata = derive_routing_metadata(
            workspace_kind="scratch", model_override="claude-opus-5"
        )
        assert metadata["model_override"] == "claude-opus-5"

    def test_provider_override_present_is_passed_through(self):
        metadata = derive_routing_metadata(
            workspace_kind="scratch",
            model_override="claude-opus-5",
            provider_override="anthropic",
        )
        assert metadata["provider_override"] == "anthropic"

    def test_no_overrides_means_no_override_keys_in_metadata(self):
        metadata = derive_routing_metadata(workspace_kind="scratch")
        assert "model_override" not in metadata
        assert "provider_override" not in metadata

    def test_derived_metadata_with_worktree_and_no_override_escalates_to_terra(self):
        metadata = derive_routing_metadata(workspace_kind="worktree")
        decision = route_task_model(metadata)
        assert decision.routed is True
        assert decision.rule == "terra"
        assert decision.reason_codes == ("cross_file",)

    def test_derived_metadata_with_scratch_and_no_override_stays_on_luna(self):
        metadata = derive_routing_metadata(workspace_kind="scratch")
        decision = route_task_model(metadata)
        assert decision.bypass is True
        assert decision.selected_model == LUNA_MODEL
        assert decision.reason_codes == ("luna_default",)

    def test_derived_metadata_with_model_override_bypasses_regardless_of_workspace_kind(self):
        metadata = derive_routing_metadata(
            workspace_kind="worktree", model_override="claude-opus-5"
        )
        decision = route_task_model(metadata)
        assert decision.bypass is True
        assert decision.explicit_pin is True
        assert decision.reason_codes == ("explicit_pin",)

    def test_derive_output_passes_validate_routing_metadata(self):
        metadata = derive_routing_metadata(
            workspace_kind="worktree",
            model_override="claude-opus-5",
            provider_override="anthropic",
        )
        # Must not raise: every key derive_routing_metadata can produce is
        # in the router's own allowlist.
        validated = validate_routing_metadata(metadata)
        assert validated == metadata

    def test_parents_is_not_a_derive_routing_metadata_parameter(self):
        # Regression guard for the explicit design decision: a task's
        # dependency chain (``parents`` in create_task) is NOT evidence
        # that the task itself needs multi-step verification. This test
        # fails loudly if a future edit adds a ``parents`` parameter that
        # feeds multi_step_verification without an explicit re-review.
        import inspect

        params = inspect.signature(derive_routing_metadata).parameters
        assert "parents" not in params
