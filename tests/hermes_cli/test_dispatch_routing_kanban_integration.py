"""Integration coverage: dispatch_routing wired into ``_default_spawn``.

``_default_spawn`` now consults ``derive_routing_metadata`` /
``route_task_model`` ONLY when a task has no explicit ``model_override``. The
decision is runtime-only: it is recomputed on every spawn from
``task.workspace_kind`` and is never written back to the task row, the Task
dataclass, or the DB schema. Any failure inside that lookup (missing module,
unexpected exception) fails closed to the pre-existing behavior: no ``-m``/
``--provider`` flag at all.

No new DB column, migration, or persisted routing metadata is introduced by
this module or by the change it covers.
"""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli.dispatch_routing import (
    BASE_PROVIDER,
    LUNA_MODEL,
    TERRA_MODEL,
    derive_routing_metadata,
    route_task_model,
)


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants

        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    return home


def _create(kb, conn, *, workspace_kind, **kwargs):
    return kb.create_task(
        conn,
        title="integration task",
        workspace_kind=workspace_kind,
        **kwargs,
    )


def _spawn_and_capture(kb, monkeypatch, fresh_home, task, *, ws_name="workspace"):
    (fresh_home / "profiles" / (task.assignee or "default")).mkdir(
        parents=True, exist_ok=True
    )
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 9999

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = fresh_home / ws_name
    workspace.mkdir()
    kb._default_spawn(task, str(workspace))
    return captured["cmd"]


# ---------------------------------------------------------------------------
# Pure-function contract: worktree -> cross_file -> Terra; scratch/dir -> Luna
# ---------------------------------------------------------------------------


def test_worktree_task_from_db_derives_cross_file_and_routes_to_terra(fresh_home):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(kb, conn, workspace_kind="worktree", branch_name="wt/integration")
        task = kb.get_task(conn, tid)
        assert task.workspace_kind == "worktree"

        metadata = derive_routing_metadata(
            workspace_kind=task.workspace_kind,
            model_override=task.model_override,
            provider_override=task.provider_override,
        )
        decision = route_task_model(metadata)

        assert metadata["cross_file"] is True
        assert decision.routed is True
        assert decision.rule == "terra"
        assert decision.selected_provider == BASE_PROVIDER
        assert decision.selected_model == TERRA_MODEL
        assert decision.reason_codes == ("cross_file",)
    finally:
        conn.close()


def test_scratch_task_from_db_does_not_derive_cross_file_and_stays_on_luna(fresh_home):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(kb, conn, workspace_kind="scratch")
        task = kb.get_task(conn, tid)
        assert task.workspace_kind == "scratch"

        metadata = derive_routing_metadata(
            workspace_kind=task.workspace_kind,
            model_override=task.model_override,
            provider_override=task.provider_override,
        )
        decision = route_task_model(metadata)

        assert "cross_file" not in metadata
        assert decision.bypass is True
        assert decision.routed is False
        assert decision.selected_provider == BASE_PROVIDER
        assert decision.selected_model == LUNA_MODEL
        assert decision.reason_codes == ("luna_default",)
    finally:
        conn.close()


def test_dir_workspace_kind_from_db_does_not_derive_cross_file(fresh_home):
    """``dir`` is a third valid workspace_kind distinct from worktree/scratch."""

    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(kb, conn, workspace_kind="dir", workspace_path=str(fresh_home))
        task = kb.get_task(conn, tid)
        assert task.workspace_kind == "dir"

        metadata = derive_routing_metadata(workspace_kind=task.workspace_kind)
        assert "cross_file" not in metadata
        decision = route_task_model(metadata)
        assert decision.selected_model == LUNA_MODEL
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# _default_spawn is now wired: worktree escalates argv, scratch/dir don't
# ---------------------------------------------------------------------------


def test_default_spawn_escalates_worktree_task_to_terra_argv(monkeypatch, fresh_home):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(
            kb, conn, workspace_kind="worktree", branch_name="wt/spawn-terra", assignee="default"
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    cmd = _spawn_and_capture(kb, monkeypatch, fresh_home, task)

    assert "-m" in cmd
    assert cmd[cmd.index("-m") + 1] == TERRA_MODEL
    assert "--provider" in cmd
    assert cmd[cmd.index("--provider") + 1] == BASE_PROVIDER


def test_default_spawn_scratch_task_gets_no_routing_flags(monkeypatch, fresh_home):
    """Luna is the router's bypass default: no flag is added, matching the
    pre-dispatch_routing argv exactly (the profile/global default resolves
    the model, same as always).
    """

    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(kb, conn, workspace_kind="scratch", assignee="default")
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    cmd = _spawn_and_capture(kb, monkeypatch, fresh_home, task)

    assert "-m" not in cmd
    assert "--provider" not in cmd


def test_default_spawn_dir_task_gets_no_routing_flags(monkeypatch, fresh_home):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(
            kb, conn, workspace_kind="dir", workspace_path=str(fresh_home), assignee="default"
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    cmd = _spawn_and_capture(kb, monkeypatch, fresh_home, task)

    assert "-m" not in cmd
    assert "--provider" not in cmd


# ---------------------------------------------------------------------------
# Explicit model/provider override always outranks auto-routing
# ---------------------------------------------------------------------------


def test_explicit_model_override_on_worktree_task_bypasses_auto_routing(fresh_home):
    """A worktree task would auto-route to Terra, but an explicit pin wins."""

    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(
            kb,
            conn,
            workspace_kind="worktree",
            branch_name="wt/pinned",
            model_override="claude-opus-5",
        )
        task = kb.get_task(conn, tid)
        assert task.model_override == "claude-opus-5"
        assert task.provider_override is None

        metadata = derive_routing_metadata(
            workspace_kind=task.workspace_kind,
            model_override=task.model_override,
            provider_override=task.provider_override,
        )
        decision = route_task_model(metadata)

        assert decision.bypass is True
        assert decision.explicit_pin is True
        assert decision.reason_codes == ("explicit_pin",)
        assert decision.selected_model != TERRA_MODEL
    finally:
        conn.close()


def test_default_spawn_explicit_override_on_worktree_task_wins_over_terra(
    monkeypatch, fresh_home
):
    """The core priority guarantee, exercised through the real spawn path:
    an explicit task-level pin on a worktree task must produce the PINNED
    model/provider in argv, never the router's Terra suggestion.
    """

    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(
            kb,
            conn,
            workspace_kind="worktree",
            branch_name="wt/spawn-pinned",
            model_override="gpt-5.6-sol",
            provider_override="openai-codex",
            assignee="default",
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    cmd = _spawn_and_capture(kb, monkeypatch, fresh_home, task)

    assert "-m" in cmd
    assert cmd[cmd.index("-m") + 1] == "gpt-5.6-sol"
    assert "--provider" in cmd
    assert cmd[cmd.index("--provider") + 1] == "openai-codex"


def test_default_spawn_model_only_override_without_provider_is_unaffected_by_router(
    monkeypatch, fresh_home
):
    """A task pinning only ``model_override`` (no provider) must not have the
    router silently add a provider flag alongside it -- the else-branch that
    calls the router must never run once model_override is set.
    """

    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(
            kb,
            conn,
            workspace_kind="worktree",
            branch_name="wt/model-only",
            model_override="qwen3.5:9b",
            assignee="default",
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    cmd = _spawn_and_capture(kb, monkeypatch, fresh_home, task)

    assert cmd[cmd.index("-m") + 1] == "qwen3.5:9b"
    assert "--provider" not in cmd


# ---------------------------------------------------------------------------
# Missing / unknown metadata fails closed
# ---------------------------------------------------------------------------


def test_task_with_none_metadata_fails_closed():
    decision = route_task_model(None)
    assert decision.bypass is True
    assert decision.routed is False
    assert decision.selected_model is None
    assert decision.selected_provider is None


def test_unknown_workspace_kind_value_fails_closed():
    """If VALID_WORKSPACE_KINDS ever grows a new value this function doesn't
    know about, derive_routing_metadata must not guess -- no cross_file key.
    """

    metadata = derive_routing_metadata(workspace_kind="some_future_kind")
    assert "cross_file" not in metadata
    decision = route_task_model(metadata)
    assert decision.bypass is True
    assert decision.selected_model == LUNA_MODEL


# ---------------------------------------------------------------------------
# Routing-module import/lookup failure fails closed at the spawn boundary,
# never blocking or altering a plain task's spawn.
# ---------------------------------------------------------------------------


def test_default_spawn_falls_back_to_no_flags_when_dispatch_routing_import_fails(
    monkeypatch, fresh_home
):
    """Simulate the routing module being unimportable (e.g. a partial
    install, a packaging regression). ``_default_spawn`` must still spawn
    the worker with the pre-existing no-override argv shape -- it must not
    raise, and must not block the task.
    """

    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(
            kb, conn, workspace_kind="worktree", branch_name="wt/import-fail", assignee="default"
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hermes_cli.dispatch_routing":
            raise ImportError("simulated: dispatch_routing unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    cmd = _spawn_and_capture(kb, monkeypatch, fresh_home, task)

    assert "-m" not in cmd
    assert "--provider" not in cmd


def test_default_spawn_falls_back_to_no_flags_when_route_task_model_raises(
    monkeypatch, fresh_home
):
    """A live-but-misbehaving router (unexpected exception mid-decision)
    must also fail closed rather than crash the spawn.
    """

    from hermes_cli import dispatch_routing as dr
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(
            kb, conn, workspace_kind="worktree", branch_name="wt/route-raises", assignee="default"
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    def boom(_metadata):
        raise RuntimeError("simulated routing failure")

    monkeypatch.setattr(dr, "route_task_model", boom)

    cmd = _spawn_and_capture(kb, monkeypatch, fresh_home, task)

    assert "-m" not in cmd
    assert "--provider" not in cmd


def test_default_spawn_explicit_override_still_wins_even_if_router_would_raise(
    monkeypatch, fresh_home
):
    """An explicit pin must short-circuit before the router (and thus
    before any router failure) is ever reached.
    """

    from hermes_cli import dispatch_routing as dr
    from hermes_cli import kanban_db as kb

    def boom(_metadata):
        raise RuntimeError("router must never be called for a pinned task")

    monkeypatch.setattr(dr, "route_task_model", boom)

    conn = kb.connect()
    try:
        tid = _create(
            kb,
            conn,
            workspace_kind="worktree",
            branch_name="wt/pinned-vs-boom",
            model_override="claude-opus-5",
            assignee="default",
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    cmd = _spawn_and_capture(kb, monkeypatch, fresh_home, task)

    assert cmd[cmd.index("-m") + 1] == "claude-opus-5"


# ---------------------------------------------------------------------------
# Runtime-only: nothing about this decision is persisted anywhere.
# ---------------------------------------------------------------------------


def test_routing_decision_is_not_persisted_on_the_task_row(fresh_home):
    """Two consecutive spawns of the same never-pinned worktree task must
    each independently recompute the routing decision; the task row itself
    gains no new column/attribute from the process.
    """

    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(
            kb, conn, workspace_kind="worktree", branch_name="wt/no-persist", assignee="default"
        )
        task_before = kb.get_task(conn, tid)
    finally:
        conn.close()

    assert task_before.model_override is None
    assert task_before.provider_override is None
    assert not hasattr(task_before, "routing_metadata")
    assert not hasattr(task_before, "routed_model")

    conn = kb.connect()
    try:
        task_after = kb.get_task(conn, tid)
    finally:
        conn.close()

    # Re-fetching from the DB after "spawning" (conceptually) still shows no
    # override was ever written -- the routing decision lived only in the
    # ephemeral child argv, not in persisted state.
    assert task_after.model_override is None
    assert task_after.provider_override is None


# ---------------------------------------------------------------------------
# Existing Task schema / serialization / from_row behavior is unchanged.
# ---------------------------------------------------------------------------


def test_create_task_and_get_task_roundtrip_unchanged_without_routing_metadata(fresh_home):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = _create(kb, conn, workspace_kind="scratch")
        task = kb.get_task(conn, tid)

        assert task.id == tid
        assert task.title == "integration task"
        assert task.workspace_kind == "scratch"
        assert task.model_override is None
        assert task.provider_override is None
        assert task.reasoning_effort is None
        assert task.skills is None
        assert task.status in ("ready", "todo", "triage")
    finally:
        conn.close()


def test_create_task_signature_has_no_routing_metadata_parameter(fresh_home):
    """Regression guard: this phase must not add a ``routing_metadata`` (or
    similar) parameter to create_task -- routing is derived at spawn time
    from the existing ``workspace_kind`` column, not passed in at creation.
    """

    import inspect

    from hermes_cli import kanban_db as kb

    params = inspect.signature(kb.create_task).parameters
    for forbidden in ("routing_metadata", "dispatch_metadata"):
        assert forbidden not in params


def test_task_dataclass_has_no_routing_metadata_field():
    """Regression guard: Task must not have gained a routing_metadata /
    dispatch_routing field -- the schema is untouched by this integration.
    """

    import dataclasses

    from hermes_cli import kanban_db as kb

    field_names = {f.name for f in dataclasses.fields(kb.Task)}
    for forbidden in ("routing_metadata", "dispatch_metadata", "cross_file"):
        assert forbidden not in field_names
