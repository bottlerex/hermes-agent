"""Focused delivery tests for the user-local morning report script."""

import importlib.util
import io
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock


SCRIPT_PATH = Path.home() / ".hermes" / "scripts" / "morning-report.py"


def load_morning_report():
    spec = importlib.util.spec_from_file_location("morning_report", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def prepare_report(module, monkeypatch, token):
    monkeypatch.setattr(
        module,
        "load_env",
        lambda: {"LINE_CHANNEL_ACCESS_TOKEN": token, "LINE_HOME_CHANNEL": "U-test"},
    )
    monkeypatch.setattr(module, "build_report", lambda: "test report")


def test_line_push_429_is_logged_without_token_and_returns_nonzero(monkeypatch):
    module = load_morning_report()
    token = "test-line-token"
    prepare_report(module, monkeypatch, token)
    def raise_http_error(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests", hdrs={}, fp=None
        )

    urlopen = Mock(side_effect=raise_http_error)
    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = module.main()

    assert result == 1
    assert "HTTP 429" in stderr.getvalue()
    assert token not in stdout.getvalue()
    assert token not in stderr.getvalue()
    urlopen.assert_called_once()


def test_line_push_success_preserves_status_and_returns_zero(monkeypatch):
    module = load_morning_report()
    prepare_report(module, monkeypatch, "test-line-token")
    urlopen = Mock(return_value=FakeResponse(200))
    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        result = module.main()

    assert result == 0
    assert "LINE push HTTP 200" in stdout.getvalue()
    urlopen.assert_called_once()
