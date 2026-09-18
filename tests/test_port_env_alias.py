"""FIDELIS_PORT must move the server, not just the clients.

`cli`, `mcp_server`, `watch_cmd`, and `augment` all resolve their target port
as FIDELIS_PORT first, COGITO_PORT second. The server resolves its bind port
through `config.load`. When that map knew only COGITO_PORT, setting
FIDELIS_PORT moved every client off a server that stayed on 19420 — the
documented alias produced a split brain instead of an override.
"""

import pytest

from fidelis.config import load

MISSING = "/nonexistent/path/.cogito.json"


@pytest.fixture(autouse=True)
def _clear_port_env(monkeypatch):
    monkeypatch.delenv("FIDELIS_PORT", raising=False)
    monkeypatch.delenv("COGITO_PORT", raising=False)


def test_server_port_defaults_to_19420():
    assert load(config_path=MISSING)["port"] == 19420


def test_fidelis_port_overrides_the_server_bind_port(monkeypatch):
    monkeypatch.setenv("FIDELIS_PORT", "19477")
    assert load(config_path=MISSING)["port"] == 19477


def test_cogito_port_still_overrides_the_server_bind_port(monkeypatch):
    monkeypatch.setenv("COGITO_PORT", "19478")
    assert load(config_path=MISSING)["port"] == 19478


def test_fidelis_port_wins_when_both_are_set(monkeypatch):
    """Same precedence the clients use, so both ends agree on one port."""
    monkeypatch.setenv("COGITO_PORT", "19478")
    monkeypatch.setenv("FIDELIS_PORT", "19477")
    assert load(config_path=MISSING)["port"] == 19477


def test_server_and_mcp_client_resolve_the_same_port(monkeypatch):
    from fidelis import mcp_server

    monkeypatch.setenv("FIDELIS_PORT", "19477")
    assert mcp_server._server_url() == f"http://127.0.0.1:{load(config_path=MISSING)['port']}"
