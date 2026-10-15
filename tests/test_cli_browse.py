import json
import sys
import urllib.error
from io import BytesIO

import pytest

from fidelis import cli


def _run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["fidelis", *args])
    cli.main()


def test_recent_routes_options_and_renders_metadata(monkeypatch, capsys):
    seen = {}

    def fake_post(path, payload, **kwargs):
        seen.update(path=path, payload=payload, kwargs=kwargs)
        return {
            "records": [
                {
                    "id": "new-id",
                    "text": "corrected text",
                    "status": "current",
                    "recorded_at": "2026-09-21T00:00:00Z",
                    "source": "ops log",
                    "supersedes": ["old-id"],
                }
            ]
        }

    monkeypatch.setattr(cli, "_post", fake_post)
    _run_cli(
        monkeypatch,
        "recent",
        "--limit",
        "3",
        "--kind",
        "corrections",
        "--since",
        "2026-09-01T00:00:00Z",
    )

    output = capsys.readouterr().out
    assert seen == {
        "path": "/recent",
        "payload": {
            "limit": 3,
            "kind": "corrections",
            "since": "2026-09-01T00:00:00Z",
        },
        "kwargs": {"structured_errors": True},
    }
    for value in (
        "new-id",
        "corrected text",
        "current",
        "2026-09-21T00:00:00Z",
        "ops log",
        "old-id",
    ):
        assert value in output


def test_get_renders_record_and_both_chain_directions(monkeypatch, capsys):
    record = {
        "id": "current-id",
        "text": "current text",
        "source": "decision log",
        "temporal": {
            "status": "superseded",
            "recorded_at": "2026-09-21T00:00:00Z",
            "superseded_by": ["next-id"],
        },
        "supersedes": [
            {
                "id": "old-id",
                "text": "old text",
                "status": "superseded",
                "recorded_at": "2026-09-20T00:00:00Z",
            }
        ],
        "superseded_by": [
            {
                "id": "next-id",
                "text": "next text",
                "status": "current",
                "recorded_at": "2026-09-22T00:00:00Z",
            }
        ],
        "index": "ok",
    }
    seen = {}

    def fake_post(path, payload, **kwargs):
        seen.update(path=path, payload=payload, kwargs=kwargs)
        return record

    monkeypatch.setattr(cli, "_post", fake_post)
    _run_cli(monkeypatch, "get", "current-id")

    output = capsys.readouterr().out
    assert seen == {
        "path": "/get",
        "payload": {"id": "current-id"},
        "kwargs": {"structured_errors": True},
    }
    for value in (
        "current-id",
        "current text",
        "decision log",
        "superseded by next-id",
        "2026-09-21T00:00:00Z",
        "old-id",
        "old text",
        "next-id",
        "next text",
    ):
        assert value in output


@pytest.mark.parametrize(
    ("args", "response"),
    [
        (("recent", "--raw"), {"kind": "all", "limit": 10, "records": []}),
        (("get", "record-id", "--raw"), {"id": "record-id", "text": "verbatim"}),
    ],
)
def test_raw_outputs_service_payload_unchanged(monkeypatch, capsys, args, response):
    monkeypatch.setattr(cli, "_post", lambda path, payload, **kwargs: response)
    _run_cli(monkeypatch, *args)
    assert json.loads(capsys.readouterr().out) == response


def test_unknown_record_is_explicit_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_post",
        lambda path, payload, **kwargs: {"error": "no record found for id 'ghost'"},
    )

    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, "get", "ghost")

    assert exc.value.code == 1
    assert "ghost" in capsys.readouterr().err


def test_raw_error_outputs_service_payload_unchanged_and_exits_nonzero(monkeypatch, capsys):
    response = {"error": "no record found for id 'ghost'"}
    monkeypatch.setattr(cli, "_post", lambda path, payload, **kwargs: response)

    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, "get", "ghost", "--raw")

    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out) == response


def test_post_preserves_structured_http_error(monkeypatch):
    response = {"error": "no record found for id 'ghost'"}

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            BytesIO(json.dumps(response).encode()),
        )

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    assert cli._post("/get", {"id": "ghost"}, structured_errors=True) == response


@pytest.mark.parametrize("raw", [False, True])
def test_existing_query_keeps_nonzero_http_failure(monkeypatch, capsys, raw):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            {},
            BytesIO(b'{"error":"store unavailable"}'),
        )

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    args = ["query", "hello"]
    if raw:
        args.append("--raw")

    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, *args)

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert captured.out == ""
    assert "unreachable or unhealthy" in captured.err


def test_unavailable_service_is_explicit_nonzero(monkeypatch, capsys):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SystemExit) as exc:
        cli._post("/recent", {"limit": 10, "kind": "all"})

    assert exc.value.code == 1
    assert "unreachable or unhealthy" in capsys.readouterr().err


@pytest.mark.parametrize("limit", ["0", "51"])
def test_recent_rejects_out_of_bounds_limit(monkeypatch, limit):
    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, "recent", "--limit", limit)
    assert exc.value.code == 2
