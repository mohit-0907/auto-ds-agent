"""Tests for serving the built React app next to the API (one process in production)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autods.api.app import create_app

INDEX = "<!doctype html><div id=root></div><script src=/assets/index-abc123.js></script>"
SECRET = "OPENAI_API_KEY=sk-secret"


@pytest.fixture
def web(tmp_path: Path) -> Path:
    """A tiny stand-in for web/dist, with a secret file next to it that must stay private."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX, encoding="utf-8")
    (dist / "assets" / "index-abc123.js").write_text("console.log('app')", encoding="utf-8")
    (dist / "assets" / "figtree-latin-400.woff2").write_bytes(b"wOF2")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / ".env").write_text(SECRET, encoding="utf-8")
    return dist


@pytest.fixture
def client(tmp_path: Path, web: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "runs", web_dir=web))


def test_the_root_serves_the_app_and_is_never_cached(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == INDEX
    assert response.headers["content-type"].startswith("text/html")
    # A rebuild changes the asset names inside index.html, so it must be re-checked.
    assert response.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize(
    "page", ["/runs", "/runs/20261103_011450_titanic-dataset", "/no/such/page"]
)
def test_page_addresses_get_the_app_so_deep_links_and_reloads_work(
    client: TestClient, page: str
) -> None:
    # These pages exist only in the React router; the server hands over index.html.
    response = client.get(page)
    assert response.status_code == 200
    assert response.text == INDEX


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/assets/index-abc123.js", "text/javascript"),
        ("/assets/figtree-latin-400.woff2", "font/woff2"),  # unknown to Python here
        ("/favicon.svg", "image/svg+xml"),
    ],
)
def test_built_files_are_served_with_the_right_type(
    client: TestClient, path: str, content_type: str
) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)
    assert response.text != INDEX


def test_api_routes_win_and_unknown_api_addresses_are_json_404s(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    for path in ("/api/nope", "/api", "/api/runs/x/nope"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.headers["content-type"] == "application/json", path
    # The generated API docs still work.
    assert client.get("/docs").status_code == 200


@pytest.mark.parametrize("path", ["/%2e%2e/.env", "/..%2f.env", "/assets/..%2f..%2f.env"])
def test_files_outside_the_build_are_never_served(client: TestClient, path: str) -> None:
    # Encoded, because HTTP clients (and browsers) collapse a plain "/../" before sending.
    response = client.get(path)
    assert SECRET not in response.text


def test_without_a_build_the_api_still_runs_and_says_how_to_build(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "runs", web_dir=tmp_path / "missing"))

    response = client.get("/runs")
    assert response.status_code == 503
    assert "npm run build" in response.text
    assert client.get("/api/health").status_code == 200


def test_web_dir_none_serves_the_api_only(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "runs", web_dir=None))
    assert client.get("/").status_code == 503
    assert client.get("/api/health").status_code == 200
