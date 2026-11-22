"""Static checks on the Docker setup: the properties a careless edit could silently break.

These read the files; they do not run Docker (CI's docker job builds and smoke-tests the
image). Each test guards one promise the README makes about the image.
"""

import shlex

from autods.config import PROJECT_ROOT

DOCKERFILE = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")


def _ignored() -> set[str]:
    lines = (line.strip() for line in DOCKERIGNORE.splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def _instructions(keyword: str) -> list[str]:
    """The arguments of every top-level Dockerfile instruction with this keyword.

    Indented lines continue the previous instruction (HEALTHCHECK's own `CMD` is one),
    so only lines starting in column 0 count.
    """
    return [
        line.split(maxsplit=1)[1]
        for line in DOCKERFILE.splitlines()
        if line[:1].strip() and line.split(maxsplit=1)[:1] == [keyword]
    ]


def test_secrets_and_local_runs_never_reach_the_build() -> None:
    # The build context is sent to Docker in full; .env holds the API key.
    assert {".env", ".env.*", "runs/"} <= _ignored()


def test_installed_packages_and_build_output_are_not_copied_from_the_host() -> None:
    # Host builds are for Windows/macOS; the image must install and build its own.
    assert {".venv/", "web/node_modules/", "web/dist/"} <= _ignored()


def test_no_api_key_is_built_into_the_image() -> None:
    code = "\n".join(line for line in DOCKERFILE.splitlines() if not line.lstrip().startswith("#"))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", ".env"):
        assert name not in code, f"{name} appears in a Dockerfile instruction"


def test_copies_named_paths_only() -> None:
    # `COPY . .` would copy whatever .dockerignore forgot.
    sources = [shlex.split(args)[0] for args in _instructions("COPY")]
    assert "." not in sources and "./" not in sources


def test_the_app_runs_as_an_ordinary_user() -> None:
    users = _instructions("USER")
    assert users, "no USER: the app would run as root"
    assert users[-1] not in {"root", "0"}


def test_serves_on_all_interfaces_inside_the_container() -> None:
    # 127.0.0.1 inside a container is unreachable from the host, even with -p.
    (cmd,) = _instructions("CMD")
    assert '"--host", "0.0.0.0"' in cmd


def test_installs_exactly_the_lockfile_without_dev_tools() -> None:
    syncs = [line for line in DOCKERFILE.splitlines() if "uv sync" in line]
    assert syncs
    for line in syncs:
        assert "--locked" in line and "--no-dev" in line
