"""Static and rendered contracts for the supported container deployment."""

from __future__ import annotations

import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.contract
def test_container_artifacts_define_a_least_privilege_locked_deployment() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "@sha256:" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "UV_PROJECT_ENVIRONMENT=/opt/venv" in dockerfile
    assert "COPY --from=builder /opt/venv /opt/venv" in dockerfile
    assert "scripts/container_health.py /usr/local/bin/ragkit-health" in dockerfile
    assert "USER 65532:65532" in dockerfile
    assert "ENTRYPOINT" in dockerfile and "HEALTHCHECK NONE" in dockerfile
    assert 'CMD ["ragkit-http", "--config", "/app/config/ragkit.toml"' in dockerfile
    assert "read_only: true" in compose
    assert "cap_drop:" in compose and "- ALL" in compose
    assert "no-new-privileges:true" in compose
    assert "/app/config/ragkit.toml:ro" in compose
    assert "/data/corpus:ro" in compose
    assert "ragkit-data:/var/lib/ragkit" in compose
    assert "healthcheck:" in compose
    assert "http://127.0.0.1:8000/readyz" in compose
    assert ".env" in ignored and "*.pem" in ignored and ".git" in ignored
    assert "tests/" in ignored and "!scripts/container_health.py" in ignored


@pytest.mark.contract
def test_compose_file_renders_without_environment_secrets() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is unavailable")
    result = subprocess.run(
        ["docker", "compose", "config"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    rendered = result.stdout
    assert "OPENAI_API_KEY" not in rendered
    assert "0.0.0.0:8000" not in rendered
    assert "127.0.0.1" in rendered


@pytest.mark.contract
def test_container_probes_reject_non_http_and_credentialed_urls() -> None:
    health = runpy.run_path(str(ROOT / "scripts" / "container_health.py"))
    smoke = runpy.run_path(str(ROOT / "scripts" / "smoke.py"))

    assert health["_valid_http_url"]("file:///etc/passwd") is False
    assert health["_valid_http_url"]("http://user:pass@localhost:8000/readyz") is False
    with pytest.raises(smoke["SmokeFailure"], match="HTTP"):
        smoke["_require_http_url"]("file:///etc/passwd")
