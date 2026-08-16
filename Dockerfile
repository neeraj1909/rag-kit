ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

FROM ${PYTHON_IMAGE} AS builder

ARG UV_VERSION=0.9.22
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build

RUN python -m pip install "uv==${UV_VERSION}"
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra http --no-editable

FROM ${PYTHON_IMAGE} AS runtime

ENV HOME=/nonexistent \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAGKIT_CONFIG=/app/config/ragkit.toml \
    RAGKIT_HOST=0.0.0.0 \
    RAGKIT_PORT=8000

RUN addgroup --system --gid 65532 ragkit \
    && adduser --system --uid 65532 --ingroup ragkit --home /nonexistent \
       --no-create-home --disabled-password ragkit \
    && install -d -o 65532 -g 65532 /var/lib/ragkit /app/config /data/corpus

COPY --from=builder /opt/venv /opt/venv
COPY --chmod=0555 deployment/entrypoint.sh /usr/local/bin/ragkit-entrypoint
COPY --chmod=0555 scripts/container_health.py /usr/local/bin/ragkit-health

WORKDIR /app
USER 65532:65532
EXPOSE 8000

# Compose owns the service-specific readiness check so operators can change the
# published port without rebuilding the image.
HEALTHCHECK NONE
ENTRYPOINT ["/usr/local/bin/ragkit-entrypoint"]
CMD ["ragkit-http", "--config", "/app/config/ragkit.toml", "--host", "0.0.0.0", "--port", "8000"]
