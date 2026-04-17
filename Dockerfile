# ---- build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml ./
# Install deps into a virtualenv so we can copy it cleanly
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -e .

COPY . .

# ---- runtime stage ----
FROM python:3.11-slim

RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source (needed for editable-style imports)
COPY --from=builder /build/nous_proxy ./nous_proxy
COPY --from=builder /build/pyproject.toml ./

# Data directory for token/key persistence
RUN mkdir -p /app/data && chown -R app:app /app/data

ENV DATA_DIR=/app/data
ENV PROXY_HOST=0.0.0.0
ENV PROXY_PORT=8090

EXPOSE 8090

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "nous_proxy.main"]
