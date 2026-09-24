# Fidelis Dockerfile — runs the fidelis MCP stdio server by default.
# Build: docker build -t fidelis:0.3.0rc1 .
# Run:   docker run -i fidelis:0.3.0rc1
#
# Default entrypoint: the MCP stdio server (`fidelis mcp serve`). Registry
# inspectors (e.g. Glama) build this image and speak MCP `initialize` /
# `tools/list` over stdio with no daemon and no Ollama required.
#
# To run the HTTP memory server instead (needs a reachable Ollama for
# embeddings — see docker-compose.yml or point OLLAMA_URL at your host):
#   docker run -p 19420:19420 -v fidelis-data:/data \
#     -e FIDELIS_ENTRYPOINT=http fidelis:0.3.0rc1

FROM python:3.12-slim

LABEL org.opencontainers.image.title="fidelis"
LABEL org.opencontainers.image.description="Local memory with verbatim records and correction history"
LABEL org.opencontainers.image.source="https://github.com/hermes-labs-ai/fidelis"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version="0.3.0rc1"

# System deps for chromadb + sqlite
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install fidelis from source (alternative: COPY pyproject.toml + pip install .)
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Persist memory store outside the container
VOLUME ["/data"]
ENV COGITO_STORE_PATH=/data/store
ENV FIDELIS_QUEUE_DIR=/data/queue
ENV MEM0_TELEMETRY=False
ENV ANONYMIZED_TELEMETRY=False
ENV CHROMA_TELEMETRY_DISABLED=True
ENV FIDELIS_PORT=19420

# Default Ollama URL points at host (override via env)
ENV COGITO_OLLAMA_URL=http://host.docker.internal:11434

# Which server to run: "mcp" (default, stdio, no Ollama needed — what a
# registry build/inspector talks to) or "http" (fidelis-server, needs a
# reachable Ollama for embeddings; see docker-compose.yml).
ENV FIDELIS_ENTRYPOINT=mcp

EXPOSE 19420

# Health check: GET /health. Only meaningful for FIDELIS_ENTRYPOINT=http;
# under the default mcp entrypoint there is no HTTP surface, so this simply
# reports unhealthy and callers running mcp mode should ignore/override it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD if [ "$FIDELIS_ENTRYPOINT" = "http" ]; then curl -fs "http://localhost:${FIDELIS_PORT}/health" || exit 1; else exit 0; fi

CMD ["sh", "-c", "if [ \"$FIDELIS_ENTRYPOINT\" = \"http\" ]; then exec fidelis-server --host 0.0.0.0; else exec fidelis mcp serve; fi"]
