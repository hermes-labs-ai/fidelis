# Fidelis Dockerfile — runs the fidelis MCP stdio server by default.
# Build: docker build -t fidelis:0.1.0 .
# Run:   docker run -i fidelis:0.1.0
#
# Default entrypoint: the MCP stdio server (`fidelis mcp serve`). Registry
# inspectors (e.g. Glama) build this image and speak MCP `initialize` /
# `tools/list` over stdio with no daemon and no Ollama required.
#
# To run the HTTP memory server instead (needs a reachable Ollama for
# embeddings — see docker-compose.yml or point OLLAMA_URL at your host):
#   docker run -p 19420:19420 -v fidelis-data:/data \
#     -e FIDELIS_ENTRYPOINT=http fidelis:0.1.0

FROM python:3.12-slim

LABEL org.opencontainers.image.title="fidelis"
LABEL org.opencontainers.image.description="Agent memory with zero-LLM retrieval and a $0-incremental QA scaffold"
LABEL org.opencontainers.image.source="https://github.com/hermes-labs-ai/fidelis"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="0.1.0"

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
ENV FIDELIS_HOME=/data
ENV FIDELIS_PORT=19420

# Default Ollama URL points at host (override via env)
ENV OLLAMA_URL=http://host.docker.internal:11434

# Which server to run: "mcp" (default, stdio, no Ollama needed — what a
# registry build/inspector talks to) or "http" (fidelis-server, needs a
# reachable Ollama for embeddings; see docker-compose.yml).
ENV FIDELIS_ENTRYPOINT=mcp

EXPOSE 19420

# Health check: GET /health. Only meaningful for FIDELIS_ENTRYPOINT=http;
# under the default mcp entrypoint there is no HTTP surface, so this simply
# reports unhealthy and callers running mcp mode should ignore/override it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fs http://localhost:19420/health || exit 1

CMD ["sh", "-c", "if [ \"$FIDELIS_ENTRYPOINT\" = \"http\" ]; then exec fidelis-server; else exec fidelis mcp serve; fi"]
