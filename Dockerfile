# Fidelis Dockerfile — runs fidelis-server in a container.
# Build: docker build -t fidelis:0.1.0 .
# Run:   docker run -p 19420:19420 -v fidelis-data:/data fidelis:0.1.0
#
# This image runs the fidelis-server only. For embedding/Ollama you point at
# a separate ollama container (see docker-compose.yml) or your host's Ollama.

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

EXPOSE 19420

# Health check: GET /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fs http://localhost:19420/health || exit 1

CMD ["fidelis-server"]
