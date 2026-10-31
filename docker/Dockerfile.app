FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r forge && useradd -r -g forge -m forge

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir --upgrade "pip>=24.0" && pip install --no-cache-dir uv==0.13.5
RUN uv sync --frozen --only-group client

COPY src/ ./src/
ENV PYTHONPATH="/app/src"

RUN chown -R forge:forge /app
USER forge

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
