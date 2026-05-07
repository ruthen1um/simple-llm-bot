FROM ghcr.io/astral-sh/uv:python3.14-alpine

ENV UV_NO_DEV=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY src/ ./src/

CMD ["uv", "run", "src/main.py"]
