FROM ghcr.io/astral-sh/uv:python3.14-alpine

ENV UV_NO_DEV=1

WORKDIR /app
COPY . .

RUN uv sync --locked

CMD ["uv", "run", "main.py"]
