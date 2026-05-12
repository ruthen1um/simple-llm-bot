# Quick LLM Bot

An easy-to-deploy Telegram bot for LLM inference using [OpenAI
API](https://github.com/openai/openai-openapi)-compatible inference servers supporting at least chat
completions endpoint.

At the moment only one model can be used and it must be preloaded.

# Running
Before running containers you need to create `.env.remote` and/or `.env.local` files according to examples: `.env.remote.example` and
`.env.local.example`.

Run docker container with remote inference provider:
```
docker compose --file compose.remote.yml --env-file .env.remote up
```

Run docker container with local inference provider (local llama-cpp server):
```
docker compose --file compose.local.yml --env-file .env.local up
```
