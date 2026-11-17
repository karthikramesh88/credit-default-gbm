# Builds the scoring API. The model is trained during the build from the public UCI
# dataset (checksum-pinned), and the build fails unless the release gate approves it,
# so a running container always serves an approved model.

FROM python:3.11-slim

# libgomp1: OpenMP runtime XGBoost needs on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ARG UCI_URL=https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
ARG UCI_SHA256=56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602
RUN mkdir -p data/raw \
    && curl -fsSL -o /tmp/uci.zip "$UCI_URL" \
    && echo "$UCI_SHA256  /tmp/uci.zip" | sha256sum -c - \
    && unzip -q /tmp/uci.zip -d data/raw \
    && rm /tmp/uci.zip

# Train; models/LATEST exists only if the gate approved the model.
RUN .venv/bin/python -m credit_default_gbm.train --data uci && test -f models/LATEST

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "exec .venv/bin/python -m uvicorn credit_default_gbm.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
