# Two images from one file:
#   base  serving only: API, Kafka worker, sink            (docker build --target base)
#   ops   adds training and orchestration dependencies      (docker build --target ops)
#
# Dependencies sit in their own layers, so editing code never reinstalls them.

FROM python:3.12-slim AS base-deps
WORKDIR /app
COPY pyproject.toml ./
RUN mkdir -p src/frauddetect && touch src/frauddetect/__init__.py \
    && pip install --no-cache-dir .

FROM base-deps AS ops-deps
RUN pip install --no-cache-dir ".[train,ops]"

FROM base-deps AS base
COPY src ./src
RUN pip install --no-cache-dir --no-deps --force-reinstall .
# Model files and data are mounted at runtime, so retraining needs no rebuild.
ENV FRAUD_HOME=/app

FROM ops-deps AS ops
COPY src ./src
RUN pip install --no-cache-dir --no-deps --force-reinstall .
ENV FRAUD_HOME=/app
