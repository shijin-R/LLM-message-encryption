FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/app

WORKDIR /app

ARG APP_UID=10001
ARG APP_GID=10001

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --home-dir /app --shell /usr/sbin/nologin --no-create-home app \
    && mkdir -p /app \
    && chown -R app:app /app


FROM base AS api

LABEL org.opencontainers.image.title="llm-messages-encryptor-api" \
      org.opencontainers.image.description="API service for preprocessing and desensitizing LLM requests" \
      org.opencontainers.image.source="Desensitize2"

# API 镜像只安装 HTTP 服务依赖；业务方启动 API 时不需要模型文件和 Paddle 依赖。
ENV HOST=0.0.0.0 \
    PORT=18001 \
    DESENSITIZE_RECOGNIZER_BACKEND=remote \
    DESENSITIZE_MODEL_SERVICE_URL=http://llm-messages-encryptor-model:18002 \
    HEALTHCHECK_PATH=/readyz

COPY requirements-api.txt .
RUN python -m pip install -r requirements-api.txt

COPY --chown=app:app app.py .
COPY --chown=app:app desensitize ./desensitize
COPY --chown=app:app example_preprocess_request.json .
COPY --chown=app:app README.md .

EXPOSE 18001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import json, os, urllib.request; port=os.getenv('PORT', '18001'); path=os.getenv('HEALTHCHECK_PATH', '/readyz'); resp=urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=3); raise SystemExit(0 if json.load(resp).get('status') in {'ok', 'ready'} else 1)"

USER app

CMD ["python", "app.py"]


FROM base AS model

LABEL org.opencontainers.image.title="llm-messages-encryptor-model" \
      org.opencontainers.image.description="Model service for sensitive entity recognition" \
      org.opencontainers.image.source="Desensitize2"

# 模型镜像独立持有 Paddle 依赖和模型目录，供多个 API 实例共享调用。
ENV MODEL_HOST=0.0.0.0 \
    MODEL_PORT=18002 \
    DESENSITIZE_MODEL_PATH=/app/resources/models/wordtag \
    DESENSITIZE_UIE_MODEL_PATH=/app/resources/models/uie-base \
    DESENSITIZE_AUTO_DOWNLOAD_MODEL=true \
    DESENSITIZE_SYNC_DOWNLOADED_MODEL=true \
    DESENSITIZE_ENABLE_UIE_CUSTOM=true \
    DESENSITIZE_PRELOAD_UIE_CUSTOM=true \
    HEALTHCHECK_PATH=/readyz

# Paddle 推理依赖 OpenMP 运行库，slim 镜像需要显式安装 libgomp1。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt requirements-model.txt ./
RUN python -m pip install -r requirements-model.txt

COPY --chown=app:app model_app.py .
COPY --chown=app:app desensitize ./desensitize
COPY --chown=app:app resources/models/wordtag/README.md ./resources/models/wordtag/README.md
COPY --chown=app:app resources/models/uie-base/README.md ./resources/models/uie-base/README.md
COPY --chown=app:app README.md .

RUN mkdir -p /app/resources/models/wordtag /app/resources/models/uie-base /app/.paddlenlp \
    && chown -R app:app /app/resources /app/.paddlenlp

VOLUME ["/app/resources/models/wordtag", "/app/resources/models/uie-base"]

EXPOSE 18002

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import json, os, urllib.request; port=os.getenv('MODEL_PORT', '18002'); path=os.getenv('HEALTHCHECK_PATH', '/readyz'); resp=urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=3); raise SystemExit(0 if json.load(resp).get('status') in {'ok', 'ready'} else 1)"

USER app

CMD ["python", "model_app.py"]
