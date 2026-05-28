FROM swr.cn-east-3.myhuaweicloud.com/weaver-qianliling/deploy:cuda11.7.1py3.10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/app \
    SERVICE_ROLE=api \
    HOST=0.0.0.0 \
    PORT=18001 \
    MODEL_HOST=0.0.0.0 \
    MODEL_PORT=18002 \
    DESENSITIZE_MODEL_SERVICE_URL=http://llm-messages-encryptor-model:18002 \
    DESENSITIZE_DEVICE=nvidia \
    DESENSITIZE_DEVICE_ID=0 \
    DESENSITIZE_MODEL_PATH=/app/resources/models/wordtag \
    DESENSITIZE_UIE_MODEL_PATH=/app/resources/models/uie-base \
    DESENSITIZE_AUTO_DOWNLOAD_MODEL=false \
    DESENSITIZE_ENABLE_UIE_CUSTOM=true \
    DESENSITIZE_PRELOAD_UIE_CUSTOM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    CPU_NUM=1 \
    MALLOC_ARENA_MAX=2 \
    FLAGS_allocator_strategy=auto_growth \
    FLAGS_fraction_of_gpu_memory_to_use=0.4 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HEALTHCHECK_PATH=/readyz

WORKDIR /app

ARG APP_UID=10001
ARG APP_GID=10001

RUN sed -i 's#http://archive.ubuntu.com/ubuntu#http://mirrors.tuna.tsinghua.edu.cn/ubuntu#g; s#http://security.ubuntu.com/ubuntu#http://mirrors.tuna.tsinghua.edu.cn/ubuntu#g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3.10 python3-pip libgomp1 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --home-dir /app --shell /usr/sbin/nologin --no-create-home app \
    && chown -R app:app /app \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt requirements-model.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && python -m pip install -r requirements-model.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

COPY --chown=app:app app.py model_app.py ./
COPY --chown=app:app desensitize ./desensitize
COPY --chown=app:app resources/models/wordtag/README.md ./resources/models/wordtag/README.md
COPY --chown=app:app resources/models/uie-base/README.md ./resources/models/uie-base/README.md
COPY --chown=app:app example_preprocess_request.json README.md ./

RUN mkdir -p /app/resources/models/wordtag /app/resources/models/uie-base /app/.paddlenlp \
    && chown -R app:app /app/resources /app/.paddlenlp

VOLUME ["/app/resources/models/wordtag", "/app/resources/models/uie-base"]

EXPOSE 18001 18002

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import json, os, urllib.request; role=os.getenv('SERVICE_ROLE', 'api'); port=os.getenv('MODEL_PORT', '18002') if role == 'model' else os.getenv('PORT', '18001'); path=os.getenv('HEALTHCHECK_PATH', '/readyz'); resp=urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=3); raise SystemExit(0 if json.load(resp).get('status') in {'ok', 'ready'} else 1)"

USER app

CMD ["sh", "-c", "if [ \"$SERVICE_ROLE\" = \"model\" ]; then exec python model_app.py; else exec python app.py; fi"]
