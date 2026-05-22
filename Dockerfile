FROM python:3.10-slim

# 运行时默认配置，容器内对外监听 18001，并从固定目录读取/同步模型。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=18001 \
    DESENSITIZE_MODEL_PATH=/app/resources/models/wordtag \
    DESENSITIZE_DICT_DIR=/app/resources/common_data/uie \
    DESENSITIZE_AUTO_DOWNLOAD_MODEL=true \
    DESENSITIZE_SYNC_DOWNLOADED_MODEL=true

WORKDIR /app

# Paddle 推理依赖 OpenMP 运行库，slim 镜像需要显式安装 libgomp1。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY desensitize ./desensitize
COPY resources/common_data ./resources/common_data
COPY resources/models/wordtag/README.md ./resources/models/wordtag/README.md
COPY example_preprocess_request.json .
COPY example_history_reuse_request.json .
COPY README.md .

# 模型文件较大，默认用 volume 持久化，避免每次重建镜像都重新内置模型。
VOLUME ["/app/resources/models/wordtag"]

EXPOSE 18001

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import json, urllib.request; resp=urllib.request.urlopen('http://127.0.0.1:18001/healthz', timeout=3); raise SystemExit(0 if json.load(resp).get('status') == 'ok' else 1)"

# 使用 app.py 内置入口启动 Flask 服务；生产环境可按需替换为 WSGI Server。
CMD ["python", "app.py"]
