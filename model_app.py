"""独立模型服务入口。"""

import logging
import os
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from desensitize.config import ServiceConfig
from desensitize.recognizer_factory import build_model_recognizer
from desensitize.types import ModelSpan


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


def create_model_app(
    config: ServiceConfig | None = None,
    recognizer: Any | None = None,
) -> Flask:
    """创建并配置独立模型服务。"""
    app = Flask(__name__)

    # 模型服务始终使用本地识别器：这里会按配置加载 wordtag / UIE 模型，
    # 对 API 服务暴露一个轻量纯推理 HTTP 后端。
    config = ServiceConfig.from_env() if config is None else config
    recognizer = build_model_recognizer(config) if recognizer is None else recognizer

    # 生产部署下可在启动阶段预热 UIE，避免首个带 uie_schema 的请求承受加载耗时。
    if config.enable_uie_custom and config.preload_uie_custom:
        recognizer.warmup_uie(config.uie_warmup_schema)

    @app.get("/healthz")
    def healthz():
        # 健康检查偏向“配置和当前状态展示”，即使模型暂未完全 ready 也返回服务状态。
        payload = {
            "status": "ok",
            "model_path": str(config.model_path),
            "strict_local_model": config.strict_local_model,
            "enable_taskflow": config.enable_taskflow,
            "auto_download_model": config.auto_download_model,
            "sync_downloaded_model": config.sync_downloaded_model,
            "model_cache_path": str(config.downloaded_model_cache_path),
            "using_taskflow": recognizer.using_taskflow,
            "enable_uie_custom": config.enable_uie_custom,
            "preload_uie_custom": config.preload_uie_custom,
            "uie_warmup_schema": list(config.uie_warmup_schema),
            "uie_model_name": config.uie_model_name,
            "uie_model_path": str(config.uie_model_path),
            "uie_position_prob": config.uie_position_prob,
            "strict_uie_model": config.strict_uie_model,
            "uie_model_cache_path": str(config.downloaded_uie_model_cache_path),
            "using_uie": recognizer.using_uie,
        }
        payload.update(_device_health(config, recognizer))
        return jsonify(payload)

    @app.get("/readyz")
    def readyz():
        # 就绪检查偏向“是否可以接流量”：wordtag 必须可用；
        # 如果配置要求预加载 UIE，则 UIE 也要加载完成。
        wordtag_ready = not config.enable_taskflow or recognizer.using_taskflow
        uie_ready = (
            not config.enable_uie_custom
            or not config.preload_uie_custom
            or recognizer.using_uie
        )
        ready = wordtag_ready and uie_ready
        status_code = 200 if ready else 503
        return (
            jsonify(
                {
                    "status": "ready" if ready else "not_ready",
                    "wordtag_ready": wordtag_ready,
                    "uie_ready": uie_ready,
                    "using_taskflow": recognizer.using_taskflow,
                    "using_uie": recognizer.using_uie,
                }
            ),
            status_code,
        )

    @app.post("/v1/infer")
    def infer():
        # 组合推理接口：只接收模型推理所需参数，不解释业务 custom_entities。
        payload = _get_json_payload()
        text = _get_text(payload)
        tasks = _get_tasks(payload)
        spans = recognizer.infer(text, tasks)
        return _ok({"spans": [_span_to_dict(span) for span in spans]})

    @app.errorhandler(ValueError)
    def bad_request(exc: ValueError):
        return jsonify({"code": 400, "message": str(exc)}), 400

    @app.errorhandler(Exception)
    def internal_error(exc: Exception):
        if isinstance(exc, HTTPException):
            return exc
        app.logger.exception("Model service request failed: %s", exc)
        return jsonify({"code": 500, "message": "Internal server error."}), 500

    return app


def _get_json_payload() -> dict[str, Any]:
    """读取并校验请求体 JSON。"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _get_text(payload: dict[str, Any]) -> str:
    """读取待识别文本；None 视为空串，非字符串直接拒绝。"""
    text = payload.get("text", "")
    if text is None:
        return ""
    if not isinstance(text, str):
        raise ValueError("`text` must be a string.")
    return text


def _get_tasks(payload: dict[str, Any]) -> dict[str, Any]:
    """读取模型推理任务。"""
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        return {"wordtag": True, "uie_schema": []}
    return {
        "wordtag": tasks.get("wordtag", True) is not False,
        "uie_schema": tasks.get("uie_schema")
        if isinstance(tasks.get("uie_schema"), list)
        else [],
    }


def _device_health(config: ServiceConfig, recognizer: Any) -> dict[str, Any]:
    """汇总设备状态；真实值优先来自模型识别器，避免只展示配置。"""
    if hasattr(recognizer, "device_info"):
        return dict(recognizer.device_info())
    return {
        "device": config.device,
        "device_id": config.device_id,
        "taskflow_device_id": config.device_id if config.device == "nvidia" else -1,
        "gpu_available": False,
        "gpu_device_count": 0,
        "gpu_compiled_with_cuda": False,
        "gpu_error": "",
    }


def _ok(data: dict[str, Any]):
    """统一包装成功响应，保持模型服务和 API 服务响应风格一致。"""
    return jsonify({"code": 0, "message": "ok", "data": data})


def _span_to_dict(span: ModelSpan) -> dict[str, Any]:
    """把内部 ModelSpan 转成可 JSON 序列化的响应对象。"""
    return {
        "label": span.label,
        "text": span.text,
        "start": span.start,
        "end": span.end,
        "source": span.source,
        "probability": span.probability,
    }


app = create_model_app()


if __name__ == "__main__":
    host = os.getenv("MODEL_HOST", os.getenv("HOST", "127.0.0.1"))
    port = int(os.getenv("MODEL_PORT", "18002"))

    app.run(host=host, port=port)
