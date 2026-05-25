"""独立模型服务入口。"""

import logging
import os
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from desensitize.config import ServiceConfig
from desensitize.recognizer_factory import build_local_recognizer
from desensitize.types import EntitySpan


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


def create_model_app() -> Flask:
    """创建并配置独立模型服务。"""
    app = Flask(__name__)

    config = ServiceConfig.from_env()
    recognizer = build_local_recognizer(config)

    if config.enable_uie_custom and config.preload_uie_custom:
        recognizer.warmup_uie(config.uie_warmup_schema)

    @app.get("/healthz")
    def healthz():
        return jsonify(
            {
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
        )

    @app.get("/readyz")
    def readyz():
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

    @app.post("/v1/recognize")
    def recognize_all():
        payload = _get_json_payload()
        text = _get_text(payload)
        custom_entities = _get_custom_entities(payload)
        spans = recognizer.recognize(text, custom_entities)
        return _ok({"spans": [_span_to_dict(span) for span in spans]})

    @app.post("/v1/recognize/builtin")
    def recognize_builtin():
        payload = _get_json_payload()
        text = _get_text(payload)
        spans = recognizer.recognize_builtin(text)
        return _ok({"spans": [_span_to_dict(span) for span in spans]})

    @app.post("/v1/recognize/custom")
    def recognize_custom():
        payload = _get_json_payload()
        text = _get_text(payload)
        custom_entities = _get_custom_entities(payload)
        spans = recognizer.recognize_custom(text, custom_entities)
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
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _get_text(payload: dict[str, Any]) -> str:
    text = payload.get("text", "")
    if text is None:
        return ""
    if not isinstance(text, str):
        raise ValueError("`text` must be a string.")
    return text


def _get_custom_entities(payload: dict[str, Any]) -> list[Any]:
    custom_entities = payload.get("custom_entities")
    if not isinstance(custom_entities, list):
        return []
    return custom_entities


def _ok(data: dict[str, Any]):
    return jsonify({"code": 0, "message": "ok", "data": data})


def _span_to_dict(span: EntitySpan) -> dict[str, Any]:
    return {
        "entity_type": span.entity_type,
        "text": span.text,
        "start": span.start,
        "end": span.end,
        "source": span.source,
    }


app = create_model_app()


if __name__ == "__main__":
    host = os.getenv("MODEL_HOST", os.getenv("HOST", "127.0.0.1"))
    port = int(os.getenv("MODEL_PORT", "18002"))

    app.run(host=host, port=port)
