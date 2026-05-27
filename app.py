"""HTTP 服务入口。"""

import logging
import os

from flask import Flask, jsonify, request

from desensitize.config import ServiceConfig
from desensitize.service import DesensitizeService


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


def create_app() -> Flask:
    """创建并配置 Flask 应用。"""
    app = Flask(__name__)

    config = ServiceConfig.from_env()
    service = DesensitizeService(config)

    def build_health_data() -> dict:
        data = {
            "status": "ok",
            "model_service_url": config.model_service_url,
            "model_service_timeout": config.model_service_timeout,
        }

        try:
            data["model_service_status"] = "ok"
            data["model_service"] = service.recognizer.health()
        except Exception as exc:
            data["model_service_status"] = "unavailable"
            data["model_service_error"] = str(exc)
        return data

    def build_ready_data() -> tuple[dict, int]:
        try:
            model_ready = service.recognizer.ready()
            return (
                {
                    "status": "ready",
                    "model_service": model_ready,
                },
                200,
            )
        except Exception as exc:
            return (
                {
                    "status": "not_ready",
                    "model_service_error": str(exc),
                },
                503,
            )

    @app.get("/healthz")
    def healthz():
        # 返回关键运行配置，便于排查环境问题。
        return jsonify(build_health_data())

    @app.get("/readyz")
    def readyz():
        ready_data, status_code = build_ready_data()
        return jsonify(ready_data), status_code

    @app.post("/v1/llm/preprocess")
    def llm_preprocess():
        # 大模型前置拦截接口：返回可直接转发的上游请求体。
        payload = request.get_json(silent=True)
        if payload is None:
            return (
                jsonify(
                    {
                        "code": 400,
                        "message": "Request body must be valid JSON.",
                    }
                ),
                400,
            )

        try:
            data = service.prepare_llm_request(payload)
        except ValueError as exc:
            return jsonify({"code": 400, "message": str(exc)}), 400
        except Exception as exc:
            app.logger.exception("LLM preprocess request failed: %s", exc)
            return jsonify({"code": 500, "message": "Internal server error."}), 500

        return jsonify({"code": 0, "message": "ok", "data": data})

    return app


app = create_app()


if __name__ == "__main__":
    # Windows/本地测试默认使用 localhost，避免 0.0.0.0 绑定权限问题。
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "18001"))

    app.run(host=host, port=port)
