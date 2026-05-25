"""调用独立模型服务的识别器客户端。"""

import json
import urllib.error
import urllib.request
from typing import Any

from .types import EntitySpan


class RemoteRecognizerError(RuntimeError):
    """模型服务调用失败。"""


class HTTPRecognizerClient:
    """通过 HTTP 调用独立模型服务，保持与本地识别器相近的接口。"""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def using_taskflow(self) -> bool:
        return bool(self.health().get("using_taskflow"))

    @property
    def using_uie(self) -> bool:
        return bool(self.health().get("using_uie"))

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/healthz")

    def ready(self) -> dict[str, Any]:
        return self._request_json("GET", "/readyz")

    def recognize(
        self,
        text: str,
        custom_entities: list[Any],
    ) -> list[EntitySpan]:
        payload = {
            "text": text,
            "custom_entities": self._normalize_custom_entities(custom_entities),
        }
        return self._request_spans("/v1/recognize", payload)

    def recognize_builtin(self, text: str) -> list[EntitySpan]:
        return self._request_spans("/v1/recognize/builtin", {"text": text})

    def recognize_custom(
        self,
        text: str,
        custom_entities: list[Any],
    ) -> list[EntitySpan]:
        payload = {
            "text": text,
            "custom_entities": self._normalize_custom_entities(custom_entities),
        }
        return self._request_spans("/v1/recognize/custom", payload)

    def _request_spans(self, path: str, payload: dict[str, Any]) -> list[EntitySpan]:
        data = self._request_json("POST", path, payload)
        return [self._span_from_dict(item) for item in data.get("spans", [])]

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteRecognizerError(
                f"Model service returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteRecognizerError(f"Model service is unavailable: {exc}") from exc

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RemoteRecognizerError("Model service returned invalid JSON.") from exc

        if not isinstance(data, dict):
            raise RemoteRecognizerError("Model service response must be a JSON object.")

        if data.get("code", 0) != 0:
            message = data.get("message", "unknown error")
            raise RemoteRecognizerError(f"Model service error: {message}")

        result = data.get("data", data)
        if not isinstance(result, dict):
            raise RemoteRecognizerError("Model service data must be a JSON object.")
        return result

    @staticmethod
    def _span_from_dict(item: Any) -> EntitySpan:
        if not isinstance(item, dict):
            raise RemoteRecognizerError("Model service span item must be an object.")
        return EntitySpan(
            entity_type=str(item.get("entity_type", "")),
            text=str(item.get("text", "")),
            start=int(item.get("start", -1)),
            end=int(item.get("end", -1)),
            source=str(item.get("source", "")),
        )

    @staticmethod
    def _normalize_custom_entities(value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        return value
