"""识别器构造逻辑。"""

from typing import Any

from .config import ServiceConfig
from .remote_recognizer import HTTPRecognizerClient


REMOTE_BACKENDS = {"remote", "http"}


def build_local_recognizer(config: ServiceConfig) -> Any:
    """构造进程内本地模型识别器。"""
    # 延迟导入，避免 API-only 部署在 remote 模式下接触本地模型代码。
    from .recognizer import LocalEntityRecognizer

    return LocalEntityRecognizer(
        model_path=config.model_path,
        device_id=config.device_id,
        max_text_len=config.max_text_len,
        enable_taskflow=config.enable_taskflow,
        strict_local_model=config.strict_local_model,
        auto_download_model=config.auto_download_model,
        sync_downloaded_model=config.sync_downloaded_model,
        downloaded_model_cache_path=config.downloaded_model_cache_path,
        enable_uie_custom=config.enable_uie_custom,
        uie_model_name=config.uie_model_name,
        uie_model_path=config.uie_model_path,
        uie_position_prob=config.uie_position_prob,
        strict_uie_model=config.strict_uie_model,
        downloaded_uie_model_cache_path=config.downloaded_uie_model_cache_path,
    )


def build_api_recognizer(config: ServiceConfig):
    """按 API 服务配置构造识别器后端。"""
    if config.recognizer_backend in {"local", ""}:
        return build_local_recognizer(config)
    if config.recognizer_backend in REMOTE_BACKENDS:
        return HTTPRecognizerClient(
            base_url=config.model_service_url,
            timeout=config.model_service_timeout,
        )
    raise ValueError(
        "`DESENSITIZE_RECOGNIZER_BACKEND` must be `local` or `remote`."
    )
