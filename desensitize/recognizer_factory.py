"""识别器构造逻辑。"""

from typing import Any

from .application_recognizer import ApplicationEntityRecognizer
from .config import ServiceConfig
from .remote_recognizer import HTTPRecognizerClient


def build_model_recognizer(config: ServiceConfig) -> Any:
    """构造模型服务进程内的本地模型识别器。"""
    from .recognizer import LocalEntityRecognizer

    return LocalEntityRecognizer(
        model_path=config.model_path,
        device_id=config.device_id,
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
    """构造 API 服务识别编排器；API 固定通过 HTTP 调用模型服务。"""
    model_client = HTTPRecognizerClient(
        base_url=config.model_service_url,
        timeout=config.model_service_timeout,
    )
    return ApplicationEntityRecognizer(
        model_client=model_client,
        max_text_len=config.max_text_len,
    )
