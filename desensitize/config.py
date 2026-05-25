"""服务配置定义。

集中管理环境变量读取逻辑，避免配置散落在业务代码中。
"""

import os
from dataclasses import dataclass
from pathlib import Path


def _env_to_bool(name: str, default: bool) -> bool:
    """把环境变量解析为布尔值。"""
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class ServiceConfig:
    """脱敏服务运行配置。"""

    # 本地模型目录（优先加载）。
    model_path: Path
    # 模型推理设备 ID。
    device_id: int = 0
    # 单条消息允许处理的最大长度。
    max_text_len: int = 10000
    # 是否启用 PaddleNLP Taskflow wordtag 模型识别。
    enable_taskflow: bool = True
    # 严格模式：模型不可用时启动报错；本服务要求必须使用 Taskflow wordtag。
    strict_local_model: bool = True
    # 本地模型缺失时是否允许自动下载默认模型。
    auto_download_model: bool = True
    # 自动下载后是否同步回本地模型目录。
    sync_downloaded_model: bool = True
    # PaddleNLP 默认下载缓存目录。
    downloaded_model_cache_path: Path = Path.home() / ".paddlenlp" / "taskflow" / "wordtag"
    # 是否启用 UIE 信息抽取旁路识别业务自定义实体；默认开启，按请求懒加载。
    enable_uie_custom: bool = True
    # UIE 信息抽取模型名。
    uie_model_name: str = "uie-base"
    # UIE 本地模型目录。
    uie_model_path: Path = Path("resources/models/uie-base")
    # UIE span 起止位置概率阈值。
    uie_position_prob: float = 0.5
    # UIE 严格模式：旁路模型不可用时是否抛错。
    strict_uie_model: bool = False
    # PaddleNLP UIE 默认下载缓存目录。
    downloaded_uie_model_cache_path: Path = (
        Path.home() / ".paddlenlp" / "taskflow" / "information_extraction" / "uie-base"
    )

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """从环境变量构建配置对象，并把相对路径标准化为绝对路径。"""
        project_root = Path(__file__).resolve().parents[1]

        def resolve_path(env_name: str, default: Path) -> Path:
            path = Path(os.getenv(env_name, str(default))).expanduser()
            if path.is_absolute():
                return path
            return (project_root / path).resolve()

        model_path = resolve_path(
            "DESENSITIZE_MODEL_PATH",
            project_root / "resources" / "models" / "wordtag",
        )
        downloaded_model_cache_path = resolve_path(
            "DESENSITIZE_MODEL_CACHE_PATH",
            Path.home() / ".paddlenlp" / "taskflow" / "wordtag",
        )
        uie_model_name = os.getenv("DESENSITIZE_UIE_MODEL_NAME", "uie-base")
        uie_model_path = resolve_path(
            "DESENSITIZE_UIE_MODEL_PATH",
            project_root / "resources" / "models" / uie_model_name,
        )
        downloaded_uie_model_cache_path = resolve_path(
            "DESENSITIZE_UIE_MODEL_CACHE_PATH",
            Path.home()
            / ".paddlenlp"
            / "taskflow"
            / "information_extraction"
            / uie_model_name,
        )

        return cls(
            model_path=model_path,
            device_id=int(os.getenv("DESENSITIZE_DEVICE_ID", "0")),
            max_text_len=int(os.getenv("DESENSITIZE_MAX_TEXT_LEN", "10000")),
            enable_taskflow=_env_to_bool("DESENSITIZE_ENABLE_TASKFLOW", True),
            strict_local_model=_env_to_bool("DESENSITIZE_STRICT_LOCAL_MODEL", True),
            auto_download_model=_env_to_bool("DESENSITIZE_AUTO_DOWNLOAD_MODEL", True),
            sync_downloaded_model=_env_to_bool("DESENSITIZE_SYNC_DOWNLOADED_MODEL", True),
            downloaded_model_cache_path=downloaded_model_cache_path,
            enable_uie_custom=_env_to_bool("DESENSITIZE_ENABLE_UIE_CUSTOM", True),
            uie_model_name=uie_model_name,
            uie_model_path=uie_model_path,
            uie_position_prob=float(os.getenv("DESENSITIZE_UIE_POSITION_PROB", "0.5")),
            strict_uie_model=_env_to_bool("DESENSITIZE_STRICT_UIE_MODEL", False),
            downloaded_uie_model_cache_path=downloaded_uie_model_cache_path,
        )
