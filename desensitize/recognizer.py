"""本地模型推理适配器。

只负责加载并调用 PaddleNLP Taskflow wordtag / UIE 模型，返回原始模型标签。
业务实体类型映射、正则补漏和占位符处理都在应用层完成。
"""

import logging
import shutil
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .types import ModelSpan


logger = logging.getLogger(__name__)


class LocalEntityRecognizer:
    # PaddleNLP 2.8.x 没有 Taskflow("wordtag") 这个顶层任务名。
    # 正确入口是 Taskflow("ner")，其 accurate 模式内部使用 wordtag 模型。
    TASKFLOW_TASK = "ner"
    UIE_TASKFLOW_TASK = "information_extraction"
    WORDTAG_SOURCE = "wordtag"
    UIE_SOURCE = "uie"
    SUPPORTED_DEVICES = {"cpu", "nvidia"}
    DEFAULT_MAX_MODEL_TOKENS = 512
    WORDTAG_SPECIAL_TOKENS = 2
    UIE_SPECIAL_TOKENS = 4
    DEFAULT_UIE_TARGET_TEXT_TOKENS = 512
    UIE_BATCH_SIZE = 8
    SEMANTIC_BOUNDARY_CHARS = "\r\n。.！？!?；;"

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        device_id: int = 0,
        enable_taskflow: bool = True,
        strict_local_model: bool = True,
        auto_download_model: bool = True,
        sync_downloaded_model: bool = False,
        downloaded_model_cache_path: Path | None = None,
        enable_uie_custom: bool = True,
        uie_model_name: str = "uie-base",
        uie_model_path: Path | None = None,
        uie_position_prob: float = 0.5,
        max_model_tokens: int = DEFAULT_MAX_MODEL_TOKENS,
        uie_target_text_tokens: int = DEFAULT_UIE_TARGET_TEXT_TOKENS,
        strict_uie_model: bool = False,
        downloaded_uie_model_cache_path: Path | None = None,
    ) -> None:
        # 初始化识别器配置。
        self.model_path = Path(model_path)
        self.device = self._normalize_device(device)
        self.device_id = int(device_id)
        self._gpu_status = self._detect_nvidia_status()
        if self.device == "nvidia":
            self._ensure_nvidia_ready()
        # PaddleNLP Taskflow 约定 CPU 使用 -1，GPU 使用具体卡号。
        self.taskflow_device_id = self.device_id if self.device == "nvidia" else -1
        self.enable_taskflow = enable_taskflow
        self.strict_local_model = strict_local_model
        self.auto_download_model = auto_download_model
        self.sync_downloaded_model = sync_downloaded_model
        self.downloaded_model_cache_path = (
            Path(downloaded_model_cache_path)
            if downloaded_model_cache_path is not None
            else Path.home() / ".paddlenlp" / "taskflow" / "wordtag"
        )
        self.enable_uie_custom = enable_uie_custom
        self.uie_model_name = uie_model_name
        self.uie_model_path = (
            Path(uie_model_path)
            if uie_model_path is not None
            else self.model_path.parent / uie_model_name
        )
        self.uie_position_prob = uie_position_prob
        self.max_model_tokens = self._normalize_positive_int(
            max_model_tokens,
            "max_model_tokens",
        )
        self.uie_target_text_tokens = self._normalize_positive_int(
            uie_target_text_tokens,
            "uie_target_text_tokens",
        )
        self.strict_uie_model = strict_uie_model
        self.downloaded_uie_model_cache_path = (
            Path(downloaded_uie_model_cache_path)
            if downloaded_uie_model_cache_path is not None
            else Path.home()
            / ".paddlenlp"
            / "taskflow"
            / "information_extraction"
            / uie_model_name
        )
        self._uie_taskflow = None
        self._uie_schema: tuple[str, ...] = ()
        self._taskflow_lock = threading.RLock()
        self._uie_lock = threading.RLock()
        self._infer_lock = threading.RLock()

        self._taskflow = self._build_taskflow()

    @property
    def using_taskflow(self) -> bool:
        """当前请求识别链路是否已启用 Taskflow wordtag。"""
        return self._taskflow is not None

    @property
    def using_uie(self) -> bool:
        """当前进程是否已懒加载 UIE 信息抽取旁路。"""
        return self._uie_taskflow is not None

    @property
    def gpu_available(self) -> bool:
        """NVIDIA GPU 是否可被当前 Paddle 运行时使用。"""
        return bool(self._gpu_status.get("available", False))

    @property
    def gpu_device_count(self) -> int:
        """当前 Paddle 运行时能看到的 NVIDIA GPU 数量。"""
        return int(self._gpu_status.get("device_count", 0))

    @property
    def gpu_compiled_with_cuda(self) -> bool:
        """当前安装的 Paddle 是否为 CUDA 版本。"""
        return bool(self._gpu_status.get("compiled_with_cuda", False))

    @property
    def gpu_error(self) -> str:
        """GPU 检测失败时的错误信息，便于健康检查排查环境。"""
        return str(self._gpu_status.get("error", ""))

    def device_info(self) -> dict[str, Any]:
        """返回健康检查需要展示的推理设备状态。"""
        return {
            "device": self.device,
            "device_id": self.device_id,
            "taskflow_device_id": self.taskflow_device_id,
            "gpu_available": self.gpu_available,
            "gpu_device_count": self.gpu_device_count,
            "gpu_compiled_with_cuda": self.gpu_compiled_with_cuda,
            "gpu_error": self.gpu_error,
        }

    @classmethod
    def _normalize_device(cls, device: Any) -> str:
        normalized = str(device or "cpu").strip().lower()
        if normalized not in cls.SUPPORTED_DEVICES:
            raise ValueError(
                "DESENSITIZE_DEVICE must be one of: cpu, nvidia. "
                f"got={normalized!r}"
            )
        return normalized

    @staticmethod
    def _normalize_positive_int(value: Any, name: str) -> int:
        normalized = int(value)
        if normalized <= 0:
            raise ValueError(f"{name} must be a positive integer. got={normalized!r}")
        return normalized

    def _detect_nvidia_status(self) -> dict[str, Any]:
        """检测 NVIDIA CUDA 能力；CPU 模式不导入 Paddle，避免影响轻量测试。"""
        if self.device != "nvidia":
            return {
                "available": False,
                "device_count": 0,
                "compiled_with_cuda": False,
                "error": "",
            }

        try:
            import paddle
        except Exception as exc:
            return {
                "available": False,
                "device_count": 0,
                "compiled_with_cuda": False,
                "error": str(exc),
            }

        try:
            compiled_with_cuda = bool(paddle.is_compiled_with_cuda())
            device_count = self._read_paddle_gpu_count(paddle) if compiled_with_cuda else 0
            return {
                "available": compiled_with_cuda and device_count > 0,
                "device_count": device_count,
                "compiled_with_cuda": compiled_with_cuda,
                "error": "",
            }
        except Exception as exc:
            return {
                "available": False,
                "device_count": 0,
                "compiled_with_cuda": False,
                "error": str(exc),
            }

    @staticmethod
    def _read_paddle_gpu_count(paddle_module: Any) -> int:
        cuda = getattr(getattr(paddle_module, "device", None), "cuda", None)
        for method_name in ("device_count", "get_device_count"):
            device_count = getattr(cuda, method_name, None)
            if callable(device_count):
                return max(0, int(device_count()))
        return 0

    def _ensure_nvidia_ready(self) -> None:
        """NVIDIA 模式采用严格启动：环境不满足就失败，避免生产静默跑 CPU。"""
        if not self.gpu_compiled_with_cuda:
            detail = f" Detail: {self.gpu_error}" if self.gpu_error else ""
            raise RuntimeError(
                "DESENSITIZE_DEVICE=nvidia requires paddlepaddle-gpu with CUDA."
                + detail
            )
        if self.gpu_device_count <= 0:
            raise RuntimeError(
                "DESENSITIZE_DEVICE=nvidia requires at least one visible NVIDIA GPU."
            )
        if self.device_id < 0 or self.device_id >= self.gpu_device_count:
            raise RuntimeError(
                "DESENSITIZE_DEVICE_ID is out of range for visible NVIDIA GPUs. "
                f"device_id={self.device_id}, gpu_device_count={self.gpu_device_count}."
            )

    def _has_local_model_files(self, model_path: Path | None = None) -> bool:
        """判断本地模型目录是否存在可用模型文件。"""
        target_path = self.model_path if model_path is None else Path(model_path)
        if not target_path.exists() or not target_path.is_dir():
            return False

        for item in target_path.rglob("*"):
            if item.is_file() and item.name.lower() not in {"readme.md", ".gitkeep"}:
                return True
        return False

    def _build_taskflow(self):
        """按配置创建 Taskflow 推理实例，不可用时自动降级。"""
        if not self.enable_taskflow:
            logger.info("Taskflow wordtag is disabled")
            return None

        try:
            from paddlenlp import Taskflow
        except ImportError as exc:
            if self.strict_local_model:
                raise RuntimeError(
                    "paddlenlp is required. Please install dependencies from requirements-model.txt"
                ) from exc
            logger.warning(
                "paddlenlp is not available, skip wordtag inference: %s",
                exc,
            )
            return None

        # 1) 优先加载本地 wordtag 模型目录。
        if self._has_local_model_files():
            local_taskflow = self._init_local_taskflow(
                Taskflow,
                raise_on_error=self.strict_local_model,
            )
            if local_taskflow is not None:
                return local_taskflow

        # 2) 本地缺失时可按配置自动下载默认 wordtag 模型。
        if self.auto_download_model:
            downloaded_taskflow = self._init_default_taskflow(
                Taskflow,
                raise_on_error=self.strict_local_model,
            )
            if downloaded_taskflow is not None:
                self._sync_downloaded_model_to_local()

                if self._has_local_model_files():
                    local_taskflow = self._init_local_taskflow(Taskflow, raise_on_error=False)
                    if local_taskflow is not None:
                        return local_taskflow
                return downloaded_taskflow

        if self.strict_local_model:
            raise FileNotFoundError(
                "Local wordtag model is unavailable and auto-download failed/disabled. "
                f"model_path={self.model_path}, auto_download_model={self.auto_download_model}."
            )

        logger.warning(
            "Local model is unavailable; skip wordtag inference. "
            "model_path=%s auto_download_model=%s",
            self.model_path,
            self.auto_download_model,
        )
        return None

    def _init_local_taskflow(self, taskflow_cls, raise_on_error: bool):
        """从本地目录初始化 Taskflow NER wordtag。"""
        try:
            self.model_path.mkdir(parents=True, exist_ok=True)
            return self._create_wordtag_taskflow(
                taskflow_cls,
                task_path=self.model_path.as_posix(),
                home_path=self.model_path.as_posix(),
            )
        except Exception as exc:
            if raise_on_error:
                raise RuntimeError(
                    "Failed to initialize local Taskflow wordtag model at "
                    f"{self.model_path}."
                ) from exc
            logger.warning("Failed to initialize local Taskflow, fallback to default: %s", exc)
            return None

    def _init_default_taskflow(self, taskflow_cls, raise_on_error: bool):
        """初始化默认 Taskflow NER wordtag（会触发默认模型下载）。"""
        try:
            return self._create_wordtag_taskflow(taskflow_cls)
        except Exception as exc:
            if raise_on_error:
                raise RuntimeError(
                    "Failed to initialize default Taskflow wordtag auto-download."
                ) from exc
            logger.warning("Failed to auto-download default Taskflow model: %s", exc)
            return None

    def _create_wordtag_taskflow(self, taskflow_cls, **kwargs):
        """创建 PaddleNLP wordtag 识别器。

        注意：这里传入的是 `ner` 任务名，不是 `wordtag`。
        `entity_only=True` 会让 Taskflow 返回精简的实体序列，便于服务层统一处理。
        """
        return taskflow_cls(
            self.TASKFLOW_TASK,
            entity_only=True,
            device_id=self.taskflow_device_id,
            **kwargs,
        )

    def _ensure_uie_taskflow(self, schema: list[str]):
        """按 schema 懒加载或更新 UIE 信息抽取模型。"""
        if not getattr(self, "enable_uie_custom", False):
            return None

        normalized_schema = tuple(
            dict.fromkeys(label.strip() for label in schema if label and label.strip())
        )
        if not normalized_schema:
            return None

        if self._uie_taskflow is None:
            self._uie_taskflow = self._build_uie_taskflow(list(normalized_schema))
            self._uie_schema = normalized_schema if self._uie_taskflow is not None else ()
            return self._uie_taskflow

        if self._uie_schema != normalized_schema:
            try:
                self._uie_taskflow.set_schema(list(normalized_schema))
                self._uie_schema = normalized_schema
            except Exception as exc:
                if self.strict_uie_model:
                    raise RuntimeError("Failed to update UIE schema.") from exc
                logger.warning("Failed to update UIE schema, skip UIE extraction: %s", exc)
                return None

        return self._uie_taskflow

    def _build_uie_taskflow(self, schema: list[str]):
        """按配置创建 Taskflow information_extraction UIE 推理实例。"""
        try:
            from paddlenlp import Taskflow
        except ImportError as exc:
            if self.strict_uie_model:
                raise RuntimeError(
                    "paddlenlp is required for UIE information extraction."
                ) from exc
            logger.warning("paddlenlp is not available, skip UIE extraction: %s", exc)
            return None

        if self._has_local_model_files(self.uie_model_path):
            local_taskflow = self._init_local_uie_taskflow(
                Taskflow,
                schema=schema,
                raise_on_error=self.strict_uie_model,
            )
            if local_taskflow is not None:
                return local_taskflow

        if self.auto_download_model:
            downloaded_taskflow = self._init_default_uie_taskflow(
                Taskflow,
                schema=schema,
                raise_on_error=self.strict_uie_model,
            )
            if downloaded_taskflow is not None:
                self._sync_downloaded_uie_model_to_local()

                if self._has_local_model_files(self.uie_model_path):
                    local_taskflow = self._init_local_uie_taskflow(
                        Taskflow,
                        schema=schema,
                        raise_on_error=False,
                    )
                    if local_taskflow is not None:
                        return local_taskflow
                return downloaded_taskflow

        if self.strict_uie_model:
            raise FileNotFoundError(
                "Local UIE model is unavailable and auto-download failed/disabled. "
                f"uie_model_path={self.uie_model_path}, "
                f"uie_model_name={self.uie_model_name}, "
                f"auto_download_model={self.auto_download_model}."
            )

        logger.warning(
            "UIE model is unavailable; skip custom UIE extraction. "
            "uie_model_path=%s uie_model_name=%s auto_download_model=%s",
            self.uie_model_path,
            self.uie_model_name,
            self.auto_download_model,
        )
        return None

    def _init_local_uie_taskflow(self, taskflow_cls, schema: list[str], raise_on_error: bool):
        """从本地目录初始化 Taskflow information_extraction UIE。"""
        try:
            self.uie_model_path.mkdir(parents=True, exist_ok=True)
            return self._create_uie_taskflow(
                taskflow_cls,
                schema=schema,
                task_path=self.uie_model_path.as_posix(),
            )
        except Exception as exc:
            if raise_on_error:
                raise RuntimeError(
                    "Failed to initialize local Taskflow UIE model at "
                    f"{self.uie_model_path}."
                ) from exc
            logger.warning("Failed to initialize local UIE, fallback to default: %s", exc)
            return None

    def _init_default_uie_taskflow(self, taskflow_cls, schema: list[str], raise_on_error: bool):
        """初始化默认 Taskflow information_extraction UIE（可能触发模型下载）。"""
        try:
            return self._create_uie_taskflow(taskflow_cls, schema=schema)
        except Exception as exc:
            if raise_on_error:
                raise RuntimeError(
                    "Failed to initialize default Taskflow UIE auto-download."
                ) from exc
            logger.warning("Failed to auto-download default UIE model: %s", exc)
            return None

    def _create_uie_taskflow(self, taskflow_cls, schema: list[str], **kwargs):
        """创建 PaddleNLP UIE 信息抽取器。"""
        return taskflow_cls(
            self.UIE_TASKFLOW_TASK,
            schema=schema,
            model=self.uie_model_name,
            device_id=self.taskflow_device_id,
            position_prob=self.uie_position_prob,
            **kwargs,
        )

    def _sync_downloaded_model_to_local(self) -> None:
        """把自动下载的模型同步到项目本地模型目录。"""
        if not self.sync_downloaded_model:
            # 多容器部署时默认使用各容器自己的 PaddleNLP 缓存，避免多个容器同时写共享模型目录。
            return

        source_dir = self.downloaded_model_cache_path
        self._sync_model_directory(source_dir, self.model_path)

    def _sync_downloaded_uie_model_to_local(self) -> None:
        """把自动下载的 UIE 模型同步到项目本地模型目录。"""
        if not self.sync_downloaded_model:
            # 多容器部署时默认使用各容器自己的 PaddleNLP 缓存，避免多个容器同时写共享模型目录。
            return

        source_dir = self._find_downloaded_uie_model_cache_path()
        self._sync_model_directory(source_dir, self.uie_model_path)

    def _find_downloaded_uie_model_cache_path(self) -> Path:
        """兼容 PaddleNLP 不同版本可能使用的 UIE 缓存目录名。"""
        taskflow_root = Path.home() / ".paddlenlp" / "taskflow"
        candidates = [
            self.downloaded_uie_model_cache_path,
            taskflow_root / "information_extraction" / self.uie_model_name,
            taskflow_root / f"information_extraction-{self.uie_model_name}",
            taskflow_root / self.uie_model_name,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return candidates[0]

    def _sync_model_directory(self, source_dir: Path, target_dir: Path) -> None:
        """把 PaddleNLP 缓存目录同步到项目模型目录。"""
        if not source_dir.exists() or not source_dir.is_dir():
            logger.warning("Auto-downloaded model cache path not found: %s", source_dir)
            return

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            for item in source_dir.iterdir():
                target = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                elif item.is_file():
                    shutil.copy2(item, target)
            logger.info(
                "Synced auto-downloaded model from %s to %s",
                source_dir,
                target_dir,
            )
        except Exception as exc:
            logger.warning(
                "Failed to sync auto-downloaded model from %s to %s: %s",
                source_dir,
                target_dir,
                exc,
            )

    def infer(self, text: str, tasks: dict[str, Any] | None = None) -> list[ModelSpan]:
        """按任务执行纯模型推理，返回原始模型标签。"""
        if not text:
            return []

        normalized_tasks = tasks if isinstance(tasks, dict) else {}
        run_wordtag = normalized_tasks.get("wordtag", True) is not False
        uie_schema = self._normalize_uie_schema(normalized_tasks.get("uie_schema"))

        spans: list[ModelSpan] = []
        # PaddleNLP Taskflow/predictor instances are process-local mutable objects.
        # Serializing the model path avoids intermittent failures when wordtag and UIE
        # enter the underlying runtime at the same time.
        lock = getattr(self, "_infer_lock", nullcontext())
        with lock:
            if run_wordtag:
                spans.extend(self.infer_wordtag(text))
            if uie_schema:
                spans.extend(self.infer_uie(text, uie_schema))

        return self._deduplicate_model_spans(spans, text)

    def infer_wordtag(self, text: str) -> list[ModelSpan]:
        """执行 wordtag 模型推理。"""
        if not text:
            return []
        if self._taskflow is None:
            return []

        spans: list[ModelSpan] = []
        for offset, chunk in self._iter_token_chunks(text, self._taskflow):
            # 模型在 chunk 内返回局部坐标，进入业务层前统一回填成全文坐标。
            for span in self._extract_with_taskflow(chunk):
                spans.append(self._offset_model_span(span, offset))
        return self._deduplicate_model_spans(spans, text)

    def warmup_uie(self, schema: list[str] | tuple[str, ...]) -> bool:
        """按给定 schema 预加载 UIE 模型。"""
        lock = getattr(self, "_uie_lock", nullcontext())
        with lock:
            return self._ensure_uie_taskflow(list(schema)) is not None

    def infer_uie(self, text: str, schema: list[str]) -> list[ModelSpan]:
        """按 schema 执行 UIE 模型推理。"""
        if not text:
            return []

        uie_schema = self._normalize_uie_schema(schema)
        if not uie_schema:
            return []

        return self._deduplicate_model_spans(
            self._extract_with_uie(text, uie_schema),
            text,
        )

    def _extract_with_taskflow(self, text: str) -> list[ModelSpan]:
        """使用 Taskflow NER wordtag 抽取模型标签片段。"""
        pairs = self._extract_taskflow_pairs(text)
        if not pairs:
            return []

        if any(self._split_bioes_tag(tag)[0] for _, tag in pairs):
            return self._extract_bioes_taskflow_spans(text, pairs)

        spans: list[ModelSpan] = []

        # Taskflow("ner") 的 accurate 模式内部使用 wordtag，通常返回 [(token, tag), ...]。
        for token, tag, start, end in self._iter_located_tokens(text, pairs):
            spans.append(ModelSpan(tag, token, start, end, self.WORDTAG_SOURCE))

        return spans

    def _extract_bioes_taskflow_spans(
        self,
        text: str,
        pairs: list[tuple[str, str]],
    ) -> list[ModelSpan]:
        """兼容 BIOES 标签序列输出。"""
        return self._extract_bioes_spans(text, pairs, self._make_wordtag_span)

    def _extract_with_uie(
        self,
        text: str,
        schema: list[str],
    ) -> list[ModelSpan]:
        """按 schema 从 UIE 输出中抽取模型标签片段。"""
        lock = getattr(self, "_uie_lock", nullcontext())
        with lock:
            # UIE Taskflow 的 schema 是实例级可变状态；set_schema 和推理必须作为一个临界区。
            # 否则多线程请求不同 uie_schema 时，后一个请求可能在前一个请求推理中途切换 schema。
            uie_taskflow = self._ensure_uie_taskflow(schema)
            if uie_taskflow is None:
                return []

            spans: list[ModelSpan] = []
            # UIE 复用吞吐优先的主切片器，再批量送入 Taskflow，减少模型调用开销。
            chunks = self._iter_uie_token_chunks(text, uie_taskflow, schema)
            for batch in self._iter_batches(chunks, self.UIE_BATCH_SIZE):
                try:
                    tagged_items = self._infer_uie_batch(
                        uie_taskflow,
                        [chunk for _, chunk in batch],
                    )
                except Exception as exc:
                    if self.strict_uie_model:
                        raise RuntimeError("Taskflow UIE inference failed.") from exc
                    logger.warning("Taskflow UIE inference failed, skip UIE extraction: %s", exc)
                    return []

                for (offset, chunk), tagged in zip(batch, tagged_items):
                    for span in self._iter_uie_spans(chunk, tagged):
                        spans.append(self._offset_model_span(span, offset))

        return spans

    def _iter_uie_token_chunks(
        self,
        text: str,
        taskflow: Any,
        prompts: list[str] | tuple[str, ...],
    ) -> Iterator[tuple[int, str]]:
        """UIE 复用主切片器，只额外扣除 schema prompt 的 token 预算。"""
        yield from self._iter_token_chunks(text, taskflow, prompts=prompts)

    def _iter_token_chunks(
        self,
        text: str,
        taskflow: Any,
        prompts: list[str] | tuple[str, ...] = (),
    ) -> Iterator[tuple[int, str]]:
        """按自然边界聚合文本，并用 Taskflow tokenizer 控制 token 预算。"""
        if not text:
            return

        tokenizer = self._taskflow_tokenizer(taskflow)
        budget = self._available_token_budget(taskflow, tokenizer, prompts)
        current_start = -1
        current_text = ""
        current_tokens = 0

        for unit_start, unit_text in self._iter_semantic_units(text):
            unit_tokens = self._token_count(unit_text, tokenizer)
            if unit_tokens > budget:
                # 单个语义单元已经超预算时，先提交已合并内容，再对该单元做 token window。
                if current_text:
                    yield current_start, current_text
                    current_start = -1
                    current_text = ""
                    current_tokens = 0
                yield from self._split_long_unit_by_tokens(
                    unit_start,
                    unit_text,
                    tokenizer,
                    budget,
                )
                continue

            if not current_text:
                current_start = unit_start
                current_text = unit_text
                current_tokens = unit_tokens
                continue

            if current_tokens + unit_tokens > budget:
                # 正常路径只在语义单元之间断开，避免把地址、证件号等实体切成半截。
                yield current_start, current_text
                current_start = unit_start
                current_text = unit_text
                current_tokens = unit_tokens
                continue

            current_text += unit_text
            current_tokens += unit_tokens

        if current_text:
            yield current_start, current_text

    @staticmethod
    def _iter_batches(
        items: Iterable[tuple[int, str]],
        batch_size: int,
    ) -> Iterator[list[tuple[int, str]]]:
        size = max(1, batch_size)
        batch: list[tuple[int, str]] = []
        for item in items:
            batch.append(item)
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _infer_uie_batch(self, uie_taskflow: Any, chunks: list[str]) -> list[Any]:
        """UIE 小 chunk 保准确率，再用 Taskflow 批量输入减少调用开销。"""
        if not chunks:
            return []
        if len(chunks) == 1:
            return [uie_taskflow(chunks[0])]

        try:
            tagged = uie_taskflow(chunks)
        except Exception:
            # 个别 Taskflow 版本或测试桩不支持 list 输入时，退回逐条调用，保持兼容。
            return [uie_taskflow(chunk) for chunk in chunks]

        if isinstance(tagged, list) and len(tagged) == len(chunks):
            return list(tagged)

        # 返回结构不符合 batch 预期时逐条重跑，避免把整批结果误套到每个 chunk 上。
        return [uie_taskflow(chunk) for chunk in chunks]

    def _iter_semantic_units(self, text: str) -> Iterator[tuple[int, str]]:
        """按自然标点拆分语义单元；只做切片，不做实体识别。"""
        start = 0
        index = 0
        while index < len(text):
            if text[index] not in self.SEMANTIC_BOUNDARY_CHARS:
                index += 1
                continue

            # 连续边界符作为同一个语义单元结尾处理，避免 "\r\n" 被拆成孤立换行片段。
            end = index + 1
            while end < len(text) and text[end] in self.SEMANTIC_BOUNDARY_CHARS:
                end += 1
            if end > start:
                yield start, text[start:end]
            start = end
            index = end
        if start < len(text):
            yield start, text[start:]

    def _split_long_unit_by_tokens(
        self,
        base_offset: int,
        text: str,
        tokenizer: Any,
        budget: int,
    ) -> Iterator[tuple[int, str]]:
        """对超长语义单元做 token window 切分，并避免切在数字/字母中间。"""
        start = 0
        while start < len(text):
            # 没有自然边界可用时，用二分减少 tokenizer 调用次数，找到预算内最长窗口。
            end = self._max_end_within_token_budget(text, start, tokenizer, budget)
            if end <= start:
                end = min(len(text), start + max(1, budget))

            safe_end = self._avoid_ascii_alnum_split(text, start, end)
            if safe_end > start:
                end = safe_end

            yield base_offset + start, text[start:end]
            start = end

    def _max_end_within_token_budget(
        self,
        text: str,
        start: int,
        tokenizer: Any,
        budget: int,
    ) -> int:
        low = start + 1
        high = len(text)
        best = low
        while low <= high:
            mid = (low + high) // 2
            token_count = self._token_count(text[start:mid], tokenizer)
            if token_count <= budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    @classmethod
    def _avoid_ascii_alnum_split(cls, text: str, start: int, end: int) -> int:
        if end <= start or end >= len(text):
            return end
        if not (cls._is_ascii_alnum(text[end - 1]) and cls._is_ascii_alnum(text[end])):
            return end

        # 银行卡、身份证等核心实体都是 ASCII 数字/字母串；能回退就不要从串中间切开。
        candidate = end
        while candidate > start and cls._is_ascii_alnum(text[candidate - 1]):
            candidate -= 1
        return candidate if candidate > start else end

    def _available_token_budget(
        self,
        taskflow: Any,
        tokenizer: Any,
        prompts: list[str] | tuple[str, ...],
    ) -> int:
        task_instance = getattr(taskflow, "task_instance", taskflow)
        max_model_tokens = self._max_model_tokens()
        max_seq_len = int(
            getattr(task_instance, "_max_seq_len", max_model_tokens)
            or max_model_tokens
        )
        max_seq_len = min(max_seq_len, max_model_tokens)

        if prompts:
            # UIE 会把 prompt 和文本拼进同一序列，文本预算要扣掉最长 prompt 的 token。
            prompt_tokens = max(self._token_count(prompt, tokenizer) for prompt in prompts)
            overhead = prompt_tokens + int(
                getattr(task_instance, "_summary_token_num", self.UIE_SPECIAL_TOKENS)
                or self.UIE_SPECIAL_TOKENS
            )
        else:
            # wordtag 没有业务 prompt，只保守扣掉 Taskflow 汇总/特殊 token 开销。
            overhead = int(
                getattr(task_instance, "summary_num", self.WORDTAG_SPECIAL_TOKENS - 1)
                or 0
            ) + 1

        return max(1, max_seq_len - overhead)

    def _max_model_tokens(self) -> int:
        return self._normalize_positive_int(
            getattr(self, "max_model_tokens", self.DEFAULT_MAX_MODEL_TOKENS),
            "max_model_tokens",
        )

    @staticmethod
    def _taskflow_tokenizer(taskflow: Any) -> Any:
        task_instance = getattr(taskflow, "task_instance", taskflow)
        return getattr(task_instance, "_tokenizer", None)

    def _token_count(self, text: str, tokenizer: Any) -> int:
        if not text:
            return 0
        if tokenizer is not None:
            for kwargs in (
                {"text": text, "add_special_tokens": False},
                {"text": text},
            ):
                try:
                    encoded = tokenizer(**kwargs)
                    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
                    if input_ids is not None:
                        return len(input_ids)
                except Exception:
                    continue
        # 测试桩或异常 tokenizer 场景下回退到字符长度，保证切片器仍可工作。
        return len(text)

    @staticmethod
    def _offset_model_span(span: ModelSpan, offset: int) -> ModelSpan:
        if offset == 0:
            return span
        return ModelSpan(
            label=span.label,
            text=span.text,
            start=span.start + offset,
            end=span.end + offset,
            source=span.source,
            probability=span.probability,
        )

    @staticmethod
    def _is_ascii_alnum(char: str) -> bool:
        return bool(char) and char.isascii() and char.isalnum()

    def _iter_uie_spans(
        self,
        text: str,
        tagged: Any,
    ) -> Iterator[ModelSpan]:
        """把 UIE 输出结构转换为 ModelSpan。"""
        cursor_by_text: dict[str, int] = {}

        for label, item in self._iter_uie_items(tagged):
            token = item.get("text") if isinstance(item, dict) else None
            if token is None:
                continue
            token = str(token)

            start, end = self._get_uie_item_span(text, item, token, cursor_by_text)
            if start < 0 or end <= start:
                continue
            probability = item.get("probability")
            if not isinstance(probability, (int, float)):
                probability = None
            yield ModelSpan(str(label), token, start, end, self.UIE_SOURCE, probability)

    @classmethod
    def _iter_uie_items(cls, tagged: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        """遍历 UIE 顶层实体和 relations 中的实体。"""
        if isinstance(tagged, list):
            for item in tagged:
                yield from cls._iter_uie_items(item)
            return

        if not isinstance(tagged, dict):
            return

        for label, value in tagged.items():
            if label == "relations":
                if isinstance(value, dict):
                    yield from cls._iter_uie_items(value)
                continue

            if not isinstance(value, list):
                continue

            for item in value:
                if not isinstance(item, dict):
                    continue
                yield str(label), item
                relations = item.get("relations")
                if isinstance(relations, dict):
                    yield from cls._iter_uie_items(relations)

    def _get_uie_item_span(
        self,
        text: str,
        item: dict[str, Any],
        token: str,
        cursor_by_text: dict[str, int],
    ) -> tuple[int, int]:
        """优先使用 UIE 返回的 start/end，缺失时退回文本定位。"""
        raw_start = item.get("start")
        raw_end = item.get("end")
        if isinstance(raw_start, int) and isinstance(raw_end, int):
            return raw_start, raw_end

        cursor = cursor_by_text.get(token, 0)
        start = self._find_token(text, token, cursor)
        if start < 0:
            return -1, -1
        end = start + len(token)
        cursor_by_text[token] = end
        return start, end

    def _extract_bioes_spans(
        self,
        text: str,
        pairs: list[tuple[str, str]],
        span_factory: Callable[[str, str, int, int], ModelSpan | None],
    ) -> list[ModelSpan]:
        """把 BIOES token/tag 序列还原为实体片段。"""
        spans: list[ModelSpan] = []
        cursor = 0
        active_label = ""
        active_start = -1
        active_text = ""

        def flush_active() -> None:
            nonlocal active_label, active_start, active_text
            if active_label and active_start >= 0 and active_text:
                end = active_start + len(active_text)
                span = span_factory(active_label, active_text, active_start, end)
                if span is not None:
                    spans.append(span)
            active_label = ""
            active_start = -1
            active_text = ""

        for token, raw_tag in pairs:
            if not token:
                continue

            start = self._find_token(text, token, cursor)
            if start < 0:
                flush_active()
                continue

            end = start + len(token)
            cursor = end
            prefix, label = self._split_bioes_tag(raw_tag)

            if not prefix or not label:
                flush_active()
                continue

            if prefix == "S":
                flush_active()
                span = span_factory(label, token, start, end)
                if span is not None:
                    spans.append(span)
            elif prefix == "B":
                flush_active()
                active_label = label
                active_start = start
                active_text = token
            elif prefix in {"I", "E"} and active_label == label:
                active_text += token
                if prefix == "E":
                    flush_active()
            else:
                flush_active()

        flush_active()
        return spans

    @staticmethod
    def _make_wordtag_span(
        tag: str,
        token: str,
        start: int,
        end: int,
    ) -> ModelSpan:
        return ModelSpan(tag, token, start, end, LocalEntityRecognizer.WORDTAG_SOURCE)

    @staticmethod
    def _normalize_uie_schema(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []

        schema: list[str] = []
        seen: set[str] = set()
        for item in value:
            label = str(item or "").strip()
            if not label or label in seen:
                continue
            seen.add(label)
            schema.append(label)
        return schema

    def _deduplicate_model_spans(
        self,
        spans: list[ModelSpan],
        text: str,
    ) -> list[ModelSpan]:
        unique: dict[tuple[str, int, int, str, str], ModelSpan] = {}
        for span in spans:
            if span.start < 0 or span.end > len(text) or span.start >= span.end:
                continue

            real_text = text[span.start : span.end]
            if not real_text.strip():
                continue

            normalized = ModelSpan(
                label=span.label,
                text=real_text,
                start=span.start,
                end=span.end,
                source=span.source,
                probability=span.probability,
            )
            key = (
                normalized.label,
                normalized.start,
                normalized.end,
                normalized.text,
                normalized.source,
            )
            unique[key] = normalized
        return list(unique.values())

    def _extract_taskflow_pairs(self, text: str) -> list[tuple[str, str]]:
        """执行 Taskflow 并归一化为 token/tag 二元组。"""
        if self._taskflow is None:
            return []

        lock = getattr(self, "_taskflow_lock", nullcontext())
        try:
            with lock:
                tagged = self._taskflow(text)
        except Exception as exc:
            if self.strict_local_model:
                raise RuntimeError("Taskflow wordtag inference failed.") from exc
            logger.warning("Taskflow inference failed, skip model extraction: %s", exc)
            return []

        return list(self._iter_taskflow_tokens(tagged))

    def _iter_located_tokens(
        self,
        text: str,
        pairs: list[tuple[str, str]],
    ) -> Iterator[tuple[str, str, int, int]]:
        """把 Taskflow token 顺序还原到原文位置。"""
        cursor = 0
        for token, tag in pairs:
            if not token:
                continue

            start = self._find_token(text, token, cursor)
            if start < 0:
                continue

            end = start + len(token)
            cursor = end
            if end <= len(text):
                yield token, tag, start, end

    @staticmethod
    def _find_token(text: str, token: str, cursor: int) -> int:
        """优先从游标后查找 token；找不到时回退到全文查找。

        Taskflow 返回的是 token 序列，服务需要还原到原文起止位置。
        游标能处理重复词的大多数情况，全文回退用于兼容个别模型切词偏移。
        """
        start = text.find(token, cursor)
        if start >= 0:
            return start
        return text.find(token)

    @staticmethod
    def _split_bioes_tag(tag: str) -> tuple[str, str]:
        if len(tag) > 2 and tag[1] == "-" and tag[0] in {"B", "I", "E", "S"}:
            return tag[0], tag[2:]
        return "", tag

    @staticmethod
    def _iter_taskflow_tokens(tagged) -> Iterator[tuple[str, str]]:
        """兼容 PaddleNLP 不同版本的 wordtag 输出结构。"""
        if not isinstance(tagged, list):
            return

        if len(tagged) == 1 and isinstance(tagged[0], list):
            tagged = tagged[0]

        for item in tagged:
            if isinstance(item, dict):
                token = (
                    item.get("item")
                    or item.get("text")
                    or item.get("word")
                    or item.get("token")
                )
                tag = (
                    item.get("wordtag_label")
                    or item.get("tag")
                    or item.get("label")
                    or item.get("type")
                )
                if token is not None and tag is not None:
                    yield str(token), str(tag)
                continue

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                yield str(item[0]), str(item[1])
