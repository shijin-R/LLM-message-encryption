"""内置实体识别器。

组合多路识别能力：
1) PaddleNLP Taskflow NER 的 wordtag 模型（优先）
2) 可选 PaddleNLP Taskflow information_extraction 的 UIE 模型旁路识别自定义实体
3) 手机号正则抽取
"""

import logging
import re
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .mapping import normalize_entity_type
from .types import EntitySpan


logger = logging.getLogger(__name__)


class LocalEntityRecognizer:
    # PaddleNLP 2.8.x 没有 Taskflow("wordtag") 这个顶层任务名。
    # 正确入口是 Taskflow("ner")，其 accurate 模式内部使用 wordtag 模型。
    TASKFLOW_TASK = "ner"
    UIE_TASKFLOW_TASK = "information_extraction"
    TASKFLOW_ENTITY_SOURCE = "model"

    MOBILE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
    CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
    PERSON_BAD_CASE = {
        "谢谢",
        "人名",
        "姓名",
        "联系人",
        "今天上午",
        "今天下午",
    }
    ORG_BAD_CASE = {"集团", "各部门"}
    ADDRESS_BAD_CASE = {
        "地址",
        "住址",
        "现住址",
        "家庭住址",
        "联系地址",
        "通信地址",
        "通讯地址",
        "收货地址",
        "户籍地址",
    }

    def __init__(
        self,
        model_path: Path,
        device_id: int = 0,
        max_text_len: int = 10000,
        enable_taskflow: bool = True,
        strict_local_model: bool = True,
        auto_download_model: bool = True,
        sync_downloaded_model: bool = True,
        downloaded_model_cache_path: Path | None = None,
        enable_uie_custom: bool = True,
        uie_model_name: str = "uie-base",
        uie_model_path: Path | None = None,
        uie_position_prob: float = 0.5,
        strict_uie_model: bool = False,
        downloaded_uie_model_cache_path: Path | None = None,
    ) -> None:
        # 初始化识别器配置。
        self.model_path = Path(model_path)
        self.device_id = device_id
        self.max_text_len = max_text_len
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

        self._taskflow = self._build_taskflow()

    @property
    def using_taskflow(self) -> bool:
        """当前请求识别链路是否已启用 Taskflow wordtag。"""
        return self._taskflow is not None

    @property
    def using_uie(self) -> bool:
        """当前进程是否已懒加载 UIE 信息抽取旁路。"""
        return self._uie_taskflow is not None

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
            logger.info("Taskflow wordtag is disabled; fallback to mobile regex")
            return None

        try:
            from paddlenlp import Taskflow
        except ImportError as exc:
            if self.strict_local_model:
                raise RuntimeError(
                    "paddlenlp is required. Please install dependencies from requirements.txt"
                ) from exc
            logger.warning(
                "paddlenlp is not available, fallback to mobile regex: %s",
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
            "Local model is unavailable; fallback to mobile regex. "
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
            device_id=self.device_id,
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
            device_id=self.device_id,
            position_prob=self.uie_position_prob,
            **kwargs,
        )

    def _sync_downloaded_model_to_local(self) -> None:
        """把自动下载的模型同步到项目本地模型目录。"""
        if not self.sync_downloaded_model:
            return

        source_dir = self.downloaded_model_cache_path
        self._sync_model_directory(source_dir, self.model_path)

    def _sync_downloaded_uie_model_to_local(self) -> None:
        """把自动下载的 UIE 模型同步到项目本地模型目录。"""
        if not self.sync_downloaded_model:
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

    def recognize_builtin(self, text: str) -> list[EntitySpan]:
        """执行内置实体识别并输出去重结果。"""
        if not text or len(text) > self.max_text_len:
            return []

        # Taskflow wordtag 为主，手机号正则补漏。
        spans = [
            *self._extract_with_taskflow(text),
            *self._extract_mobile(text),
        ]
        return self._deduplicate_spans(spans, text)

    def recognize_custom(
        self,
        text: str,
        custom_entities: list[Any],
    ) -> list[EntitySpan]:
        """使用 UIE 信息抽取模型识别业务方声明的自定义实体。

        这里不会执行业务方传入的 regex/pattern/value 字符串匹配；自定义实体只通过
        UIE 模型返回结果命中。`uie_schema` 用于声明 UIE 信息抽取目标；
        未提供 UIE schema 时不会触发自定义实体识别。
        """
        if not text or len(text) > self.max_text_len:
            return []
        if not isinstance(custom_entities, list):
            return []

        uie_rules, uie_schema = self._normalize_custom_uie_rules(custom_entities)
        if not uie_rules:
            return []

        spans = self._extract_custom_with_uie(text, uie_rules, uie_schema)
        spans = self._merge_adjacent_custom_spans(spans, text)
        return self._deduplicate_spans(spans, text)

    def _extract_with_taskflow(self, text: str) -> list[EntitySpan]:
        """使用 Taskflow NER wordtag 抽取人名/机构。"""
        pairs = self._extract_taskflow_pairs(text)
        if not pairs:
            return []

        if any(self._split_bioes_tag(tag)[0] for _, tag in pairs):
            return self._extract_bioes_taskflow_spans(text, pairs)

        spans: list[EntitySpan] = []

        # Taskflow("ner") 的 accurate 模式内部使用 wordtag，通常返回 [(token, tag), ...]。
        for token, tag, start, end in self._iter_located_tokens(text, pairs):
            self._append_model_span(spans, tag, token, start, end)

        return spans

    def _extract_bioes_taskflow_spans(
        self,
        text: str,
        pairs: list[tuple[str, str]],
    ) -> list[EntitySpan]:
        """兼容 BIOES 标签序列输出。"""
        return self._extract_bioes_spans(text, pairs, self._make_model_span)

    def _extract_custom_with_uie(
        self,
        text: str,
        rules: list[tuple[str, set[str]]],
        schema: list[str],
    ) -> list[EntitySpan]:
        """按自定义 schema 从 UIE 输出中抽取实体。"""
        uie_taskflow = self._ensure_uie_taskflow(schema)
        if uie_taskflow is None:
            return []

        try:
            tagged = uie_taskflow(text)
        except Exception as exc:
            if self.strict_uie_model:
                raise RuntimeError("Taskflow UIE inference failed.") from exc
            logger.warning("Taskflow UIE inference failed, skip UIE extraction: %s", exc)
            return []

        return list(self._iter_uie_spans(text, tagged, rules))

    def _iter_uie_spans(
        self,
        text: str,
        tagged: Any,
        rules: list[tuple[str, set[str]]],
    ) -> Iterator[EntitySpan]:
        """把 UIE 输出结构转换为 EntitySpan。"""
        cursor_by_text: dict[str, int] = {}

        for label, item in self._iter_uie_items(tagged):
            entity_type = self._match_custom_entity_type(label, rules)
            if not entity_type:
                continue

            token = item.get("text") if isinstance(item, dict) else None
            if token is None:
                continue
            token = str(token)
            if not self._looks_valid_entity(entity_type, token):
                continue

            start, end = self._get_uie_item_span(text, item, token, cursor_by_text)
            if start < 0 or end <= start:
                continue
            yield EntitySpan(entity_type, token, start, end, "custom")

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
        span_factory: Callable[[str, str, int, int], EntitySpan | None],
    ) -> list[EntitySpan]:
        """把 BIOES token/tag 序列还原为实体片段。"""
        spans: list[EntitySpan] = []
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

    def _normalize_custom_uie_rules(
        self,
        custom_entities: list[Any],
    ) -> tuple[list[tuple[str, set[str]]], list[str]]:
        """把 custom_entities 转成 UIE schema 和输出标签匹配规则。"""
        if not getattr(self, "enable_uie_custom", False):
            return [], []

        rules: list[tuple[str, set[str]]] = []
        schema: list[str] = []
        seen_schema: set[str] = set()

        for rule in custom_entities:
            if not isinstance(rule, dict):
                continue

            raw_entity_type = rule.get("entity_type", rule.get("type", "CUSTOM"))
            entity_type = normalize_entity_type(raw_entity_type)
            raw_labels = list(
                self._iter_label_values(
                    rule.get("uie_schema")
                    or rule.get("uie_labels")
                    or rule.get("uie_label")
                )
            )
            if not raw_labels:
                continue

            labels = {self._normalize_label(label) for label in raw_labels}
            labels = {label for label in labels if label}
            if not labels:
                continue

            for label in raw_labels:
                clean_label = str(label or "").strip()
                if not clean_label or clean_label in seen_schema:
                    continue
                seen_schema.add(clean_label)
                schema.append(clean_label)

            rules.append((entity_type, labels))

        return rules, schema

    @classmethod
    def _match_custom_entity_type(
        cls,
        tag: str,
        rules: list[tuple[str, set[str]]],
    ) -> str:
        normalized_tag = cls._normalize_label(tag)
        if not normalized_tag:
            return ""

        for entity_type, labels in rules:
            if any(
                label == normalized_tag
                or label in normalized_tag
                or normalized_tag in label
                for label in labels
            ):
                return entity_type
        return ""

    def _append_model_span(
        self,
        spans: list[EntitySpan],
        tag: str,
        token: str,
        start: int,
        end: int,
    ) -> None:
        span = self._make_model_span(tag, token, start, end)
        if span is not None:
            spans.append(span)

    def _make_model_span(
        self,
        tag: str,
        token: str,
        start: int,
        end: int,
    ) -> EntitySpan | None:
        if self._is_org_tag(tag) and self._looks_valid_org(token):
            return EntitySpan("ORG", token, start, end, self.TASKFLOW_ENTITY_SOURCE)
        if self._is_person_tag(tag) and self._looks_valid_person(token):
            return EntitySpan("PERSON", token, start, end, self.TASKFLOW_ENTITY_SOURCE)
        return None

    def _extract_taskflow_pairs(self, text: str) -> list[tuple[str, str]]:
        """执行 Taskflow 并归一化为 token/tag 二元组。"""
        if self._taskflow is None:
            return []

        try:
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
    def _normalize_label(label: Any) -> str:
        """标准化模型标签，便于中英文别名做包含匹配。"""
        normalized = str(label or "").strip().lower()
        return re.sub(r"\s+", "", normalized)

    @staticmethod
    def _ensure_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if value is None:
            return []
        if isinstance(value, dict):
            label = (
                value.get("name")
                or value.get("label")
                or value.get("type")
                or value.get("entity_type")
            )
            return [label] if label is not None else []
        return [value]

    @classmethod
    def _iter_label_values(cls, value: Any) -> Iterator[Any]:
        for item in cls._ensure_list(value):
            if isinstance(item, dict):
                label = (
                    item.get("name")
                    or item.get("label")
                    or item.get("type")
                    or item.get("entity_type")
                )
                if label is not None:
                    yield label
                continue
            yield item

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

    def _extract_mobile(self, text: str) -> list[EntitySpan]:
        """使用正则抽取手机号。"""
        spans: list[EntitySpan] = []
        for match in self.MOBILE_PATTERN.finditer(text):
            spans.append(
                EntitySpan(
                    "MOBILE",
                    match.group(),
                    match.start(),
                    match.end(),
                    "regex",
                )
            )
        return spans

    @classmethod
    def _is_person_tag(cls, tag: str) -> bool:
        if "概念" in tag:
            return False
        if tag in {"人物类_实体", "文化类_姓氏与人名"}:
            return True
        return any(keyword in tag for keyword in ("人名", "人物", "姓氏"))

    @classmethod
    def _is_org_tag(cls, tag: str) -> bool:
        if "概念" in tag:
            return False
        if tag == "品牌名":
            return True
        return any(keyword in tag for keyword in ("组织", "机构", "公司", "企业", "品牌"))

    def _looks_valid_person(self, token: str) -> bool:
        """过滤明显噪声的人名候选。"""
        text = token.strip()
        if len(text) <= 1 or len(text) > 20:
            return False
        if text in self.PERSON_BAD_CASE:
            return False
        if not self.CHINESE_PATTERN.search(text):
            return False
        if re.search(r"\d", text):
            return False
        return True

    def _looks_valid_org(self, token: str) -> bool:
        """过滤明显噪声的机构候选。"""
        text = token.strip()
        if len(text) <= 1:
            return False
        if text in self.ORG_BAD_CASE:
            return False
        if not re.search(r"[\u4e00-\u9fffA-Za-z]", text):
            return False
        return True

    def _looks_valid_entity(self, entity_type: str, token: str) -> bool:
        """按实体类型复用内置候选过滤，避免自定义 UIE 候选放大噪声。"""
        normalized_type = normalize_entity_type(entity_type)
        if normalized_type == "PERSON":
            return self._looks_valid_person(token)
        if normalized_type == "ORG":
            return self._looks_valid_org(token)
        if normalized_type == "ADDRESS":
            return self._looks_valid_address(token)
        return self._looks_valid_custom(token)

    def _looks_valid_address(self, token: str) -> bool:
        """过滤字段名，保留更像真实地点/地址的候选。"""
        text = token.strip()
        if len(text) <= 1:
            return False
        if text in self.ADDRESS_BAD_CASE:
            return False
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
            return False
        if re.search(r"\d", text):
            return True
        return bool(re.search(r"[省市区县镇乡村街路巷道号楼室园场站口]", text))

    def _merge_adjacent_custom_spans(
        self,
        spans: list[EntitySpan],
        text: str,
    ) -> list[EntitySpan]:
        """合并模型切开的连续地址片段。"""
        if not spans:
            return []

        merged: list[EntitySpan] = []
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            if merged and self._should_merge_custom_span(merged[-1], span, text):
                previous = merged[-1]
                merged[-1] = EntitySpan(
                    entity_type=previous.entity_type,
                    text=text[previous.start : span.end],
                    start=previous.start,
                    end=span.end,
                    source=previous.source,
                )
                continue
            merged.append(span)
        return merged

    @staticmethod
    def _should_merge_custom_span(
        left: EntitySpan,
        right: EntitySpan,
        text: str,
    ) -> bool:
        if left.entity_type != "ADDRESS" or right.entity_type != "ADDRESS":
            return False
        if left.source != "custom" or right.source != "custom":
            return False
        return text[left.end : right.start] == ""

    @staticmethod
    def _looks_valid_custom(token: str) -> bool:
        """过滤空白和纯标点的自定义实体候选。"""
        text = token.strip()
        if not text:
            return False
        return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text))

    @staticmethod
    def _deduplicate_spans(spans: list[EntitySpan], text: str) -> list[EntitySpan]:
        """按实体类型 + 位置 + 文本去重。"""
        unique: dict[tuple[str, int, int, str], EntitySpan] = {}

        for span in spans:
            if span.start < 0 or span.end > len(text) or span.start >= span.end:
                continue

            real_text = text[span.start : span.end]
            if not real_text.strip():
                continue

            normalized_span = EntitySpan(
                entity_type=span.entity_type,
                text=real_text,
                start=span.start,
                end=span.end,
                source=span.source,
            )
            key = (
                normalized_span.entity_type,
                normalized_span.start,
                normalized_span.end,
                normalized_span.text,
            )
            unique[key] = normalized_span

        return list(unique.values())
