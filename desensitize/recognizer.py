"""内置实体识别器。

组合三路识别能力：
1) PaddleNLP Taskflow NER 的 wordtag 模型（优先）
2) jieba 词性识别
3) 手机号正则抽取
"""

import logging
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jieba
from jieba import posseg

from .mapping import normalize_entity_type
from .types import EntitySpan


logger = logging.getLogger(__name__)


class LocalEntityRecognizer:
    # PaddleNLP 2.8.x 没有 Taskflow("wordtag") 这个顶层任务名。
    # 正确入口是 Taskflow("ner")，其 accurate 模式内部使用 wordtag 模型。
    TASKFLOW_TASK = "ner"
    TASKFLOW_ENTITY_SOURCE = "model"

    MOBILE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
    CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
    CUSTOM_ENTITY_LABEL_ALIASES = {
        "PERSON": {"person", "姓名", "人名", "人物", "姓氏"},
        "ORG": {"org", "organization", "组织", "机构", "公司", "企业", "品牌"},
        "MOBILE": {"mobile", "phone", "手机号", "手机号码", "电话"},
    }

    PERSON_BAD_CASE = {
        "谢谢",
        "人名",
        "姓名",
        "联系人",
        "今天上午",
        "今天下午",
    }
    ORG_BAD_CASE = {"集团", "各部门"}

    def __init__(
        self,
        model_path: Path,
        dict_dir: Path,
        device_id: int = 0,
        max_text_len: int = 10000,
        enable_taskflow: bool = True,
        strict_local_model: bool = True,
        auto_download_model: bool = True,
        sync_downloaded_model: bool = True,
        downloaded_model_cache_path: Path | None = None,
    ) -> None:
        # 初始化识别器配置。
        self.model_path = Path(model_path)
        self.dict_dir = Path(dict_dir)
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

        self._load_jieba_dicts()
        self._taskflow = self._build_taskflow()

    @property
    def using_taskflow(self) -> bool:
        """当前请求识别链路是否已启用 Taskflow wordtag。"""
        return self._taskflow is not None

    def _load_jieba_dicts(self) -> None:
        """加载本地词典，提升 jieba 对业务词的人名/机构识别效果。"""
        name_dict = self.dict_dir / "name.txt"
        user_dict = self.dict_dir / "user.txt"

        if name_dict.exists():
            with name_dict.open(encoding="utf-8") as file:
                jieba.load_userdict(file)
        if user_dict.exists():
            with user_dict.open(encoding="utf-8") as file:
                jieba.load_userdict(file)

    def _has_local_model_files(self) -> bool:
        """判断本地模型目录是否存在可用模型文件。"""
        if not self.model_path.exists() or not self.model_path.is_dir():
            return False

        for item in self.model_path.rglob("*"):
            if item.is_file() and item.name.lower() not in {"readme.md", ".gitkeep"}:
                return True
        return False

    def _build_taskflow(self):
        """按配置创建 Taskflow 推理实例，不可用时自动降级。"""
        if not self.enable_taskflow:
            logger.info("Taskflow wordtag is disabled; fallback to jieba+regex only")
            return None

        try:
            from paddlenlp import Taskflow
        except ImportError as exc:
            if self.strict_local_model:
                raise RuntimeError(
                    "paddlenlp is required. Please install dependencies from requirements.txt"
                ) from exc
            logger.warning("paddlenlp is not available, fallback to jieba+regex only: %s", exc)
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
            "Local model is unavailable; fallback to jieba+regex only. "
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

    def _sync_downloaded_model_to_local(self) -> None:
        """把自动下载的模型同步到项目本地模型目录。"""
        if not self.sync_downloaded_model:
            return

        source_dir = self.downloaded_model_cache_path
        if not source_dir.exists() or not source_dir.is_dir():
            logger.warning("Auto-downloaded model cache path not found: %s", source_dir)
            return

        try:
            self.model_path.mkdir(parents=True, exist_ok=True)
            for item in source_dir.iterdir():
                target = self.model_path / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                elif item.is_file():
                    shutil.copy2(item, target)
            logger.info(
                "Synced auto-downloaded model from %s to %s",
                source_dir,
                self.model_path,
            )
        except Exception as exc:
            logger.warning(
                "Failed to sync auto-downloaded model from %s to %s: %s",
                source_dir,
                self.model_path,
                exc,
            )

    def recognize_builtin(self, text: str) -> list[EntitySpan]:
        """执行内置实体识别并输出去重结果。"""
        if not text or len(text) > self.max_text_len:
            return []

        # 三路抽取并合并：Taskflow wordtag 为主，jieba 和手机号正则补漏。
        spans = [
            *self._extract_with_taskflow(text),
            *self._extract_with_jieba(text),
            *self._extract_mobile(text),
        ]
        return self._deduplicate_spans(spans, text)

    def recognize_custom(
        self,
        text: str,
        custom_entities: list[Any],
    ) -> list[EntitySpan]:
        """使用本地 Taskflow 模型识别业务方声明的自定义实体。

        这里不会执行业务方传入的 regex/pattern/value 字符串匹配；自定义实体只通过
        本地模型返回的标签命中。`model_labels`、`labels` 或 `schema` 可用于声明要
        匹配的模型标签；未提供时会根据 `entity_type` 使用内置别名兜底。
        """
        if not text or len(text) > self.max_text_len:
            return []
        if not isinstance(custom_entities, list):
            return []

        rules = self._normalize_custom_model_rules(custom_entities)
        if not rules or self._taskflow is None:
            return []

        spans = self._extract_custom_with_taskflow(text, rules)
        return self._deduplicate_spans(spans, text)

    def _extract_with_taskflow(self, text: str) -> list[EntitySpan]:
        """使用 Taskflow NER wordtag 抽取人名/机构。"""
        if self._taskflow is None:
            return []

        try:
            tagged = self._taskflow(text)
        except Exception as exc:
            if self.strict_local_model:
                raise RuntimeError("Taskflow wordtag inference failed.") from exc
            logger.warning("Taskflow inference failed, skip model extraction: %s", exc)
            return []

        pairs = list(self._iter_taskflow_tokens(tagged))
        if any(self._split_bioes_tag(tag)[0] for _, tag in pairs):
            return self._extract_bioes_taskflow_spans(text, pairs)

        spans: list[EntitySpan] = []
        cursor = 0

        # Taskflow("ner") 的 accurate 模式内部使用 wordtag，通常返回 [(token, tag), ...]。
        for token, tag in pairs:
            if not token:
                continue

            start = self._find_token(text, token, cursor)
            if start < 0:
                continue
            end = start + len(token)
            cursor = end

            if end > len(text):
                continue

            self._append_model_span(spans, tag, token, start, end)

        return spans

    def _extract_bioes_taskflow_spans(
        self,
        text: str,
        pairs: list[tuple[str, str]],
    ) -> list[EntitySpan]:
        """兼容 BIOES 标签序列输出。"""
        spans: list[EntitySpan] = []
        cursor = 0
        active_label = ""
        active_start = -1
        active_text = ""

        def flush_active() -> None:
            nonlocal active_label, active_start, active_text
            if active_label and active_start >= 0 and active_text:
                end = active_start + len(active_text)
                self._append_model_span(spans, active_label, active_text, active_start, end)
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
                self._append_model_span(spans, label, token, start, end)
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
                if prefix == "B":
                    active_label = label
                    active_start = start
                    active_text = token

        flush_active()
        return spans

    def _extract_custom_with_taskflow(
        self,
        text: str,
        rules: list[tuple[str, set[str]]],
    ) -> list[EntitySpan]:
        """按自定义标签规则从 Taskflow 输出中抽取实体。"""
        try:
            tagged = self._taskflow(text)
        except Exception as exc:
            if self.strict_local_model:
                raise RuntimeError("Taskflow wordtag inference failed.") from exc
            logger.warning("Taskflow inference failed, skip custom extraction: %s", exc)
            return []

        pairs = list(self._iter_taskflow_tokens(tagged))
        if any(self._split_bioes_tag(tag)[0] for _, tag in pairs):
            return self._extract_bioes_custom_taskflow_spans(text, pairs, rules)

        spans: list[EntitySpan] = []
        cursor = 0

        for token, tag in pairs:
            if not token:
                continue

            start = self._find_token(text, token, cursor)
            if start < 0:
                continue
            end = start + len(token)
            cursor = end

            entity_type = self._match_custom_entity_type(tag, rules)
            if (
                entity_type
                and self._looks_valid_entity(entity_type, token)
                and end <= len(text)
            ):
                spans.append(EntitySpan(entity_type, token, start, end, "custom"))

        return spans

    def _extract_bioes_custom_taskflow_spans(
        self,
        text: str,
        pairs: list[tuple[str, str]],
        rules: list[tuple[str, set[str]]],
    ) -> list[EntitySpan]:
        """兼容 BIOES 标签序列的自定义实体抽取。"""
        spans: list[EntitySpan] = []
        cursor = 0
        active_label = ""
        active_start = -1
        active_text = ""

        def flush_active() -> None:
            nonlocal active_label, active_start, active_text
            entity_type = self._match_custom_entity_type(active_label, rules)
            if (
                entity_type
                and active_start >= 0
                and active_text
                and self._looks_valid_entity(entity_type, active_text)
            ):
                end = active_start + len(active_text)
                spans.append(EntitySpan(entity_type, active_text, active_start, end, "custom"))
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
                entity_type = self._match_custom_entity_type(label, rules)
                if entity_type and self._looks_valid_entity(entity_type, token):
                    spans.append(EntitySpan(entity_type, token, start, end, "custom"))
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
                if prefix == "B":
                    active_label = label
                    active_start = start
                    active_text = token

        flush_active()
        return spans

    def _normalize_custom_model_rules(
        self,
        custom_entities: list[Any],
    ) -> list[tuple[str, set[str]]]:
        """把 custom_entities 转成模型标签匹配规则。"""
        rules: list[tuple[str, set[str]]] = []
        for rule in custom_entities:
            if not isinstance(rule, dict):
                continue

            raw_entity_type = rule.get("entity_type", rule.get("type", "CUSTOM"))
            entity_type = normalize_entity_type(raw_entity_type)
            labels = {
                self._normalize_label(label)
                for label in self._iter_label_values(
                    rule.get("model_labels")
                    or rule.get("labels")
                    or rule.get("label")
                    or rule.get("schema")
                )
            }
            labels = {label for label in labels if label}
            if not labels:
                labels.add(self._normalize_label(raw_entity_type))

            labels.update(self.CUSTOM_ENTITY_LABEL_ALIASES.get(entity_type, set()))
            labels = {label for label in labels if label}
            if labels:
                rules.append((entity_type, labels))
        return rules

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
        if self._is_org_tag(tag) and self._looks_valid_org(token):
            spans.append(EntitySpan("ORG", token, start, end, self.TASKFLOW_ENTITY_SOURCE))
        elif self._is_person_tag(tag) and self._looks_valid_person(token):
            spans.append(EntitySpan("PERSON", token, start, end, self.TASKFLOW_ENTITY_SOURCE))

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

    def _extract_with_jieba(self, text: str) -> list[EntitySpan]:
        """使用 jieba 词性标注抽取人名/机构。"""
        spans: list[EntitySpan] = []
        cursor = 0

        for seg in posseg.lcut(text):
            raw_word = seg.word
            flag = seg.flag

            start = cursor
            end = cursor + len(raw_word)
            cursor = end

            if end > len(text):
                break

            if flag in {"nr", "nrt"} and self._looks_valid_person(raw_word):
                spans.append(EntitySpan("PERSON", raw_word, start, end, "jieba"))
            elif flag in {"nt"} and self._looks_valid_org(raw_word):
                spans.append(EntitySpan("ORG", raw_word, start, end, "jieba"))

        return spans

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
        """按实体类型复用内置候选过滤，避免自定义模型规则放大噪声。"""
        normalized_type = normalize_entity_type(entity_type)
        if normalized_type == "PERSON":
            return self._looks_valid_person(token)
        if normalized_type == "ORG":
            return self._looks_valid_org(token)
        return self._looks_valid_custom(token)

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
