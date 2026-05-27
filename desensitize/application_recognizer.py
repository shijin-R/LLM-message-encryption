"""应用层实体识别编排。

模型层只负责返回原始模型标签；本模块负责把模型标签、正则补漏和业务
custom_entities 规则转换为脱敏服务需要的 EntitySpan。
"""

import re
from collections.abc import Iterator
from typing import Any

from .mapping import normalize_entity_type
from .types import EntitySpan, ModelSpan


class ApplicationEntityRecognizer:
    """把纯模型推理结果编排成脱敏业务实体。"""

    DEFAULT_MAX_TEXT_LEN = 512
    CHUNK_OVERLAP = 128
    CHUNK_BOUNDARY_CHARS = "\r\n。！？!?；;，,、：: \t"

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

    def __init__(self, model_client: Any, max_text_len: int = DEFAULT_MAX_TEXT_LEN):
        self.model_client = model_client
        self.max_text_len = max(1, int(max_text_len))

    @property
    def using_taskflow(self) -> bool:
        return bool(getattr(self.model_client, "using_taskflow", False))

    @property
    def using_uie(self) -> bool:
        return bool(getattr(self.model_client, "using_uie", False))

    def health(self) -> dict[str, Any]:
        health = getattr(self.model_client, "health", None)
        if callable(health):
            return dict(health())
        return {
            "using_taskflow": self.using_taskflow,
            "using_uie": self.using_uie,
        }

    def ready(self) -> dict[str, Any]:
        ready = getattr(self.model_client, "ready", None)
        if callable(ready):
            return dict(ready())
        return {"using_taskflow": self.using_taskflow, "using_uie": self.using_uie}

    def recognize(self, text: str, custom_entities: list[Any]) -> list[EntitySpan]:
        """一次性执行模型推理和应用层补漏，减少远程模型服务往返。"""
        if not text:
            return []

        uie_rules, uie_schema = self._parse_custom_uie_rules(custom_entities)
        spans: list[EntitySpan] = []
        for model_span in self._infer_chunks(text, wordtag=True, uie_schema=uie_schema):
            entity_span = self._model_span_to_entity(text, model_span, uie_rules)
            if entity_span is not None:
                spans.append(entity_span)

        spans.extend(self._extract_mobile(text))
        spans = self._merge_adjacent_custom_spans(spans, text)
        return self._deduplicate_spans(spans, text)

    def _infer_chunks(
        self,
        text: str,
        wordtag: bool,
        uie_schema: list[str],
    ) -> Iterator[ModelSpan]:
        # 这里只切分“识别输入”，不拆业务消息；返回 span 时统一回填到全文坐标。
        for offset, chunk in self._iter_text_chunks(text):
            spans = self.model_client.infer(
                chunk,
                {
                    "wordtag": wordtag,
                    "uie_schema": uie_schema,
                },
            )
            for span in spans:
                if span.start < 0 or span.end > len(chunk) or span.start >= span.end:
                    continue
                yield ModelSpan(
                    label=span.label,
                    text=span.text,
                    start=span.start + offset,
                    end=span.end + offset,
                    source=span.source,
                    probability=span.probability,
                )

    def _model_span_to_entity(
        self,
        text: str,
        span: ModelSpan,
        uie_rules: list[tuple[str, set[str]]],
    ) -> EntitySpan | None:
        if span.source == "uie":
            return self._uie_span_to_entity(text, span, uie_rules)
        return self._wordtag_span_to_entity(text, span)

    def _wordtag_span_to_entity(
        self,
        text: str,
        span: ModelSpan,
    ) -> EntitySpan | None:
        real_text = text[span.start : span.end]
        if self._is_org_tag(span.label) and self._looks_valid_org(real_text):
            return EntitySpan("ORG", real_text, span.start, span.end, "model")
        if self._is_person_tag(span.label) and self._looks_valid_person(real_text):
            return EntitySpan("PERSON", real_text, span.start, span.end, "model")
        return None

    def _uie_span_to_entity(
        self,
        text: str,
        span: ModelSpan,
        rules: list[tuple[str, set[str]]],
    ) -> EntitySpan | None:
        entity_type = self._match_uie_label_entity_type(span.label, rules)
        if not entity_type:
            return None

        real_text = text[span.start : span.end]
        if not self._looks_valid_entity(entity_type, real_text):
            return None
        return EntitySpan(entity_type, real_text, span.start, span.end, "custom")

    def _iter_text_chunks(self, text: str) -> Iterator[tuple[int, str]]:
        """按配置长度切分文本，分片之间保留重叠区避免边界漏识别。"""
        if len(text) <= self.max_text_len:
            yield 0, text
            return

        start = 0
        text_len = len(text)
        while start < text_len:
            hard_end = min(start + self.max_text_len, text_len)
            end = self._find_chunk_end(text, start, hard_end)
            if end <= start:
                end = hard_end

            yield start, text[start:end]

            if end >= text_len:
                break

            overlap = min(self.CHUNK_OVERLAP, self.max_text_len // 2, end - start - 1)
            next_start = end - max(0, overlap)
            if next_start <= start:
                next_start = end
            start = next_start

    def _find_chunk_end(self, text: str, start: int, hard_end: int) -> int:
        """优先在靠近分片末尾的自然边界断开，找不到时使用硬切点。"""
        if hard_end >= len(text):
            return hard_end

        search_window = min(self.CHUNK_OVERLAP, (hard_end - start) // 2)
        search_start = max(start + 1, hard_end - search_window)
        for index in range(hard_end - 1, search_start - 1, -1):
            if text[index] in self.CHUNK_BOUNDARY_CHARS:
                return index + 1
        return hard_end

    def _extract_mobile(self, text: str) -> list[EntitySpan]:
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

    def _parse_custom_uie_rules(
        self,
        custom_entities: list[Any],
    ) -> tuple[list[tuple[str, set[str]]], list[str]]:
        """解析 custom_entities，返回业务映射规则和需要下发给模型层的 UIE schema。"""
        if not isinstance(custom_entities, list):
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
                self._iter_schema_labels(
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
    def _match_uie_label_entity_type(
        cls,
        tag: str,
        rules: list[tuple[str, set[str]]],
    ) -> str:
        """把 UIE label 映射成业务实体类型；先精确匹配，再用最长 label 模糊匹配。"""
        normalized_tag = cls._normalize_label(tag)
        if not normalized_tag:
            return ""

        for entity_type, labels in rules:
            if normalized_tag in labels:
                return entity_type

        best_type = ""
        best_label_len = -1
        for entity_type, labels in rules:
            for label in labels:
                if label in normalized_tag or normalized_tag in label:
                    # 同样长度时保留 custom_entities 中更靠前的规则，避免结果被字典序影响。
                    if len(label) > best_label_len:
                        best_type = entity_type
                        best_label_len = len(label)
        return best_type

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
        text = token.strip()
        if len(text) <= 1:
            return False
        if text in self.ORG_BAD_CASE:
            return False
        if not re.search(r"[\u4e00-\u9fffA-Za-z]", text):
            return False
        return True

    def _looks_valid_entity(self, entity_type: str, token: str) -> bool:
        normalized_type = normalize_entity_type(entity_type)
        if normalized_type == "PERSON":
            return self._looks_valid_person(token)
        if normalized_type == "ORG":
            return self._looks_valid_org(token)
        if normalized_type == "ADDRESS":
            return self._looks_valid_address(token)
        return self._looks_valid_custom(token)

    def _looks_valid_address(self, token: str) -> bool:
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
        text = token.strip()
        if not text:
            return False
        return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text))

    @staticmethod
    def _deduplicate_spans(spans: list[EntitySpan], text: str) -> list[EntitySpan]:
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

    @staticmethod
    def _normalize_label(label: Any) -> str:
        normalized = str(label or "").strip().lower()
        return re.sub(r"\s+", "", normalized)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """把 schema 的多种写法统一成列表，便于后续逐项抽取 label。"""
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
    def _iter_schema_labels(cls, value: Any) -> Iterator[Any]:
        """从字符串、字典或列表形式的 uie_schema 中逐个取出 label。"""
        for item in cls._as_list(value):
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
