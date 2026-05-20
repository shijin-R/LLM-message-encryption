import logging
import re
from typing import Any

from .config import ServiceConfig
from .mapping import MappingStore, is_placeholder_token, normalize_entity_type
from .recognizer import LocalEntityRecognizer
from .types import EntitySpan


logger = logging.getLogger(__name__)


class DesensitizeService:
    # 实体冲突消解时的来源优先级：值越大优先级越高。
    SPAN_PRIORITY = {
        "history": 5,
        "custom": 4,
        "model": 3,
        "regex": 3,
        "jieba": 2,
    }

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.recognizer = LocalEntityRecognizer(
            model_path=config.model_path,
            dict_dir=config.dict_dir,
            device_id=config.device_id,
            max_text_len=config.max_text_len,
            enable_taskflow=config.enable_taskflow,
            strict_local_model=config.strict_local_model,
            auto_download_model=config.auto_download_model,
            sync_downloaded_model=config.sync_downloaded_model,
            downloaded_model_cache_path=config.downloaded_model_cache_path,
        )

    def desensitize(self, payload: dict[str, Any]) -> dict[str, Any]:
        """对输入的 messages 执行脱敏并返回映射信息。"""
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")

        normalized_payload = self._normalize_payload(payload)
        raw_messages = normalized_payload["messages"]
        self._validate_no_message_history_mappings(raw_messages)

        # 读取上下文：顶层历史映射 + 自定义规则 + 内置识别。
        mapping_store = MappingStore(self._get_history_mappings(normalized_payload))
        custom_entities = self._normalize_custom_entities(
            normalized_payload.get("custom_entities")
        )

        # 控制哪些消息需要处理，避免重复脱敏。
        desensitized_indexes = self._to_index_set(
            normalized_payload.get("desensitized_message_indexes", [])
        )
        target_indexes = None
        if "target_message_indexes" in normalized_payload:
            target_indexes = self._to_index_set(
                normalized_payload.get("target_message_indexes", [])
            )

        output_messages: list[Any] = []
        processed_indexes: list[int] = []
        total_replacements = 0

        for index, raw_message in enumerate(raw_messages):
            normalized_message, original_is_str = self._normalize_message(raw_message)
            content = normalized_message["content"]

            should_process = self._should_process_message(
                index=index,
                message=normalized_message,
                desensitized_indexes=desensitized_indexes,
                target_indexes=target_indexes,
            )

            if should_process and content:
                masked_content, replacement_count = self._mask_single_text(
                    text=content,
                    mapping_store=mapping_store,
                    custom_entities=custom_entities,
                )
                normalized_message["content"] = masked_content
                normalized_message["desensitized"] = True
                processed_indexes.append(index)
                total_replacements += replacement_count

            # 兼容输入可能是字符串列表或对象列表两种形式。
            output_messages.append(
                normalized_message["content"] if original_is_str else normalized_message
            )

        new_mapping = mapping_store.new_items()

        return {
            "messages": output_messages,
            "mapping": mapping_store.as_dict(),
            "stats": self._build_stats(
                total_messages=len(raw_messages),
                processed_indexes=processed_indexes,
                replacements=total_replacements,
                new_mapping=new_mapping,
            ),
        }

    def prepare_llm_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """拦截并脱敏即将发送给大模型的请求体。"""
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")

        llm_request = self._get_llm_request(payload)

        raw_messages = llm_request.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("`llm_request.messages` must be a list.")
        self._validate_no_message_history_mappings(raw_messages)

        history_mappings = self._get_history_mappings(payload)
        custom_entities = self._normalize_custom_entities(payload.get("custom_entities"))

        upstream_request = dict(llm_request)

        desensitize_payload: dict[str, Any] = {
            "messages": raw_messages,
            "history_mappings": history_mappings,
            "custom_entities": custom_entities,
            "desensitized_message_indexes": payload.get("desensitized_message_indexes", []),
        }
        if "target_message_indexes" in payload:
            desensitize_payload["target_message_indexes"] = payload.get(
                "target_message_indexes", []
            )

        desensitized = self.desensitize(desensitize_payload)

        upstream_request["messages"] = desensitized["messages"]

        return {
            "desensitized_request": upstream_request,
            "upstream_request": upstream_request,
            "mapping": desensitized["mapping"],
            "stats": desensitized["stats"],
        }

    @staticmethod
    def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """校验并标准化主脱敏接口入参。"""
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("`messages` must be a list.")
        return payload

    @staticmethod
    def _get_llm_request(payload: dict[str, Any]) -> dict[str, Any]:
        """兼容不同字段名，取出原始大模型请求体。"""
        for field_name in ("llm_request", "model_request", "request"):
            llm_request = payload.get(field_name)
            if llm_request is not None:
                if not isinstance(llm_request, dict):
                    raise ValueError(
                        "`llm_request` (or `model_request` / `request`) "
                        "must be a JSON object."
                    )
                return llm_request

        raise ValueError(
            "`llm_request` (or `model_request` / `request`) must be a JSON object."
        )

    @staticmethod
    def _build_stats(
        total_messages: int,
        processed_indexes: list[int],
        replacements: int,
        new_mapping: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        """统一构造响应统计信息，避免主流程里混入汇总细节。"""
        return {
            "total_messages": total_messages,
            "processed_messages": len(processed_indexes),
            "processed_message_indexes": processed_indexes,
            "replacements": replacements,
            "new_entities": sum(len(item_map) for item_map in new_mapping.values()),
        }

    @staticmethod
    def _get_history_mappings(payload: dict[str, Any]) -> Any:
        """读取顶层历史映射；`history_mapping` 仅作为兼容别名。"""
        has_primary = "history_mappings" in payload
        has_alias = "history_mapping" in payload
        if has_primary and has_alias:
            raise ValueError(
                "Use only one of `history_mappings` or `history_mapping`; "
                "`history_mappings` is recommended."
            )
        if has_primary:
            return payload.get("history_mappings")
        return payload.get("history_mapping")

    @staticmethod
    def _validate_no_message_history_mappings(raw_messages: list[Any]) -> None:
        """历史映射只能放在请求顶层，不能混入 message。"""
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, dict):
                continue
            if "history_mappings" in raw_message or "history_mapping" in raw_message:
                raise ValueError(
                    "`history_mappings` must be provided at the top level, "
                    f"not inside `messages[{index}]`."
                )

    def _mask_single_text(
        self,
        text: str,
        mapping_store: MappingStore,
        custom_entities: list[Any],
    ) -> tuple[str, int]:
        """对单条文本执行实体抽取、冲突消解和占位符替换。"""
        # 三路实体来源：历史映射、自定义规则、内置识别。
        # 历史映射用于保证占位符复用；内置识别里 Taskflow wordtag 是主识别链路。
        history_spans = self._extract_history_spans(text, mapping_store.mapping)
        custom_spans = self._extract_custom_spans(text, custom_entities)
        builtin_spans = self.recognizer.recognize_builtin(text)

        selected_spans = self._resolve_overlaps(
            text,
            history_spans + custom_spans + builtin_spans,
        )

        if not selected_spans:
            return text, 0

        masked_text = text
        replacement_count = 0

        # 从后往前替换，避免前面的替换影响后续索引。
        for span in sorted(selected_spans, key=lambda item: item.start, reverse=True):
            placeholder = mapping_store.get_or_create(span.entity_type, span.text)
            if not placeholder:
                continue
            masked_text = (
                masked_text[: span.start] + placeholder + masked_text[span.end :]
            )
            replacement_count += 1

        return masked_text, replacement_count

    def _extract_history_spans(
        self,
        text: str,
        history_mapping: dict[str, dict[str, str]],
    ) -> list[EntitySpan]:
        """从历史映射中回查当前文本命中的实体。"""
        spans: list[EntitySpan] = []

        for entity_type, item_map in history_mapping.items():
            for source_text in item_map.keys():
                if not isinstance(source_text, str):
                    continue
                candidate = source_text.strip()
                if not candidate or is_placeholder_token(candidate):
                    continue

                for start, end in self._find_all_occurrences(text, candidate):
                    spans.append(
                        EntitySpan(
                            entity_type=entity_type,
                            text=candidate,
                            start=start,
                            end=end,
                            source="history",
                        )
                    )

        return spans

    def _extract_custom_spans(
        self,
        text: str,
        custom_entities: list[Any],
    ) -> list[EntitySpan]:
        """按业务方传入的 values/patterns 提取自定义实体。"""
        if not isinstance(custom_entities, list):
            return []

        spans: list[EntitySpan] = []

        for rule in custom_entities:
            if not isinstance(rule, dict):
                continue

            entity_type = normalize_entity_type(rule.get("entity_type", "CUSTOM"))

            for raw_value in self._ensure_list(rule.get("values", [])):
                if not isinstance(raw_value, str):
                    continue
                value = raw_value.strip()
                if not value:
                    continue
                for start, end in self._find_all_occurrences(text, value):
                    spans.append(
                        EntitySpan(
                            entity_type=entity_type,
                            text=value,
                            start=start,
                            end=end,
                            source="custom",
                        )
                    )

            for raw_pattern in self._ensure_list(rule.get("patterns", [])):
                pattern, group = self._parse_pattern(raw_pattern)
                if not pattern:
                    continue

                try:
                    matches = re.finditer(pattern, text)
                except re.error as exc:
                    logger.warning("Invalid custom regex `%s`: %s", pattern, exc)
                    continue

                for match in matches:
                    try:
                        start, end = match.span(group)
                        matched_text = match.group(group)
                    except IndexError:
                        continue

                    if matched_text is None:
                        continue

                    # 去除分组捕获中的首尾空白，保证替换边界准确。
                    left_trim = len(matched_text) - len(matched_text.lstrip())
                    right_trim = len(matched_text) - len(matched_text.rstrip())
                    start += left_trim
                    end -= right_trim

                    if start >= end:
                        continue

                    candidate = text[start:end]
                    if not candidate.strip() or is_placeholder_token(candidate.strip()):
                        continue

                    spans.append(
                        EntitySpan(
                            entity_type=entity_type,
                            text=candidate,
                            start=start,
                            end=end,
                            source="custom",
                        )
                    )

        return spans

    def _resolve_overlaps(self, text: str, spans: list[EntitySpan]) -> list[EntitySpan]:
        """对重叠实体做冲突消解，输出最终替换集合。"""
        if not spans:
            return []

        valid_spans: list[EntitySpan] = []
        for span in spans:
            if span.start < 0 or span.end > len(text) or span.start >= span.end:
                continue

            real_text = text[span.start : span.end]
            if not real_text.strip():
                continue
            if is_placeholder_token(real_text.strip()):
                continue

            valid_spans.append(
                EntitySpan(
                    entity_type=span.entity_type,
                    text=real_text,
                    start=span.start,
                    end=span.end,
                    source=span.source,
                )
            )

        if not valid_spans:
            return []

        # 历史映射优先，避免复用关系被更长的模型候选覆盖；同优先级取更长跨度。
        ranked_spans = sorted(
            valid_spans,
            key=lambda item: (
                -self.SPAN_PRIORITY.get(item.source, 0),
                -item.length,
                item.start,
            ),
        )

        occupied = [False] * len(text)
        selected: list[EntitySpan] = []

        # 贪心选择不重叠区间。
        for span in ranked_spans:
            if any(occupied[index] for index in range(span.start, span.end)):
                continue
            for index in range(span.start, span.end):
                occupied[index] = True
            selected.append(span)

        return sorted(selected, key=lambda item: item.start)

    @staticmethod
    def _find_all_occurrences(text: str, candidate: str) -> list[tuple[int, int]]:
        """查找子串在文本中的所有不重叠位置。"""
        positions: list[tuple[int, int]] = []
        start = 0
        while True:
            index = text.find(candidate, start)
            if index == -1:
                break
            end = index + len(candidate)
            positions.append((index, end))
            start = end
        return positions

    @staticmethod
    def _parse_pattern(raw_pattern: Any) -> tuple[str, int]:
        """兼容自定义正则的字符串/对象两种格式。"""
        if isinstance(raw_pattern, str):
            return raw_pattern, 0
        if isinstance(raw_pattern, dict):
            pattern = raw_pattern.get("regex") or raw_pattern.get("pattern")
            if not isinstance(pattern, str):
                return "", 0
            group = raw_pattern.get("group", 0)
            try:
                return pattern, int(group)
            except (TypeError, ValueError):
                return pattern, 0
        return "", 0

    @staticmethod
    def _ensure_list(value: Any) -> list[Any]:
        """把单值/空值统一成列表，简化后续遍历。"""
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return [value]

    @staticmethod
    def _normalize_custom_entities(value: Any) -> list[Any]:
        """清洗自定义实体规则，避免非列表入参干扰识别流程。"""
        if not isinstance(value, list):
            return []
        return value

    @staticmethod
    def _normalize_message(raw_message: Any) -> tuple[dict[str, Any], bool]:
        """兼容字符串消息与对象消息，统一为内部结构。"""
        if isinstance(raw_message, str):
            return {"role": "user", "content": raw_message, "desensitized": False}, True

        if isinstance(raw_message, dict):
            message = dict(raw_message)
            content = message.get("content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = str(content)
            message["content"] = content
            return message, False

        raise ValueError("Each message must be either a string or an object.")

    @staticmethod
    def _to_index_set(raw_indexes: Any) -> set[int]:
        """把索引数组清洗成非负整数集合。"""
        if not isinstance(raw_indexes, list):
            return set()

        indexes: set[int] = set()
        for item in raw_indexes:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if index >= 0:
                indexes.add(index)
        return indexes

    @staticmethod
    def _should_process_message(
        index: int,
        message: dict[str, Any],
        desensitized_indexes: set[int],
        target_indexes: set[int] | None,
    ) -> bool:
        """判断当前消息是否需要本次脱敏处理。"""
        if target_indexes is not None and index not in target_indexes:
            return False
        if index in desensitized_indexes:
            return False
        return not bool(message.get("desensitized", False))
