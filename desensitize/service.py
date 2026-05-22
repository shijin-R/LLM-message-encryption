from typing import Any

from .config import ServiceConfig
from .mapping import MappingStore, is_placeholder_token
from .recognizer import LocalEntityRecognizer
from .types import EntitySpan


class DesensitizeService:
    MESSAGE_MAPPING_FIELD = "mapping"
    MESSAGE_ENCRYPTED_FIELD = "encrypted"

    # 实体冲突消解时的来源优先级：值越大优先级越高。
    SPAN_PRIORITY = {
        "mapping": 5,
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
            enable_jieba_fallback=config.enable_jieba_fallback,
            strict_local_model=config.strict_local_model,
            auto_download_model=config.auto_download_model,
            sync_downloaded_model=config.sync_downloaded_model,
            downloaded_model_cache_path=config.downloaded_model_cache_path,
        )

    def _desensitize_messages(
        self,
        raw_messages: list[Any],
        custom_entities: list[Any],
    ) -> dict[str, Any]:
        """对预处理请求中的 messages 执行内部脱敏并返回映射信息。"""
        mapping_store = MappingStore(self._collect_message_mappings(raw_messages))

        output_messages: list[Any] = []
        processed_indexes: list[int] = []
        total_replacements = 0

        for index, raw_message in enumerate(raw_messages):
            normalized_message, original_is_str = self._normalize_message(raw_message)
            content = normalized_message["content"]

            should_process = self._should_process_message(
                message=normalized_message,
            )

            if should_process and content:
                masked_content, replacement_count = self._mask_single_text(
                    text=content,
                    mapping_store=mapping_store,
                    custom_entities=custom_entities,
                )
                normalized_message["content"] = masked_content
                processed_indexes.append(index)
                total_replacements += replacement_count

            self._strip_internal_message_fields(normalized_message)

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

        custom_entities = self._normalize_custom_entities(payload.get("custom_entities"))

        desensitized = self._desensitize_messages(raw_messages, custom_entities)

        desensitized_request = dict(llm_request)
        desensitized_request["messages"] = desensitized["messages"]

        return {
            "desensitized_request": desensitized_request,
            "mapping": desensitized["mapping"],
            "stats": desensitized["stats"],
        }

    @staticmethod
    def _get_llm_request(payload: dict[str, Any]) -> dict[str, Any]:
        """取出原始大模型请求体。"""
        llm_request = payload.get("llm_request")
        if not isinstance(llm_request, dict):
            raise ValueError("`llm_request` must be a JSON object.")
        return llm_request

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

    @classmethod
    def _get_message_mapping(cls, message: dict[str, Any]) -> Any:
        """读取单条 user message 内携带的历史映射。"""
        return message.get(cls.MESSAGE_MAPPING_FIELD)

    @classmethod
    def _collect_message_mappings(
        cls,
        raw_messages: list[Any],
    ) -> dict[str, dict[str, str]]:
        """合并已脱敏 user message 内的映射，供本次请求复用占位符。"""
        combined: dict[str, dict[str, str]] = {}
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                continue
            if not cls._is_user_message(raw_message):
                continue
            if not cls._is_message_encrypted(raw_message):
                continue
            cls._merge_mappings(combined, cls._get_message_mapping(raw_message))

        return combined

    @staticmethod
    def _merge_mappings(
        target: dict[str, dict[str, str]],
        mapping: Any,
    ) -> None:
        """把一份映射归一化后合入目标字典；已有项保持优先。"""
        normalized = MappingStore(mapping).as_dict()
        for entity_type, item_map in normalized.items():
            target_map = target.setdefault(entity_type, {})
            for source_text, placeholder in item_map.items():
                target_map.setdefault(source_text, placeholder)

    def _mask_single_text(
        self,
        text: str,
        mapping_store: MappingStore,
        custom_entities: list[Any],
    ) -> tuple[str, int]:
        """对单条文本执行实体抽取、冲突消解和占位符替换。"""
        # 三路实体来源：已有映射、自定义模型规则、内置识别。
        # 已有映射用于保证占位符复用；内置识别里 Taskflow wordtag 是主识别链路。
        mapping_spans = self._extract_mapping_spans(text, mapping_store.mapping)
        custom_spans = self.recognizer.recognize_custom(text, custom_entities)
        builtin_spans = self.recognizer.recognize_builtin(text)

        selected_spans = self._resolve_overlaps(
            text,
            mapping_spans + custom_spans + builtin_spans,
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

    def _extract_mapping_spans(
        self,
        text: str,
        known_mapping: dict[str, dict[str, str]],
    ) -> list[EntitySpan]:
        """从已有映射中回查当前文本命中的实体。"""
        spans: list[EntitySpan] = []

        for entity_type, item_map in known_mapping.items():
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
                            source="mapping",
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

        # 已有映射优先，避免复用关系被更长的模型候选覆盖；同优先级取更长跨度。
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
    def _normalize_custom_entities(value: Any) -> list[Any]:
        """清洗自定义实体规则，避免非列表入参干扰识别流程。"""
        if not isinstance(value, list):
            return []
        return value

    @staticmethod
    def _normalize_message(raw_message: Any) -> tuple[dict[str, Any], bool]:
        """兼容字符串消息与对象消息，统一为内部结构。"""
        if isinstance(raw_message, str):
            return {"role": "user", "content": raw_message}, True

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
    def _is_user_message(message: dict[str, Any]) -> bool:
        """判断是否是需要脱敏服务关注的 user 消息。"""
        role = message.get("role", "user")
        return isinstance(role, str) and role.lower() == "user"

    @classmethod
    def _is_message_encrypted(cls, message: dict[str, Any]) -> bool:
        """判断业务方标记的 user message 是否已经完成脱敏处理。"""
        return message.get(cls.MESSAGE_ENCRYPTED_FIELD) is True

    @classmethod
    def _strip_internal_message_fields(cls, message: dict[str, Any]) -> None:
        """返回给上游模型前移除脱敏服务内部控制字段。"""
        message.pop(cls.MESSAGE_MAPPING_FIELD, None)
        message.pop(cls.MESSAGE_ENCRYPTED_FIELD, None)

    @classmethod
    def _should_process_message(
        cls,
        message: dict[str, Any],
    ) -> bool:
        """判断当前消息是否需要本次脱敏处理。"""
        if not cls._is_user_message(message):
            return False
        return not cls._is_message_encrypted(message)
