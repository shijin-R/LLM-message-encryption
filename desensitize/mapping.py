"""映射字典管理模块。

负责历史映射归一化、占位符分配与增量映射输出。
"""

import re
from typing import Any


# 占位符格式：[[ENTITY_TYPE_001]]
PLACEHOLDER_FULL_PATTERN = re.compile(r"^\[\[([A-Z0-9_]+)_(\d+)\]\]$")


def normalize_entity_type(entity_type: Any) -> str:
    """把实体类型标准化为大写下划线形式。"""
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(entity_type).strip().upper())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "ENTITY"


def is_placeholder_token(value: str) -> bool:
    """判断文本是否已经是占位符，避免重复替换。"""
    return bool(PLACEHOLDER_FULL_PATTERN.match(value.strip()))


class MappingStore:
    """维护完整映射与本次新增映射。"""

    def __init__(self, history_mappings: Any = None) -> None:
        # mapping: {ENTITY_TYPE: {原文: 占位符}}
        self.mapping = self._normalize_history(history_mappings)
        # new_mapping: 本次请求中新建的映射项。
        self.new_mapping: dict[str, dict[str, str]] = {
            entity_type: {} for entity_type in self.mapping
        }
        # 每种实体类型的下一个编号游标。
        self._next_index = self._build_next_index()

    def _normalize_history(self, history_mappings: Any) -> dict[str, dict[str, str]]:
        """把历史映射统一转换为标准字典结构。"""
        normalized: dict[str, dict[str, str]] = {}
        if not isinstance(history_mappings, dict):
            return normalized

        for raw_entity_type, raw_mapping in history_mappings.items():
            entity_type = normalize_entity_type(raw_entity_type)
            normalized.setdefault(entity_type, {})

            if isinstance(raw_mapping, dict):
                iterator = raw_mapping.items()
            elif isinstance(raw_mapping, list):
                # 兼容列表结构：[{source/origin/text/value, placeholder}, ...]
                iterator = []
                for item in raw_mapping:
                    if not isinstance(item, dict):
                        continue
                    source_text = (
                        item.get("source")
                        or item.get("origin")
                        or item.get("text")
                        or item.get("value")
                    )
                    placeholder = item.get("placeholder")
                    iterator.append((source_text, placeholder))
            else:
                continue

            for source_text, placeholder in iterator:
                if not isinstance(source_text, str) or not source_text.strip():
                    continue
                if not isinstance(placeholder, str) or not placeholder.strip():
                    continue
                normalized[entity_type][source_text] = placeholder

        return normalized

    def _build_next_index(self) -> dict[str, int]:
        """根据已有占位符计算每类实体的下一个序号。"""
        next_index: dict[str, int] = {}
        for entity_type, item_map in self.mapping.items():
            max_index = 0
            for placeholder in item_map.values():
                match = PLACEHOLDER_FULL_PATTERN.match(placeholder.strip())
                if not match:
                    continue
                if match.group(1) != entity_type:
                    continue
                max_index = max(max_index, int(match.group(2)))
            if max_index > 0:
                next_index[entity_type] = max_index + 1
            else:
                next_index[entity_type] = len(item_map) + 1
        return next_index

    def get_or_create(self, entity_type: str, source_text: str) -> str:
        """获取已存在占位符，或为新实体创建占位符。"""
        normalized_entity_type = normalize_entity_type(entity_type)
        clean_source_text = source_text.strip()
        if not clean_source_text:
            return ""

        entity_mapping = self.mapping.setdefault(normalized_entity_type, {})
        self.new_mapping.setdefault(normalized_entity_type, {})

        if clean_source_text in entity_mapping:
            return entity_mapping[clean_source_text]

        current_index = self._next_index.get(normalized_entity_type, 1)
        placeholder = f"[[{normalized_entity_type}_{current_index:03d}]]"

        entity_mapping[clean_source_text] = placeholder
        self.new_mapping[normalized_entity_type][clean_source_text] = placeholder
        self._next_index[normalized_entity_type] = current_index + 1

        return placeholder

    def as_dict(self) -> dict[str, dict[str, str]]:
        """返回完整映射（可用于回传给调用方）。"""
        return {
            entity_type: dict(item_map)
            for entity_type, item_map in self.mapping.items()
        }

    def new_items(self) -> dict[str, dict[str, str]]:
        """返回本次请求新增的映射项。"""
        return {
            entity_type: dict(item_map)
            for entity_type, item_map in self.new_mapping.items()
            if item_map
        }