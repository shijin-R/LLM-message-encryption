"""核心数据类型定义。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EntitySpan:
    """文本中的实体片段。

    entity_type: 实体类型（如 PERSON/ORG/MOBILE）
    text: 实体原文
    start/end: 在原文中的字符区间（左闭右开）
    source: 命中来源（mapping/custom/model/jieba/regex）
    """

    entity_type: str
    text: str
    start: int
    end: int
    source: str

    @property
    def length(self) -> int:
        """返回实体跨度长度，供冲突消解排序使用。"""
        return self.end - self.start
