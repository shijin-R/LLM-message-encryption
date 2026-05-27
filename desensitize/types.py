"""核心数据类型定义。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpan:
    """模型推理返回的原始标签片段。

    label: 模型标签或 UIE schema 标签
    text: 命中文本
    start/end: 在输入文本中的字符区间（左闭右开）
    source: 推理来源（wordtag/uie）
    probability: 模型置信度；无置信度时为 None
    """

    label: str
    text: str
    start: int
    end: int
    source: str
    probability: float | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class EntitySpan:
    """文本中的实体片段。

    entity_type: 实体类型（如 PERSON/ORG/MOBILE）
    text: 实体原文
    start/end: 在原文中的字符区间（左闭右开）
    source: 命中来源（mapping/custom/model/regex）
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
