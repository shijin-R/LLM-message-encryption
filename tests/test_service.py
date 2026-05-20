from pathlib import Path
import unittest

from desensitize.config import ServiceConfig
from desensitize.recognizer import LocalEntityRecognizer
from desensitize.service import DesensitizeService
from desensitize.types import EntitySpan


class EmptyRecognizer:
    def recognize_builtin(self, text: str) -> list[EntitySpan]:
        return []


class FakeRecognizer:
    def recognize_builtin(self, text: str) -> list[EntitySpan]:
        spans: list[EntitySpan] = []
        for entity_type, value, source in (
            ("ORG", "上海泛微网络科技股份有限公司", "model"),
            ("PERSON", "张三", "model"),
            ("MOBILE", "13800138000", "regex"),
        ):
            start = text.find(value)
            if start >= 0:
                spans.append(
                    EntitySpan(entity_type, value, start, start + len(value), source)
                )
        return spans


class DesensitizeServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        config = ServiceConfig(
            model_path=Path("resources/models/wordtag"),
            dict_dir=Path("resources/common_data/uie"),
            enable_taskflow=False,
            strict_local_model=False,
        )
        self.service = DesensitizeService(config)
        self.service.recognizer = FakeRecognizer()

    def test_preprocess_returns_masked_request_and_reuses_history(self) -> None:
        payload = {
            "llm_request": {
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": "合同甲方：上海泛微网络科技股份有限公司，联系人张三，手机号13800138000。",
                    }
                ],
            },
            "history_mappings": {"PERSON": {"张三": "[[PERSON_009]]"}},
        }

        result = self.service.prepare_llm_request(payload)

        self.assertEqual(
            result["desensitized_request"]["messages"][0]["content"],
            "合同甲方：[[ORG_001]]，联系人[[PERSON_009]]，手机号[[MOBILE_001]]。",
        )
        self.assertEqual(result["mapping"]["PERSON"]["张三"], "[[PERSON_009]]")
        self.assertNotIn("new_mapping", result)

    def test_history_mapping_alias_is_supported(self) -> None:
        result = self.service.prepare_llm_request(
            {
                "llm_request": {
                    "model": "demo",
                    "messages": [
                        {
                            "role": "user",
                            "content": "联系人张三，手机号13800138000。",
                        }
                    ],
                },
                "history_mapping": {"PERSON": {"张三": "[[PERSON_007]]"}},
            }
        )

        self.assertEqual(
            result["desensitized_request"]["messages"][0]["content"],
            "联系人[[PERSON_007]]，手机号[[MOBILE_001]]。",
        )

    def test_rejects_both_history_field_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Use only one"):
            self.service.prepare_llm_request(
                {
                    "llm_request": {
                        "model": "demo",
                        "messages": [{"role": "user", "content": "张三"}],
                    },
                    "history_mappings": {"PERSON": {"张三": "[[PERSON_001]]"}},
                    "history_mapping": {"PERSON": {"张三": "[[PERSON_001]]"}},
                }
            )

    def test_rejects_history_mappings_inside_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "top level"):
            self.service.prepare_llm_request(
                {
                    "llm_request": {
                        "model": "demo",
                        "messages": [
                            {
                                "role": "user",
                                "content": "张三",
                                "history_mappings": {
                                    "PERSON": {"张三": "[[PERSON_001]]"}
                                },
                            }
                        ],
                    }
                }
            )

    def test_desensitize_masks_without_custom_rules(self) -> None:
        result = self.service.desensitize(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "手机号13800138000",
                    }
                ]
            }
        )

        self.assertEqual(
            result["messages"][0]["content"],
            "手机号[[MOBILE_001]]",
        )
        self.assertEqual(result["mapping"]["MOBILE"]["13800138000"], "[[MOBILE_001]]")

    def test_custom_pattern_group_masks_capture_only(self) -> None:
        self.service.recognizer = EmptyRecognizer()

        result = self.service.desensitize(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "请联系联系人：李四确认合同。",
                    }
                ],
                "custom_entities": [
                    {
                        "entity_type": "PERSON",
                        "patterns": [
                            {
                                "regex": "联系人[：:\\s]*([\\u4e00-\\u9fff]{2})",
                                "group": 1,
                            }
                        ],
                    }
                ],
            }
        )

        self.assertEqual(
            result["messages"][0]["content"],
            "请联系联系人：[[PERSON_001]]确认合同。",
        )
        self.assertEqual(result["mapping"]["PERSON"]["李四"], "[[PERSON_001]]")

    def test_custom_value_masks_without_model_result(self) -> None:
        self.service.recognizer = EmptyRecognizer()

        result = self.service.desensitize(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "请核对北京测试科技有限公司的合同。",
                    }
                ],
                "custom_entities": [
                    {
                        "entity_type": "ORG",
                        "values": ["北京测试科技有限公司"],
                    }
                ],
            }
        )

        self.assertEqual(
            result["messages"][0]["content"],
            "请核对[[ORG_001]]的合同。",
        )
        self.assertEqual(
            result["mapping"]["ORG"]["北京测试科技有限公司"],
            "[[ORG_001]]",
        )

    def test_target_message_indexes_only_process_selected_messages(self) -> None:
        result = self.service.desensitize(
            {
                "messages": [
                    {"role": "user", "content": "手机号13800138000"},
                    {"role": "user", "content": "手机号13800138000"},
                ],
                "target_message_indexes": [1],
            }
        )

        self.assertEqual(result["messages"][0]["content"], "手机号13800138000")
        self.assertEqual(result["messages"][1]["content"], "手机号[[MOBILE_001]]")
        self.assertEqual(result["stats"]["processed_message_indexes"], [1])

    def test_desensitized_message_indexes_skip_selected_messages(self) -> None:
        result = self.service.desensitize(
            {
                "messages": [
                    {"role": "user", "content": "手机号13800138000"},
                    {"role": "user", "content": "手机号13800138000"},
                ],
                "desensitized_message_indexes": [0],
            }
        )

        self.assertEqual(result["messages"][0]["content"], "手机号13800138000")
        self.assertEqual(result["messages"][1]["content"], "手机号[[MOBILE_001]]")
        self.assertEqual(result["stats"]["processed_message_indexes"], [1])

    def test_mobile_regex_does_not_match_inside_long_number(self) -> None:
        text = "订单号9913800138000123，手机号13800138000。"

        matches = [
            match.group()
            for match in LocalEntityRecognizer.MOBILE_PATTERN.finditer(text)
        ]

        self.assertEqual(matches, ["13800138000"])


if __name__ == "__main__":
    unittest.main()
