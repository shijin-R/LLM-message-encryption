from pathlib import Path
import unittest

from desensitize.config import ServiceConfig
from desensitize.recognizer import LocalEntityRecognizer
from desensitize.service import DesensitizeService
from desensitize.types import EntitySpan


class EmptyRecognizer:
    def recognize_builtin(self, text: str) -> list[EntitySpan]:
        return []

    def recognize_custom(
        self,
        text: str,
        custom_entities: list[dict],
    ) -> list[EntitySpan]:
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

    def recognize_custom(
        self,
        text: str,
        custom_entities: list[dict],
    ) -> list[EntitySpan]:
        if not isinstance(custom_entities, list):
            return []

        requested_types = {
            str(rule.get("entity_type", "CUSTOM")).upper()
            for rule in custom_entities
            if isinstance(rule, dict)
        }
        spans: list[EntitySpan] = []
        for entity_type, value in (
            ("PERSON", "李四"),
            ("ORG", "北京测试科技有限公司"),
        ):
            if entity_type not in requested_types:
                continue
            start = text.find(value)
            if start >= 0:
                spans.append(
                    EntitySpan(entity_type, value, start, start + len(value), "custom")
                )
        return spans


class FakeUIETaskflow:
    def __init__(self, output_by_label: dict[str, list[dict]]) -> None:
        self.output_by_label = output_by_label
        self.schema: list[str] = []

    def set_schema(self, schema: list[str]) -> None:
        self.schema = schema

    def __call__(self, text: str) -> list[dict]:
        result: dict[str, list[dict]] = {}
        for label in self.schema:
            if label in self.output_by_label:
                result[label] = self.output_by_label[label]
        return [result]


class CombinedRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, text: str, custom_entities: list[dict]) -> list[EntitySpan]:
        self.calls += 1
        start = text.find("13800138000")
        if start < 0:
            return []
        return [
            EntitySpan(
                "MOBILE",
                "13800138000",
                start,
                start + len("13800138000"),
                "regex",
            )
        ]

    def recognize_builtin(self, text: str) -> list[EntitySpan]:
        raise AssertionError("combined recognizer should use recognize()")

    def recognize_custom(
        self,
        text: str,
        custom_entities: list[dict],
    ) -> list[EntitySpan]:
        raise AssertionError("combined recognizer should use recognize()")


class DesensitizeServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        config = ServiceConfig(
            model_path=Path("resources/models/wordtag"),
            enable_taskflow=False,
            strict_local_model=False,
        )
        self.service = DesensitizeService(config)
        self.service.recognizer = FakeRecognizer()

    def preprocess(self, messages: list[dict], **extra_payload):
        payload = {
            "llm_request": {
                "model": "demo",
                "messages": messages,
            }
        }
        payload.update(extra_payload)
        return self.service.prepare_llm_request(payload)

    def test_preprocess_returns_masked_request_and_reuses_message_mapping(self) -> None:
        payload = {
            "llm_request": {
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": "上一轮联系人[[PERSON_009]]。",
                        "encrypted": True,
                        "mapping": {"PERSON": {"张三": "[[PERSON_009]]"}},
                    },
                    {"role": "assistant", "content": "已记录。"},
                    {
                        "role": "user",
                        "content": "合同甲方：上海泛微网络科技股份有限公司，联系人张三，手机号13800138000。",
                        "encrypted": False,
                    }
                ],
            },
        }

        result = self.service.prepare_llm_request(payload)

        messages = result["desensitized_request"]["messages"]
        self.assertNotIn("mapping", messages[0])
        self.assertNotIn("encrypted", messages[0])
        self.assertNotIn("encrypted", messages[2])
        self.assertEqual(messages[0]["content"], "上一轮联系人[[PERSON_009]]。")
        self.assertEqual(
            messages[2]["content"],
            "合同甲方：[[ORG_001]]，联系人[[PERSON_009]]，手机号[[MOBILE_001]]。",
        )
        self.assertEqual(result["mapping"]["PERSON"]["张三"], "[[PERSON_009]]")
        self.assertEqual(result["stats"]["processed_message_indexes"], [2])
        self.assertEqual(set(result), {"desensitized_request", "mapping", "stats"})
        self.assertNotIn("new_mapping", result)

    def test_uses_message_mapping_and_strips_from_response(self) -> None:
        result = self.service.prepare_llm_request(
            {
                "llm_request": {
                    "model": "demo",
                    "messages": [
                        {
                            "role": "user",
                            "content": "历史联系人[[PERSON_001]]。",
                            "encrypted": True,
                            "mapping": {
                                "PERSON": {"张三": "[[PERSON_001]]"}
                            },
                        },
                        {"role": "assistant", "content": "已记录。"},
                        {
                            "role": "user",
                            "content": "这次还是张三，手机号13800138000。",
                            "encrypted": False,
                        },
                    ],
                }
            }
        )

        messages = result["desensitized_request"]["messages"]
        self.assertNotIn("mapping", messages[0])
        self.assertNotIn("encrypted", messages[0])
        self.assertNotIn("encrypted", messages[2])
        self.assertEqual(messages[0]["content"], "历史联系人[[PERSON_001]]。")
        self.assertEqual(
            messages[2]["content"],
            "这次还是[[PERSON_001]]，手机号[[MOBILE_001]]。",
        )
        self.assertEqual(result["mapping"]["PERSON"]["张三"], "[[PERSON_001]]")
        self.assertEqual(result["stats"]["processed_message_indexes"], [2])

    def test_unencrypted_message_mapping_is_not_used_as_history(self) -> None:
        result = self.preprocess(
            [
                {
                    "role": "user",
                    "content": "联系人张三。",
                    "encrypted": False,
                    "mapping": {"PERSON": {"张三": "[[PERSON_009]]"}},
                }
            ]
        )

        message = result["desensitized_request"]["messages"][0]
        self.assertNotIn("mapping", message)
        self.assertNotIn("encrypted", message)
        self.assertEqual(message["content"], "联系人[[PERSON_001]]。")
        self.assertEqual(result["mapping"]["PERSON"]["张三"], "[[PERSON_001]]")

    def test_all_unencrypted_user_messages_are_processed(self) -> None:
        result = self.preprocess(
            [
                {"role": "user", "content": "第一轮联系人张三。"},
                {"role": "assistant", "content": "已记录。"},
                {
                    "role": "user",
                    "content": "第二轮还是张三，手机号13800138000。",
                    "encrypted": False,
                },
            ]
        )

        messages = result["desensitized_request"]["messages"]
        self.assertEqual(messages[0]["content"], "第一轮联系人[[PERSON_001]]。")
        self.assertEqual(
            messages[2]["content"],
            "第二轮还是[[PERSON_001]]，手机号[[MOBILE_001]]。",
        )
        self.assertEqual(result["stats"]["processed_message_indexes"], [0, 2])

    def test_encrypted_user_messages_are_skipped(self) -> None:
        result = self.preprocess(
            [
                {
                    "role": "user",
                    "content": "历史联系人[[PERSON_001]]，手机号[[MOBILE_001]]。",
                    "encrypted": True,
                    "mapping": {
                        "PERSON": {"张三": "[[PERSON_001]]"},
                        "MOBILE": {"13800138000": "[[MOBILE_001]]"},
                    },
                }
            ]
        )

        message = result["desensitized_request"]["messages"][0]
        self.assertEqual(message["content"], "历史联系人[[PERSON_001]]，手机号[[MOBILE_001]]。")
        self.assertNotIn("mapping", message)
        self.assertNotIn("encrypted", message)
        self.assertEqual(result["stats"]["processed_message_indexes"], [])

    def test_desensitize_masks_without_custom_rules(self) -> None:
        result = self.preprocess(
            [
                {
                    "role": "user",
                    "content": "手机号13800138000",
                    "encrypted": False,
                }
            ]
        )

        self.assertEqual(
            result["desensitized_request"]["messages"][0]["content"],
            "手机号[[MOBILE_001]]",
        )
        self.assertEqual(result["mapping"]["MOBILE"]["13800138000"], "[[MOBILE_001]]")

    def test_combined_recognizer_is_used_when_available(self) -> None:
        recognizer = CombinedRecognizer()
        self.service.recognizer = recognizer

        result = self.preprocess(
            [
                {
                    "role": "user",
                    "content": "手机号13800138000",
                    "encrypted": False,
                }
            ]
        )

        self.assertEqual(recognizer.calls, 1)
        self.assertEqual(
            result["desensitized_request"]["messages"][0]["content"],
            "手机号[[MOBILE_001]]",
        )

    def test_all_unencrypted_user_content_is_processed(self) -> None:
        result = self.preprocess(
            [
                {"role": "system", "content": "系统手机号13800138000"},
                {"role": "user", "content": "历史用户手机号13800138000"},
                {"role": "assistant", "content": "助手提到张三"},
                {"role": "user", "content": "用户手机号13800138000", "encrypted": False},
            ]
        )

        messages = result["desensitized_request"]["messages"]
        self.assertEqual(messages[0]["content"], "系统手机号13800138000")
        self.assertEqual(messages[1]["content"], "历史用户手机号[[MOBILE_001]]")
        self.assertEqual(messages[2]["content"], "助手提到张三")
        self.assertEqual(messages[3]["content"], "用户手机号[[MOBILE_001]]")
        self.assertEqual(result["stats"]["processed_message_indexes"], [1, 3])

    def test_custom_pattern_regex_is_not_used_without_model_result(self) -> None:
        self.service.recognizer = EmptyRecognizer()

        result = self.preprocess(
            [
                {
                    "role": "user",
                    "content": "请联系联系人：李四确认合同。",
                    "encrypted": False,
                }
            ],
            custom_entities=[
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
        )

        self.assertEqual(
            result["desensitized_request"]["messages"][0]["content"],
            "请联系联系人：李四确认合同。",
        )
        self.assertNotIn("PERSON", result["mapping"])

    def test_custom_entity_masks_model_result(self) -> None:
        result = self.preprocess(
            [
                {
                    "role": "user",
                    "content": "请联系联系人：李四确认合同。",
                    "encrypted": False,
                }
            ],
            custom_entities=[
                {
                    "entity_type": "PERSON",
                    "uie_schema": ["人名"],
                }
            ],
        )

        self.assertEqual(
            result["desensitized_request"]["messages"][0]["content"],
            "请联系联系人：[[PERSON_001]]确认合同。",
        )
        self.assertEqual(result["mapping"]["PERSON"]["李四"], "[[PERSON_001]]")

    def test_custom_value_is_not_string_matched_without_model_result(self) -> None:
        self.service.recognizer = EmptyRecognizer()

        result = self.preprocess(
            [
                {
                    "role": "user",
                    "content": "请核对北京测试科技有限公司的合同。",
                    "encrypted": False,
                }
            ],
            custom_entities=[
                {
                    "entity_type": "ORG",
                    "values": ["北京测试科技有限公司"],
                }
            ],
        )

        self.assertEqual(
            result["desensitized_request"]["messages"][0]["content"],
            "请核对北京测试科技有限公司的合同。",
        )
        self.assertNotIn("ORG", result["mapping"])

    def test_mobile_regex_does_not_match_inside_long_number(self) -> None:
        text = "订单号9913800138000123，手机号13800138000。"

        matches = [
            match.group()
            for match in LocalEntityRecognizer.MOBILE_PATTERN.finditer(text)
        ]

        self.assertEqual(matches, ["13800138000"])

    def test_custom_model_labels_are_ignored_by_uie_custom_path(self) -> None:
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer.max_text_len = 10000
        recognizer.strict_local_model = True
        recognizer.strict_uie_model = True
        recognizer.enable_uie_custom = True
        recognizer._uie_taskflow = None
        recognizer._uie_schema = ()

        def fail_if_uie_loaded(schema: list[str]):
            raise AssertionError("model_labels must not trigger UIE loading")

        recognizer._ensure_uie_taskflow = fail_if_uie_loaded

        spans = recognizer.recognize_custom(
            "联系人张三",
            [{"entity_type": "PERSON", "model_labels": ["人名"]}],
        )

        self.assertEqual(spans, [])

    def test_custom_uie_address_filters_field_word_and_keeps_real_address(self) -> None:
        fake_uie = FakeUIETaskflow(
            {
                "地址": [
                    {
                        "text": "地址",
                        "start": 0,
                        "end": 2,
                        "probability": 0.99,
                    },
                    {
                        "text": "北京市海淀区中关村大街27号",
                        "start": 3,
                        "end": 17,
                        "probability": 0.98,
                    },
                ]
            }
        )
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer.max_text_len = 10000
        recognizer.strict_local_model = True
        recognizer.strict_uie_model = True
        recognizer.enable_uie_custom = True
        recognizer._uie_taskflow = fake_uie
        recognizer._uie_schema = ()
        recognizer._ensure_uie_taskflow = lambda schema: (
            fake_uie.set_schema(schema) or fake_uie
        )

        spans = recognizer.recognize_custom(
            "地址：北京市海淀区中关村大街27号",
            [{"entity_type": "ADDRESS", "uie_schema": ["地址"]}],
        )

        self.assertEqual(
            [(span.entity_type, span.text, span.source) for span in spans],
            [
                ("ADDRESS", "北京市海淀区中关村大街27号", "custom"),
            ],
        )

    def test_custom_uie_masks_id_card_and_bank_card(self) -> None:
        text = "客户身份证号110101199003071234，银行卡号6222020202020202020。"
        fake_uie = FakeUIETaskflow(
            {
                "身份证号": [
                    {
                        "text": "110101199003071234",
                        "start": 6,
                        "end": 24,
                        "probability": 0.98,
                    }
                ],
                "银行卡号": [
                    {
                        "text": "6222020202020202020",
                        "start": 29,
                        "end": 48,
                        "probability": 0.97,
                    }
                ],
            }
        )

        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer.max_text_len = 10000
        recognizer.strict_local_model = True
        recognizer.strict_uie_model = True
        recognizer.enable_uie_custom = True
        recognizer._taskflow = None
        recognizer._uie_taskflow = fake_uie
        recognizer._uie_schema = ()
        recognizer._ensure_uie_taskflow = lambda schema: (
            fake_uie.set_schema(schema) or fake_uie
        )

        spans = recognizer.recognize_custom(
            text,
            [
                {"entity_type": "ID_CARD", "uie_schema": ["身份证号"]},
                {"entity_type": "BANK_CARD", "uie_schema": ["银行卡号"]},
            ],
        )

        self.assertEqual(
            [(span.entity_type, span.text, span.source) for span in spans],
            [
                ("ID_CARD", "110101199003071234", "custom"),
                ("BANK_CARD", "6222020202020202020", "custom"),
            ],
        )

    def test_preprocess_masks_uie_custom_id_card_and_bank_card(self) -> None:
        text = "客户身份证号110101199003071234，银行卡号6222020202020202020。"
        fake_uie = FakeUIETaskflow(
            {
                "身份证号": [
                    {
                        "text": "110101199003071234",
                        "start": 6,
                        "end": 24,
                        "probability": 0.98,
                    }
                ],
                "银行卡号": [
                    {
                        "text": "6222020202020202020",
                        "start": 29,
                        "end": 48,
                        "probability": 0.97,
                    }
                ],
            }
        )
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer.max_text_len = 10000
        recognizer.strict_local_model = True
        recognizer.strict_uie_model = True
        recognizer.enable_uie_custom = True
        recognizer._taskflow = None
        recognizer._uie_taskflow = fake_uie
        recognizer._uie_schema = ()
        recognizer._ensure_uie_taskflow = lambda schema: (
            fake_uie.set_schema(schema) or fake_uie
        )
        recognizer._extract_with_taskflow = lambda content: []
        recognizer._extract_mobile = lambda content: []

        self.service.recognizer = recognizer

        result = self.preprocess(
            [{"role": "user", "content": text, "encrypted": False}],
            custom_entities=[
                {"entity_type": "ID_CARD", "uie_schema": ["身份证号"]},
                {"entity_type": "BANK_CARD", "uie_schema": ["银行卡号"]},
            ],
        )

        self.assertEqual(
            result["desensitized_request"]["messages"][0]["content"],
            "客户身份证号[[ID_CARD_001]]，银行卡号[[BANK_CARD_001]]。",
        )
        self.assertEqual(
            result["mapping"]["ID_CARD"]["110101199003071234"],
            "[[ID_CARD_001]]",
        )
        self.assertEqual(
            result["mapping"]["BANK_CARD"]["6222020202020202020"],
            "[[BANK_CARD_001]]",
        )


if __name__ == "__main__":
    unittest.main()
