from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

from desensitize.application_recognizer import ApplicationEntityRecognizer
from desensitize.config import ServiceConfig
from desensitize.recognizer import LocalEntityRecognizer
from desensitize.remote_recognizer import HTTPRecognizerClient, RemoteRecognizerError
from desensitize.service import DesensitizeService
from desensitize.types import EntitySpan, ModelSpan


class EmptyRecognizer:
    def recognize(self, text: str, custom_entities: list[dict]) -> list[EntitySpan]:
        return []


class FakeRecognizer:
    def recognize(self, text: str, custom_entities: list[dict]) -> list[EntitySpan]:
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

        if not isinstance(custom_entities, list):
            return spans

        requested_types = {
            str(rule.get("entity_type", "CUSTOM")).upper()
            for rule in custom_entities
            if isinstance(rule, dict)
        }
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


class SlowWordtagTaskflow:
    def __init__(self) -> None:
        self.active_calls = 0
        self.max_active_calls = 0
        self._calls_lock = threading.Lock()

    def __call__(self, text: str) -> list[tuple[str, str]]:
        with self._calls_lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(0.02)
            if "李四" in text:
                return [("李四", "人名")]
            return [("张三", "人名")]
        finally:
            with self._calls_lock:
                self.active_calls -= 1


class SlowSchemaAwareUIETaskflow:
    def __init__(self, output_by_label: dict[str, list[dict]]) -> None:
        self.output_by_label = output_by_label
        self.schema: list[str] = []
        self.schema_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self._calls_lock = threading.Lock()

    def set_schema(self, schema: list[str]) -> None:
        self.schema = list(schema)

    def __call__(self, text: str) -> list[dict]:
        start_schema = tuple(self.schema)
        time.sleep(0.02)
        end_schema = tuple(self.schema)
        with self._calls_lock:
            self.schema_pairs.append((start_schema, end_schema))

        result: dict[str, list[dict]] = {}
        for label in end_schema:
            if label in self.output_by_label:
                result[label] = self.output_by_label[label]
        return [result]


class CountingTokenizer:
    def __call__(self, text: str, **kwargs) -> dict:
        return {"input_ids": list(range(len(text)))}


class FakeTokenTaskInstance:
    def __init__(
        self,
        max_seq_len: int = 512,
        summary_num: int = 0,
        summary_token_num: int = 4,
    ) -> None:
        self._max_seq_len = max_seq_len
        self._tokenizer = CountingTokenizer()
        self.summary_num = summary_num
        self._summary_token_num = summary_token_num


class FakeTokenTaskflow:
    def __init__(
        self,
        max_seq_len: int = 512,
        summary_num: int = 0,
        summary_token_num: int = 4,
    ) -> None:
        self.task_instance = FakeTokenTaskInstance(
            max_seq_len=max_seq_len,
            summary_num=summary_num,
            summary_token_num=summary_token_num,
        )


class FindingUIETaskflow:
    def __init__(
        self,
        token_by_label: dict[str, str],
        max_seq_len: int = 512,
        summary_token_num: int = 4,
    ) -> None:
        self.token_by_label = token_by_label
        self.task_instance = FakeTokenTaskInstance(
            max_seq_len=max_seq_len,
            summary_token_num=summary_token_num,
        )
        self.schema: list[str] = []
        self.calls: list[str] = []

    def set_schema(self, schema: list[str]) -> None:
        self.schema = list(schema)

    def __call__(self, text: str | list[str]) -> list[dict]:
        if isinstance(text, list):
            return [self._build_result(item) for item in text]
        return [self._build_result(text)]

    def _build_result(self, text: str) -> dict:
        self.calls.append(text)
        result: dict[str, list[dict]] = {}
        for label in self.schema:
            token = self.token_by_label.get(label)
            if token and token in text:
                result[label] = [{"text": token, "probability": 0.98}]
        return result


class EmptyModelClient:
    using_taskflow = False
    using_uie = False

    def infer(self, text: str, tasks: dict) -> list[ModelSpan]:
        return []


class FindingModelClient:
    using_taskflow = False
    using_uie = True

    def __init__(self, token_by_label: dict[str, str]) -> None:
        self.token_by_label = token_by_label

    def infer(self, text: str, tasks: dict) -> list[ModelSpan]:
        spans: list[ModelSpan] = []
        for label in tasks.get("uie_schema", []):
            token = self.token_by_label.get(label)
            if not token:
                continue
            start = text.find(token)
            if start >= 0:
                spans.append(
                    ModelSpan(label, token, start, start + len(token), "uie", 0.98)
                )
        return spans


class StaticModelClient:
    using_taskflow = True
    using_uie = True

    def __init__(self, spans: list[ModelSpan]) -> None:
        self.spans = spans

    def infer(self, text: str, tasks: dict) -> list[ModelSpan]:
        spans: list[ModelSpan] = []
        uie_schema = set(tasks.get("uie_schema", []))
        for span in self.spans:
            if span.source == "wordtag" and not tasks.get("wordtag", True):
                continue
            if span.source == "uie" and span.label not in uie_schema:
                continue
            if text[span.start : span.end] == span.text:
                spans.append(span)
        return spans


class RecordingModelClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def infer(self, text: str, tasks: dict) -> list[ModelSpan]:
        self.calls.append(dict(tasks))
        return []


class StubHTTPRecognizerClient(HTTPRecognizerClient):
    def __init__(self, response_data: dict) -> None:
        self.response_data = response_data

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict:
        return self.response_data


class FakeHealthRecognizer:
    using_taskflow = True
    using_uie = True

    def device_info(self) -> dict:
        return {
            "device": "nvidia",
            "device_id": 1,
            "taskflow_device_id": 1,
            "gpu_available": True,
            "gpu_device_count": 2,
            "gpu_compiled_with_cuda": True,
            "gpu_error": "",
        }


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


def build_mobile_only_recognizer() -> ApplicationEntityRecognizer:
    return ApplicationEntityRecognizer(EmptyModelClient())


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

    def test_preprocess_mapping_is_request_scoped_under_concurrent_calls(self) -> None:
        def run_request(messages: list[dict]) -> dict:
            return self.service.prepare_llm_request(
                {
                    "llm_request": {
                        "model": "demo",
                        "messages": messages,
                    }
                }
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_with_history = executor.submit(
                run_request,
                [
                    {
                        "role": "user",
                        "content": "历史联系人[[PERSON_009]]。",
                        "encrypted": True,
                        "mapping": {"PERSON": {"张三": "[[PERSON_009]]"}},
                    },
                    {
                        "role": "user",
                        "content": "本轮联系人张三，手机号13800138000。",
                        "encrypted": False,
                    },
                ],
            )
            future_without_history = executor.submit(
                run_request,
                [
                    {
                        "role": "user",
                        "content": "另一个会话联系人张三，手机号13800138000。",
                        "encrypted": False,
                    }
                ],
            )

        result_with_history = future_with_history.result()
        result_without_history = future_without_history.result()

        self.assertEqual(
            result_with_history["mapping"]["PERSON"]["张三"],
            "[[PERSON_009]]",
        )
        self.assertEqual(
            result_without_history["mapping"]["PERSON"]["张三"],
            "[[PERSON_001]]",
        )
        self.assertEqual(
            result_with_history["desensitized_request"]["messages"][1]["content"],
            "本轮联系人[[PERSON_009]]，手机号[[MOBILE_001]]。",
        )
        self.assertEqual(
            result_without_history["desensitized_request"]["messages"][0]["content"],
            "另一个会话联系人[[PERSON_001]]，手机号[[MOBILE_001]]。",
        )
        self.assertEqual(result_with_history["stats"]["processed_message_indexes"], [1])
        self.assertEqual(
            result_without_history["stats"]["processed_message_indexes"],
            [0],
        )

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

    def test_new_mapping_replaces_repeated_entities_in_same_message(self) -> None:
        id_card = "110101199003071234"
        bank_card = "6222020202020202020"

        class FirstOccurrenceRecognizer:
            def recognize(
                self,
                text: str,
                custom_entities: list[dict],
            ) -> list[EntitySpan]:
                spans: list[EntitySpan] = []
                for entity_type, value in (
                    ("ID_CARD", id_card),
                    ("BANK_CARD", bank_card),
                ):
                    start = text.find(value)
                    if start >= 0:
                        spans.append(
                            EntitySpan(
                                entity_type,
                                value,
                                start,
                                start + len(value),
                                "custom",
                            )
                        )
                return spans

        self.service.recognizer = FirstOccurrenceRecognizer()
        content = (
            f"第一处身份证号{id_card}，银行卡号{bank_card}。"
            f"第二处身份证号{id_card}，银行卡号{bank_card}。"
        )

        result = self.preprocess(
            [{"role": "user", "content": content, "encrypted": False}]
        )

        masked = result["desensitized_request"]["messages"][0]["content"]
        self.assertNotIn(id_card, masked)
        self.assertNotIn(bank_card, masked)
        self.assertEqual(masked.count("[[ID_CARD_001]]"), 2)
        self.assertEqual(masked.count("[[BANK_CARD_001]]"), 2)
        self.assertEqual(result["stats"]["replacements"], 4)
        self.assertEqual(
            result["mapping"]["ID_CARD"][id_card],
            "[[ID_CARD_001]]",
        )
        self.assertEqual(
            result["mapping"]["BANK_CARD"][bank_card],
            "[[BANK_CARD_001]]",
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
            for match in ApplicationEntityRecognizer.MOBILE_PATTERN.finditer(text)
        ]

        self.assertEqual(matches, ["13800138000"])

    def test_model_infer_wordtag_returns_raw_labels_without_mobile_regex(self) -> None:
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer._taskflow = lambda content: [("张三", "人名")]
        recognizer.strict_local_model = True

        spans = recognizer.infer(
            "联系人张三，手机号13800138000。",
            {"wordtag": True, "uie_schema": []},
        )

        self.assertEqual(
            [(span.label, span.text, span.source) for span in spans],
            [("人名", "张三", "wordtag")],
        )
        self.assertFalse(any(span.text == "13800138000" for span in spans))

    def test_model_infer_wordtag_serializes_taskflow_calls(self) -> None:
        fake_wordtag = SlowWordtagTaskflow()
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer._taskflow = fake_wordtag
        recognizer._taskflow_lock = threading.RLock()
        recognizer.strict_local_model = True

        with ThreadPoolExecutor(max_workers=2) as executor:
            zhang_future = executor.submit(recognizer.infer_wordtag, "联系人张三。")
            li_future = executor.submit(recognizer.infer_wordtag, "联系人李四。")

        zhang_spans = zhang_future.result()
        li_spans = li_future.result()

        self.assertEqual(
            [(span.label, span.text) for span in zhang_spans],
            [("人名", "张三")],
        )
        self.assertEqual(
            [(span.label, span.text) for span in li_spans],
            [("人名", "李四")],
        )
        self.assertEqual(fake_wordtag.max_active_calls, 1)

    def test_model_infer_uie_accepts_schema_without_business_entity_type(self) -> None:
        fake_uie = FakeUIETaskflow(
            {
                "身份证号": [
                    {
                        "text": "110101199003071234",
                        "start": 6,
                        "end": 24,
                        "probability": 0.98,
                    }
                ]
            }
        )
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer.strict_uie_model = True
        recognizer._uie_taskflow = fake_uie
        recognizer._uie_schema = ()
        recognizer._ensure_uie_taskflow = lambda schema: (
            fake_uie.set_schema(schema) or fake_uie
        )

        spans = recognizer.infer(
            "客户身份证号110101199003071234。",
            {"wordtag": False, "uie_schema": ["身份证号"]},
        )

        self.assertEqual(fake_uie.schema, ["身份证号"])
        self.assertEqual(spans[0].label, "身份证号")
        self.assertEqual(spans[0].source, "uie")
        self.assertEqual(spans[0].probability, 0.98)
        self.assertFalse(hasattr(spans[0], "entity_type"))

    def test_model_infer_uie_serializes_schema_switch_and_inference(self) -> None:
        fake_uie = SlowSchemaAwareUIETaskflow(
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
                        "start": 6,
                        "end": 25,
                        "probability": 0.97,
                    }
                ],
            }
        )
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer.enable_uie_custom = True
        recognizer.strict_uie_model = True
        recognizer._uie_taskflow = fake_uie
        recognizer._uie_schema = ()
        recognizer._uie_lock = threading.RLock()

        def infer_label(text: str, label: str) -> list[ModelSpan]:
            return recognizer.infer(
                text,
                {"wordtag": False, "uie_schema": [label]},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            id_future = executor.submit(
                infer_label,
                "客户身份证号110101199003071234。",
                "身份证号",
            )
            bank_future = executor.submit(
                infer_label,
                "客户银行卡号6222020202020202020。",
                "银行卡号",
            )

        id_spans = id_future.result()
        bank_spans = bank_future.result()

        self.assertEqual(
            [(span.label, span.text) for span in id_spans],
            [("身份证号", "110101199003071234")],
        )
        self.assertEqual(
            [(span.label, span.text) for span in bank_spans],
            [("银行卡号", "6222020202020202020")],
        )
        self.assertEqual(
            fake_uie.schema_pairs,
            [(("身份证号",), ("身份证号",)), (("银行卡号",), ("银行卡号",))],
        )

    def test_token_chunks_batch_short_semantic_units(self) -> None:
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        taskflow = FakeTokenTaskflow(max_seq_len=64)
        text = "one!two?three;four."

        chunks = list(recognizer._iter_token_chunks(text, taskflow))

        self.assertEqual(chunks, [(0, text)])

    def test_semantic_units_keep_consecutive_boundaries_together(self) -> None:
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)

        units = list(recognizer._iter_semantic_units("first.\r\nsecond."))

        self.assertEqual(units, [(0, "first.\r\n"), (8, "second.")])

    def test_uie_chunks_ignore_compat_target_window_for_throughput(self) -> None:
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer.uie_target_text_tokens = 12
        taskflow = FakeTokenTaskflow(max_seq_len=64)
        text = "aaaa!bbbb!cccc!dddd!"

        chunks = list(recognizer._iter_uie_token_chunks(text, taskflow, ["label"]))

        self.assertEqual(chunks, [(0, text)])

    def test_uie_chunks_follow_max_token_budget(self) -> None:
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        taskflow = FakeTokenTaskflow(max_seq_len=512)
        text = (("a" * 50) + "!") * 5

        chunks = list(recognizer._iter_uie_token_chunks(text, taskflow, ["label"]))

        self.assertEqual(chunks, [(0, text)])

    def test_token_chunks_stay_within_budget_when_batching(self) -> None:
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        taskflow = FakeTokenTaskflow(max_seq_len=13)
        text = "aaaa!bbbb!cccc!"

        chunks = list(recognizer._iter_token_chunks(text, taskflow))

        self.assertEqual(chunks, [(0, "aaaa!bbbb!"), (10, "cccc!")])
        for _, chunk in chunks:
            self.assertLessEqual(len(chunk), 12)

    def test_uie_token_chunks_reserve_prompt_budget(self) -> None:
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        taskflow = FakeTokenTaskflow(max_seq_len=20, summary_token_num=2)
        text = "aaaaa!bbbbb!ccccc!"

        chunks = list(recognizer._iter_uie_token_chunks(text, taskflow, ["label"]))
        shared_chunks = list(
            recognizer._iter_token_chunks(text, taskflow, prompts=["label"])
        )

        self.assertEqual(chunks, shared_chunks)
        self.assertEqual(chunks, [(0, "aaaaa!bbbbb!"), (12, "ccccc!")])
        for _, chunk in chunks:
            self.assertLessEqual(len(chunk), 13)

    def test_token_chunks_avoid_ascii_identifier_boundary_when_possible(self) -> None:
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        taskflow = FakeTokenTaskflow(max_seq_len=16)
        text = "prefix 622202020202 tail;"

        chunks = list(recognizer._iter_token_chunks(text, taskflow))

        self.assertEqual(chunks, [(0, "prefix "), (7, "622202020202 "), (20, "tail;")])
        for offset, chunk in chunks[:-1]:
            end = offset + len(chunk)
            self.assertFalse(
                text[end - 1].isascii()
                and text[end - 1].isalnum()
                and text[end].isascii()
                and text[end].isalnum()
            )

    def test_uie_chunk_spans_are_offset_to_original_text(self) -> None:
        token = "622202020202"
        text = f"prefix {token} tail;"
        fake_uie = FindingUIETaskflow(
            {"bank": token},
            max_seq_len=22,
            summary_token_num=1,
        )
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer.strict_uie_model = True
        recognizer._ensure_uie_taskflow = lambda schema: (
            fake_uie.set_schema(schema) or fake_uie
        )

        spans = recognizer._extract_with_uie(text, ["bank"])

        self.assertEqual(
            [(span.label, span.text, span.start, span.end) for span in spans],
            [("bank", token, text.find(token), text.find(token) + len(token))],
        )
        self.assertGreater(len(fake_uie.calls), 1)

    def test_cpu_device_uses_taskflow_cpu_id(self) -> None:
        recognizer = LocalEntityRecognizer(
            model_path=Path("resources/models/wordtag"),
            device="cpu",
            device_id=3,
            enable_taskflow=False,
            enable_uie_custom=False,
            strict_local_model=False,
        )

        self.assertEqual(recognizer.device, "cpu")
        self.assertEqual(recognizer.device_id, 3)
        self.assertEqual(recognizer.taskflow_device_id, -1)
        self.assertFalse(recognizer.gpu_available)

    def test_nvidia_device_uses_configured_gpu_id_when_available(self) -> None:
        status = {
            "available": True,
            "device_count": 2,
            "compiled_with_cuda": True,
            "error": "",
        }

        with patch.object(
            LocalEntityRecognizer,
            "_detect_nvidia_status",
            return_value=status,
        ):
            recognizer = LocalEntityRecognizer(
                model_path=Path("resources/models/wordtag"),
                device="nvidia",
                device_id=1,
                enable_taskflow=False,
                enable_uie_custom=False,
                strict_local_model=False,
            )

        self.assertEqual(recognizer.device, "nvidia")
        self.assertEqual(recognizer.taskflow_device_id, 1)
        self.assertTrue(recognizer.gpu_available)
        self.assertEqual(recognizer.gpu_device_count, 2)

    def test_nvidia_device_fails_when_paddle_is_not_cuda_build(self) -> None:
        status = {
            "available": False,
            "device_count": 0,
            "compiled_with_cuda": False,
            "error": "paddle is cpu only",
        }

        with patch.object(
            LocalEntityRecognizer,
            "_detect_nvidia_status",
            return_value=status,
        ):
            with self.assertRaisesRegex(RuntimeError, "paddlepaddle-gpu"):
                LocalEntityRecognizer(
                    model_path=Path("resources/models/wordtag"),
                    device="nvidia",
                    enable_taskflow=False,
                    enable_uie_custom=False,
                    strict_local_model=False,
                )

    def test_nvidia_device_fails_when_no_gpu_is_visible(self) -> None:
        status = {
            "available": False,
            "device_count": 0,
            "compiled_with_cuda": True,
            "error": "",
        }

        with patch.object(
            LocalEntityRecognizer,
            "_detect_nvidia_status",
            return_value=status,
        ):
            with self.assertRaisesRegex(RuntimeError, "at least one visible"):
                LocalEntityRecognizer(
                    model_path=Path("resources/models/wordtag"),
                    device="nvidia",
                    enable_taskflow=False,
                    enable_uie_custom=False,
                    strict_local_model=False,
                )

    def test_nvidia_device_fails_when_device_id_is_out_of_range(self) -> None:
        status = {
            "available": True,
            "device_count": 1,
            "compiled_with_cuda": True,
            "error": "",
        }

        with patch.object(
            LocalEntityRecognizer,
            "_detect_nvidia_status",
            return_value=status,
        ):
            with self.assertRaisesRegex(RuntimeError, "out of range"):
                LocalEntityRecognizer(
                    model_path=Path("resources/models/wordtag"),
                    device="nvidia",
                    device_id=2,
                    enable_taskflow=False,
                    enable_uie_custom=False,
                    strict_local_model=False,
                )

    def test_taskflow_creation_receives_resolved_device_id(self) -> None:
        calls: list[dict] = []

        class CapturingTaskflow:
            def __init__(self, *args, **kwargs) -> None:
                calls.append(kwargs)

        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer.taskflow_device_id = -1
        recognizer._create_wordtag_taskflow(CapturingTaskflow)

        recognizer.taskflow_device_id = 1
        recognizer.uie_model_name = "uie-base"
        recognizer.uie_position_prob = 0.5
        recognizer._create_uie_taskflow(CapturingTaskflow, schema=["身份证号"])

        self.assertEqual(calls[0]["device_id"], -1)
        self.assertEqual(calls[1]["device_id"], 1)

    def test_model_healthz_returns_device_status(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DESENSITIZE_DEVICE": "cpu",
                "DESENSITIZE_ENABLE_TASKFLOW": "false",
                "DESENSITIZE_ENABLE_UIE_CUSTOM": "false",
                "DESENSITIZE_PRELOAD_UIE_CUSTOM": "false",
                "DESENSITIZE_STRICT_LOCAL_MODEL": "false",
            },
        ):
            from model_app import create_model_app

        config = ServiceConfig(
            model_path=Path("resources/models/wordtag"),
            device="nvidia",
            device_id=1,
            enable_taskflow=False,
            enable_uie_custom=False,
            preload_uie_custom=False,
            strict_local_model=False,
        )
        app = create_model_app(config=config, recognizer=FakeHealthRecognizer())

        response = app.test_client().get("/healthz")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["device"], "nvidia")
        self.assertEqual(data["device_id"], 1)
        self.assertEqual(data["taskflow_device_id"], 1)
        self.assertEqual(data["max_model_tokens"], 512)
        self.assertEqual(data["uie_target_text_tokens"], 512)
        self.assertTrue(data["gpu_available"])
        self.assertEqual(data["gpu_device_count"], 2)
        self.assertTrue(data["gpu_compiled_with_cuda"])

    def test_config_reads_and_validates_device(self) -> None:
        with patch.dict(
            os.environ,
            {"DESENSITIZE_DEVICE": "nvidia", "DESENSITIZE_DEVICE_ID": "2"},
        ):
            config = ServiceConfig.from_env()

        self.assertEqual(config.device, "nvidia")
        self.assertEqual(config.device_id, 2)

        with patch.dict(os.environ, {"DESENSITIZE_DEVICE": "amd"}):
            with self.assertRaisesRegex(ValueError, "cpu, nvidia"):
                ServiceConfig.from_env()

    def test_config_reads_token_window_settings(self) -> None:
        config = ServiceConfig(model_path=Path("resources/models/wordtag"))

        self.assertEqual(config.max_model_tokens, 512)
        self.assertEqual(config.uie_target_text_tokens, 512)

        with patch.dict(
            os.environ,
            {
                "DESENSITIZE_MAX_MODEL_TOKENS": "384",
                "DESENSITIZE_UIE_TARGET_TEXT_TOKENS": "256",
            },
        ):
            config = ServiceConfig.from_env()

        self.assertEqual(config.max_model_tokens, 384)
        self.assertEqual(config.uie_target_text_tokens, 256)

    def test_downloaded_model_sync_is_disabled_by_default(self) -> None:
        config = ServiceConfig(model_path=Path("resources/models/wordtag"))

        self.assertFalse(config.sync_downloaded_model)

    def test_uie_preload_is_disabled_by_default(self) -> None:
        config = ServiceConfig(model_path=Path("resources/models/wordtag"))

        self.assertFalse(config.preload_uie_custom)

    def test_downloaded_model_sync_respects_flag(self) -> None:
        recognizer = LocalEntityRecognizer.__new__(LocalEntityRecognizer)
        recognizer.downloaded_model_cache_path = Path("cache/wordtag")
        recognizer.model_path = Path("resources/models/wordtag")
        calls: list[tuple[Path, Path]] = []
        recognizer._sync_model_directory = lambda source, target: calls.append(
            (source, target)
        )

        recognizer.sync_downloaded_model = False
        recognizer._sync_downloaded_model_to_local()
        self.assertEqual(calls, [])

        recognizer.sync_downloaded_model = True
        recognizer._sync_downloaded_model_to_local()
        self.assertEqual(
            calls,
            [(Path("cache/wordtag"), Path("resources/models/wordtag"))],
        )

    def test_application_maps_wordtag_labels_to_business_entities(self) -> None:
        text = "联系人张三来自上海泛微网络科技股份有限公司。"
        person = "张三"
        org = "上海泛微网络科技股份有限公司"
        person_start = text.find(person)
        org_start = text.find(org)
        recognizer = ApplicationEntityRecognizer(
            StaticModelClient(
                [
                    ModelSpan(
                        "人名",
                        person,
                        person_start,
                        person_start + len(person),
                        "wordtag",
                    ),
                    ModelSpan(
                        "组织机构",
                        org,
                        org_start,
                        org_start + len(org),
                        "wordtag",
                    ),
                ]
            )
        )

        spans = recognizer.recognize(text, [])

        self.assertEqual(
            [(span.entity_type, span.text, span.source) for span in spans],
            [
                ("PERSON", "张三", "model"),
                ("ORG", "上海泛微网络科技股份有限公司", "model"),
            ],
        )

    def test_long_builtin_text_is_recognized_without_api_chunking(self) -> None:
        self.service.recognizer = build_mobile_only_recognizer()
        text = "说明：" + ("甲" * 520) + "手机号13800138000。"

        result = self.preprocess(
            [{"role": "user", "content": text, "encrypted": False}]
        )

        masked = result["desensitized_request"]["messages"][0]["content"]
        self.assertIn("手机号[[MOBILE_001]]", masked)
        self.assertNotIn("13800138000", masked)
        self.assertEqual(result["mapping"]["MOBILE"]["13800138000"], "[[MOBILE_001]]")

    def test_long_text_entity_keeps_original_coordinates(self) -> None:
        recognizer = build_mobile_only_recognizer()
        text = ("A" * 58) + "13800138000" + "完成"

        spans = recognizer.recognize(text, [])

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].start, 58)
        self.assertEqual(spans[0].end, 69)
        self.assertEqual(text[spans[0].start : spans[0].end], "13800138000")

    def test_long_text_does_not_create_duplicate_mapping(self) -> None:
        self.service.recognizer = build_mobile_only_recognizer()
        text = ("A" * 40) + "13800138000" + ("B" * 50)

        result = self.preprocess(
            [{"role": "user", "content": text, "encrypted": False}]
        )

        masked = result["desensitized_request"]["messages"][0]["content"]
        self.assertEqual(masked.count("[[MOBILE_001]]"), 1)
        self.assertEqual(result["stats"]["replacements"], 1)
        self.assertEqual(
            result["mapping"]["MOBILE"],
            {"13800138000": "[[MOBILE_001]]"},
        )

    def test_long_custom_uie_text_is_recognized_without_api_chunking(self) -> None:
        id_card = "110101199003071234"
        bank_card = "6222020202020202020"
        self.service.recognizer = ApplicationEntityRecognizer(
            FindingModelClient(
                {
                    "身份证号": id_card,
                    "银行卡号": bank_card,
                }
            )
        )
        text = (
            "客户资料："
            + ("甲" * 520)
            + f"身份证号{id_card}，银行卡号{bank_card}。"
        )

        result = self.preprocess(
            [{"role": "user", "content": text, "encrypted": False}],
            custom_entities=[
                {"entity_type": "ID_CARD", "uie_schema": ["身份证号"]},
                {"entity_type": "BANK_CARD", "uie_schema": ["银行卡号"]},
            ],
        )

        masked = result["desensitized_request"]["messages"][0]["content"]
        self.assertIn("身份证号[[ID_CARD_001]]", masked)
        self.assertIn("银行卡号[[BANK_CARD_001]]", masked)
        self.assertNotIn(id_card, masked)
        self.assertNotIn(bank_card, masked)
        self.assertEqual(result["mapping"]["ID_CARD"][id_card], "[[ID_CARD_001]]")
        self.assertEqual(
            result["mapping"]["BANK_CARD"][bank_card],
            "[[BANK_CARD_001]]",
        )

    def test_custom_model_labels_are_ignored_by_uie_custom_path(self) -> None:
        model_client = RecordingModelClient()
        recognizer = ApplicationEntityRecognizer(model_client)

        spans = recognizer.recognize(
            "联系人张三",
            [{"entity_type": "PERSON", "model_labels": ["人名"]}],
        )

        self.assertEqual(spans, [])
        self.assertEqual(model_client.calls[0]["uie_schema"], [])

    def test_custom_uie_address_filters_field_word_and_keeps_real_address(self) -> None:
        recognizer = ApplicationEntityRecognizer(
            StaticModelClient(
                [
                    ModelSpan("地址", "地址", 0, 2, "uie", 0.99),
                    ModelSpan("地址", "北京市海淀区中关村大街27号", 3, 17, "uie", 0.98),
                ]
            )
        )

        spans = recognizer.recognize(
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
        recognizer = ApplicationEntityRecognizer(
            StaticModelClient(
                [
                    ModelSpan("身份证号", "110101199003071234", 6, 24, "uie", 0.98),
                    ModelSpan("银行卡号", "6222020202020202020", 29, 48, "uie", 0.97),
                ]
            )
        )

        spans = recognizer.recognize(
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

    def test_custom_uie_rejects_embedded_id_and_bank_card_fragments(self) -> None:
        id_card = "110101199003071234"
        bank_card = "6222020202020202020"
        id_fragment = "10119900307123"
        bank_fragment = "020202020202020"
        text = f"id {id_card}; bank {bank_card}."
        id_start = text.find(id_fragment)
        bank_start = text.find(bank_fragment)
        recognizer = ApplicationEntityRecognizer(
            StaticModelClient(
                [
                    ModelSpan(
                        "id",
                        id_fragment,
                        id_start,
                        id_start + len(id_fragment),
                        "uie",
                        0.98,
                    ),
                    ModelSpan(
                        "bank",
                        bank_fragment,
                        bank_start,
                        bank_start + len(bank_fragment),
                        "uie",
                        0.97,
                    ),
                ]
            )
        )

        spans = recognizer.recognize(
            text,
            [
                {"entity_type": "ID_CARD", "uie_schema": ["id"]},
                {"entity_type": "BANK_CARD", "uie_schema": ["bank"]},
            ],
        )

        self.assertEqual(spans, [])

    def test_custom_uie_prefers_exact_label_when_schema_names_overlap(self) -> None:
        bank_card = "6222020202020202020"
        text = f"客户银行卡号{bank_card}。"
        recognizer = ApplicationEntityRecognizer(
            StaticModelClient(
                [
                    ModelSpan(
                        "银行卡号",
                        bank_card,
                        text.find(bank_card),
                        text.find(bank_card) + len(bank_card),
                        "uie",
                        0.97,
                    ),
                ]
            )
        )

        spans = recognizer.recognize(
            text,
            [
                {"entity_type": "CARD", "uie_schema": ["卡号"]},
                {"entity_type": "BANK_CARD", "uie_schema": ["银行卡号"]},
            ],
        )

        self.assertEqual(
            [(span.entity_type, span.text) for span in spans],
            [("BANK_CARD", bank_card)],
        )

    def test_remote_client_rejects_invalid_spans_as_model_service_error(self) -> None:
        client = StubHTTPRecognizerClient(
            {
                "spans": [
                    {
                        "label": "身份证号",
                        "text": "110101199003071234",
                        "start": "bad",
                        "end": 24,
                        "source": "uie",
                    }
                ]
            }
        )

        with self.assertRaises(RemoteRecognizerError):
            client.infer("客户身份证号110101199003071234。")

    def test_remote_client_requires_spans_to_be_a_list(self) -> None:
        client = StubHTTPRecognizerClient({"spans": "bad"})

        with self.assertRaises(RemoteRecognizerError):
            client.infer("任意文本")

    def test_preprocess_masks_uie_custom_id_card_and_bank_card(self) -> None:
        text = "客户身份证号110101199003071234，银行卡号6222020202020202020。"
        self.service.recognizer = ApplicationEntityRecognizer(
            StaticModelClient(
                [
                    ModelSpan("身份证号", "110101199003071234", 6, 24, "uie", 0.98),
                    ModelSpan("银行卡号", "6222020202020202020", 29, 48, "uie", 0.97),
                ]
            )
        )

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
