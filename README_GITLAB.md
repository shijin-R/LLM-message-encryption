# 大模型请求脱敏服务

这是一个独立的 HTTP 服务，用于在业务请求发送给大模型前，对 `messages` 中的敏感实体做脱敏替换，并返回脱敏后的请求体与原文映射字典。

## 功能范围

- 处理 `llm_request.messages` 中未标记为已脱敏的 `user` 消息。
- 将姓名、组织/公司、手机号替换为结构化占位符，例如 `[[PERSON_001]]`、`[[ORG_001]]`、`[[MOBILE_001]]`。
- 返回完整 `mapping`，由调用方自行保存并在后续请求中带回，以复用同一套占位符。
- 默认使用本地 PaddleNLP `Taskflow("ner", entity_only=True)` 的 wordtag/accurate 模型；手机号使用正则补充识别。

本服务不保存会话状态，不持久化映射字典，不转发大模型请求，也不提供大模型响应还原接口。

## 环境准备

建议使用 Python 3.10+。安装依赖：

```powershell
python -m pip install -r requirements.txt
```

依赖版本中 `paddlenlp==2.8.1` 需要 `aistudio-sdk==0.2.6`，不要随意升级该包。

## 模型文件

默认模型目录：

```text
resources/models/wordtag
```

模型文件体积较大，不建议提交到代码仓库。部署环境应通过内网制品库、共享目录或部署脚本准备模型文件。目录中至少需要包含 PaddleNLP wordtag 推理所需文件，例如：

- `model_state.pdparams`
- `config.json`
- `vocab.txt`
- `static/`

相关环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DESENSITIZE_MODEL_PATH` | `resources/models/wordtag` | 本地模型目录 |
| `DESENSITIZE_DICT_DIR` | `resources/common_data/uie` | jieba 用户词典目录 |
| `DESENSITIZE_ENABLE_TASKFLOW` | `true` | 是否启用 Taskflow |
| `DESENSITIZE_STRICT_LOCAL_MODEL` | `true` | 模型不可用时是否启动失败 |
| `DESENSITIZE_AUTO_DOWNLOAD_MODEL` | `true` | 本地模型缺失时是否尝试自动下载 |
| `DESENSITIZE_SYNC_DOWNLOADED_MODEL` | `true` | 自动下载后是否同步回本地模型目录 |

## 启动服务

```powershell
python app.py
```

默认监听：

```text
http://127.0.0.1:18001
```

健康检查：

```powershell
Invoke-RestMethod "http://127.0.0.1:18001/healthz" | ConvertTo-Json -Depth 10
```

## 接口

```text
POST /v1/llm/preprocess
```

请求体格式：

```json
{
  "llm_request": {
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "合同甲方：上海泛微网络科技股份有限公司，联系人张三，手机号13800138000。",
        "encrypted": false
      }
    ]
  }
}
```

关键字段：

- `llm_request`：原始大模型请求体，服务会保留其中除 `messages` 外的其他字段。
- `messages[].encrypted`：`true` 表示该条 user 消息已经脱敏，本次跳过；`false` 或缺失表示需要本次脱敏。
- `messages[].mapping`：历史已脱敏 user 消息携带的映射字典，用于复用占位符。
- `custom_entities`：可选，声明需要本地模型额外关注的实体标签。

返回结果中的 `data.desensitized_request` 可继续发送给上游大模型，`data.mapping` 应由调用方保存，供下一轮对话复用。

## 示例请求

项目提供两个示例 JSON，便于联调和理解历史映射复用流程。

`example_preprocess_request.json` 演示普通预处理请求：

- 包含一条已脱敏的历史 user 消息和一条新的未脱敏 user 消息。
- 服务会复用历史 `mapping`，把新消息中的相同实体替换为已有占位符。
- 适合验证 `encrypted=false` 的 user 消息是否会被正确脱敏。

调用示例：

```powershell
$body = Get-Content .\example_preprocess_request.json -Raw -Encoding UTF8

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:18001/v1/llm/preprocess" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body | ConvertTo-Json -Depth 20
```

`example_history_reuse_request.json` 演示多轮对话中的历史映射复用：

- 历史 user 消息携带上一次保存的 `mapping`。
- 新 user 消息再次出现相同姓名、组织时，会继续使用原来的占位符。
- 新出现的手机号会分配新的占位符，例如 `[[MOBILE_002]]`。

调用示例：

```powershell
$body = Get-Content .\example_history_reuse_request.json -Raw -Encoding UTF8

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:18001/v1/llm/preprocess" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body | ConvertTo-Json -Depth 20
```

业务方实际接入时，建议每次请求后保存接口返回的 `data.mapping`；下一轮请求中，把该映射放到对应的历史 user message 的 `mapping` 字段，并将该历史消息标记为 `encrypted=true`。

成功响应格式：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "desensitized_request": {},
    "mapping": {},
    "stats": {}
  }
}
```

## 测试

```powershell
python -m unittest discover -s tests
```
