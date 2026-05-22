# 大模型请求脱敏服务

## 项目用途

本项目提供一个独立的 HTTP 前置服务，用于在业务请求发送给大模型前，对 `llm_request.messages` 中未标记为已脱敏的 `user` 消息执行敏感实体替换，并返回可继续转发给上游模型的请求体。

服务会将姓名、组织/公司、手机号等实体替换为结构化占位符，例如 `[[PERSON_001]]`、`[[ORG_001]]`、`[[MOBILE_001]]`。调用方需要保存接口返回的 `mapping`，并在下一轮请求的历史 `user` 消息中带回，以复用同一套占位符。

本服务不保存会话状态，不持久化映射字典，不转发大模型请求，也不提供大模型响应还原接口。实体识别、本地调试和模型准备细节见 [本地开发与模型准备](docs/local-development.md)。

## 依赖前提

- Python 3.10。
- 依赖包见 `requirements.txt`，核心依赖包括 Flask、PaddlePaddle、PaddleNLP 和 jieba。
- 默认模型目录为 `resources/models/wordtag`；大模型文件不进入 Git 仓库，只保留目录说明文件。
- 部署环境建议使用 Docker/Linux；Windows 本地开发可使用虚拟环境。

关键环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | 本地启动监听地址；Dockerfile 内默认 `0.0.0.0` |
| `PORT` | `18001` | 服务端口 |
| `DESENSITIZE_MODEL_PATH` | `resources/models/wordtag` | 本地 wordtag 模型目录 |
| `DESENSITIZE_AUTO_DOWNLOAD_MODEL` | `true` | 本地模型缺失时是否尝试自动下载 |
| `DESENSITIZE_SYNC_DOWNLOADED_MODEL` | `true` | 自动下载后是否同步回本地模型目录 |
| `DESENSITIZE_ENABLE_JIEBA_FALLBACK` | `false` | 是否启用 jieba 人名/机构补漏 |

## 启动方式

本地启动：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -u app.py
```

默认监听：

```text
http://127.0.0.1:18001
```

健康检查：

```powershell
Invoke-RestMethod "http://127.0.0.1:18001/healthz" | ConvertTo-Json -Depth 10
```

Docker 启动：

```powershell
docker build -t llm-messages-encryptor:latest .
docker volume create llm_messages_encryptor_model

docker run --rm `
  -p 18001:18001 `
  -v llm_messages_encryptor_model:/app/resources/models/wordtag `
  --name llm-messages-encryptor `
  llm-messages-encryptor:latest
```

## 接口示例

接口地址：

```text
POST /v1/llm/preprocess
```

请求示例：

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
  },
  "custom_entities": [
    {
      "entity_type": "ADDRESS",
      "model_labels": ["住址", "地址"]
    }
  ]
}
```

字段说明：

- `llm_request`：原始大模型请求体，服务会保留其中除 `messages` 内部控制字段外的其他内容。
- `messages[].encrypted`：`true` 表示该条 `user` 消息已经脱敏，本次跳过；`false` 或缺失表示需要本次脱敏。
- `messages[].mapping`：历史已脱敏 `user` 消息携带的映射字典，仅在 `encrypted=true` 时用于复用占位符。
- `custom_entities`：可选，声明需要本地模型额外关注的实体标签；当前不会执行 `patterns`、`regex` 或 `values` 字符串匹配。

成功响应示例：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "desensitized_request": {
      "model": "gpt-4o-mini",
      "messages": [
        {
          "role": "user",
          "content": "合同甲方：[[ORG_001]]，联系人[[PERSON_001]]，手机号[[MOBILE_001]]。"
        }
      ]
    },
    "mapping": {
      "PERSON": {
        "张三": "[[PERSON_001]]"
      },
      "ORG": {
        "上海泛微网络科技股份有限公司": "[[ORG_001]]"
      },
      "MOBILE": {
        "13800138000": "[[MOBILE_001]]"
      }
    },
    "stats": {
      "total_messages": 1,
      "processed_messages": 1,
      "processed_message_indexes": [0],
      "replacements": 3,
      "new_entities": 3
    }
  }
}
```

联调时可直接使用仓库内示例文件：

```powershell
$body = Get-Content .\example_preprocess_request.json -Raw -Encoding UTF8

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:18001/v1/llm/preprocess" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body | ConvertTo-Json -Depth 20
```

## 部署注意事项

- 内网或离线环境需要提前准备 `resources/models/wordtag`，或挂载包含模型文件的 Docker volume；启动后通过 `/healthz` 确认 `using_taskflow=true`。
- `resources/models/wordtag` 下的模型权重、缓存和推理文件较大，仓库只保留 `README.md`，不要提交实际模型文件。
- 服务是无状态的，调用方必须保存 `data.mapping`，并在后续历史 `user` 消息中携带该映射和 `encrypted=true` 标记。
- 返回的 `data.desensitized_request.messages` 会移除 `encrypted` 和 `mapping` 字段，可直接继续请求上游大模型。
- jieba 补漏默认关闭，只有在明确接受更高误报风险时再设置 `DESENSITIZE_ENABLE_JIEBA_FALLBACK=true`。
- 仓库文件保留/排除规则见 [仓库文件检查](docs/repository-files.md)。
