# Desensitize2 - 大模型请求脱敏服务

这是一个独立的 HTTP 服务，用于在业务请求发送给大模型前，对 `messages` 中的敏感实体做脱敏替换，并把**脱敏后的请求体**和**映射字典**返回给业务方。

## 服务边界

本服务只负责请求前脱敏：

```text
业务方原始大模型请求
  -> 脱敏服务
  -> 脱敏后的大模型请求体 + mapping 映射字典
```

本服务不做以下事情：

- 不保存会话状态。
- 不持久化映射字典。
- 不提供大模型响应还原接口。
- 不负责把请求转发给大模型。

业务方需要自行保存服务返回的 `mapping`，并在下一次请求时通过 `history_mappings` 传回服务，以便复用同一套占位符。

## 当前能力

- 请求脱敏：将 `messages` 中的敏感实体替换成结构化占位符。
- 映射回传：返回最新完整 `mapping`，业务方保存后可在下次请求中复用。
- 历史复用：相同实体在后续请求中继续使用历史占位符。
- 本地模型识别：强制使用 PaddleNLP `Taskflow("ner")` 的 accurate/wordtag 模型入口。
- 补充识别：手机号使用正则识别；jieba 用于补充人名、组织识别。

当前只考虑三类实体：

| 实体 | 类型 | 占位符示例 |
| --- | --- | --- |
| 姓名 | `PERSON` | `[[PERSON_001]]` |
| 组织/公司 | `ORG` | `[[ORG_001]]` |
| 手机号 | `MOBILE` | `[[MOBILE_001]]` |

## 接口一：大模型请求预处理

推荐业务方使用该接口。

```text
POST /v1/llm/preprocess
```

用途：接收原始大模型请求体，返回可以继续发给大模型的脱敏请求体。

### 请求字段

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `llm_request` | 是 | object | 原始大模型请求体，必须包含 `messages` |
| `llm_request.messages` | 是 | array | 待脱敏消息列表 |
| `history_mappings` | 否 | object | 历史映射字典，用于复用占位符 |
| `history_mapping` | 否 | object | `history_mappings` 的兼容别名，含义完全相同；新接入方不建议使用 |
| `custom_entities` | 否 | array | 业务方补充的固定值或正则规则 |
| `target_message_indexes` | 否 | array | 只处理指定下标的消息 |
| `desensitized_message_indexes` | 否 | array | 跳过已脱敏的消息下标 |

历史映射只应放在请求顶层。`messages` 应保持为原始大模型消息内容，不要在单条 message 内携带 `history_mappings` 或 `history_mapping`。`history_mappings` 和 `history_mapping` 不能同时传；如果需要兼容旧调用方，可单独使用 `history_mapping`。

请求示例见 [example_preprocess_request.json](example_preprocess_request.json)。

### 响应字段

接口统一返回：

```json
{
  "code": 0,
  "message": "ok",
  "data": {}
}
```

`data` 字段说明：

| 字段 | 说明 |
| --- | --- |
| `desensitized_request` | 脱敏后的大模型请求体，推荐业务方使用该字段继续调用大模型 |
| `upstream_request` | 可直接转发给上游大模型的请求体；当前与 `desensitized_request` 内容相同，保留用于兼容旧调用方 |
| `mapping` | 完整映射字典，业务方应保存该字段 |
| `stats` | 处理消息数、替换次数、新增实体数等统计信息 |

推荐新接入方读取 `desensitized_request`。`upstream_request` 是兼容字段，用来强调该请求体可以继续作为上游模型调用参数；当前两个字段内容一致，业务方选择其中一个使用即可。

### 响应示例

原始内容：

```text
合同甲方：上海泛微网络科技股份有限公司，联系人张三，手机号13800138000。
```

脱敏后：

```text
合同甲方：[[ORG_001]]，联系人[[PERSON_001]]，手机号[[MOBILE_001]]。
```

返回的 `mapping`：

```json
{
  "PERSON": {
    "张三": "[[PERSON_001]]"
  },
  "ORG": {
    "上海泛微网络科技股份有限公司": "[[ORG_001]]"
  },
  "MOBILE": {
    "13800138000": "[[MOBILE_001]]"
  }
}
```

## 接口二：基础 messages 脱敏

```text
POST /v1/desensitize
```

用途：直接接收 `messages`，返回脱敏后的 `messages`、`mapping` 和 `stats`。

请求示例见 [example_request.json](example_request.json)。

## 历史映射复用

服务本身不保存状态。业务方需要按下面方式复用历史映射。

### 第一次请求

服务返回：

```json
{
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
  }
}
```

业务方保存完整 `mapping`。

### 第二次请求

业务方把上次保存的 `mapping` 放到 `history_mappings`：

```json
{
  "llm_request": {
    "messages": [
      {
        "role": "user",
        "content": "请核对上海泛微网络科技股份有限公司的合同，联系人还是张三，新手机号15800158000。"
      }
    ]
  },
  "history_mappings": {
    "PERSON": {
      "张三": "[[PERSON_001]]"
    },
    "ORG": {
      "上海泛微网络科技股份有限公司": "[[ORG_001]]"
    },
    "MOBILE": {
      "13800138000": "[[MOBILE_001]]"
    }
  }
}
```

完整示例见 [example_history_reuse_request.json](example_history_reuse_request.json)。

脱敏结果：

```text
请核对[[ORG_001]]的合同，联系人还是[[PERSON_001]]，新手机号[[MOBILE_002]]。
```

可以看到：

- `上海泛微网络科技股份有限公司` 继续复用 `[[ORG_001]]`
- `张三` 继续复用 `[[PERSON_001]]`
- 新手机号 `15800158000` 新增为 `[[MOBILE_002]]`

## 自定义实体规则

`custom_entities` 用于补充模型识别结果，适合业务中结构化较强的文本，例如“联系人”“甲方”等上下文。

示例：

```json
{
  "entity_type": "ORG",
  "patterns": [
    {
      "regex": "甲方[：:\\s]*([^，。\\n]+)",
      "group": 1
    }
  ]
}
```

当前项目的业务范围仍建议只使用：

```text
PERSON / ORG / MOBILE
```

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DESENSITIZE_MODEL_PATH` | `resources/models/wordtag` | 本地 wordtag 模型目录 |
| `DESENSITIZE_DICT_DIR` | `resources/common_data/uie` | jieba 用户词典目录 |
| `DESENSITIZE_ENABLE_TASKFLOW` | `true` | 是否启用 Taskflow wordtag |
| `DESENSITIZE_STRICT_LOCAL_MODEL` | `true` | 模型不可用时是否启动失败 |
| `DESENSITIZE_AUTO_DOWNLOAD_MODEL` | `true` | 本地模型不可用时是否尝试下载默认模型 |
| `DESENSITIZE_SYNC_DOWNLOADED_MODEL` | `true` | 自动下载后是否同步回本地模型目录 |

注意：PaddleNLP 2.8.1 中没有 `Taskflow("wordtag")` 这个顶层任务名。当前代码使用的是：

```python
Taskflow("ner", entity_only=True)
```

该入口的 accurate 模式内部使用 wordtag 模型。

## 安装依赖

```powershell
cd D:\code\PythonProjects\Desensitize2
uv run python -m pip install -r requirements.txt
```

注意：`paddlenlp==2.8.1` 需要 `aistudio-sdk==0.2.6`。新版本 `aistudio-sdk` 缺少 PaddleNLP 2.8.1 依赖的 `aistudio_sdk.hub.download` 接口。

## 准备本地模型

当前演示脚本默认设置 `DESENSITIZE_AUTO_DOWNLOAD_MODEL=false`，因此运行前需要确认本地模型文件已经放在：

```text
resources/models/wordtag
```

至少应包含 `model_state.pdparams`、`config.json`、`vocab.txt` 和 `static` 推理文件等内容。可以用下面命令快速检查：

```powershell
Get-ChildItem .\resources\models\wordtag
```

如果本地模型缺失，服务会在严格模式下启动失败。项目的 `.gitignore` 默认忽略模型文件，提交代码时只保留模型目录说明，避免把大体积模型文件直接提交到代码仓库。

## 启动服务

```powershell
cd D:\code\PythonProjects\Desensitize2
uv run python app.py
```

服务默认监听：

```text
http://127.0.0.1:18001
```

## 确认本地模型已启用

访问健康检查接口：

```powershell
Invoke-RestMethod "http://127.0.0.1:18001/healthz" | ConvertTo-Json -Depth 10
```

重点检查：

```json
{
  "enable_taskflow": true,
  "strict_local_model": true,
  "using_taskflow": true,
  "model_path": "D:\\code\\PythonProjects\\Desensitize2\\resources\\models\\wordtag"
}
```

含义：

- `enable_taskflow=true`：配置启用了 Taskflow。
- `strict_local_model=true`：模型不可用时服务会启动失败。
- `using_taskflow=true`：当前进程已经成功初始化 Taskflow wordtag。
- `model_path`：当前加载的本地模型目录。

## 手动测试与演示

推荐按下面三轮演示，能够完整展示服务启动、基础脱敏和历史映射复用。

### 第一轮：启动服务并确认健康状态

直接运行脚本并保留服务进程：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_and_test.ps1 -KeepService
```

脚本会自动完成：

- 启动本地服务
- 检查 `/healthz`
- 调用一次 `/v1/llm/preprocess`
- 打印完整响应

如果需要单独查看健康状态：

```powershell
Invoke-RestMethod "http://127.0.0.1:18001/healthz" | ConvertTo-Json -Depth 10
```

重点关注：

- `status` 是否为 `ok`
- `enable_taskflow` 是否为 `true`
- `using_taskflow` 是否为 `true`
- `model_path` 是否指向本地 `resources/models/wordtag`

### 第二轮：展示基础脱敏效果

使用 [example_preprocess_request.json](example_preprocess_request.json) 演示普通请求脱敏：

```powershell
$body = Get-Content .\example_preprocess_request.json -Raw -Encoding UTF8

$resp = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:18001/v1/llm/preprocess" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body

$resp.data.desensitized_request.messages | ConvertTo-Json -Depth 10
$resp.data.mapping | ConvertTo-Json -Depth 10
$resp.data.stats | ConvertTo-Json -Depth 10
```

示例原文：

```text
合同甲方：上海泛微网络科技股份有限公司，联系人张三，手机号13800138000。
```

预期脱敏效果：

```text
合同甲方：[[ORG_001]]，联系人[[PERSON_001]]，手机号[[MOBILE_001]]。
```

这一轮重点展示三类实体都能被识别并替换：

| 原文 | 类型 | 脱敏后 |
| --- | --- | --- |
| 上海泛微网络科技股份有限公司 | `ORG` | `[[ORG_001]]` |
| 张三 | `PERSON` | `[[PERSON_001]]` |
| 13800138000 | `MOBILE` | `[[MOBILE_001]]` |

### 第三轮：展示历史映射复用

使用 [example_history_reuse_request.json](example_history_reuse_request.json) 演示第二次请求复用历史占位符：

```powershell
$body = Get-Content .\example_history_reuse_request.json -Raw -Encoding UTF8

$resp = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:18001/v1/llm/preprocess" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body

$resp.data.desensitized_request.messages[0].content
$resp.data.mapping | ConvertTo-Json -Depth 10
$resp.data.stats | ConvertTo-Json -Depth 10
```

示例原文：

```text
请核对上海泛微网络科技股份有限公司的合同，联系人还是张三，新手机号15800158000。
```

预期脱敏效果：

```text
请核对[[ORG_001]]的合同，联系人还是[[PERSON_001]]，新手机号[[MOBILE_002]]。
```

这一轮重点说明：

- `上海泛微网络科技股份有限公司` 来自 `history_mappings`，继续复用 `[[ORG_001]]`
- `张三` 来自 `history_mappings`，继续复用 `[[PERSON_001]]`
- `15800158000` 是本次新出现的手机号，所以生成 `[[MOBILE_002]]`

### 临时输入一段文本测试

当前服务没有纯文本接口。临时文本需要放到 `llm_request.messages[].content` 里：

```powershell
$payload = @{
  llm_request = @{
    model = "demo"
    messages = @(
      @{
        role = "user"
        content = "请联系张三，手机号13912345678，单位是北京测试科技有限公司。"
      }
    )
  }
} | ConvertTo-Json -Depth 20

$resp = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:18001/v1/llm/preprocess" `
  -ContentType "application/json; charset=utf-8" `
  -Body $payload

$resp.data.desensitized_request.messages[0].content
$resp.data.mapping | ConvertTo-Json -Depth 10
```

## 常见问题

### 1. `using_taskflow` 是 `false`

说明 Taskflow 没有初始化成功。检查：

- `DESENSITIZE_ENABLE_TASKFLOW` 是否为 `true`
- `DESENSITIZE_STRICT_LOCAL_MODEL` 是否为 `true`
- `resources/models/wordtag` 是否存在模型文件
- Paddle/PaddleNLP 依赖版本是否与 `requirements.txt` 一致

### 2. 服务启动失败

当前服务要求必须使用本地 Taskflow wordtag。模型或依赖不可用时，服务会启动失败，这是预期行为。

优先执行：

```powershell
uv run python -m pip install -r requirements.txt
```

### 3. 历史映射没有复用

检查下一次请求是否把上次返回的完整 `mapping` 放到了顶层 `history_mappings` 字段。

正确：

```json
{
  "history_mappings": {
    "PERSON": {
      "张三": "[[PERSON_001]]"
    }
  }
}
```

### 4. `stats.new_entities` 为什么小于替换次数

如果实体来自 `history_mappings`，说明它复用了历史占位符，不属于本次新增。`stats.new_entities` 只统计本次新创建的映射数量。
