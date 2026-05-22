# 本地开发与模型准备

本文档承接 README 中不适合放在 GitLab 首页的本地调试、模型准备和实体识别细节。

## 实体识别逻辑

服务当前使用 PaddleNLP `Taskflow("ner", entity_only=True)` 的 wordtag 模型链路，本地模型目录默认为 `resources/models/wordtag`。

一次脱敏会合并以下实体来源：

- 历史 `mapping`：已脱敏 `user` 消息携带的映射会被优先复用，确保同一实体继续使用原占位符。
- 自定义模型标签：`custom_entities` 声明业务实体类型与需要匹配的 wordtag 模型标签，命中后按声明的 `entity_type` 生成占位符。
- 内置模型识别：wordtag 默认用于识别人名、组织/公司等中文实体。
- 规则补漏：手机号使用正则识别，避免通用模型漏掉纯数字手机号。
- 可选补漏：jieba 人名/机构补漏默认关闭，可通过 `DESENSITIZE_ENABLE_JIEBA_FALLBACK=true` 显式开启。

冲突处理优先级：

```text
history mapping > custom entity > model/regex > jieba
```

同一优先级下倾向保留更长片段，避免短片段覆盖完整实体。

`custom_entities` 中的 `model_labels`、`labels` 或 `schema` 是“模型标签匹配规则”，不是独立的正则或字面值匹配。当前服务不会执行 `custom_entities.patterns`、`regex` 或 `values` 字段；这些字段即使传入，也只有在模型本身返回对应片段时才可能产生脱敏效果。

地址类实体可声明为 `ADDRESS`。为了兼容业务写法，`ADDRESS` 内置了 `住址`、`地址`、`场所类`、`世界地区类`、`位置方位` 等标签别名，并会过滤 `住址`、`地址` 这类字段名。模型如果把连续地址切成多个片段，例如 `北京市海淀区` 与 `中关村大街27号`，服务会在两段紧邻且同为 `ADDRESS` 时合并成完整地址后再替换。

示例：

```json
{
  "custom_entities": [
    {
      "entity_type": "ADDRESS",
      "model_labels": ["住址", "地址"]
    }
  ]
}
```

对于账号、卡号等数字类隐私实体，当前只有手机号作为内置规则。后续如需支持身份证号、银行卡号、平台账号、对公账户等，建议另行设计专门的格式化敏感实体识别链路。

## 环境准备

推荐使用 Python 3.10。项目当前依赖已在 Python 3.10.20 下验证。

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

依赖版本中 `paddlenlp==2.8.1` 需要 `aistudio-sdk==0.2.6`，不要随意升级该包。`tool-helpers` 是 PaddleNLP 声明的预训练辅助依赖，上游只发布 Linux wheel；Windows 本地如果安装时被该包阻塞，部署交付优先使用 Docker/Linux 环境。

## 准备模型

服务默认从下面目录加载 wordtag 模型：

```text
resources/models/wordtag
```

推荐让服务首次启动时自动下载并同步模型：

```powershell
$env:DESENSITIZE_AUTO_DOWNLOAD_MODEL="true"
$env:DESENSITIZE_SYNC_DOWNLOADED_MODEL="true"
python -u app.py
```

首次启动会通过 PaddleNLP 下载默认 wordtag 模型。下载完成后，服务会把模型同步到 `resources/models/wordtag`，后续启动会优先读取该目录。

如果运行环境无法直接访问外网，可以先在一台能访问模型源的机器上执行：

```powershell
python -c "from paddlenlp import Taskflow; Taskflow('ner', entity_only=True)"
```

下载完成后，将用户目录下的 PaddleNLP 缓存复制到项目模型目录：

```powershell
New-Item -ItemType Directory -Force .\resources\models\wordtag
Copy-Item "$env:USERPROFILE\.paddlenlp\taskflow\wordtag\*" .\resources\models\wordtag -Recurse -Force
```

模型目录准备好后，可用下面命令检查：

```powershell
Get-ChildItem .\resources\models\wordtag
```

常见文件包括 `model_state.pdparams`、`config.json`、`vocab.txt` 和 `static` 推理文件。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DESENSITIZE_MODEL_PATH` | `resources/models/wordtag` | 本地模型目录 |
| `DESENSITIZE_DICT_DIR` | `resources/common_data/uie` | jieba 用户词典目录 |
| `DESENSITIZE_DEVICE_ID` | `0` | PaddleNLP 推理设备 ID |
| `DESENSITIZE_MAX_TEXT_LEN` | `10000` | 单条消息允许处理的最大长度 |
| `DESENSITIZE_ENABLE_TASKFLOW` | `true` | 是否启用 Taskflow |
| `DESENSITIZE_ENABLE_JIEBA_FALLBACK` | `false` | 是否启用 jieba 人名/机构补漏 |
| `DESENSITIZE_STRICT_LOCAL_MODEL` | `true` | 模型不可用时是否启动失败 |
| `DESENSITIZE_AUTO_DOWNLOAD_MODEL` | `true` | 本地模型缺失时是否尝试自动下载 |
| `DESENSITIZE_SYNC_DOWNLOADED_MODEL` | `true` | 自动下载后是否同步回本地模型目录 |
| `DESENSITIZE_MODEL_CACHE_PATH` | `$HOME/.paddlenlp/taskflow/wordtag` | PaddleNLP 默认下载缓存目录 |

## 本地启动与检查

请先确认当前 `python` 指向项目虚拟环境：

```powershell
python -c "import sys; print(sys.executable)"
```

已激活虚拟环境时启动服务：

```powershell
python -u app.py
```

没有激活虚拟环境时，也可以直接使用项目内解释器：

```powershell
.\.venv\Scripts\python.exe -u app.py
```

默认监听：

```text
http://127.0.0.1:18001
```

健康检查：

```powershell
Invoke-RestMethod "http://127.0.0.1:18001/healthz" | ConvertTo-Json -Depth 10
```

重点关注：

```json
{
  "status": "ok",
  "using_taskflow": true,
  "model_path": "resources/models/wordtag"
}
```

`using_taskflow=true` 表示本地模型已经成功初始化。

## Docker 调试

构建镜像：

```powershell
docker build -t llm-messages-encryptor:latest .
```

推荐使用 Docker volume 保存模型。容器首次启动时会自动下载 wordtag 模型，并同步到 `/app/resources/models/wordtag`；后续再次启动会复用 volume 中的模型文件。

```powershell
docker volume create llm_messages_encryptor_model

docker run --rm `
  -p 18001:18001 `
  -v llm_messages_encryptor_model:/app/resources/models/wordtag `
  --name llm-messages-encryptor `
  llm-messages-encryptor:latest
```

如果运行环境不能直接下载模型，可以先在宿主机准备好 `resources/models/wordtag`，再把本机模型目录挂载到容器：

```powershell
docker run --rm `
  -p 18001:18001 `
  -v ${PWD}/resources/models/wordtag:/app/resources/models/wordtag `
  --name llm-messages-encryptor `
  llm-messages-encryptor:latest
```

## 示例请求

`example_preprocess_request.json` 演示普通预处理请求：

- 包含一条已脱敏的历史 `user` 消息和一条新的未脱敏 `user` 消息。
- 服务会复用历史 `mapping`，把新消息中的相同实体替换为已有占位符。
- 适合验证 `encrypted=false` 的 `user` 消息是否会被正确脱敏。

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

- 历史 `user` 消息携带上一次保存的 `mapping`。
- 新 `user` 消息再次出现相同姓名、组织时，会继续使用原来的占位符。
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

业务方实际接入时，建议每次请求后保存接口返回的 `data.mapping`；下一轮请求中，把该映射放到对应历史 `user` message 的 `mapping` 字段，并将该历史消息标记为 `encrypted=true`。

## 测试

```powershell
python -m unittest discover -s tests
```
