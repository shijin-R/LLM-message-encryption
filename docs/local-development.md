# 本地开发与模型准备

本文档承接 README 中不适合放在 GitLab 首页的本地调试、模型准备和实体识别细节。

## 实体识别逻辑

服务当前使用 PaddleNLP `Taskflow("ner", entity_only=True)` 的 wordtag 模型链路，本地模型目录默认为 `resources/models/wordtag`。业务自定义实体默认启用 `Taskflow("information_extraction", model="uie-base")` 旁路，本地模型目录默认为 `resources/models/uie-base`。UIE 按请求懒加载，服务启动时不会立即加载模型。

一次脱敏会合并以下实体来源：

- 历史 `mapping`：已脱敏 `user` 消息携带的映射会被优先复用，确保同一实体继续使用原占位符。
- 内置模型识别：wordtag 默认用于识别人名、组织/公司等中文实体。
- 规则补漏：手机号使用正则识别，避免通用模型漏掉纯数字手机号。
- 自定义 UIE schema：`custom_entities[].uie_schema` 会作为 UIE 信息抽取目标，适合身份证号、银行卡号、平台账号等上下文依赖更强的实体。可通过 `DESENSITIZE_ENABLE_UIE_CUSTOM=false` 关闭。

冲突处理优先级：

```text
history mapping > custom entity > model/regex
```

同一优先级下倾向保留更长片段，避免短片段覆盖完整实体。

`custom_entities` 只用于 UIE 自定义实体旁路，`uie_schema` 是 UIE 的“信息抽取目标”。当前服务不会执行 `custom_entities.patterns`、`regex`、`values` 或 `model_labels` 字段；没有 `uie_schema` 的自定义规则不会触发 UIE。人名和组织/公司等内置实体由 wordtag 链路负责。

地址类实体可声明为 `ADDRESS`，并在 `uie_schema` 中显式写出业务字段名，例如 `地址`、`住址`。服务会过滤 `住址`、`地址` 这类字段名。模型如果把连续地址切成多个片段，例如 `北京市海淀区` 与 `中关村大街27号`，服务会在两段紧邻且同为 `ADDRESS` 时合并成完整地址后再替换。

示例：

```json
{
  "custom_entities": [
    {
      "entity_type": "ADDRESS",
      "uie_schema": ["地址", "住址"]
    }
  ]
}
```

对于账号、卡号等数字类隐私实体，手机号仍作为唯一内置正则补漏。身份证号、银行卡号、平台账号、对公账户等建议通过 `uie_schema` 交给 UIE 旁路或后续自训模型识别，避免继续堆叠难维护的正则规则。

UIE 自定义实体示例：

```json
{
  "custom_entities": [
    {
      "entity_type": "ID_CARD",
      "uie_schema": ["身份证号"]
    },
    {
      "entity_type": "BANK_CARD",
      "uie_schema": ["银行卡号", "银行账号"]
    }
  ]
}
```

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

UIE 自定义实体旁路默认开启；如需指定模型目录或模型名，可在启动前设置：

```powershell
$env:DESENSITIZE_UIE_MODEL_NAME="uie-base"
$env:DESENSITIZE_UIE_MODEL_PATH="resources/models/uie-base"
python -u app.py
```

首次命中带 `uie_schema` 的请求时，会懒加载 UIE 模型；本地目录缺失且 `DESENSITIZE_AUTO_DOWNLOAD_MODEL=true` 时会下载 `uie-base` 并尝试同步到 `resources/models/uie-base`。当前本地实测 `uie-base` 权重下载约 450 MB，静态推理文件生成后整体占用约 900 MB。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DESENSITIZE_MODEL_PATH` | `resources/models/wordtag` | 本地模型目录 |
| `DESENSITIZE_DEVICE_ID` | `0` | PaddleNLP 推理设备 ID |
| `DESENSITIZE_MAX_TEXT_LEN` | `10000` | 单条消息允许处理的最大长度 |
| `DESENSITIZE_ENABLE_TASKFLOW` | `true` | 是否启用 Taskflow |
| `DESENSITIZE_ENABLE_UIE_CUSTOM` | `true` | 是否启用 UIE 信息抽取旁路识别业务自定义实体；开启后仍按请求懒加载 |
| `DESENSITIZE_UIE_MODEL_NAME` | `uie-base` | UIE 信息抽取模型名 |
| `DESENSITIZE_UIE_MODEL_PATH` | `resources/models/uie-base` | 本地 UIE 模型目录 |
| `DESENSITIZE_UIE_POSITION_PROB` | `0.5` | UIE 起止位置概率阈值 |
| `DESENSITIZE_STRICT_UIE_MODEL` | `false` | UIE 不可用时是否抛错 |
| `DESENSITIZE_UIE_MODEL_CACHE_PATH` | `$HOME/.paddlenlp/taskflow/information_extraction/uie-base` | PaddleNLP UIE 默认下载缓存目录 |
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
  "enable_uie_custom": true,
  "using_uie": false,
  "model_path": "resources/models/wordtag"
}
```

`using_taskflow=true` 表示 wordtag 模型已经成功初始化。`using_uie=true` 表示 UIE 旁路已在当前进程中懒加载成功；服务刚启动且尚未处理 `uie_schema` 请求时通常为 `false`。

## Docker 调试

构建镜像：

```powershell
docker build -t llm-messages-encryptor:latest .
```

推荐使用 Docker volume 保存模型。容器首次启动时会自动下载 wordtag 模型，并同步到 `/app/resources/models/wordtag`；启用 UIE 时也建议为 `/app/resources/models/uie-base` 单独挂载 volume。后续再次启动会复用 volume 中的模型文件。

```powershell
docker volume create llm_messages_encryptor_model
docker volume create llm_messages_encryptor_uie_model

docker run --rm `
  -p 18001:18001 `
  -v llm_messages_encryptor_model:/app/resources/models/wordtag `
  -v llm_messages_encryptor_uie_model:/app/resources/models/uie-base `
  --name llm-messages-encryptor `
  llm-messages-encryptor:latest
```

如果运行环境不能直接下载模型，可以先在宿主机准备好 `resources/models/wordtag`，再把本机模型目录挂载到容器：

```powershell
docker run --rm `
  -p 18001:18001 `
  -v ${PWD}/resources/models/wordtag:/app/resources/models/wordtag `
  -v ${PWD}/resources/models/uie-base:/app/resources/models/uie-base `
  --name llm-messages-encryptor `
  llm-messages-encryptor:latest
```

## 示例请求

`example_preprocess_request.json` 演示完整预处理请求：

- 包含一条已脱敏的历史 `user` 消息和一条新的未脱敏 `user` 消息。
- 服务会复用历史 `mapping`，把新消息中的相同实体替换为已有占位符。
- 覆盖内置 wordtag 人名/组织识别、手机号补漏，以及 UIE 身份证号/卡号自定义实体。

调用示例：

```powershell
$body = Get-Content .\example_preprocess_request.json -Raw -Encoding UTF8

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:18001/v1/llm/preprocess" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body | ConvertTo-Json -Depth 20
```

成功时，最后一条 user 消息中的敏感实体会被替换成类似：

```text
本次合同甲方仍是[[ORG_001]]，联系人[[PERSON_001]]，新手机号[[MOBILE_002]]。身份证号[[ID_CARD_001]]，卡号[[BANK_CARD_001]]。
```

业务方实际接入时，建议每次请求后保存接口返回的 `data.mapping`；下一轮请求中，把该映射放到对应历史 `user` message 的 `mapping` 字段，并将该历史消息标记为 `encrypted=true`。

## 测试

```powershell
python -m unittest discover -s tests
```
