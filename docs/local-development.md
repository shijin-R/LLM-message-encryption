# 开发、部署与模型准备

本文档承接 README 中不适合放在首页的实体识别细节、本地调试、Docker 部署、模型准备和仓库文件规则。

## 实体识别逻辑

服务当前使用 PaddleNLP `Taskflow("ner", entity_only=True)` 的 wordtag 模型链路，本地模型目录默认为 `resources/models/wordtag`。业务自定义实体默认启用 `Taskflow("information_extraction", model="uie-base")` 旁路，本地模型目录默认为 `resources/models/uie-base`。

一次脱敏会合并以下实体来源：

- 历史 `mapping`：已脱敏 `user` 消息携带的映射会被优先复用，确保同一实体继续使用原占位符。
- 内置模型识别：wordtag 默认用于识别人名、组织/公司等中文实体。
- 规则补漏：手机号使用正则识别，避免通用模型漏掉纯数字手机号。
- 自定义 UIE schema：`custom_entities[].uie_schema` 会作为 UIE 信息抽取目标，适合身份证号、银行卡号、平台账号等上下文依赖更强的实体。

冲突处理优先级：

```text
history mapping > custom entity > model/regex
```

同一优先级下倾向保留更长片段，避免短片段覆盖完整实体。

`custom_entities` 只用于 UIE 自定义实体旁路，`uie_schema` 是 UIE 的“信息抽取目标”。当前服务不会执行 `custom_entities.patterns`、`regex`、`values` 或 `model_labels` 字段；没有 `uie_schema` 的自定义规则不会触发 UIE。

地址类实体可声明为 `ADDRESS`，并在 `uie_schema` 中显式写出业务字段名，例如 `地址`、`住址`。服务会过滤 `住址`、`地址` 这类字段名。模型如果把连续地址切成多个片段，例如 `北京市海淀区` 与 `中关村大街27号`，服务会在两段紧邻且同为 `ADDRESS` 时合并成完整地址后再替换。

对于账号、卡号等数字类隐私实体，手机号仍作为唯一内置正则补漏。身份证号、银行卡号、平台账号、对公账户等建议通过 `uie_schema` 交给 UIE 旁路或后续自训模型识别。

## 环境准备

推荐使用 Python 3.10。项目当前依赖已在 Python 3.10.20 下验证。

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装模型服务依赖：

```powershell
python -m pip install -r requirements-model.txt
```

如果只启动 API 服务并连接远程模型服务，可以只安装轻量依赖：

```powershell
python -m pip install -r requirements-api.txt
```

依赖版本中 `paddlenlp==2.8.1` 需要 `aistudio-sdk==0.2.6`，不要随意升级该包。`tool-helpers` 是 PaddleNLP 声明的预训练辅助依赖，上游只发布 Linux wheel；Windows 本地如果安装时被该包阻塞，部署交付优先使用 Docker/Linux 环境。

## 配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DESENSITIZE_RECOGNIZER_BACKEND` | `remote` | API 服务识别后端；`remote` 为独立模型服务，`local` 为进程内模型 |
| `DESENSITIZE_MODEL_SERVICE_URL` | `http://127.0.0.1:18002` | 独立模型服务地址 |
| `DESENSITIZE_MODEL_SERVICE_TIMEOUT` | `30` | API 调用模型服务的超时时间，单位秒 |
| `DESENSITIZE_MODEL_PATH` | `resources/models/wordtag` | wordtag 本地模型目录 |
| `DESENSITIZE_UIE_MODEL_PATH` | `resources/models/uie-base` | UIE 本地模型目录 |
| `DESENSITIZE_ENABLE_UIE_CUSTOM` | `true` | 是否启用 UIE 信息抽取旁路识别业务自定义实体 |
| `DESENSITIZE_PRELOAD_UIE_CUSTOM` | `true` | 模型服务启动时是否预加载 UIE |
| `DESENSITIZE_UIE_WARMUP_SCHEMA` | `身份证号,卡号,银行卡号,地址,住址` | UIE 预热 schema，逗号分隔 |
| `DESENSITIZE_AUTO_DOWNLOAD_MODEL` | `true` | 本地模型缺失时是否尝试自动下载 |
| `DESENSITIZE_SYNC_DOWNLOADED_MODEL` | `true` | 自动下载后是否同步回本地模型目录 |
| `DESENSITIZE_STRICT_LOCAL_MODEL` | `true` | wordtag 模型不可用时是否启动失败 |
| `DESENSITIZE_STRICT_UIE_MODEL` | `false` | UIE 不可用时是否抛错 |
| `DESENSITIZE_MAX_TEXT_LEN` | `10000` | 单条消息允许处理的最大长度 |
| `DESENSITIZE_DEVICE_ID` | `0` | PaddleNLP 推理设备 ID |
| `DESENSITIZE_UIE_POSITION_PROB` | `0.5` | UIE 起止位置概率阈值 |

## 本地启动与检查

单进程本地模式：

```powershell
$env:DESENSITIZE_RECOGNIZER_BACKEND="local"
python -u app.py
```

生产联调建议拆成两个进程。先启动模型服务：

```powershell
$env:MODEL_PORT="18002"
python -u model_app.py
```

再启动 API 服务：

```powershell
$env:DESENSITIZE_RECOGNIZER_BACKEND="remote"
$env:DESENSITIZE_MODEL_SERVICE_URL="http://127.0.0.1:18002"
python -u app.py
```

健康检查：

```powershell
Invoke-RestMethod "http://127.0.0.1:18001/healthz" | ConvertTo-Json -Depth 10
Invoke-RestMethod "http://127.0.0.1:18001/readyz" | ConvertTo-Json -Depth 10
Invoke-RestMethod "http://127.0.0.1:18002/readyz" | ConvertTo-Json -Depth 10
```

`using_taskflow=true` 表示 wordtag 模型已经成功初始化。单进程本地模式下，`using_uie=true` 表示 UIE 旁路已在当前进程中懒加载成功；服务刚启动且尚未处理 `uie_schema` 请求时通常为 `false`。模型服务模式下，如果开启了 `DESENSITIZE_PRELOAD_UIE_CUSTOM=true`，`/readyz` 会等 UIE 预加载完成后才返回 ready。

## Docker 部署

镜像拆成两个 target：

- `model`：模型服务镜像，安装 Paddle/PaddleNLP，负责加载 wordtag 和 UIE 模型。
- `api`：API 服务镜像，只安装 Flask 等轻量依赖，负责请求校验、映射复用和占位符替换。

构建镜像：

```powershell
docker build --target model -t llm-messages-encryptor-model:latest .
docker build --target api -t llm-messages-encryptor-api:latest .
```

正式交付建议使用明确版本号替代 `latest`，例如 `llm-messages-encryptor-api:v0.1.0`。

### 同一台 Docker 主机

当模型服务容器和 API 服务容器运行在同一台 Docker 主机，并加入同一个 Docker network 时，API 服务可以用模型容器名访问模型服务：

```text
http://llm-messages-encryptor-model:18002
```

PowerShell：

```powershell
docker network create llm_messages_encryptor_net
docker volume create llm_messages_encryptor_model
docker volume create llm_messages_encryptor_uie_model

docker run -d `
  --restart unless-stopped `
  --network llm_messages_encryptor_net `
  -e MODEL_HOST=0.0.0.0 `
  -e MODEL_PORT=18002 `
  -v llm_messages_encryptor_model:/app/resources/models/wordtag `
  -v llm_messages_encryptor_uie_model:/app/resources/models/uie-base `
  --name llm-messages-encryptor-model `
  llm-messages-encryptor-model:latest

docker run -d `
  --restart unless-stopped `
  --network llm_messages_encryptor_net `
  -p 18001:18001 `
  -e DESENSITIZE_MODEL_SERVICE_URL=http://llm-messages-encryptor-model:18002 `
  --name llm-messages-encryptor-api `
  llm-messages-encryptor-api:latest
```

Bash：

```bash
docker network create llm_messages_encryptor_net
docker volume create llm_messages_encryptor_model
docker volume create llm_messages_encryptor_uie_model

docker run -d \
  --restart unless-stopped \
  --network llm_messages_encryptor_net \
  -e MODEL_HOST=0.0.0.0 \
  -e MODEL_PORT=18002 \
  -v llm_messages_encryptor_model:/app/resources/models/wordtag \
  -v llm_messages_encryptor_uie_model:/app/resources/models/uie-base \
  --name llm-messages-encryptor-model \
  llm-messages-encryptor-model:latest

docker run -d \
  --restart unless-stopped \
  --network llm_messages_encryptor_net \
  -p 18001:18001 \
  -e DESENSITIZE_MODEL_SERVICE_URL=http://llm-messages-encryptor-model:18002 \
  --name llm-messages-encryptor-api \
  llm-messages-encryptor-api:latest
```

### 不同主机

普通 Docker bridge network 只在单台主机内生效，不能直接跨主机使用容器名访问。如果模型服务和 API 服务部署在不同主机上，应使用模型服务机器的内网 IP 或内网 DNS。

示例：

```text
模型服务主机：10.10.1.20
API 服务主机：10.10.1.30
```

模型服务主机：

```bash
docker run -d \
  --restart unless-stopped \
  -p 18002:18002 \
  -e MODEL_HOST=0.0.0.0 \
  -e MODEL_PORT=18002 \
  -v /data/models/wordtag:/app/resources/models/wordtag \
  -v /data/models/uie-base:/app/resources/models/uie-base \
  --name llm-messages-encryptor-model \
  llm-messages-encryptor-model:latest
```

API 服务主机：

```bash
docker run -d \
  --restart unless-stopped \
  -p 18001:18001 \
  -e DESENSITIZE_MODEL_SERVICE_URL=http://10.10.1.20:18002 \
  --name llm-messages-encryptor-api \
  llm-messages-encryptor-api:latest
```

跨主机部署时，建议用防火墙或安全组限制只有 API 服务主机可以访问模型服务主机的 `18002` 端口。

### 离线交付

导出镜像包：

```powershell
docker save -o llm-messages-encryptor-model.tar llm-messages-encryptor-model:latest
docker save -o llm-messages-encryptor-api.tar llm-messages-encryptor-api:latest
```

业务方服务器导入镜像：

```powershell
docker load -i llm-messages-encryptor-model.tar
docker load -i llm-messages-encryptor-api.tar
```

离线环境建议提前准备模型目录，并关闭自动下载：

```bash
docker run -d \
  --restart unless-stopped \
  -e MODEL_HOST=0.0.0.0 \
  -e MODEL_PORT=18002 \
  -e DESENSITIZE_AUTO_DOWNLOAD_MODEL=false \
  -e DESENSITIZE_SYNC_DOWNLOADED_MODEL=false \
  -v /data/models/wordtag:/app/resources/models/wordtag \
  -v /data/models/uie-base:/app/resources/models/uie-base \
  --name llm-messages-encryptor-model \
  llm-messages-encryptor-model:latest
```

模型目录要求：

- `wordtag` 模型目录挂载到 `/app/resources/models/wordtag`。
- `uie-base` 模型目录挂载到 `/app/resources/models/uie-base`。
- 模型文件只需要放在模型服务机器或模型服务容器的 volume 中。
- API 服务机器不需要模型文件，也不需要安装 PaddlePaddle/PaddleNLP。
- Linux 宿主机目录挂载时，请确保容器用户 `10001:10001` 对挂载目录有读写权限，或至少对已有模型文件有读取权限。

## 示例请求

`example_preprocess_request.json` 演示完整预处理请求：

- 包含一条已脱敏的历史 `user` 消息和一条新的未脱敏 `user` 消息。
- 服务会复用历史 `mapping`，把新消息中的相同实体替换为已有占位符。
- 覆盖内置 wordtag 人名/组织识别、手机号补漏，以及 UIE 身份证号/卡号/地址自定义实体。

调用示例：

```powershell
$body = Get-Content .\example_preprocess_request.json -Raw -Encoding UTF8

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

## 仓库瘦身规则

仓库只保留源码、依赖声明、Dockerfile、测试、示例请求和模型目录说明文件。实际模型权重、推理文件和缓存不要提交到 Git。

应保留的交付文件：

- `app.py`：API 服务入口。
- `model_app.py`：独立模型服务入口。
- `desensitize/*.py`：脱敏、识别、远程调用、映射和配置代码。
- `requirements-api.txt`、`requirements-model.txt`、`requirements.txt`：依赖入口。
- `Dockerfile`、`.dockerignore`、`.gitignore`：容器构建和本地文件排除规则。
- `example_preprocess_request.json`：综合联调示例。
- `tests/test_service.py`：服务行为回归测试。
- `resources/models/wordtag/README.md`、`resources/models/uie-base/README.md`：模型目录占位说明。

不应提交的本地文件：

- `.venv/`、`venv/`、`.uv-cache/`、`.paddlenlp/`。
- `__pycache__/`、`*.pyc`、`.pytest_cache/`、`.coverage`、`htmlcov/`。
- `logs/`、`*.log`、`run_service.*.log`。
- `.env`、`.env.*`、个人 IDE 配置。
- `resources/models/wordtag/*` 和 `resources/models/uie-base/*` 下的实际模型文件。
- `run_and_test.ps1` 这类本地临时联调脚本。
