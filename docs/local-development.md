# 开发、部署与模型准备

本文档承接 README 中不适合放在首页的实体识别细节、本地调试、Docker 部署、模型准备和仓库文件规则。

## 实体识别逻辑

模型服务当前使用 PaddleNLP `Taskflow("ner", entity_only=True)` 的 wordtag 模型链路，本地模型目录默认为 `resources/models/wordtag`。业务自定义实体默认启用 `Taskflow("information_extraction", model="uie-base")` 旁路，本地模型目录默认为 `resources/models/uie-base`。

模型服务是纯推理层，只接收 `text` 和 `tasks`，通过内部接口 `POST /v1/infer` 返回模型原始标签片段，并在模型内部按自然边界和 token 预算处理长文本；wordtag 和 UIE 共用吞吐优先的 `512` token 上限切片策略，UIE 只额外扣除 schema prompt 预算。API 服务是应用层，负责手机号正则补漏、`custom_entities` 解析、历史 `mapping`、占位符、冲突消解和最终替换。

一次脱敏会合并以下实体来源：

- 历史 `mapping`：应用层从已脱敏 `user` 消息携带的映射中复用占位符。
- 内置模型识别：模型层返回 wordtag 原始标签，应用层映射为人名、组织/公司等业务实体。
- 规则补漏：应用层用手机号正则识别，避免通用模型漏掉纯数字手机号。
- 自定义 UIE schema：应用层把 `custom_entities[].uie_schema` 传给模型层，再把 UIE label 映射回业务实体类型。

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

安装模型服务依赖。模型推理默认部署在 NVIDIA/CUDA 11.7 服务器上，模型依赖统一使用 `requirements-model.txt`：

```powershell
python -m pip install -r requirements-model.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果只启动 API 服务并连接远程模型服务，可以只安装轻量依赖：

```powershell
python -m pip install -r requirements-api.txt
```

依赖版本中 `paddlenlp==2.8.1` 需要 `aistudio-sdk==0.2.6`，不要随意升级该包。`tool-helpers` 是 PaddleNLP 声明的预训练辅助依赖，上游只发布 Linux wheel；Windows 本地如果安装时被该包阻塞，部署交付优先使用 Docker/Linux 环境。

## 配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DESENSITIZE_MODEL_SERVICE_URL` | `http://127.0.0.1:18002` | 独立模型服务地址 |
| `DESENSITIZE_MODEL_SERVICE_TIMEOUT` | `30` | API 调用模型服务的超时时间，单位秒 |
| `DESENSITIZE_MODEL_PATH` | `resources/models/wordtag` | wordtag 本地模型目录 |
| `DESENSITIZE_UIE_MODEL_PATH` | `resources/models/uie-base` | UIE 本地模型目录 |
| `DESENSITIZE_ENABLE_UIE_CUSTOM` | `true` | 是否启用 UIE 信息抽取旁路识别业务自定义实体 |
| `DESENSITIZE_PRELOAD_UIE_CUSTOM` | `false` | 模型服务启动时是否预加载 UIE；容器默认关闭，避免启动阶段资源冲高 |
| `DESENSITIZE_UIE_WARMUP_SCHEMA` | `身份证号,卡号,银行卡号,地址,住址` | UIE 预热 schema，逗号分隔 |
| `DESENSITIZE_AUTO_DOWNLOAD_MODEL` | `true` | 本地模型缺失时是否尝试自动下载 |
| `DESENSITIZE_SYNC_DOWNLOADED_MODEL` | `false` | 自动下载后是否同步回本地模型目录；生产多容器默认使用各容器私有缓存 |
| `DESENSITIZE_STRICT_LOCAL_MODEL` | `true` | wordtag 模型不可用时是否启动失败 |
| `DESENSITIZE_STRICT_UIE_MODEL` | `false` | UIE 不可用时是否抛错 |
| `DESENSITIZE_MAX_MODEL_TOKENS` | `512` | 模型侧最大 token 上限；wordtag 和 UIE 都不会超过该预算 |
| `DESENSITIZE_UIE_TARGET_TEXT_TOKENS` | `512` | 兼容保留字段；吞吐优先切片不再使用 |
| `DESENSITIZE_DEVICE` | `cpu` | 模型服务推理设备类型，首版只支持 `cpu` 或 `nvidia` |
| `DESENSITIZE_DEVICE_ID` | `0` | NVIDIA GPU 编号；CPU 模式下会传给 PaddleNLP `-1` |
| `DESENSITIZE_UIE_POSITION_PROB` | `0.5` | UIE 起止位置概率阈值 |

## 本地启动与检查

开发、测试和生产联调统一使用两个进程。先启动模型服务：

```powershell
$env:MODEL_PORT="18002"
python -u model_app.py
```

再启动 API 服务，并通过 `DESENSITIZE_MODEL_SERVICE_URL` 指向模型服务：

```powershell
$env:DESENSITIZE_MODEL_SERVICE_URL="http://127.0.0.1:18002"
python -u app.py
```

健康检查：

```powershell
Invoke-RestMethod "http://127.0.0.1:18001/healthz" | ConvertTo-Json -Depth 10
Invoke-RestMethod "http://127.0.0.1:18001/readyz" | ConvertTo-Json -Depth 10
Invoke-RestMethod "http://127.0.0.1:18002/healthz" | ConvertTo-Json -Depth 10
Invoke-RestMethod "http://127.0.0.1:18002/readyz" | ConvertTo-Json -Depth 10
```

`using_taskflow=true` 表示 wordtag 模型已经成功初始化。模型服务如果开启了 `DESENSITIZE_PRELOAD_UIE_CUSTOM=true`，`/readyz` 会等 UIE 预加载完成后才返回 ready。NVIDIA GPU 模式下，模型服务 `/healthz` 还应看到 `device=nvidia`、`gpu_available=true` 和大于 `0` 的 `gpu_device_count`。

## 多实例部署

API 服务是无状态应用，所有会话延续信息都来自请求中历史 `user` 消息携带的 `mapping`。可以部署多个 API 容器或多台 API 服务器，通过负载均衡访问，不需要 sticky session。

模型服务也可以部署多个实例。默认情况下，模型缺失时仍允许自动下载，但下载结果保留在各容器自己的 PaddleNLP 缓存中，不同步写回 `/app/resources/models/*`。如果需要让多个模型容器共享模型文件，建议提前准备模型目录并以只读方式挂载。

## Docker 部署

交付镜像统一使用根目录 `Dockerfile`。同一个镜像包含 API 服务和模型服务所需依赖，运行时通过 `SERVICE_ROLE` 区分进程角色：

- `SERVICE_ROLE=model`：启动 `model_app.py`，加载 Paddle/PaddleNLP 和本地模型，对 API 容器暴露纯推理接口。
- `SERVICE_ROLE=api`：启动 `app.py`，负责请求校验、映射复用和占位符替换，通过 HTTP 调用模型容器。

构建镜像：

```bash
sudo docker build -t llm-messages-encryptor:latest .
```

正式交付建议使用明确版本号替代 `latest`，例如 `llm-messages-encryptor:v0.1.0`。

API 角色容器和模型角色容器可以来自同一个镜像，但模型目录只需要挂载到 `SERVICE_ROLE=model` 的容器。

### NVIDIA GPU 模型服务

首版 GPU 方案只支持 NVIDIA/CUDA，不引入昇腾、寒武纪、AMD、XPU/NPU 等厂商抽象。模型角色默认关闭启动阶段 UIE 预热，避免容器启动时同时加载 wordtag 和 UIE 导致服务器资源打满。

你的服务器是 NVIDIA GeForce RTX 2080 Ti，当前 `memory.free` 约 `8872 MiB`。模型容器启动时必须加资源护栏：限制只使用第 `0` 张 GPU，限制宿主机内存和 CPU，并通过 Paddle 的 `FLAGS_fraction_of_gpu_memory_to_use` 限制显存预分配比例。建议先用 `0.4`，确认稳定后再小幅调整。

宿主机需要提前安装：

- NVIDIA 驱动。
- NVIDIA Container Toolkit。

GPU 模式采用严格启动：`DESENSITIZE_DEVICE=nvidia` 时，如果容器内 Paddle 不是 CUDA 版本、没有可见 GPU，或 `DESENSITIZE_DEVICE_ID` 超出可见 GPU 编号范围，模型服务会直接启动失败，不会静默降级到 CPU。

启动模型角色容器：

```powershell
docker run -d `
  --gpus '"device=0"' `
  --restart unless-stopped `
  -p 18002:18002 `
  --memory 8g `
  --cpus 4 `
  -e SERVICE_ROLE=model `
  -e DESENSITIZE_DEVICE_ID=0 `
  -e FLAGS_fraction_of_gpu_memory_to_use=0.4 `
  -v /data/models/wordtag:/app/resources/models/wordtag:ro `
  -v /data/models/uie-base:/app/resources/models/uie-base:ro `
  --name llm-messages-encryptor-model `
  llm-messages-encryptor:latest
```

不建议使用 `--gpus all`。RTX 2080 Ti 单卡场景下固定 `--gpus '"device=0"'` 更容易控制显存和故障范围。

启动后检查：

```powershell
Invoke-RestMethod "http://127.0.0.1:18002/healthz" | ConvertTo-Json -Depth 10
```

重点字段：

- `device` 应为 `nvidia`。
- `taskflow_device_id` 应等于 `DESENSITIZE_DEVICE_ID`。
- `gpu_available` 应为 `true`。
- `gpu_device_count` 应大于 `0`。

### 同一台 Docker 主机

当模型服务容器和 API 服务容器运行在同一台 Docker 主机，并加入同一个 Docker network 时，API 服务可以用模型容器名访问模型服务：

```text
http://llm-messages-encryptor-model:18002
```

Bash：

```bash
sudo docker network create llm_messages_encryptor_net || true

sudo docker run -d \
  --gpus '"device=0"' \
  --restart unless-stopped \
  --network llm_messages_encryptor_net \
  -p 18002:18002 \
  --memory 8g \
  --cpus 4 \
  -e SERVICE_ROLE=model \
  -e DESENSITIZE_DEVICE_ID=0 \
  -e FLAGS_fraction_of_gpu_memory_to_use=0.4 \
  -v /data/models/wordtag:/app/resources/models/wordtag \
  -v /data/models/uie-base:/app/resources/models/uie-base \
  --name llm-messages-encryptor-model \
  llm-messages-encryptor:latest

sudo docker run -d \
  --restart unless-stopped \
  --network llm_messages_encryptor_net \
  -p 18001:18001 \
  --memory 1g \
  --cpus 1 \
  -e SERVICE_ROLE=api \
  -e DESENSITIZE_MODEL_SERVICE_URL=http://llm-messages-encryptor-model:18002 \
  --name llm-messages-encryptor-api \
  llm-messages-encryptor:latest
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
sudo docker run -d \
  --gpus '"device=0"' \
  --restart unless-stopped \
  -p 18002:18002 \
  --memory 8g \
  --cpus 4 \
  -e SERVICE_ROLE=model \
  -e DESENSITIZE_DEVICE_ID=0 \
  -e FLAGS_fraction_of_gpu_memory_to_use=0.4 \
  -v /data/models/wordtag:/app/resources/models/wordtag \
  -v /data/models/uie-base:/app/resources/models/uie-base \
  --name llm-messages-encryptor-model \
  llm-messages-encryptor:latest
```

API 服务主机：

```bash
sudo docker run -d \
  --restart unless-stopped \
  -p 18001:18001 \
  --memory 1g \
  --cpus 1 \
  -e SERVICE_ROLE=api \
  -e DESENSITIZE_MODEL_SERVICE_URL=http://10.10.1.20:18002 \
  --name llm-messages-encryptor-api \
  llm-messages-encryptor:latest
```

跨主机部署时，建议用防火墙或安全组限制只有 API 服务主机可以访问模型服务主机的 `18002` 端口。

### 离线交付

导出镜像包：

```powershell
docker save -o llm-messages-encryptor.tar llm-messages-encryptor:latest
```

业务方服务器导入镜像：

```powershell
docker load -i llm-messages-encryptor.tar
```

离线环境建议提前准备模型目录，并关闭自动下载：

```bash
sudo docker run -d \
  --gpus '"device=0"' \
  --restart unless-stopped \
  --memory 8g \
  --cpus 4 \
  -e SERVICE_ROLE=model \
  -e DESENSITIZE_DEVICE_ID=0 \
  -e FLAGS_fraction_of_gpu_memory_to_use=0.4 \
  -e DESENSITIZE_AUTO_DOWNLOAD_MODEL=false \
  -e DESENSITIZE_SYNC_DOWNLOADED_MODEL=false \
  -v /data/models/wordtag:/app/resources/models/wordtag \
  -v /data/models/uie-base:/app/resources/models/uie-base \
  --name llm-messages-encryptor-model \
  llm-messages-encryptor:latest
```

模型目录要求：

- `wordtag` 模型目录挂载到 `/app/resources/models/wordtag`。
- `uie-base` 模型目录挂载到 `/app/resources/models/uie-base`。
- 模型文件只需要放在模型服务机器或模型服务容器的 volume 中；多容器共享时建议只读挂载。
- API 角色容器不需要挂载模型目录，也不会在 API 进程里加载 PaddlePaddle/PaddleNLP。
- Linux 宿主机目录挂载时，如果只使用预置模型，容器用户 `10001:10001` 只需要读取权限；只有显式开启同步写回时才需要写权限。

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

仓库只保留源码、依赖声明、单一交付 Dockerfile、单元测试、示例请求和模型目录说明文件。实际模型权重、推理文件、缓存和本地压测脚本不要提交到 Git。

应保留的交付文件：

- `app.py`：API 服务入口。
- `model_app.py`：独立模型服务入口。
- `desensitize/*.py`：脱敏、识别、远程调用、映射和配置代码。
- `requirements-api.txt`、`requirements-model.txt`：依赖入口，分别用于 API 服务和 CUDA 11.7 GPU 模型服务。
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
- `run_and_test.ps1`、`tools/concurrency_test.py` 这类本地临时联调或压测脚本。
