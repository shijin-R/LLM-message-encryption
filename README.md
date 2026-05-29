# 大模型请求脱敏服务

## 项目说明

本项目提供一个 HTTP 前置服务，用于在业务请求发送给大模型前，对 `llm_request.messages` 中未脱敏的 `user` 消息执行敏感实体替换，并返回可直接转发给上游模型的请求体。

服务会把姓名、组织/公司、手机号，以及通过 UIE 声明的身份证号、银行卡号、地址等实体替换为结构化占位符，例如 `[[PERSON_001]]`、`[[ORG_001]]`、`[[MOBILE_001]]`。

服务本身不保存会话状态，不持久化 `mapping`，不转发大模型请求，也不提供大模型响应还原接口。调用方需要保存接口返回的 `data.mapping`，并在后续历史 `user` 消息中带回，以复用同一套占位符。

## 服务架构

生产部署建议拆成两个服务：

```text
业务系统 -> API 服务 app.py:18001 -> 模型服务 model_app.py:18002
```

- API 服务：应用层，负责请求校验、手机号正则补漏、`custom_entities` 解析、历史 `mapping` 复用、冲突消解和占位符替换，默认通过 HTTP 调用模型服务，只需要 `requirements-api.txt`。
- 模型服务：纯推理层，常驻加载 wordtag 和 `uie-base`，只接收 `text` 与推理 `tasks`，返回模型原始标签片段，可被多个 API 服务共享；依赖统一使用 `requirements-model.txt`，默认面向 NVIDIA/CUDA 11.7 服务器部署。
- API 服务不保存会话状态，可部署多个容器或多台服务器并通过负载均衡访问，不需要 sticky session。
- 模型服务可部署多个实例；默认自动下载到各容器私有缓存，不把下载结果写回共享模型目录，避免多容器抢写。

当前识别策略：

- 内置实体：API 服务编排 wordtag 模型结果，并在应用层用手机号规则补漏。
- 自定义实体：API 服务把 `custom_entities[].uie_schema` 转成模型服务的 UIE schema，再把模型标签映射回业务实体类型。
- 长文本由模型服务内部使用 Taskflow tokenizer 按自然边界合并/切分；wordtag 和 UIE 共用 `512` token 上限并继续批量推理，API 服务不再做字符窗口重叠分片。
- 除手机号外，不再执行自定义正则、固定值、字符串模式或 `model_labels` 匹配。

## 依赖与模型

- 推荐 Python 3.10。
- `requirements-api.txt`：API 服务轻量依赖。
- `requirements-model.txt`：模型服务完整依赖，包含 API 依赖、PaddlePaddle GPU 和 PaddleNLP，默认使用 NVIDIA/CUDA 11.7 路线。
- wordtag 默认目录：`resources/models/wordtag`。
- UIE 默认目录：`resources/models/uie-base`。

实际模型权重、缓存和推理文件不进入 Git 仓库。模型文件只需要挂载到模型服务容器；API 角色容器虽然来自同一个镜像，但不需要挂载模型目录，也不会在 API 进程里加载 PaddlePaddle/PaddleNLP。

## 快速启动

先启动模型服务：

```powershell
python -m pip install -r requirements-model.txt
$env:MODEL_PORT="18002"
python -u model_app.py
```

再启动 API 服务：

```powershell
python -m pip install -r requirements-api.txt
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

重点关注：

- `model_service_status` 应为 `ok`。
- `model_service.using_taskflow` 应为 `true`。
- `model_service.enable_uie_custom` 应为 `true`。
- 开启 UIE 预加载时，`model_service.using_uie` 应为 `true`。
- NVIDIA GPU 模式下，模型服务 `/healthz` 中 `device` 应为 `nvidia`，`gpu_available` 应为 `true`。

## Docker 快速部署

交付只构建一个镜像，同一镜像通过 `SERVICE_ROLE` 区分 API 服务和模型服务：

```bash
sudo docker build -t llm-messages-encryptor:latest .
```

镜像内包含 API 服务和模型服务所需的全部依赖。模型服务默认使用 NVIDIA/CUDA 11.7，关闭启动阶段 UIE 预热，并限制 Paddle 线程和 GPU 显存预分配，避免容器启动时一次性占满服务器资源。

同一台 Docker 主机部署时，两个容器加入同一个 Docker network，API 服务通过模型容器名访问模型服务：

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

业务系统只需要调用 API 服务：

```text
POST http://API服务IP:18001/v1/llm/preprocess
```

如果服务器资源充足，并且希望模型服务启动阶段就加载 UIE，可以额外设置：

```text
-e DESENSITIZE_PRELOAD_UIE_CUSTOM=true
```

RTX 2080 Ti 当前可用显存约 `8872 MiB` 时，建议先保持 `FLAGS_fraction_of_gpu_memory_to_use=0.4`，并用 `--memory 8g --cpus 4` 给模型容器加护栏；确认稳定后再小幅上调。`--memory` 是宿主机内存上限，GPU 显存主要由 `--gpus '"device=0"'` 和 Paddle 的 `FLAGS_fraction_of_gpu_memory_to_use` 控制。

跨主机部署时，普通 Docker network 不能跨主机使用容器名，API 服务应配置模型服务机器的内网 IP 或内网 DNS：

```text
DESENSITIZE_MODEL_SERVICE_URL=http://模型服务机器内网IP:18002
```

完整镜像交付、Bash 命令、跨主机部署、离线模型挂载和生产环境变量说明见 [开发、部署与模型准备](docs/local-development.md)。

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
        "content": "合同甲方：上海泛微网络科技股份有限公司，联系人张三，手机号13800138000。身份证号110101199003071234，银行卡号6222020202020202020。",
        "encrypted": false
      }
    ]
  },
  "custom_entities": [
    {
      "entity_type": "ID_CARD",
      "uie_schema": ["身份证号"]
    },
    {
      "entity_type": "BANK_CARD",
      "uie_schema": ["银行卡号", "卡号"]
    }
  ]
}
```

响应中的核心字段：

- `data.desensitized_request`：已脱敏后的大模型请求体，`messages` 内会移除 `encrypted` 和 `mapping` 控制字段。
- `data.mapping`：原始实体到占位符的映射，调用方需要自行保存。
- `data.stats`：本次处理消息数、替换次数、新增实体数等统计信息。

接口约定：

- 只处理 `role=user` 的消息；`system` 和 `assistant` 消息不会被脱敏。
- `encrypted=true` 的历史 `user` 消息表示已经脱敏，本次不会再次处理，但其中的 `mapping` 会被用于复用占位符。
- `encrypted=false` 或缺失 `encrypted` 的 `user` 消息会参与本次脱敏。
- 返回给上游大模型的 `data.desensitized_request.messages` 会移除 `encrypted` 和 `mapping` 控制字段。
- `custom_entities` 当前只支持通过 `uie_schema` 声明 UIE 信息抽取目标，不执行 `patterns`、`regex`、`values` 或 `model_labels`。
- 服务不保存 `mapping`，调用方需要在会话侧保存 `data.mapping`，并在后续历史 `user` 消息中带回。

常见状态：

- 请求体不是合法 JSON：返回 `400`。
- `llm_request` 不是对象或 `llm_request.messages` 不是数组：返回 `400`。
- API 服务无法连接模型服务或模型服务内部异常：返回 `500`。
- `/readyz` 在模型未就绪或模型服务不可达时返回 `503`。

联调可直接使用仓库内示例：

```powershell
$body = Get-Content .\example_preprocess_request.json -Raw -Encoding UTF8

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:18001/v1/llm/preprocess" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body | ConvertTo-Json -Depth 20
```

## 接入注意事项

- 业务方只调用 API 服务时，不需要准备或挂载模型目录。
- 模型下载、模型挂载和 UIE 预加载都由模型服务侧负责。
- 如果身份证号、银行卡号或地址未脱敏，先检查 `/healthz` 中的 `model_service.enable_uie_custom` 和 `model_service.using_uie`。
- 服务是无状态的，调用方必须保存 `data.mapping`，并在下一轮历史 `user` 消息中携带该映射和 `encrypted=true`。
- 内网或离线环境需要提前准备模型文件，或为模型服务挂载包含模型文件的只读 Docker volume。

更多本地调试、模型准备和仓库瘦身规则见 [开发、部署与模型准备](docs/local-development.md)。
