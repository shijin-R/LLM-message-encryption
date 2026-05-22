# 仓库文件检查

检查范围限定为 `D:\code\PythonProjects\Desensitize2`。

## 当前结论

- 当前没有发现未被 `.gitignore` 覆盖的未跟踪文件。
- 本地存在虚拟环境、uv 缓存、PaddleNLP 缓存、Python 缓存和 wordtag 模型文件，均已被 `.gitignore` 忽略。
- 仓库已跟踪文件整体合理；实际模型权重没有进入 Git 跟踪，只保留 `resources/models/wordtag/README.md` 作为目录说明。

## 应保留

| 路径 | 用途 |
| --- | --- |
| `app.py` | Flask HTTP 入口，提供 `/healthz` 和 `/v1/llm/preprocess` |
| `desensitize/*.py` | 脱敏服务、实体识别、映射和配置代码 |
| `requirements.txt` | Python 依赖声明 |
| `Dockerfile` | Docker 镜像构建入口 |
| `.dockerignore` | 控制镜像构建上下文，排除缓存、虚拟环境、模型文件和测试目录 |
| `.gitignore` | 控制本地缓存、模型文件、日志和环境配置不进入 Git |
| `example_preprocess_request.json` | 普通预处理联调示例 |
| `example_history_reuse_request.json` | 历史映射复用联调示例 |
| `tests/test_service.py` | 服务行为回归测试 |
| `resources/common_data/uie/name.txt` | jieba 补漏可选词典 |
| `resources/common_data/uie/user.txt` | jieba 补漏可选词典 |
| `resources/models/wordtag/README.md` | 模型目录说明，占位保留 |
| `README.md` | GitLab 首页展示文档 |
| `docs/*.md` | 本地开发、模型准备和仓库文件说明 |

## 不应保留或不应提交

| 路径或模式 | 原因 |
| --- | --- |
| `.venv/`, `venv/` | 本地虚拟环境，体积大且与机器相关 |
| `.uv-cache/` | 本地依赖缓存 |
| `.paddlenlp/` | PaddleNLP 下载缓存，应由运行环境自行准备或挂载 |
| `__pycache__/`, `*/__pycache__/`, `*.pyc` | Python 运行缓存 |
| `.pytest_cache/`, `.coverage`, `htmlcov/` | 测试和覆盖率产物 |
| `logs/`, `*.log`, `run_service.*.log` | 本地运行日志 |
| `.env`, `.env.*` | 本地配置，可能包含敏感信息 |
| `.idea/`, `.vscode/` | 个人 IDE 配置 |
| `resources/models/wordtag/*` | wordtag 模型权重、推理文件和缓存体积大，不适合进入 Git |
| `run_and_test.ps1` | 本地临时联调脚本，不属于维护交付物 |

## 本次看到的本地产物

以下内容存在于工作树内，但已经被忽略，不建议提交：

- `.venv/`
- `.uv-cache/`
- `.paddlenlp/`
- `__pycache__/`
- `desensitize/__pycache__/`
- `tests/__pycache__/`
- `resources/models/wordtag/.cache_info`
- `resources/models/wordtag/config.json`
- `resources/models/wordtag/model_state.pdparams`
- `resources/models/wordtag/special_tokens_map.json`
- `resources/models/wordtag/spo_config.pkl`
- `resources/models/wordtag/static/`
- `resources/models/wordtag/tags.txt`
- `resources/models/wordtag/termtree_data`
- `resources/models/wordtag/termtree_type.csv`
- `resources/models/wordtag/tokenizer_config.json`
- `resources/models/wordtag/vocab.txt`
- `run_and_test.ps1`
