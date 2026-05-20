本目录用于存放本地 `wordtag` 模型文件。

当前服务的模型策略：

1. 优先读取本目录（`DESENSITIZE_MODEL_PATH`）
2. 若本目录没有模型文件，且 `DESENSITIZE_AUTO_DOWNLOAD_MODEL=true`，会调用 `Taskflow("ner")` 的 wordtag 模型入口触发默认下载到 `$HOME/.paddlenlp/taskflow/wordtag`
3. 若 `DESENSITIZE_SYNC_DOWNLOADED_MODEL=true`，会把下载内容同步回本目录

可根据环境变量调整：
- `DESENSITIZE_MODEL_PATH`
- `DESENSITIZE_AUTO_DOWNLOAD_MODEL`
- `DESENSITIZE_SYNC_DOWNLOADED_MODEL`
- `DESENSITIZE_MODEL_CACHE_PATH`
- `DESENSITIZE_STRICT_LOCAL_MODEL`
