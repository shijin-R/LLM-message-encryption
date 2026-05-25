本目录用于存放本地 `uie-base` 信息抽取模型文件。

UIE 旁路用于业务方通过 `custom_entities[].uie_schema` 声明的自定义实体抽取，
例如身份证号、银行卡号、银行账号、平台账号等上下文依赖较强的敏感实体。

默认配置：

1. 默认启用旁路；如需关闭可设置 `DESENSITIZE_ENABLE_UIE_CUSTOM=false`
2. 优先读取本目录（`DESENSITIZE_UIE_MODEL_PATH`）
3. 若本目录没有模型文件，且 `DESENSITIZE_AUTO_DOWNLOAD_MODEL=true`，会调用
   `Taskflow("information_extraction", model="uie-base")` 触发默认下载
4. 若 `DESENSITIZE_SYNC_DOWNLOADED_MODEL=true`，会尝试把下载内容同步回本目录

模型权重和静态推理文件较大，不提交到 Git 仓库。
