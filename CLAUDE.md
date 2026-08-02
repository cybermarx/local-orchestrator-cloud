# E:\claude-nvidia — NVIDIA ↔ Claude Code 桥接项目

## 这是什么
本目录让 **Claude Code** 通过本地代理，调用 **NVIDIA 托管的开源模型**（无需 Anthropic 账号）。
架构：Claude Code → `nvidia-claude-proxy.js`（把 Anthropic Messages API 翻译成 OpenAI 格式）→ NVIDIA `integrate.api.nvidia.com`。

## 关键文件
- `nvidia-claude-proxy.js` — 本地代理（Node，监听 `127.0.0.1:3456`）。负责模型名解析、503/429/超时降级、用量日志。
- `nvidia-proxy-config.json` — 代理配置：`nvidiaApiKey`、`model`(默认模型)、`listenPort`、`upstreamTimeout`(ms)。
- `start-claude-nvidia.bat` — 启动脚本：先确认代理在跑，再启动 `claude`。
- `bing_search_mcp.py` — 免 key 的 Bing 搜索 MCP（爬 HTML）。
- `nvidia-proxy.log` — 代理运行日志。`usage.jsonl` — 每次请求的 token/耗时记录。

## 约定
- 默认模型：`deepseek-ai/deepseek-v4-flash`（稳定快速，见全局 `settings.json` 的 `ANTHROPIC_MODEL`）。
- `/model` 选择器只暴露这三个 NVIDIA 模型（由 `settings.json` 的 `ANTHROPIC_DEFAULT_*_MODEL` 控制，代理同步映射）：
  - `haiku` → `deepseek-ai/deepseek-v4-flash`（快/便宜）
  - `sonnet` → `z-ai/glm-5.2`（通用；NVIDIA 上偶发 120s 超时）
  - `opus` → `deepseek-ai/deepseek-v4-pro`（强）
- 代理对 503/429/超时做**自动降级+重试**：主模型失败/被打满时切到备用模型链，不用人工重试。
- API key 写在 `nvidia-proxy-config.json` 与 bat 脚本里，**不要提交到仓库或泄露**。

## 常用调试
- 看模型列表：`curl http://127.0.0.1:3456/v1/models`
- 看用量统计：`curl http://127.0.0.1:3456/v1/stats`（来自 `usage.jsonl`，含每模型 token/耗时）
- 代理没起：`start-claude-nvidia.bat` 会自动拉起；或手动 `node nvidia-claude-proxy.js`。
- 改端口/超时：编辑 `nvidia-proxy-config.json`（`listenPort` / `upstreamTimeout`，单位毫秒）。
- 已接入 MCP：`bing-search`(免key搜索)、`filesystem`、`memory`；`github` 需先在 `~/.claude.json` 填 `GITHUB_PERSONAL_ACCESS_TOKEN`。

## 环境
- Node: WorkBuddy 托管版（`C:\Users\LEGION\.workbuddy\binaries\node\versions\22.22.2`）。
- Python: 同托管版 venv（`C:\Users\LEGION\.workbuddy\binaries\python\envs\default`），bing_search_mcp 用它跑。
