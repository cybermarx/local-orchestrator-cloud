# local-orchestrator-cloud

本地 Ollama 35B 编排器 + 云端 NVIDIA subagent + 本地 linker 的「契约优先 / 盲 subagent」agent。

> 这是 **云端 subagent 版本** 的快照备份。核心思路：本地 35B 只做规划与调度（对话历史只存「指针」，不存大段代码），每个子任务派给云端模型无状态地独立生成，最后由本地确定性连接器组装、校验、并自动修复。

## 架构（三层）

```
① 本地 35B（ollama_agent.py）
   规划 / 调度。对话历史只放「指针」，永不撑爆上下文。
   带递归摘要压缩 maybe_compact（默认 Qwen3 思考 + 撑爆自动关思考兜底）。

② 云端 subagent（NVIDIA，本仓库版本）
   把子目标 + 契约 作为一次性 /api/chat 请求发出，拿回 {module, code, provides, depends_on}。
   每个 subagent 不知道总目标（信息隔离），1M ctx。

③ 本地 linker.py
   确定性「动态连接器」：解析依赖图、落盘、生成 requirements/main 脚手架、
   语法校验、导入冒烟。零 token、纯标准库。
```

## 关键设计

- **盲 subagent（信息隔离）**：编排器把 `子目标 + 契约(provides/depends_on)` 传给 subagent，
  **总目标从不进 subagent 提示**。每个 subagent 只见自己这块，互不串扰。
- **契约优先**：编排器先设计模块契约（provides=必暴露符号；depends_on=**含读写两端**的依赖），
  再逐个 delegate。写依赖（如 `user_db:create_user`）必须写进 depends_on，否则 subagent 可能自创不落库实现。
- **自动修复闭环**：`link` / `verify` 在导入冒烟或集成断言失败后，把报错回灌给对应云端 subagent 修复
  （原地更新侧边存储条目）→ 重链 → 再校验，直到通过或达到 `AGENT_REPAIR_ROUNDS`。

## 文件

| 文件 | 说明 |
|---|---|
| `ollama_agent.py` | 核心：本地 35B 编排器 + 云端 subagent 调度 + 自动修复闭环 |
| `linker.py` | 本地确定性连接器：依赖解析/落盘/语法校验/导入冒烟 |
| `nvidia-claude-proxy.js` | 云端路由代理（NVIDIA↔OpenAI 翻译、503/429 降级，被本 agent 复用的工具链一部分） |
| `CLAUDE.md` | 项目说明与约定 |

## 用法

```bash
# 交互模式（像对话一样用）
python ollama_agent.py

# 单次任务
python ollama_agent.py "写一个带登录的博客系统"

# 需要云端 subagent 时提供 NVIDIA key
NVIDIA_API_KEY=nvapi-xxx python ollama_agent.py "写一个带登录的博客系统"
```

编排器会：理解总目标 → 设计模块契约 → 逐个 delegate → link 组装 →（可选）verify 集成校验。

## 环境变量

**编排器 / Ollama（沿用原有，略）**：`OLLAMA_URL` / `AGENT_MODEL` / `AGENT_NUM_CTX` /
`AGENT_MAX_ITER` / `AGENT_TEMP` / `AGENT_THINKING` / `AGENT_SUMMARIZE` /
`AGENT_SUMMARY_MODE` / `AGENT_SUMMARIZER_MODEL` / `AGENT_TRUNCATE_CAP` /
`AGENT_KEEP_RECENT` / `AGENT_COMPACT_ROUNDS`

**NVIDIA 凭证与超时**：
- `NVIDIA_API_KEY`：可选；未设置时自动从 `~/.workbuddy/models.json` / `~/.claude.json` 等回退读取 `nvapi-` key
- `NVIDIA_KEY_CONFIG`：可选，显式指定含 key 的 JSON config 路径
- `NVIDIA_BASE_URL`：默认 `https://integrate.api.nvidia.com/v1`
- `AGENT_NVIDIA_FLASH`：flash 档候选模型（逗号分隔，默认 deepseek-v4-flash 等）
- `AGENT_NVIDIA_PRO`：pro 档候选模型
- `AGENT_DELEGATE_TIMEOUT`：单轮超时（默认 120s，超时自动切换下一候选模型）
- `AGENT_LATENCY_WARN` / `AGENT_MODEL_COOLDOWN` / `AGENT_MODEL_MAX_FAILS`：延迟感知 + 熔断器
- `AGENT_DELEGATE_TRIES`：failover 总尝试次数
- `AGENT_429_BACKOFF`：账号级限速退避
- `AGENT_PROJECT_DIR`：项目输出目录（默认 `./projects`）
- `AGENT_REPAIR_ROUNDS`：自动修复闭环最大轮次（默认 3）

## 安全说明

本仓库**不含任何密钥或配置文件**：`nvidia-proxy-config.json`、`settings.*.json`、`*.bat`、
`.workbuddy/`、`.claude/`、`*.db`、`*.log` 等一律未提交。NVIDIA key 仅从运行时环境变量或本地
config 读取，文档中的 `nvapi-xxx` 仅为占位示例。
