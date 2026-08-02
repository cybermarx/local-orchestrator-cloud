#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地编排器 + 云端 subagent + 本地冒烟验证器(去 linker)
直连 Ollama(本地35B)作总指挥,唯一能力是把子任务 delegate 给云端模型(NVIDIA 或 SiliconFlow,可切换),
最后 assemble 把所有子任务组装成项目。纯标准库 + 本地 smoke.py,零额外依赖。

设计要点(去 linker):Python 的 import 系统本身就是 linker——模块=文件=命名空间,
同目录放好文件即连通,无需独立接线。本脚本只负责:① 编排器设计模块契约 + 写 main.py
(组合根);② 把各模块交给云端 subagent 实现;③ assemble 时把所有文件落到同目录,用
smoke.py 做「导入全部模块 + 真正调用 main 入口」的契约感知冒烟,失败按 manifest 回灌修复。

架构三层:
  ① 本地 35B(本脚本):规划/调度,对话历史只放"指针",不存大段代码 → 永不撑爆。
  ② 云端 subagent(NVIDIA 或 SiliconFlow,用 AGENT_DELEGATE_BACKEND 切换):产模块 + manifest(结构化输出),各自独立、长上下文。
  ③ 本地 smoke.py:确定性"冒烟验证器",落盘 + 导入/执行校验,零 token。

上下文管理:递归摘要压缩 maybe_compact + 默认 Qwen3 思考 + 撑爆自动关思考兜底(沿用)。

用法:
  python ollama_agent.py                 # 交互模式(像对话界面一样用)
  python ollama_agent.py "任务"          # 单次任务
  NVIDIA_API_KEY=nvapi-xxx python ollama_agent.py "写一个博客系统"
  SILICONFLOW_API_KEY=sk-xxx python ollama_agent.py "写一个博客系统"   # SiliconFlow 为默认后端(设 AGENT_DELEGATE_BACKEND=nvidia 可切回 NVIDIA);key 不写也可放 local_config.json

环境变量:
  OLLAMA_URL / AGENT_MODEL / AGENT_NUM_CTX / AGENT_MAX_ITER / AGENT_TEMP / AGENT_THINKING / AGENT_SUMMARIZE / AGENT_SUMMARY_MODE / AGENT_SUMMARIZER_MODEL / AGENT_TRUNCATE_CAP / AGENT_KEEP_RECENT / AGENT_COMPACT_ROUNDS  (沿用原有)
  NVIDIA_API_KEY        可选;未设置时自动回退读取 Claude/WorkBuddy config 中的 nvapi- key
  NVIDIA_KEY_CONFIG     可选;显式指定一个含 NVIDIA key 的 JSON config 路径(回退用)
  NVIDIA_BASE_URL       默认 https://integrate.api.nvidia.com/v1(也可由 config 的 url 字段回退)
  AGENT_NVIDIA_FLASH    默认 "deepseek-ai/deepseek-v4-flash, z-ai/glm-5.2, minimaxai/minimax-m3, stepfun-ai/step-3.7-flash"
  AGENT_NVIDIA_PRO      默认 "deepseek-ai/deepseek-v4-pro, nvidia/llama-3.1-nemotron-ultra-253b-v1"
  AGENT_DELEGATE_BACKEND 默认 siliconflow(用 SiliconFlow 作云端 subagent 后端);设 nvidia 可切回 NVIDIA
  SILICONFLOW_API_KEY   可选;优先读环境变量,否则读项目根目录 git-ignored 的 local_config.json(避免每次手输;不写入本文件 / 不入库)
  SILICONFLOW_BASE_URL  默认 https://api.siliconflow.cn/v1
  AGENT_SILICONFLOW_FLASH 默认 "deepseek-ai/DeepSeek-V3, Qwen/Qwen3.5-35B-A3B, Qwen/Qwen3.5-9B"
  AGENT_SILICONFLOW_PRO   默认 "deepseek-ai/DeepSeek-V3.2, deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-V3.1-Terminus"
  AGENT_DELEGATE_TIMEOUT 默认 120 (秒,单轮内失败会自动切换到下一个候选模型)
  AGENT_LATENCY_WARN    默认 30 (秒,超过记降级)
  AGENT_MODEL_COOLDOWN  默认 60 (秒,熔断冷却)
  AGENT_MODEL_MAX_FAILS 默认 3 (连续失败次数触发熔断)
  AGENT_DELEGATE_TRIES  默认 6 (failover 总尝试次数)
  AGENT_429_BACKOFF     默认 5 (秒,账号级限速退避)
  AGENT_PROJECT_DIR     默认 ./projects
  AGENT_REPAIR_ROUNDS   默认 3 (链接后自动修复闭环的最大轮次)
"""

import json
import os
import sys
import re
import ast
import time
import subprocess
import urllib.request
import urllib.error
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("AGENT_MODEL", "fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:IQ2_M")
NUM_CTX = int(os.environ.get("AGENT_NUM_CTX", "8192"))
MAX_ITER = int(os.environ.get("AGENT_MAX_ITER", "30"))
TEMPERATURE = float(os.environ.get("AGENT_TEMP", "0.7"))

# --- 输出预算(截断防护) ---------------------------------------------------
# num_ctx 是 prompt + 输出「共用」的窗口。不显式给 num_predict 时,历史一涨
# 输出空间就被挤压,35B 写到一半就被硬截断(尤其 write_main 的 main_code)。
# 编排器每轮按「剩余窗口」动态计算输出预算,并保底 ORCH_MIN_PREDICT。
ORCH_MAX_PREDICT = int(os.environ.get("AGENT_ORCH_MAX_PREDICT", "2048"))
ORCH_MIN_PREDICT = int(os.environ.get("AGENT_ORCH_MIN_PREDICT", "768"))
CTX_MARGIN = int(os.environ.get("AGENT_CTX_MARGIN", "256"))  # 给模板/特殊 token 留白

# --- 本地 subagent(串行递归:同一个 35B 兼任 subagent) ----------------------
# 本机通常只常驻一个模型,无法并行(Ollama 默认 OLLAMA_NUM_PARALLEL=1),故串行。
LOCAL_SUBAGENT_MODEL = os.environ.get("AGENT_LOCAL_SUBAGENT_MODEL", "").strip() or MODEL
LOCAL_DELEGATE_TRIES = int(os.environ.get("AGENT_LOCAL_TRIES", "2"))  # 本地慢,少重试
# 注意:本 GGUF 不遵守 enable_thinking:false,think:false 仍会思考 8K+ token;
# num_ctx 是 prompt+输出共用窗口,故必须 > 思考链长度,否则思考吃光窗口后
# 模型在产出 content 前就被 done_reason=length 截断 → content 为空(已实测复现)。
SUB_NUM_CTX = int(os.environ.get("AGENT_SUB_NUM_CTX", "16384"))
SUB_NUM_PREDICT = int(os.environ.get("AGENT_SUB_NUM_PREDICT", "4096"))
SUB_TEMPERATURE = float(os.environ.get("AGENT_SUB_TEMP", "0.3"))  # 写代码宜低温,减少退化
MAX_CONTINUE = int(os.environ.get("AGENT_MAX_CONTINUE", "3"))     # 截断后最多续写几次
LOCAL_TIMEOUT = int(os.environ.get("AGENT_LOCAL_TIMEOUT", "900"))

# --- 粒度硬约束(低比特量化下,缩短单次输出是保证完整性的主要手段) -----------
MAX_MODULE_LINES = int(os.environ.get("AGENT_MAX_MODULE_LINES", "60"))
MAX_MAIN_LINES = int(os.environ.get("AGENT_MAX_MAIN_LINES", "40"))
MAX_TEST_LINES = int(os.environ.get("AGENT_MAX_TEST_LINES", "20"))

# --- 递归分治 delegate(任务过大→subagent 自拆成更小子任务,还大则继续拆) ------
# 触发条件:code 回来后被判"过大"(截断/语法错/超行数)。深度到顶强制出代码,保证收敛。
# 本地串行下每深一层调用数 ×N,默认深度 2(拆一层)以控成本;设 0 关闭递归。
MAX_DELEGATE_DEPTH = int(os.environ.get("AGENT_MAX_DELEGATE_DEPTH", "2"))
SPLIT_MIN_SUBTASKS = int(os.environ.get("AGENT_SPLIT_MIN", "2"))   # 拆分至少几块
SPLIT_MAX_SUBTASKS = int(os.environ.get("AGENT_SPLIT_MAX", "4"))   # 拆分最多几块(控爆炸)

# --- 选择性推理 ------------------------------------------------------------
# 思考链在本 GGUF 上恒 8K+ token,且与输出共享 num_ctx。故只在"决策/规划"这类
# 输出短的步骤开思考(装得下思考+短输出);"写代码"这类长输出步骤一律关思考,
# 否则思考撑爆窗口→content 为空(上一轮修过的 bug)。开关逐调用即时生效、调用完即
# 失效(每次请求无状态传 think),不存在"忘了关"的残留。
THINK_ON_SPLIT = os.environ.get("AGENT_THINK_ON_SPLIT", "1").lower() in ("1", "true", "yes", "on")
THINK_ON_REPAIR = os.environ.get("AGENT_THINK_ON_REPAIR", "0").lower() in ("1", "true", "yes", "on")
THINK_NUM_CTX = int(os.environ.get("AGENT_THINK_NUM_CTX", "16384"))  # 思考步骤的 num_ctx 下限

# ----------------------------------------------------------------------------
# NVIDIA 凭证加载(优先 env,否则回退读取常见 Claude/WorkBuddy config)
# ----------------------------------------------------------------------------
def _search_nvidia_key(path):
    """递归扫描 JSON config,返回 (nvapi-key, url-or-None)。找不到返回 (None, None)。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None
    result = {}

    def walk(node):
        if isinstance(node, dict):
            for fk in ("apiKey", "api_key"):
                v = node.get(fk)
                if isinstance(v, str) and v.startswith("nvapi-"):
                    result.setdefault("key", v)
                    u = node.get("url") or node.get("baseUrl") or node.get("base_url")
                    if isinstance(u, str):
                        result.setdefault("url", u)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for it in node:
                walk(it)

    walk(data)
    return result.get("key"), result.get("url")


def _load_nvidia_credentials():
    """解析 NVIDIA 凭证:优先环境变量,否则从常见 config 文件回退读取 nvapi- key。"""
    env_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if env_key:
        return env_key, None, "环境变量 NVIDIA_API_KEY"
    candidates = []
    cfg = os.environ.get("NVIDIA_KEY_CONFIG", "").strip()
    if cfg:
        candidates.append(cfg)
    candidates += [
        os.path.expanduser("~/.workbuddy/models.json"),
        os.path.expanduser("~/.claude.json"),
        os.path.expanduser("~/.claude/settings.json"),
    ]
    for p in candidates:
        if not p or not os.path.isfile(p):
            continue
        key, url = _search_nvidia_key(p)
        if key:
            return key, url, p
    return "", None, None


def _load_local_secret(name, cfg_path=None):
    """从项目根目录的 git-ignored local_config.json 读取密钥/配置(避免每次手输)。
    仅作环境变量的兜底;密钥不写入任何被 git 跟踪的源码文件。找不到或解析失败返回空串。"""
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get(name, "")
        return v.strip() if isinstance(v, str) else ""
    except Exception:
        return ""


_NVIDIA_API_KEY, _FALLBACK_URL, _KEY_SOURCE = _load_nvidia_credentials()
NVIDIA_API_KEY = _NVIDIA_API_KEY
NVIDIA_BASE_URL = (
    os.environ.get("NVIDIA_BASE_URL", "").strip()
    or _FALLBACK_URL
    or "https://integrate.api.nvidia.com/v1"
).rstrip("/")
NVIDIA_FLASH = [x.strip() for x in os.environ.get(
    "AGENT_NVIDIA_FLASH",
    "deepseek-ai/deepseek-v4-flash, z-ai/glm-5.2, minimaxai/minimax-m3, stepfun-ai/step-3.7-flash").split(",") if x.strip()]
NVIDIA_PRO = [x.strip() for x in os.environ.get(
    "AGENT_NVIDIA_PRO",
    "deepseek-ai/deepseek-v4-pro, nvidia/llama-3.1-nemotron-ultra-253b-v1").split(",") if x.strip()]
DELEGATE_TIMEOUT = int(os.environ.get("AGENT_DELEGATE_TIMEOUT", "120"))
LATENCY_WARN = float(os.environ.get("AGENT_LATENCY_WARN", "30"))
COOLDOWN = float(os.environ.get("AGENT_MODEL_COOLDOWN", "60"))
MAX_FAILS = int(os.environ.get("AGENT_MODEL_MAX_FAILS", "3"))
MAX_DELEGATE_TRIES = int(os.environ.get("AGENT_DELEGATE_TRIES", "6"))

# ----------------------------------------------------------------------------
# SiliconFlow 后端(临时可切换的云端 subagent 后端,OpenAI 兼容 /chat/completions)
# ----------------------------------------------------------------------------
DELEGATE_BACKEND = os.environ.get("AGENT_DELEGATE_BACKEND", "siliconflow").strip().lower() or "siliconflow"
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "").strip() or _load_local_secret("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
SILICONFLOW_FLASH = [x.strip() for x in os.environ.get(
    "AGENT_SILICONFLOW_FLASH",
    "deepseek-ai/DeepSeek-V3, Qwen/Qwen3.5-35B-A3B, Qwen/Qwen3.5-9B").split(",") if x.strip()]
SILICONFLOW_PRO = [x.strip() for x in os.environ.get(
    "AGENT_SILICONFLOW_PRO",
    "deepseek-ai/DeepSeek-V3.2, deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-V3.1-Terminus").split(",") if x.strip()]
BACKOFF = float(os.environ.get("AGENT_429_BACKOFF", "5"))
PROJECT_DIR = os.environ.get("AGENT_PROJECT_DIR", "./projects")
REPAIR_ROUNDS = int(os.environ.get("AGENT_REPAIR_ROUNDS", "3"))

MAX_OUTPUT_CHARS = 6000

# compact 相关
USE_SUMMARY = os.environ.get("AGENT_SUMMARIZE", "1").lower() not in ("0", "false", "no")
SUMMARIZER_MODEL = os.environ.get("AGENT_SUMMARIZER_MODEL", MODEL)
KEEP_RECENT = int(os.environ.get("AGENT_KEEP_RECENT", "6"))
COMPACT_ROUNDS = int(os.environ.get("AGENT_COMPACT_ROUNDS", "8"))
SUMMARY_MODE = os.environ.get("AGENT_SUMMARY_MODE", "model").lower()  # model | truncate
TRUNCATE_CAP = int(os.environ.get("AGENT_TRUNCATE_CAP", "400"))
# thinking：Qwen3 类思考模型默认先输出长链思考，可能把生成预算吃光，
# 导致最终回答 content 为空（done_reason=length）。设 AGENT_THINKING=0 可关闭以提速并
# 保证 content 落地；默认开启（纯文本轮次若返回空会自动重试关闭思考）。
ENABLE_THINKING = os.environ.get("AGENT_THINKING", "1").lower() in ("1", "true", "yes", "on")
# 安全预算：留出 20% 给模型生成，其余才给上下文
BUDGET = int(NUM_CTX * 0.8)


# ----------------------------------------------------------------------------
# 侧边存储 + 路由器(上下文外,不进 35B 历史)
# ----------------------------------------------------------------------------
SUBTASKS = []  # [{"id","manifest":{...},"model","latency"}]
MAIN_CODE = ""  # 编排器写的主函数(组合根),由 write_main 工具暂存,assemble 时一并落盘
MAIN_SUBTASK_ID = "__MAIN__"  # 组合根 main.py 在 SUBTASKS 中的特殊条目 id(可修复路由用)



class Nvidia429(Exception):
    pass


class ModelRouter:
    """延迟感知 + 熔断的 subagent 路由器。"""
    def __init__(self, flash, pro):
        self.tiers = {"flash": list(flash), "pro": list(pro)}
        self.stats = {}  # model -> {ema, fails, cooldown_until}
        self._rr = 0     # 跨调用轮转指针,实现多模型自动切换(负载分散,而非永远打候选第一个)

    def _stat(self, m):
        return self.stats.setdefault(m, {"ema": 0.0, "fails": 0, "cooldown_until": 0.0})

    def select(self, tier, exclude=None):
        now = time.time()
        exclude = exclude or set()
        cands = self.tiers.get(tier) or self.tiers["flash"]
        healthy = [m for m in cands
                   if m not in exclude
                   and now >= self._stat(m)["cooldown_until"]
                   and self._stat(m)["fails"] < MAX_FAILS]
        if not healthy:
            # 全部熔断/冷却/本轮回避:退而求其次,挑未冷却的;再不行就放开 exclude 随便给一个
            healthy = [m for m in cands
                       if m not in exclude and now >= self._stat(m)["cooldown_until"]] \
                      or [m for m in cands if m not in exclude] \
                      or list(cands)
        # 优先 EMA 延迟最低(已知慢模型降优先级),再按候选顺序;
        # 在此基础上用轮转指针在健康候选间循环,实现跨模型「自动切」(负载分散,而非永远打第一个)
        healthy.sort(key=lambda m: (self._stat(m)["ema"], cands.index(m)))
        pick = healthy[self._rr % len(healthy)]
        self._rr = (self._rr + 1) % len(healthy)
        return pick

    def report(self, model, latency, error):
        s = self._stat(model)
        if error:
            s["fails"] += 1
            if s["fails"] >= MAX_FAILS:
                s["cooldown_until"] = time.time() + COOLDOWN
        else:
            s["fails"] = 0
            if s["ema"] == 0:
                s["ema"] = latency
            else:
                s["ema"] = s["ema"] * 0.7 + latency * 0.3
            if latency > LATENCY_WARN:
                s["ema"] = max(s["ema"], LATENCY_WARN * 1.5)  # 软降级:轻微惩罚

    def status_line(self):
        now = time.time()
        parts = []
        for tier, models in self.tiers.items():
            for m in models:
                s = self._stat(m)
                flag = "⏸" if now < s["cooldown_until"] else ("✗" if s["fails"] >= MAX_FAILS else "✓")
                ema = f"{s['ema']:.1f}s" if s["ema"] else "-"
                parts.append(f"{m.split('/')[-1]}={flag}{ema}")
        return " | ".join(parts)


ROUTER = ModelRouter(NVIDIA_FLASH, NVIDIA_PRO)
ROUTER_SF = ModelRouter(SILICONFLOW_FLASH, SILICONFLOW_PRO)


class LocalRouter:
    """本地单模型「路由器」:接口与 ModelRouter 对齐,但只有一个候选。
    本机通常只常驻一个模型(Ollama 默认不并行),所以无从路由——串行递归即由此而来。"""

    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.fails = 0
        self.ema = 0.0

    def select(self, tier="flash", exclude=None):
        return self.model  # 唯一候选:失败也只能重试它(本地失败多为截断,重试有效)

    def report(self, model, latency, err=None):
        self.calls += 1
        if err:
            self.fails += 1
        self.ema = latency if self.calls == 1 else 0.7 * self.ema + 0.3 * latency

    def status_line(self):
        return (f"{self.model} | 调用={self.calls} 失败={self.fails} "
                f"平均={self.ema:.1f}s (本地串行)")


ROUTER_LOCAL = LocalRouter(LOCAL_SUBAGENT_MODEL)


SYSTEM_PROMPT = """你是一个运行在用户本机、由 Ollama(本地35B)提供算力的「任务编排器」。
你掌握用户的**总目标**,但你派出去的每一个云端 subagent 都**只知道自己的子目标**,不知道总目标——这是刻意的信息隔离,让每个 subagent 专注于自己的子任务、互不串扰。

你有这些工具:
- delegate(task, tier?, name?, provides?, depends_on?): 把一个子任务的**子目标**派给云端 subagent 完成。返回简短指针——真实成果存入「侧边存储」,不占对话上下文。
- write_main(project_name, main_code): 你(编排器,掌握总目标)亲自写**主函数 main.py(组合根)**。它 import 各模块、按你设计的契约调用它们,把整个项目串起来。main.py 必须严格按 provides/depends_on 约定的**名字与签名**调用各模块符号——这是「去 linker」设计下唯一需要你保证接口一致的地方(同目录放文件即可,import 会自行接线)。这里只暂存代码,落盘由 assemble 统一做。
- assemble(project_name): 所有模块 delegate 完成、且你已 write_main 后调用。它把所有模块文件 + main.py 落到同一目录(无需接线,Python import 即 linker),然后做「导入全部模块 + 真正调用 main 入口」的契约感知冒烟;对冒烟失败(符号名/签名漂移、缺失模块、循环依赖)自动启动修复闭环,返回项目树。
- verify(project_name, test_code, name?): 【可选】assemble 后,用一个**短**Python 片段对生成的项目做**一片**集成校验(你掌握总目标,应写出能验证核心流程的断言,如 register 后 login 能拿到 token)。失败会自动回灌 subagent 修复并重新组装。**校验要分片**:一次只验一个切面,分多次调用,用 name 标注这片验什么。

工作方式(契约优先,去 linker):
1. 理解用户的**总目标**;
2. 先做**模块契约设计**(这是唯一需要总目标的地方):把项目拆成若干**小**模块(见下方「输出长度纪律」),为每个模块确定
   - module: 文件名
   - provides: 它必须对外暴露的符号列表(函数/类名,尽量带签名,如 "get_user(id) -> User")
   - depends_on: 它**实际会调用**的其它模块符号——**既要列读依赖,也要列写依赖**(例如注册模块既要依赖 "user_db:authenticate" 也要依赖 "user_db:create_user",因为注册必须把用户写进库)。格式 "模块:符号(签名)"。
   契约是 subagent 之间对接的**唯一依据**,务必让 provides 与 depends_on 互相吻合(谁提供、谁消费要一致;尤其注意写流程的两端都要连上)。
3. 逐个 delegate:每次只把**该模块的「子目标」+「契约」**(必提供的符号/签名、可依赖的符号/签名)传给 subagent。**绝不要把总目标写进 subagent 的提示**——subagent 只该看到自己的子目标与契约。
4. 你亲自写 main.py:基于上面设计的契约,import 各模块、按约定名字/签名调用它们,串成完整流程。用 write_main 暂存(务必让 import 名与 provides 完全一致)。
5. 调用 assemble 组装(会自动冒烟与修复);若任务有明显 happy-path,再**分多次**调用 verify,每次跑一小片集成断言,让系统把逻辑错误也自动修掉。若 assemble/verify 报告「缺失模块 X」或「No module named X」,说明 X 从未被 delegate——你必须在下一轮用 delegate 创建 X 模块(给出子目标与契约),然后再次 assemble,直到不再有缺失模块。
6. 简单聊天可直接回答,不必 delegate。

**输出长度纪律(硬约束,必须遵守)**:
你和 subagent 可能都跑在本机的低比特量化模型上,**单次输出越长,被截断成半截代码的概率越高**。所以:
- **每个模块**:单一职责、只暴露 1-3 个符号、目标 ≤%MAXMOD% 行。宁可多拆 3 个小模块,也绝不要 1 个大模块。若发现某个子目标要写很多代码,先把它再拆开。
- **main.py**:只做「import 各模块 + 按契约把流程串起来」,目标 ≤%MAXMAIN% 行。**任何实际逻辑都要下沉成新模块 delegate 出去**,不要写在 main.py 里。
- **verify**:不要写一个大测试。每次只验**一个切面**(≤%MAXTEST% 行),分多次调用并用 name 标注(如 "注册流程"、"登录流程")。
- 若工具返回里出现「⚠ …被截断 / 语法错误 / 超过上限行」,说明**这一块太大了**:把它拆成更小的子目标重新 delegate,不要原样重试。

规则:
- delegate 的 task 就是该模块的「子目标」,要自包含、清晰。
- provides/depends_on 是你(编排器)预先设计好的契约;若省略,subagent 会自行声明,但你可能需要在 assemble 前核对一致性。**务必把写依赖也写进 depends_on**,否则 subagent 可能自创一个不落库的实现(如注册只用内存字典)。
- 若 delegate 返回错误,换种描述重试,或把任务拆更细再 delegate。
- 你无法自己联网;需要实时信息时,云端模型会用其训练知识回答(可能非最新),请向用户说明。
- 最终回答要简洁,引用 assemble/verify 返回的项目路径与校验结论。
"""

SUBAGENT_SYS = """你是一个代码 subagent。你只负责完成分配给你的**单个子任务**,你**不知道、也不需要知道**整个项目的总目标。

你会收到:
- [子目标] 这个模块要做什么(用你自己的理解实现即可)。
- [必须提供的接口 provides] 你必须定义这些符号,签名须严格匹配(名字、参数、返回含义)。
- [可依赖的接口 depends_on] 你实现时**只允许** import/调用这些外部符号,格式 "模块:符号(签名)"。除此之外不要假设存在任何其它模块或符号。**depends_on 里列出的每一个符号都必须在代码中真正被调用**(包括写操作,如把数据写回数据库的函数);如果你声明了某个依赖却没用到,说明契约没兑现。

请只输出一个 JSON 对象(不要任何解释文字、不要用 markdown 代码块包裹):
{
  "module": "文件名,如 auth.py",
  "language": "python",
  "provides": ["你实际提供的符号,应与要求的 provides 一致"],
  "depends_on": ["你实际依赖的符号,格式 '模块:符号'"],
  "summary": "一句话说明这个子任务做了什么",
  "code": "完整的源代码字符串"
}
要求:
- code 必须定义 provides 中列出的每一个符号,且签名一致。
- code 只可使用 depends_on 中列出的外部符号(其它一律当不存在),不要去"猜"或"协调"其它子任务。
- code 必须是自洽、可独立存在的完整文件内容。
- **务必简短**:目标 ≤%MAXMOD% 行(不含空行)。只实现 provides 要求的符号,**不要**添加额外功能、不要写长篇 docstring、不要写逐行注释、不要写 `if __name__ == "__main__"` 演示块、不要写自测代码。输出越长越可能被截断成半截代码——宁可写得紧凑,也不能写不完。
- 只输出 JSON。
"""

SUBAGENT_SPLIT_SYS = """你是一个任务拆分器。你刚收到的这个子任务**太大**,一个 ≤%MAXMOD% 行的模块装不下(可能被截断或职责过多)。
你现在的工作**不是写代码**,而是把它拆成 %SPLITMIN%~%SPLITMAX% 个更小、职责单一的子任务,每个都能用 ≤%MAXMOD% 行代码独立完成。

请只输出一个 JSON 对象(不要任何解释文字、不要用 markdown 代码块包裹):
{
  "subtasks": [
    {
      "name": "简短英文名,如 probe(会自动加父任务前缀,不要重复父名)",
      "task": "这个子模块具体要做什么,写清楚到能独立实现",
      "provides": ["本子模块对外提供的符号,格式 'func(args)->ret'"],
      "depends_on": ["依赖的兄弟子模块符号,格式 '兄弟name:func',没有则留空"]
    }
  ]
}
要求:
- 拆成 %SPLITMIN%~%SPLITMAX% 个子任务,每个职责单一、可用 ≤%MAXMOD% 行完成。
- 子任务之间用 provides/depends_on 串成清晰的数据流(前一个的产出作为后一个的输入)。
- 必须覆盖原任务的全部职责,不遗漏、不新增无关功能。
- **绝不输出任何代码**,只输出这份拆分计划(计划很短,不会被截断)。
- 只输出 JSON。
"""

# 粒度上限注入 prompt(prompt 含字面花括号,不能用 f-string,故用占位符替换)
for _ph, _v in (("%MAXMOD%", MAX_MODULE_LINES), ("%MAXMAIN%", MAX_MAIN_LINES),
                ("%MAXTEST%", MAX_TEST_LINES),
                ("%SPLITMIN%", SPLIT_MIN_SUBTASKS), ("%SPLITMAX%", SPLIT_MAX_SUBTASKS)):
    SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_ph, str(_v))
    SUBAGENT_SYS = SUBAGENT_SYS.replace(_ph, str(_v))
    SUBAGENT_SPLIT_SYS = SUBAGENT_SPLIT_SYS.replace(_ph, str(_v))


# ----------------------------------------------------------------------------
# 工具 schema(精简,控制总 token 数)
# ----------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": "把一个子任务派给英伟达云端模型(subagent)完成。用于需要生成代码/内容/计算的子任务。返回简短指针(真实成果存入侧边存储,不占上下文)。复杂任务请先自行分解为多个子任务,逐个 delegate。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "该模块的「子目标」:清晰、自包含、只描述本模块要做什么(不要写入总目标)。务必足够小——单一职责、只暴露 1-3 个符号,代码目标 60 行内,过大会被截断"},
                    "tier": {"type": "string", "enum": ["flash", "pro"], "description": "模型档位:flash=快/便宜,pro=强/慢。默认 flash"},
                    "name": {"type": "string", "description": "可选的子任务名/编号,便于后续引用"},
                    "provides": {"type": "array", "items": {"type": "string"}, "description": "本模块必须对外暴露的符号契约,如 ['get_user(id) -> User'];subagent 须严格按此实现。总目标不传于此。"},
                    "depends_on": {"type": "array", "items": {"type": "string"}, "description": "本模块可依赖的其它模块符号契约,格式 '模块:符号(签名)'。subagent 只允许调用这些外部符号。"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_main",
            "description": "编排器亲自写主函数 main.py(组合根):import 各模块、按契约调用它们串起整个项目。main.py 必须严格按 provides/depends_on 约定的名字与签名调用各模块符号。这里只把代码暂存到侧边存储,落盘由 assemble 统一做。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "项目目录名(须与后续 assemble 一致)"},
                    "main_code": {"type": "string", "description": "完整的 main.py 源码字符串(须定义 def main(): 且 if __name__=='__main__': main())。只做 import + 流程编排,目标 40 行内;逻辑一律下沉成模块 delegate 出去,写长了会被截断"},
                },
                "required": ["project_name", "main_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assemble",
            "description": "把所有已完成的子任务(侧边存储中)与 write_main 暂存的 main.py 组装成一个项目:落到同一目录(Python import 即 linker,不做接线)、做语法校验,并自动做「导入全部模块 + 调用 main 入口」的契约感知冒烟——冒烟失败(符号名/签名漂移、缺失模块、循环依赖)会启动自动修复闭环(回灌云端 subagent 修复后重新组装)。用于「写一个大项目」类任务收尾。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "项目目录名"},
                },
                "required": ["project_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify",
            "description": "对已 assemble 的项目做【一片】集成校验:写一个简短 Python 片段导入生成模块并对某一个核心流程断言(如 register 后 login 能拿到 token)。校验失败会自动把报错回灌给 subagent 修复并重新组装。请分多次调用,每次只验一个切面——一次写太长会被输出上限截断,而截断的测试会被误判成项目 bug。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "项目目录名(须已 assemble)"},
                    "test_code": {"type": "string", "description": "本片校验的 Python 代码(导入项目模块并断言;项目目录已在 sys.path)。只验一个切面,目标 20 行内"},
                    "name": {"type": "string", "description": "本片校验的名字,如 '注册流程'、'登录流程',便于区分多次调用"},
                },
                "required": ["project_name", "test_code"],
            },
        },
    },
]


# ----------------------------------------------------------------------------
# 云端 subagent 调用(NVIDIA)
# ----------------------------------------------------------------------------
def _openai_chat(key, base_url, backend_label, model, messages, timeout=DELEGATE_TIMEOUT, use_json=True):
    """通用 OpenAI 兼容 /chat/completions 调用:NVIDIA 与 SiliconFlow 共用。"""
    if not key:
        raise RuntimeError(
            f"未找到 {backend_label} API key：请设置对应的环境变量（NVIDIA_API_KEY 或 SILICONFLOW_API_KEY）"
        )
    payload = {"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 4096}
    if use_json:
        payload["response_format"] = {"type": "json_object"}
    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    req = urllib.request.Request(
        base_url + "/chat/completions", data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        if e.code == 429:
            raise Nvidia429(f"429 限速: {body}")
        if e.code in (503, 529):
            # 服务临时过载(免费层常见):当作可重试的限速,触发退避而非换模型
            raise Nvidia429(f"{e.code} 服务过载: {body}")
        if e.code == 400 and use_json:
            # 该模型可能不支持 json_object,重试不带格式
            return _openai_chat(key, base_url, backend_label, model, messages, timeout, use_json=False)
        raise RuntimeError(f"{backend_label} HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"{backend_label} 连接失败: {e}")
    return d["choices"][0]["message"]["content"]


def nvidia_chat(model, messages, timeout=DELEGATE_TIMEOUT, use_json=True):
    return _openai_chat(NVIDIA_API_KEY, NVIDIA_BASE_URL, "NVIDIA", model, messages, timeout, use_json)


def siliconflow_chat(model, messages, timeout=DELEGATE_TIMEOUT, use_json=True):
    return _openai_chat(SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, "SiliconFlow", model, messages, timeout, use_json)


def _decode_json_string_prefix(s, start):
    """从 s[start](起始引号之后)解码 JSON 字符串内容,遇未转义 `"` 结束。
    字符串因截断而未闭合时,返回已解出的部分与 None。返回 (value, end_index_or_None)。"""
    out, i, n = [], start, len(s)
    esc = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
           "n": "\n", "r": "\r", "t": "\t"}
    while i < n:
        c = s[i]
        if c == "\\":
            if i + 1 >= n:
                break
            e = s[i + 1]
            if e in esc:
                out.append(esc[e]); i += 2; continue
            if e == "u":
                h = s[i + 2:i + 6]
                if len(h) == 4:
                    try:
                        out.append(chr(int(h, 16))); i += 6; continue
                    except ValueError:
                        pass
                break
            out.append(e); i += 2; continue
        if c == '"':
            return "".join(out), i
        out.append(c); i += 1
    return "".join(out), None


def _salvage_truncated_json(t):
    """从被截断/残缺的 JSON 中抢救字段(低比特本地模型的常见故障)。
    code 通常是最后一个字段,截断就发生在它中间——逐字段解码可救回大部分源码。
    返回 dict(含 _truncated 标记)或 None。"""
    m_code = re.search(r'"code"\s*:\s*"', t)
    if not m_code:
        return None
    code, end = _decode_json_string_prefix(t, m_code.end())
    if not code.strip():
        return None
    obj = {"code": code, "_truncated": end is None}
    for key in ("module", "language", "summary"):
        m = re.search(r'"%s"\s*:\s*"' % key, t)
        if m:
            val, _ = _decode_json_string_prefix(t, m.end())
            obj[key] = val
    for key in ("provides", "depends_on"):
        m = re.search(r'"%s"\s*:\s*\[' % key, t)
        if m:
            seg = t[m.end():]
            close = seg.find("]")
            obj[key] = re.findall(r'"((?:[^"\\]|\\.)*)"', seg if close < 0 else seg[:close])
    return obj


def parse_subagent_output(text):
    if not text:
        return {"module": "module.py", "code": "", "provides": [], "depends_on": [],
                "summary": "", "truncated": False}
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    truncated = False
    try:
        obj = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        obj = {}
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = {}
        if not obj:
            # JSON 解析失败:多半是被截断。逐字段抢救,避免把整段原始文本当成代码落盘。
            salvaged = _salvage_truncated_json(t)
            if salvaged:
                truncated = salvaged.pop("_truncated", False)
                obj = salvaged
    if not isinstance(obj, dict):
        obj = {}
    code = obj.get("code", "")
    if not code:
        # 连 JSON 都没有:尝试从 ```python``` 代码块里取,再退回原文
        m = re.search(r"```(?:python)?\s*\n(.*?)(?:\n```|\Z)", t, re.S)
        code = m.group(1) if m else t
    return {
        "module": obj.get("module") or "module.py",
        "language": obj.get("language", "python"),
        "provides": obj.get("provides") or [],
        "depends_on": obj.get("depends_on") or [],
        "summary": obj.get("summary", ""),
        "code": code,
        "truncated": truncated,
    }


def subagent_messages(task, contract=None):
    parts = [f"[子目标]\n{task}"]
    if contract:
        provides = contract.get("provides") or []
        depends_on = contract.get("depends_on") or []
        if provides:
            parts.append("[必须提供的接口 provides(严格按此签名实现)]\n" +
                         "\n".join(f"- {p}" for p in provides))
        if depends_on:
            parts.append("[可依赖的接口 depends_on(只允许调用这些外部符号)]\n" +
                         "\n".join(f"- {d}" for d in depends_on))
    return [
        {"role": "system", "content": SUBAGENT_SYS},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def subagent_repair_messages(repair, contract=None):
    """修复模式:把原始子目标 + 当前代码 + 报错 喂给 subagent,让它基于现有代码修正。
    总目标仍不传入——subagent 依旧只见自己这块的上下文。"""
    parts = [
        "[修复任务] 你是一个代码 subagent。下面这个模块存在缺陷,需要在不改变其职责与契约的前提下修正它。",
        f"[原始子目标]\n{repair.get('original_task', '')}",
        f"[当前代码]\n```python\n{repair.get('current_code', '')}\n```",
        f"[报错 / 校验失败信息]\n{repair.get('error', '')}",
    ]
    if contract:
        provides = contract.get("provides") or []
        depends_on = contract.get("depends_on") or []
        if provides:
            parts.append("[必须保持的接口 provides(签名严格不变)]\n" +
                         "\n".join(f"- {p}" for p in provides))
        if depends_on:
            parts.append("[可依赖的接口 depends_on(只允许调用这些外部符号)]\n" +
                         "\n".join(f"- {d}" for d in depends_on))
    parts.append(
        "[要求] 只修正错误,不要改变模块职责,保持模块名与 provides/depends_on 不变;"
        "保持 depends_on 中声明的每个外部符号都被真正调用(含写操作)。"
        "只输出修正后的完整 JSON(同原结构: module / language / provides / depends_on / summary / code)。"
    )
    return [
        {"role": "system", "content": SUBAGENT_SYS},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def subagent_split_messages(task, contract=None, prev_manifest=None):
    """递归分治:任务过大时,让同一 subagent 把它拆成更小子任务(只输出计划,不写代码)。
    总目标仍不传入——subagent 只在自己这块子目标范围内拆分。"""
    parts = [f"[要拆分的子目标]\n{task}"]
    if contract:
        provides = contract.get("provides") or []
        if provides:
            parts.append("[原本要求提供的接口 provides(拆分后由各子模块合起来满足)]\n" +
                         "\n".join(f"- {p}" for p in provides))
    if prev_manifest is not None:
        if prev_manifest.get("truncated"):
            why = "被截断/不完整"
        else:
            why = f"{_code_lines(prev_manifest.get('code', ''))} 行,超出上限"
        parts.append(f"[为何要拆] 上次直接实现产出的代码{why},说明任务过大,必须拆小。")
    return [
        {"role": "system", "content": SUBAGENT_SPLIT_SYS},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_split_output(text):
    """解析拆分计划,返回 [{name,task,provides,depends_on}, ...](解析失败或无子任务返回 [])。"""
    if not text:
        return []
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    obj = {}
    try:
        obj = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = {}
    subs = obj.get("subtasks") if isinstance(obj, dict) else None
    if not isinstance(subs, list):
        return []
    out = []
    for s in subs:
        if not isinstance(s, dict):
            continue
        task = (s.get("task") or "").strip()
        if not task:
            continue
        out.append({
            "name": (s.get("name") or "").strip(),
            "task": task,
            "provides": s.get("provides") or [],
            "depends_on": s.get("depends_on") or [],
        })
    return out


def _wait_printer(stop, model, label, interval=20):
    """在阻塞的网络调用期间周期性打印等待提示,避免看起来像卡死。"""
    waited = 0
    while not stop.wait(interval):
        waited += interval
        print(f"   ⌛ 仍在等待 {model} 生成子任务'{label}'… 已 {waited}s", flush=True)


def _code_lines(code):
    return len([ln for ln in (code or "").splitlines() if ln.strip()])


def _grain_warning(manifest):
    """粒度 / 完整性体检。语法错误是「被截断」最确定的信号——低比特量化下,
    与其让半截代码流进 assemble 再去猜,不如当场告诉编排器:拆得更小、重来。"""
    notes = []
    code = manifest.get("code", "") or ""
    if manifest.get("truncated"):
        notes.append("输出被截断(已从残缺 JSON 抢救,可能不完整)")
    n = _code_lines(code)
    if n > MAX_MODULE_LINES:
        notes.append(f"模块 {n} 行 > 上限 {MAX_MODULE_LINES} 行,截断风险高,"
                     f"建议拆成 2 个以上更小模块重新 delegate")
    try:
        compile(code, manifest.get("module", "m.py"), "exec")
    except SyntaxError as e:
        notes.append(f"代码语法错误(第 {e.lineno} 行: {e.msg}),典型截断特征,"
                     f"请用更小的子目标重新 delegate")
    except Exception:
        pass
    return (" | ⚠ " + " ; ".join(notes)) if notes else ""


def _too_big(manifest):
    """判定 subagent 产出是否"过大到该拆"——与 _grain_warning 同源:截断/语法错/超行。"""
    code = manifest.get("code", "") or ""
    if manifest.get("truncated"):
        return True
    if _code_lines(code) > MAX_MODULE_LINES:
        return True
    try:
        compile(code, manifest.get("module", "m.py"), "exec")
    except SyntaxError:
        return True
    except Exception:
        pass
    return False


def _subagent_once(router, chat_fn, tier, tmo, msgs, label, think=False):
    """发起一次 subagent 调用(带等待提示),返回 content。用于拆分决策等辅助调用(不做 failover)。
    think=True 仅对本地 ollama 后端开思考(云端后端思考由服务端控制,忽略此参数)。"""
    model = router.select(tier)
    stop = threading.Event()
    wp = threading.Thread(target=_wait_printer, args=(stop, model, label), daemon=True)
    t0 = time.perf_counter()
    try:
        print(f"   ⌛ 调用 {model} 拆分'{label}'…", flush=True)
        wp.start()
        extra = {"think": True} if (think and chat_fn is ollama_subagent_chat) else {}
        content = chat_fn(model, msgs, timeout=tmo, **extra)
        router.report(model, time.perf_counter() - t0, None)
        return content
    finally:
        stop.set(); wp.join(timeout=1)


def _split_and_recurse(task, name, contract, prev_manifest, tier, depth,
                       router, chat_fn, tmo):
    """任务过大→让同一 subagent 拆成子任务→逐个递归 delegate→(有 provides 则)生成聚合模块。
    返回给编排器看的字符串;拆分未成功则返回 None,由调用方落回"直接使用原代码"。"""
    parent = (name or task[:12]).strip() or "part"
    print(f"   ⚙ 子任务'{parent}'判定过大,拆分分摊(深度 {depth + 1}/{MAX_DELEGATE_DEPTH})…", flush=True)
    try:
        content = _subagent_once(
            router, chat_fn, tier, tmo,
            subagent_split_messages(task, contract, prev_manifest), f"{parent}:split",
            think=THINK_ON_SPLIT)  # 拆分=规划步骤,输出短,可安全开思考
    except Exception as e:
        print(f"   ⚠ 拆分调用失败({type(e).__name__}),落回直接使用原代码", flush=True)
        return None
    subs = parse_split_output(content)
    if len(subs) < SPLIT_MIN_SUBTASKS:
        print(f"   ⚠ 未得到有效拆分计划(<{SPLIT_MIN_SUBTASKS} 子任务),落回直接使用原代码", flush=True)
        return None
    subs = subs[:SPLIT_MAX_SUBTASKS]

    child_names, child_provides, seen = [], [], set()
    for i, s in enumerate(subs):
        raw = s.get("name") or f"part{i + 1}"
        cname = raw if raw.startswith(parent + "_") else f"{parent}_{raw}"
        cname = re.sub(r"[^0-9A-Za-z_]", "_", cname) or f"{parent}_part{i + 1}"
        while cname in seen:                # 防重名覆盖
            cname += f"_{i + 1}"
        seen.add(cname)
        child_names.append(cname)
        # 递归:子任务还太大会在 depth+1 层继续自拆
        delegate(s["task"], tier=tier, name=cname,
                 provides=s.get("provides"), depends_on=s.get("depends_on"),
                 _depth=depth + 1)
        # 取回该子模块实际 provides,供聚合模块 depends_on
        st = next((x for x in reversed(SUBTASKS) if x.get("name") == cname), None)
        got = (st.get("manifest", {}).get("provides") if st else None) or s.get("provides") or []
        for p in got:
            child_provides.append(p if ":" in p else f"{cname}:{p}")

    # 聚合:父任务若对外承诺了 provides,拆分后须有人把它们合起来,否则破坏契约
    agg_note = ""
    parent_provides = (contract or {}).get("provides") or []
    if parent_provides:
        agg_task = (
            f"编写聚合模块:import 下列兄弟子模块并组合它们,对外实现原接口。"
            f"只做组装与串联,不要重新实现子模块内部逻辑,保持 ≤{MAX_MODULE_LINES} 行。\n"
            f"[原任务]\n{task}")
        delegate(agg_task, tier=tier, name=parent,
                 provides=parent_provides, depends_on=child_provides,
                 _depth=MAX_DELEGATE_DEPTH)   # 深度到顶:强制出代码,聚合不再拆
        agg_note = f" + 聚合模块 {parent}"

    tag = f"[{name}] " if name else ""
    return (f"{tag}任务过大,已递归拆成 {len(child_names)} 个子模块"
            f"({', '.join(child_names)}){agg_note},均已存入侧边存储(不占上下文)")


def delegate(task: str, tier: str = "flash", name: str = None,
             provides: list = None, depends_on: list = None,
             repair: dict = None, replace_id: int = None, _depth: int = 0) -> str:
    """把子任务派给云端 subagent(延迟感知路由器 + 单轮内跨模型 failover)。
    provides/depends_on 为编排器预先设计的契约(可选);传给 subagent 作接口约束,总目标不传入。
    repair: 修复模式,传 {"original_task","current_code","error"} 让 subagent 基于现有代码修正(总目标仍不传)。
    replace_id: 指定则原地更新该 SUBTASKS 条目(用于修复),否则追加新条目。
    单轮内若某模型超时/失败会切换到下一个候选,避免反复卡在同一个过载模型上。"""
    backend = DELEGATE_BACKEND
    if backend == "ollama":
        router, chat_fn = ROUTER_LOCAL, ollama_subagent_chat
        tries, tmo = LOCAL_DELEGATE_TRIES, LOCAL_TIMEOUT
    elif backend == "siliconflow":
        router, chat_fn = ROUTER_SF, siliconflow_chat
        tries, tmo = MAX_DELEGATE_TRIES, DELEGATE_TIMEOUT
    else:
        router, chat_fn = ROUTER, nvidia_chat
        tries, tmo = MAX_DELEGATE_TRIES, DELEGATE_TIMEOUT
    last_err = "未知错误"
    contract = None
    if provides or depends_on:
        contract = {"provides": provides or [], "depends_on": depends_on or []}
    tried = set()  # 本轮回避:同一模型本轮不再重试
    label = name or task[:14]
    for _ in range(tries):
        model = router.select(tier, exclude=tried)
        t0 = time.perf_counter()
        stop = threading.Event()
        wp = threading.Thread(target=_wait_printer, args=(stop, model, label), daemon=True)
        try:
            print(f"   ⌛ 调用 {model} 生成子任务'{label}'…", flush=True)
            wp.start()
            msgs = subagent_repair_messages(repair, contract) if repair else subagent_messages(task, contract)
            # 修复=诊断步骤,可选开思考(默认关:修复输出含整段代码,长输出+思考有撑爆窗口风险);
            # 写新模块=纯长输出,恒关思考。
            extra = ({"think": True}
                     if (repair and THINK_ON_REPAIR and chat_fn is ollama_subagent_chat) else {})
            content = chat_fn(model, msgs, timeout=tmo, **extra)
            latency = time.perf_counter() - t0
            stop.set(); wp.join(timeout=1)
            router.report(model, latency, None)
            manifest = parse_subagent_output(content)
            warn = _grain_warning(manifest)
            # 递归分治:任务过大(截断/语法错/超行)且深度未满、非修复模式 → 自拆分摊。
            # 子任务还太大会在更深一层继续自拆,深度到顶强制出代码,天然收敛。
            if (repair is None and replace_id is None
                    and _depth < MAX_DELEGATE_DEPTH and _too_big(manifest)):
                recursed = _split_and_recurse(
                    task, name, contract, manifest, tier, _depth,
                    router, chat_fn, tmo)
                if recursed is not None:
                    return recursed
            if replace_id is not None:
                for st in SUBTASKS:
                    if st.get("id") == replace_id:
                        st["manifest"] = manifest
                        st["contract"] = contract
                        st["model"] = model
                        st["latency"] = round(latency, 1)
                        tag = f"[{name}] " if name else ""
                        return (f"{tag}子任务#{replace_id} 修复完成 | model={model} ({latency:.1f}s) | "
                                f"module={manifest['module']} | 已更新侧边存储{warn}")
            sid = len(SUBTASKS) + 1
            SUBTASKS.append({
                "id": sid,
                "name": name,
                "task": task,
                "manifest": manifest,
                "contract": contract,
                "model": model,
                "tier": tier,
                "latency": round(latency, 1),
            })
            tag = f"[{name}] " if name else ""
            return (f"{tag}子任务#{sid} 完成 | model={model} ({latency:.1f}s) | "
                    f"module={manifest['module']} | provides={manifest['provides']} | "
                    f"{_code_lines(manifest.get('code'))}行 | 已存入侧边存储(不占上下文){warn}")
        except Nvidia429 as e:
            stop.set(); wp.join(timeout=1)
            latency = time.perf_counter() - t0
            # 限速/过载是账号级且暂时的,不计入熔断(避免误冷却),仅退避后重试(不加入 tried,允许同模型)
            last_err = str(e)
            time.sleep(BACKOFF)
            continue
        except KeyboardInterrupt:
            stop.set(); wp.join(timeout=1)
            return f"[delegate 已取消] 子任务'{label}'被用户中断"
        except Exception as e:
            stop.set(); wp.join(timeout=1)
            latency = time.perf_counter() - t0
            router.report(model, latency, True)
            tried.add(model)  # 本轮内不再重试同一模型,切换到下一个候选
            last_err = f"{type(e).__name__}: {e}"
            continue
    return f"[delegate 失败] 所有候选模型均不可用({last_err})。可换种描述重试,或拆分更细的子任务。"


def _get_main_code():
    """组合根 main.py 的当前代码:优先取 SUBTASKS 中 MAIN 条目(assemble/repair 后会更新),
    否则回退到 MAIN_CODE 全局。保证修复闭环重写 main.py 后 assemble 用最新版本。"""
    for st in SUBTASKS:
        if st.get("id") == MAIN_SUBTASK_ID:
            return st.get("manifest", {}).get("code", "") or ""
    return MAIN_CODE


def write_main(project_name: str, main_code: str) -> str:
    """编排器(掌握总目标)亲自写组合根 main.py,注册为 SUBTASKS 中的 MAIN 条目(可修复路由),
    assemble 时一并落盘。"""
    global MAIN_CODE
    MAIN_CODE = main_code or ""
    if not MAIN_CODE.strip():
        return "[write_main] 收到空代码,未暂存。请传入完整的 main.py 源码。"
    # 注册/更新 MAIN 条目(供自动修复闭环路由:组合根 import 错误可被重写)
    for st in SUBTASKS:
        if st.get("id") == MAIN_SUBTASK_ID:
            st["manifest"] = {"module": "main.py", "code": MAIN_CODE}
            st["task"] = "组合根 main.py（导入各业务模块、串起整个项目流程）"
            break
    else:
        SUBTASKS.append({
            "id": MAIN_SUBTASK_ID,
            "task": "组合根 main.py（导入各业务模块、串起整个项目流程）",
            "manifest": {"module": "main.py", "code": MAIN_CODE},
            "contract": None,
            "model": None,
            "tier": "pro",
            "latency": None,
        })
    notes = []
    # 组合根由编排器在「工具参数」里输出,是最容易被输出上限截断的地方——当场体检。
    try:
        compile(MAIN_CODE, "main.py", "exec")
    except SyntaxError as e:
        notes.append(f"语法错误(第 {e.lineno} 行: {e.msg})。这几乎必然是输出被截断——"
                     f"请把 main.py 写得更短(只做 import 与流程编排,把逻辑下沉到模块)后重发")
    if "def main(" not in MAIN_CODE:
        notes.append("未定义 def main(),smoke 将报『main.py 未定义 main() 入口』;"
                     "请补上 def main(): 与 if __name__ == '__main__': main()")
    n = _code_lines(MAIN_CODE)
    if n > MAX_MAIN_LINES:
        notes.append(f"{n} 行 > 组合根上限 {MAX_MAIN_LINES} 行。main.py 应当只做"
                     f"「import 各模块 + 按契约串流程」,任何实际逻辑都该 delegate 成新模块")
    head = f"[write_main] 已暂存 main.py({n} 行)并注册为可修复条目。"
    if notes:
        return head + " ⚠ " + " ; ".join(notes)
    return (head + "调用 assemble 即可与所有子任务模块组装到同一目录并做契约感知冒烟；"
            "组装失败时组合根可被自动重写修复。")


def assemble(project_name: str) -> str:
    biz_subtasks = [st for st in SUBTASKS if st.get("manifest", {}).get("module") != "main.py"]
    if not biz_subtasks:
        return "[assemble] 侧边存储中没有业务子任务,请先 delegate 若干子任务(并 write_main 组合根)。"
    out_dir = os.path.join(PROJECT_DIR, project_name or "project")
    try:
        # main.py 由 _get_main_code 单独落盘;业务模块排除 main.py 条目,避免双重写
        main_code = _get_main_code()
        written = smoke.assemble(out_dir, biz_subtasks, main_code)
        ok, failures = smoke.smoke_project(out_dir)
        lines = [smoke.format_report(out_dir, written, ok, failures)]
        if not ok:
            lines.append(f"   ⚠ 冒烟发现 {len(failures)} 个失败点,启动自动修复闭环")
            rlog, fixed, missing = repair_loop(out_dir, lambda d: smoke.smoke_project(d))
            lines += rlog
            if missing:
                lines.append("   ⚠ 缺失模块: " + ", ".join(sorted(missing)) +
                             " — 请用 delegate 创建这些模块后重新 assemble")
            lines.append("   " + ("✅ 修复后冒烟通过" if fixed else
                       "⚠ 自动修复未完全解决;缺失模块请用 delegate 补齐后重新 assemble,其余失败点见上方"))
        else:
            lines.append("   ✅ 组装 + 冒烟通过,可直接 python main.py 运行")
        return "\n".join(lines)
    except Exception as e:
        return f"[assemble 错误] {e}"


# ----------------------------------------------------------------------------
# 校验 + 自动修复闭环(组装→校验→把错误回灌云端 subagent→重新组装)
# ----------------------------------------------------------------------------
def _module_from_traceback(tb_text, module_names):
    """从 traceback 中定位失败所属的项目模块。
    优先用出错帧的文件名;其次扫描 traceback 源码行里出现的 '模块名.'(如 auth.login),
    这样集成测试(test_smoke.py 抛错)也能定位到真正出问题的业务模块。"""
    for line in (tb_text or "").splitlines():
        m = re.search(r'File "([^"]+)"', line)
        if m:
            fn = os.path.basename(m.group(1))
            if fn in module_names:
                return fn
    for name in module_names:
        if re.search(r'\b' + re.escape(name) + r'\.', tb_text or ""):
            return name
    return None


def _map_module_to_subtask(module_name):
    if not module_name:
        return None
    name = module_name if module_name.endswith(".py") else module_name + ".py"
    for st in SUBTASKS:
        mm = (st.get("manifest", {}).get("module") or "")
        if mm == name or mm.endswith("/" + name) or mm == module_name:
            return st
    return None


def _missing_module_of(failure, delivered):
    """判断失败点是否为『缺失模块』:error 含 No module named 'X' 且 X 的顶层名不在已交付
    模块(delivered)中、也不是 main。返回顶层模块名或 None。仅依据 No module named 信号,
    避免把『变量未定义』(name 'X' is not defined)等误判为缺失模块。"""
    err = failure.get("error") or ""
    m = re.search(r"No module named ['\"]?([\w.]+)['\"]?", err)
    if not m:
        return None
    top = m.group(1).split(".")[0]
    if top in delivered or top == "main":
        return None
    return top


def _check_import(out_dir):
    # 契约感知冒烟:导入全部模块 + 调用 main 入口(会暴露符号名/签名漂移)
    ok, failures = smoke.smoke_project(out_dir)
    return (ok, [{"module": f["module"], "symbol": f["symbol"], "error": f["error"]} for f in failures])


def _check_test(out_dir, test_code):
    # 路径必须绝对化:out_dir 来自相对 PROJECT_DIR(如 ./projects),
    # 若 path 用相对路径 + cwd=out_dir,子进程会把相对 path 再次基于 cwd 解析,造成路径翻倍。
    abs_out = os.path.abspath(out_dir)
    path = os.path.join(abs_out, "test_smoke.py")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(test_code)
        r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           cwd=abs_out, timeout=DELEGATE_TIMEOUT)
        ok = r.returncode == 0
        tb = (r.stderr or "") + "\n" + (r.stdout or "")
        fails = [] if ok else [{"module": None, "error": tb[-2500:]}]
        return ok, fails
    except subprocess.TimeoutExpired:
        return False, [{"module": None, "error": "集成测试超时(>%ds)" % DELEGATE_TIMEOUT}]
    except Exception as e:
        return False, [{"module": None, "error": f"{type(e).__name__}: {e}"}]


def _delivered_provides_text():
    """拼出当前已交付业务模块及其 provides 符号,供组合根修复时给 subagent 参考。"""
    lines = []
    for st in SUBTASKS:
        mm = st.get("manifest", {})
        if mm.get("module") == "main.py":
            continue
        mod = mm.get("module", "?")
        provides = mm.get("provides") or []
        pv = "; ".join(str(p) for p in provides) if provides else "(未在契约声明 provides)"
        lines.append(f"- {mod}: {pv}")
    return "\n".join(lines) if lines else "(无已交付业务模块)"


def _repair_module(st, error_text):
    """用云端 subagent 修复单个子任务(原地更新 SUBTASKS 条目)。
    MAIN 条目(组合根 main.py)特判:带已交付模块符号清单,让 subagent 重写 main.py
    正确引用这些符号(缺失功能在 main.py 内联实现),从而自愈组合根 import 错误。"""
    manifest = st.get("manifest", {})
    if st.get("id") == MAIN_SUBTASK_ID:
        repair = {
            "original_task": st.get("task") or "组合根 main.py（导入各业务模块、串起整个项目流程）",
            "current_code": manifest.get("code", ""),
            "error": (
                "【已交付的模块与提供的符号 —— 组合根必须只引用这些;缺失的功能请在 main.py 内联实现】\n"
                + _delivered_provides_text()
                + "\n\n【报错 / 校验失败信息】\n" + error_text
            ),
        }
        delegate(
            "修复组合根 main.py",
            tier="pro",
            name="main.py",
            contract={"provides": [], "depends_on": []},
            repair=repair,
            replace_id=st.get("id"),
        )
        return
    repair = {
        "original_task": st.get("task") or manifest.get("summary") or "",
        "current_code": manifest.get("code", ""),
        "error": error_text,
    }
    contract = st.get("contract") or {}
    delegate(
        f"修复模块 {manifest.get('module')}",
        tier=st.get("tier", "flash"),
        name=manifest.get("module"),
        provides=contract.get("provides"),
        depends_on=contract.get("depends_on"),
        repair=repair,
        replace_id=st.get("id"),
    )


def _modules_imported_by_test(test_code):
    """从集成测试代码里提取它 import 的顶层模块名(用于 verify 失败时的修复范围)。"""
    names = set()
    try:
        tree = ast.parse(test_code)
    except Exception:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                names.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def repair_loop(out_dir, check_fn, rounds=REPAIR_ROUNDS, scope=None):
    """check_fn(out_dir) -> (ok, failures[{module?,error}])。
    逐轮:定位失败模块→云端修复→重链→再校验,直到通过或轮次耗尽。
    返回 (log, ok, missing):missing 为自动闭环无法新建的「缺失模块」集合(从未被 delegate),
    需交由编排器(35B)用 delegate 补齐后再 assemble。
    scope: 当无法从 traceback 定位时,回退修复这些模块(verify 用于指定测试 import 的模块)。"""
    log = []
    ok, fails = check_fn(out_dir)
    if ok:
        return log, True, set()
    # 模块名集合:以 SUBTASKS 中实际委托的模块为准(权威),避免把 verify 写入的 test_smoke.py 也算进来
    module_names = {os.path.basename(st.get("manifest", {}).get("module", ""))
                    for st in SUBTASKS if st.get("manifest", {}).get("module")}
    delivered = {m[:-3] if m.endswith(".py") else m for m in module_names}
    missing = set()
    for rnd in range(1, rounds + 1):
        log.append(f"   🔧 修复轮次 {rnd}/{rounds}: {len(fails)} 个失败点")
        targets = []
        missing = set()
        for f in fails:
            # 缺失模块(No module named 'X' 且 X 从未被 delegate):自动闭环无法新建,
            # 不路由到任何已存在条目(含组合根 MAIN),收集后交由 35B 补 delegate。
            miss = _missing_module_of(f, delivered)
            if miss:
                missing.add(miss)
                continue
            # 组合根(main.py)入口失败:直接归属 MAIN 条目,由重写修复(不再静默失败)
            sym = f.get("symbol")
            owner = None
            if f.get("module") == "main.py":
                owner = "main.py"
            elif sym and "." in sym:
                # 优先用错误里解析出的精确符号(如 auth.login)定位归属模块;
                # main.py 入口失败但符号指向 auth.login 时,归属 auth 而非 main。
                owner = sym.split(".", 1)[0]
            elif f.get("module"):
                owner = f["module"][:-3] if f["module"].endswith(".py") else f["module"]
            if not owner:
                owner = _module_from_traceback(f.get("error", ""), module_names)
            if owner:
                st = _map_module_to_subtask(owner)
                if st and st not in targets:
                    targets.append(st)
        if missing:
            log.append("     - ⚠ 缺失模块: " + ", ".join(sorted(missing)) +
                       " — 从未被 delegate,自动闭环无法新建;请编排器用 delegate 创建后重新 assemble")
        if not targets and scope:
            # 集成测试(断言失败)的 traceback 往往不点名模块,改用测试 import 的模块范围
            for mod in scope:
                st = _map_module_to_subtask(mod if mod.endswith(".py") else mod + ".py")
                if st and st not in targets:
                    targets.append(st)
        if not targets:
            if missing:
                log.append("     - 缺失模块需编排器补 delegate,自动修复暂停")
            else:
                log.append("     - 无法把失败映射到任何子任务,停止自动修复")
            break
        full_err = "\n".join(f.get("error", "") for f in fails)[:3000]
        for st in targets:
            _repair_module(st, full_err)
        smoke.assemble(out_dir, [st for st in SUBTASKS if st.get("manifest", {}).get("module") != "main.py"], _get_main_code())  # 用修复后的代码重新组装(业务模块 + 最新 main.py)
        log.append("     - 已用修复后的代码重新组装项目")
        ok, fails = check_fn(out_dir)
        if ok:
            log.append(f"   ✅ 第 {rnd} 轮修复后校验通过")
            return log, True, missing
    return log, ok, missing


def verify(project_name: str, test_code: str, name: str = None) -> str:
    """对已 assemble 的项目做一「片」集成校验。
    分片语义:每次只验一个切面(建议 ≤MAX_TEST_LINES 行),可多次调用。
    这不只是为了定位精度——test_code 由编排器自己生成,写长了同样会被输出上限截断,
    而截断的测试是语法错误,会被误判成「项目有 bug」,触发一整轮无谓修复。"""
    if not SUBTASKS:
        return "[verify] 侧边存储为空,请先 delegate 并 assemble。"
    out_dir = os.path.join(PROJECT_DIR, project_name or "project")
    if not os.path.isdir(out_dir):
        return f"[verify] 项目目录不存在: {out_dir},请先 assemble。"
    tag = f"[{name}] " if name else ""
    # 关键:先体检测试代码本身。语法不通 = 测试被截断,而非项目有问题——
    # 此时绝不能进入修复闭环去「修」一个其实没坏的项目。
    try:
        compile(test_code or "", "test_smoke.py", "exec")
    except SyntaxError as e:
        return (f"{tag}[verify 未执行] 你提供的 test_code 语法错误"
                f"(第 {e.lineno} 行: {e.msg})——这通常是输出被截断。"
                f"请把校验拆成更小的分片(每片 ≤{MAX_TEST_LINES} 行、只验一个切面),"
                f"分多次调用 verify,不要一次写一个大测试。")
    n = _code_lines(test_code)
    over = (f" ⚠ 本片 {n} 行 > 建议上限 {MAX_TEST_LINES} 行,下次请拆更细"
            if n > MAX_TEST_LINES else "")
    try:
        ok, fails = _check_test(out_dir, test_code)
        if ok:
            return f"{tag}✅ 集成校验通过({n} 行 | 项目: {project_name}){over}"
        lines = [f"{tag}⚠ 集成校验失败,启动自动修复闭环(项目: {project_name}){over}"]
        # 集成测试的 traceback 通常不点名模块,改用测试 import 的模块作为修复范围。
        # scope 统一存成 manifest 里的文件名(如 auth.py),与失败点/映射函数的口径一致。
        scope = set()
        for m in _modules_imported_by_test(test_code):
            for st in SUBTASKS:
                mod = st.get("manifest", {}).get("module", "")
                if mod and (mod == m + ".py" or mod.endswith("/" + m + ".py")):
                    scope.add(mod)
        rlog, fixed, missing = repair_loop(out_dir, lambda d: _check_test(d, test_code), scope=scope)
        lines += rlog
        if missing:
            lines.append("   ⚠ 缺失模块: " + ", ".join(sorted(missing)) +
                         " — 请用 delegate 创建后重新 assemble")
        lines.append("   " + ("✅ 修复后集成校验通过" if fixed else "⚠ 自动修复未完全解决,见上方失败点"))
        return "\n".join(lines)
    except Exception as e:
        return f"[verify 错误] {e}"


DISPATCH = {
    "delegate": delegate,
    "write_main": write_main,
    "assemble": assemble,
    "verify": verify,
}


# ----------------------------------------------------------------------------
# token 估算(CJK 感知)
# ----------------------------------------------------------------------------
def _est_text(s: str) -> int:
    if not s:
        return 0
    cjk = 0
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            cjk += 1
    n = len(s) - cjk
    return int(cjk * 0.6 + n * 0.25) + 1


def _msg_tokens(m: dict) -> int:
    c = m.get("content") or ""
    extra = ""
    if m.get("tool_calls"):
        extra = json.dumps(m["tool_calls"], ensure_ascii=False)
    return _est_text(c) + _est_text(extra)


def history_tokens(messages: list) -> int:
    total = 0
    if messages and messages[0].get("role") == "system":
        total += _est_text(messages[0].get("content", ""))
    total += _est_text(json.dumps(TOOLS, ensure_ascii=False))
    for m in messages:
        total += _msg_tokens(m)
    return total


# ----------------------------------------------------------------------------
# Ollama 通信(原生 /api/chat)
# ----------------------------------------------------------------------------
def _ollama_raw(payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _output_budget(messages: list) -> int:
    """按「剩余上下文窗口」算本轮可用的输出预算。
    num_ctx 是 prompt+输出共用的:不显式限制时,历史一涨输出就被硬截断
    (write_main 的 main_code 断在半截就是这么来的)。"""
    used = history_tokens(messages)
    room = NUM_CTX - used - CTX_MARGIN
    if room < ORCH_MIN_PREDICT:
        print(f"   ⚠ 输出预算仅剩 ~{max(room, 0)} token(< {ORCH_MIN_PREDICT}),"
              f"生成极易被截断;建议降低 AGENT_BUDGET 触发压缩或调大 AGENT_NUM_CTX", flush=True)
        return max(room, ORCH_MIN_PREDICT)  # 仍给保底,宁可溢出也不要生成半截
    return min(ORCH_MAX_PREDICT, room)


def _ollama_chat(messages: list, force_no_think: bool = False) -> dict:
    opts = {
        "num_ctx": NUM_CTX,
        "temperature": TEMPERATURE,
        "num_predict": _output_budget(messages),
    }
    if force_no_think or not ENABLE_THINKING:
        # think:false 是 Ollama 对 Qwen3 真正生效的关思考开关(enable_thinking 对本 GGUF 无效)
        opts["enable_thinking"] = False
        opts["think"] = False
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "options": opts,
        "stream": False,
    }
    resp = _ollama_raw(payload)
    if resp.get("done_reason") == "length":
        print("   ⚠ 编排器本轮输出被长度上限截断(工具参数可能不完整)。"
              "请让它把模块/主函数拆得更小,或调大 AGENT_NUM_CTX", flush=True)
    return resp


# ----------------------------------------------------------------------------
# 本地 subagent(串行递归:同一个 35B 兼任 subagent)
#   低比特量化(如 IQ2_M)下长输出极易退化/截断,故:
#   ① 显式 num_predict,把窗口留给输出;② 检测 done_reason=="length";
#   ③ 用 assistant prefill 续写,并按重叠去重拼接。
# ----------------------------------------------------------------------------
def _stitch(acc: str, nxt: str) -> str:
    """拼接续写结果:模型常会重复尾部若干字符,取最长重叠去重,避免代码被写重。"""
    if not acc:
        return nxt
    if not nxt:
        return acc
    # 取「最长」重叠:从大到小扫,首个命中即最长。下限 12 字符——再低容易被
    # "\n        " 这类缩进串误判成重叠而吃掉真代码,再高则漏掉真实的短重复。
    max_ov = min(len(acc), len(nxt), 400)
    for n in range(max_ov, 11, -1):
        if acc[-n:] == nxt[:n]:
            return acc + nxt[n:]
    return acc + nxt


def _ollama_complete(model, messages, num_ctx, num_predict, temperature, timeout, think=False):
    opts = {"num_ctx": num_ctx, "temperature": temperature, "num_predict": num_predict}
    if think:
        opts["think"] = True             # 决策/规划步骤:显式开思考(输出短,窗口装得下思考链)
    else:
        opts["enable_thinking"] = False  # 写代码步骤:关思考(think:false 才对本 GGUF 真正生效)
        opts["think"] = False
    payload = {"model": model, "messages": messages, "options": opts, "stream": False}
    resp = _ollama_raw(payload, timeout=timeout)
    return (resp.get("message", {}) or {}).get("content", "") or "", resp.get("done_reason", "")


def ollama_subagent_chat(model, messages, timeout=None, use_json=True, think=False):
    """本地 35B 兼任 subagent(串行递归)。截断则自动 prefill 续写,最多 MAX_CONTINUE 次。
    think=True 用于"拆分决策"等短输出的规划步骤(开思考并把 num_ctx 抬到 THINK_NUM_CTX
    下限,装下 8K+ 思考链);写代码步骤保持 think=False。签名与 nvidia_chat /
    siliconflow_chat 一致,便于 delegate 统一调度。"""
    timeout = timeout or LOCAL_TIMEOUT
    base_ctx = max(SUB_NUM_CTX, THINK_NUM_CTX) if think else SUB_NUM_CTX
    acc, reason = "", ""
    for attempt in range(MAX_CONTINUE + 1):
        # 已有内容时把它作为 assistant prefill,让模型「接着写」而不是重新开始
        msgs = list(messages) + ([{"role": "assistant", "content": acc}] if acc else [])
        content, reason = _ollama_complete(
            model, msgs, base_ctx, SUB_NUM_PREDICT, SUB_TEMPERATURE, timeout, think=think)
        if not content.strip():
            break
        acc = _stitch(acc, content)
        if reason != "length":
            break
        if attempt < MAX_CONTINUE:
            print(f"   ↻ 输出被截断,续写第 {attempt + 1}/{MAX_CONTINUE} 次"
                  f"(已 {len(acc)} 字符)…", flush=True)
    if reason == "length":
        print(f"   ⚠ 续写 {MAX_CONTINUE} 次后仍未收尾,将尝试从残缺 JSON 中抢救代码", flush=True)
    # 兜底:content 为空且因长度截断,多半是思考链(本 GGUF 即便 think:false 仍思考 8K+ token)
    # 吃光了 num_ctx 窗口。放大 num_ctx 给思考留空间,再试一次(不计入 MAX_CONTINUE 续写额度)。
    if not acc.strip() and reason == "length":
        bigger = max(base_ctx, 32768)
        if bigger > base_ctx:
            print(f"   ↻ 疑似思考链耗光上下文窗口,放大 num_ctx={bigger} 重试一次…", flush=True)
            content, reason = _ollama_complete(
                model, messages, bigger, SUB_NUM_PREDICT, SUB_TEMPERATURE, timeout, think=think)
            if content.strip():
                acc = content
                if reason == "length":
                    print(f"   ⚠ 放大后仍被长度截断,将尝试从残缺 JSON 中抢救代码", flush=True)
    if not acc.strip():
        raise RuntimeError("本地 subagent 返回空内容")
    return acc


def _dispatch_tool(tc: dict) -> str:
    fn = tc.get("function", {})
    name = fn.get("name", "")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except Exception:
            args = {}
    if name not in DISPATCH:
        return f"[未知工具] {name}"
    try:
        return DISPATCH[name](**args)
    except TypeError as e:
        return f"[参数错误] 工具 {name}: {e}"


# ----------------------------------------------------------------------------
# 递归摘要压缩(对标 Claude Code 混合代理的 summary-based compaction)
# ----------------------------------------------------------------------------
_SUM_SYS = (
    "你是对话压缩器。把下面这段对话记录（用户指令、子任务委派、subagent 返回、模型回应）"
    "压缩成一段简洁中文摘要，保留：用户的最终目标、已委派/完成的子任务及其产出模块、关键结论、"
    "任何未解决的要点。不要添加原文没有的信息。直接输出摘要文本，不要使用 markdown 标题。"
)


def summarize_chunk(chunk: list) -> str:
    if not chunk:
        return ""
    lines = []
    for m in chunk:
        role = m.get("role", "?")
        c = m.get("content") or ""
        if role == "tool":
            c = f"[工具 {m.get('name', '')} 返回] {c}"
        elif role == "assistant" and m.get("tool_calls"):
            names = ", ".join(
                tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]
            )
            c = (c + f" [调用工具: {names}]") if c else f"[调用工具: {names}]"
        lines.append(f"{role}: {c}")
    text = "\n".join(lines)

    payload = {
        "model": SUMMARIZER_MODEL,
        "messages": [
            {"role": "system", "content": _SUM_SYS},
            {"role": "user", "content": text},
        ],
        "options": {"num_ctx": min(NUM_CTX, 4096), "temperature": 0.2},
        "stream": False,
    }
    try:
        resp = _ollama_raw(payload)
        d = resp.get("message", {}).get("content", "").strip()
        return d or text[:1200]
    except Exception:
        return text[:1200]


def truncate_chunk(chunk: list, cap: int = TRUNCATE_CAP) -> str:
    if not chunk:
        return ""
    lines = []
    for m in chunk:
        role = m.get("role", "?")
        c = m.get("content") or ""
        if len(c) > cap:
            c = f"{c[:cap]}\n...[该条已抽取式截断，原 {len(c)} 字符]"
        if role == "tool":
            c = f"[工具 {m.get('name', '')} 返回] {c}"
        elif role == "assistant" and m.get("tool_calls"):
            names = ", ".join(
                tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]
            )
            c = (c + f" [调用工具: {names}]") if c else f"[调用工具: {names}]"
        lines.append(f"{role}: {c}")
    return "\n".join(lines)


def maybe_compact(messages: list) -> None:
    for _ in range(COMPACT_ROUNDS):
        if history_tokens(messages) <= BUDGET:
            return
        if len(messages) <= 1 + KEEP_RECENT:
            break
        tail = messages[-KEEP_RECENT:]
        head = messages[1:-KEEP_RECENT]
        if not head:
            break
        if USE_SUMMARY:
            if SUMMARY_MODE == "truncate":
                digest = truncate_chunk(head)
            else:
                digest = summarize_chunk(head)
            digest_msg = {"role": "assistant", "content": f"[历史摘要]\n{digest}"}
        else:
            digest_msg = {"role": "assistant", "content": "[历史摘要] (已省略早期对话以节省上下文)"}
        messages[:] = [messages[0]] + [digest_msg] + tail

    if history_tokens(messages) > BUDGET and len(messages) > 1 + 4:
        messages[:] = [messages[0]] + messages[-4:]


# ----------------------------------------------------------------------------
# Agent 主循环
# ----------------------------------------------------------------------------
def run_agent(query: str, history: list = None) -> str:
    messages = history if history is not None else [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": query})

    for step in range(1, MAX_ITER + 1):
        maybe_compact(messages)
        try:
            resp = _ollama_chat(messages)
        except urllib.error.URLError as e:
            return f"[无法连接 Ollama] 确认 Ollama 已启动且监听 {OLLAMA_URL}：{e}"

        msg = resp.get("message", {})
        if ENABLE_THINKING:
            had_error = "error" in resp
            empty_final = (not msg.get("tool_calls")) and not msg.get("content", "").strip()
            if had_error or empty_final:
                try:
                    resp2 = _ollama_chat(messages, force_no_think=True)
                    if "error" not in resp2:
                        resp, msg = resp2, resp2.get("message", {})
                except Exception:
                    pass

        if "error" in resp:
            return f"[Ollama 错误] {resp['error']}"

        if msg.get("tool_calls"):
            content = msg.get("content", "")
            print(f"\n🤖 思考: {content}" if content else f"\n🤖 第 {step} 步: 调用工具")
            messages.append(msg)
            for tc in msg["tool_calls"]:
                name = tc.get("function", {}).get("name", "?")
                args = tc.get("function", {}).get("arguments", {})
                print(f"   🔧 {name}({args if not isinstance(args, str) else args})")
                result = _dispatch_tool(tc)
                print(f"   ↳ {result.splitlines()[0][:140]}")
                messages.append({"role": "tool", "content": result, "name": name})
            continue

        final = msg.get("content", "").strip()
        messages.append(msg)
        if history is not None:
            history[:] = messages
        return final or "[模型未返回内容]"

    return f"[已达最大迭代 {MAX_ITER}，未完成]"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    global DELEGATE_BACKEND
    parser = argparse.ArgumentParser(description="本地编排器 + subagent(云端或本地) + 契约感知组装")
    parser.add_argument("query", nargs="*", help="单次任务（不填则进入交互模式）")
    parser.add_argument("--local", action="store_true",
                        help="全本地模式:subagent 也用本地 Ollama(串行递归),完全不联网")
    args = parser.parse_args()
    if args.local:
        DELEGATE_BACKEND = "ollama"

    print(f"🦙 本地编排器 | 模型={MODEL} | num_ctx={NUM_CTX} | Ollama={OLLAMA_URL}")
    print(f"   compact: mode={SUMMARY_MODE if USE_SUMMARY else 'off'} | thinking={'on' if ENABLE_THINKING else 'off'} | 预算={BUDGET} token")
    print(f"   选择性推理: 顶层规划={'on' if ENABLE_THINKING else 'off'}"
          f" · 拆分决策={'on' if THINK_ON_SPLIT else 'off'}"
          f" · 修复诊断={'on' if THINK_ON_REPAIR else 'off'}"
          f" · 写代码=off(恒关) | 思考步骤 num_ctx≥{THINK_NUM_CTX}")
    print(f"   输出预算: 编排器≤{ORCH_MAX_PREDICT} / subagent≤{SUB_NUM_PREDICT} token"
          f" | 截断自动续写×{MAX_CONTINUE}"
          f" | 粒度上限: 模块{MAX_MODULE_LINES}行 main{MAX_MAIN_LINES}行 测试{MAX_TEST_LINES}行")
    if DELEGATE_BACKEND == "ollama":
        print(f"   工具: delegate(本地subagent/串行递归 · {LOCAL_SUBAGENT_MODEL})"
              f" + assemble(组装+冒烟) + verify(分片校验)")
        print(f"   🔒 全本地模式:不联网,编排器与 subagent 共用同一个本地模型,串行执行")
        if args.query:
            q = " ".join(args.query)
            print(f"\n👤 {q}")
            try:
                ans = run_agent(q)
            except KeyboardInterrupt:
                print("\n   ⏹ 已取消")
                return
            print(f"\n🤖 {ans}")
            print(f"   [路由器状态/本地] {ROUTER_LOCAL.status_line()}")
            return
        _interactive(ROUTER_LOCAL, "本地")
        return
    if DELEGATE_BACKEND == "siliconflow":
        backend_key, backend_flash, backend_pro, backend_name = SILICONFLOW_API_KEY, SILICONFLOW_FLASH, SILICONFLOW_PRO, "SiliconFlow"
        backend_src = "local_config.json" if (SILICONFLOW_API_KEY and not os.environ.get("SILICONFLOW_API_KEY", "").strip()) else ("环境变量" if SILICONFLOW_API_KEY else "无")
        backend_router = ROUTER_SF
    else:
        backend_key, backend_flash, backend_pro, backend_name = NVIDIA_API_KEY, NVIDIA_FLASH, NVIDIA_PRO, "NVIDIA"
        backend_src = _KEY_SOURCE or "无"
        backend_router = ROUTER
    print(f"   工具: delegate(云端subagent/{backend_name}) + assemble(组装+冒烟) + verify | 档位: flash={len(backend_flash)} pro={len(backend_pro)}")
    if not backend_key:
        print(f"   ⚠ 未检测到 {backend_name} API key —— delegate 将报错,请先设置对应环境变量或写入 local_config.json")
    else:
        masked = backend_key[:10] + "…" + backend_key[-4:]
        print(f"   🔑 {backend_name} key: {masked} (来源: {backend_src})")

    if args.query:
        q = " ".join(args.query)
        print(f"\n👤 {q}")
        try:
            ans = run_agent(q)
        except KeyboardInterrupt:
            print("\n   ⏹ 已取消")
            return
        print(f"\n🤖 {ans}")
        return

    _interactive(backend_router, backend_name)


def _interactive(backend_router, backend_name):
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("\n交互模式（输入 exit / quit / 按 Ctrl-C 退出）\n")
    while True:
        try:
            q = input("👤 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("👋 再见")
            break
        try:
            ans = run_agent(q, history=history)
        except KeyboardInterrupt:
            print("\n   ⏹ 已取消当前任务(已完成的子任务仍保留在内存,可继续 assemble 或重新 delegate)")
            continue
        print(f"\n🤖 {ans}")
        print(f"   [路由器状态/{backend_name}] {backend_router.status_line()}")


if __name__ == "__main__":
    main()
