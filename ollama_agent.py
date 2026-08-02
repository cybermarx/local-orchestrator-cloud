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
import shutil
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
# 空内容兜底时放大 num_ctx 的上限。8GB 显存 + IQ2_M 35B 上,32768 会 OOM 让 Ollama 崩(HTTP 500),
# 故默认降到 24576;放大调用若仍抛错(OOM/500)会被捕获,不再连累整个 delegate。
SUB_CTX_MAX = int(os.environ.get("AGENT_SUB_CTX_MAX", "24576"))
# 影子模块守卫:subagent/编排器若把模块命名成与 stdlib 或已安装第三方库同名(如 requests.py),
# 本地文件会遮蔽真库,且生成代码常 import 同名库 → 自引用递归。默认开启,delegate 时告警。
GUARD_STDLIB_SHADOW = os.environ.get("AGENT_GUARD_SHADOW", "1").lower() in ("1", "true", "yes", "on")
# 运行时契约校验:assemble 时对「模块是否真定义其 provides 符号 / depends_on 目标是否存在 /
# main 是否接线各模块」做静态 AST 检查,暴露符号级契约漂移(孤儿/改名/影子/未接线)。
# 默认开启;数据形状错配(如 main 平铺 dict 而消费方读嵌套)属运行时行为,需 verify 真实断言暴露。
CONTRACT_CHECK = os.environ.get("AGENT_CONTRACT_CHECK", "1").lower() in ("1", "true", "yes", "on")

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
# THINK_ON_REPAIR:非两阶段(云端/回退)时,单次修复调用是否开思考(默认关:修复输出含整段代码,长输出+思考有撑爆窗口风险)。
THINK_ON_REPAIR = os.environ.get("AGENT_THINK_ON_REPAIR", "0").lower() in ("1", "true", "yes", "on")
THINK_NUM_CTX = int(os.environ.get("AGENT_THINK_NUM_CTX", "16384"))  # 思考步骤的 num_ctx 下限

# --- 两阶段 repair(诊断→修复):把"需推理的短诊断"与"不需推理的长修复"拆开 ------
# 修复本质是"诊断(为什么错、怎么改)+ 改代码"两种性质相反的工作。捆一次调用 → 想开思考
# 诊断却被长代码撑爆窗口。拆成:①诊断(开思考·短输出,实测175字符/34s/更准) ②修复(关思考·
# 长输出)。诊断若判定 needs_rewrite(模块烂到要重写)且深度未满 → 递归拆子模块重写(契约不破)。
TWO_STAGE_REPAIR = os.environ.get("AGENT_TWO_STAGE_REPAIR", "1").lower() in ("1", "true", "yes", "on")
THINK_ON_REPAIR_DIAG = os.environ.get("AGENT_THINK_ON_REPAIR_DIAG", "1").lower() in ("1", "true", "yes", "on")

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
_CURRENT_PROJECT = ""  # 当前项目名(由 write_main/assemble 设置,供子编排器归位子系统目录)
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
- delegate_subsystem(task, subdir, name?, provides?, depends_on?, project_name?): **无损 scope 分解**——把一簇**内聚的模块**打包成一个「子系统」,派给一个**子编排器**递归 delegate+assemble 构建。子编排器只把『对外薄契约』(provides 符号)返回给你,你这边只占 1 个节点,**内部模块不进你的上下文**,从而避免契约列表膨胀、上下文爆。当你要设计的模块太多(契约列表装不下)、或你正基于一大坨**已有代码**改进(已有代码已被载入侧边存储)而改动面很大时,用本工具:把内聚的一簇任务/已有模块归成一个子系统。你后续用 `from subdir import <符号>` 调用它。
- write_main(project_name, main_code): 你(编排器,掌握总目标)亲自写**主函数 main.py(组合根)**。它 import 各模块、按你设计的契约调用它们,把整个项目串起来。main.py 必须严格按 provides/depends_on 约定的**名字与签名**调用各模块符号——这是「去 linker」设计下唯一需要你保证接口一致的地方(同目录放文件即可,import 会自行接线)。这里只暂存代码,落盘由 assemble 统一做。
- assemble(project_name): 所有模块 delegate 完成、且你已 write_main 后调用。它把所有模块文件 + main.py 落到同一目录(无需接线,Python import 即 linker),然后做「导入全部模块 + 真正调用 main 入口」的契约感知冒烟;对冒烟失败(符号名/签名漂移、缺失模块、循环依赖)自动启动修复闭环,返回项目树。
- verify(project_name, test_code, name?): 【可选】assemble 后,用一个**短**Python 片段对生成的项目做**一片**集成校验(你掌握总目标,应写出能验证核心流程的断言,如 register 后 login 能拿到 token)。失败会自动回灌 subagent 修复并重新组装。**校验要分片**:一次只验一个切面,分多次调用,用 name 标注这片验什么。**防作弊硬约束**:你写的断言必须是**真实校验**(如 `assert "token" in output`、`assert user.id == 1`),**严禁**在 verify 失败后把断言改弱来强行变绿(如 `assert len(output)>10 or "Error" not in output`、`assert True`、只用 `len(x)>0` 验存在性不验内容);同一 name 重复 verify 应修正测试逻辑或修复项目,而不是放宽断言。

工作方式(契约优先,去 linker):
1. 理解用户的**总目标**;
2. 先做**模块契约设计**(这是唯一需要总目标的地方):把项目拆成若干**小**模块(见下方「输出长度纪律」),为每个模块确定
   - module: 文件名
   - provides: 它必须对外暴露的符号列表(函数/类名,尽量带签名,如 "get_user(id) -> User")
   - depends_on: 它**实际会调用**的其它模块符号——**既要列读依赖,也要列写依赖**(例如注册模块既要依赖 "user_db:authenticate" 也要依赖 "user_db:create_user",因为注册必须把用户写进库)。格式 "模块:符号(签名)"。
   契约是 subagent 之间对接的**唯一依据**,务必让 provides 与 depends_on 互相吻合(谁提供、谁消费要一致;尤其注意写流程的两端都要连上)。
3. 逐个 delegate:每次只把**该模块的「子目标」+「契约」**(必提供的符号/签名、可依赖的符号/签名)传给 subagent。**绝不要把总目标写进 subagent 的提示**——subagent 只该看到自己的子目标与契约。若你规划的模块太多、契约列表开始撑爆上下文,或你正基于**已有代码**做大面改进:把内聚的一簇任务/已有模块归成一个子系统,改用 **delegate_subsystem**(子编排器只把薄契约返回给你,内部模块不进你的上下文)。
4. 你亲自写 main.py:基于上面设计的契约,import 各模块、按约定名字/签名调用它们,串成完整流程。用 write_main 暂存(务必让 import 名与 provides 完全一致)。
5. 调用 assemble 组装(会自动冒烟与修复);若任务有明显 happy-path,再**分多次**调用 verify,每次跑一小片集成断言,让系统把逻辑错误也自动修掉。若 assemble/verify 报告「缺失模块 X」或「No module named X」,说明 X 从未被 delegate——你必须在下一轮用 delegate 创建 X 模块(给出子目标与契约),然后再次 assemble,直到不再有缺失模块。
6. 简单聊天可直接回答,不必 delegate。

**输出长度纪律(硬约束,必须遵守)**:
你和 subagent 可能都跑在本机的低比特量化模型上,**单次输出越长,被截断成半截代码的概率越高**。所以:
- **每个模块**:单一职责、只暴露 1-3 个符号、目标 ≤%MAXMOD% 行。宁可多拆 3 个小模块,也绝不要 1 个大模块。若发现某个子目标要写很多代码,先把它再拆开。
- **main.py**:只做「import 各模块 + 按契约把流程串起来」,目标 ≤%MAXMAIN% 行。**任何实际逻辑都要下沉成新模块 delegate 出去**,不要写在 main.py 里。
- **verify**:不要写一个大测试。每次只验**一个切面**(≤%MAXTEST% 行),分多次调用并用 name 标注(如 "注册流程"、"登录流程")。**断言必须真实**:写 `assert` 校验核心行为(如 `assert result.ok`),**绝不允许断言恒真或只验存在性来强行通过**——verify 失败说明项目可能真有问题,应去修项目或改测试逻辑,不要放宽断言。
- 若工具返回里出现「⚠ …被截断 / 语法错误 / 超过上限行」,说明**这一块太大了**:把它拆成更小的子目标重新 delegate,不要原样重试。

规则:
- delegate 的 task 就是该模块的「子目标」,要自包含、清晰。
- provides/depends_on 是你(编排器)预先设计好的契约;若省略,subagent 会自行声明,但你可能需要在 assemble 前核对一致性。**务必把写依赖也写进 depends_on**,否则 subagent 可能自创一个不落库的实现(如注册只用内存字典)。
- 若 delegate 返回错误,换种描述重试,或把任务拆更细再 delegate。
- 你无法自己联网;需要实时信息时,云端模型会用其训练知识回答(可能非最新),请向用户说明。
- 最终回答要简洁,引用 assemble/verify 返回的项目路径与校验结论。
- **verify 防作弊**:绝不允许为了「让校验变绿」而把断言改弱——同一 name 重复 verify 时若仍失败,应修正测试逻辑或把失败信息回灌修复项目,而不是把断言改成恒真/弱化(如 `assert len(output)>0`、`assert "Error" not in output`)。被系统判定为「过弱断言」的校验会被标记警告,不算真实通过。
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

SUBAGENT_DIAG_SYS = """你是一个代码诊断专家。下面给你一个有缺陷的 Python 模块和它的报错信息。
你现在的工作**不是修代码**,而是精准诊断:错在哪、为什么错、该怎么改。
请只输出一个 JSON 对象(不要解释文字、不要用 markdown 代码块包裹):
{
  "cause": "一句话说清根本病因",
  "plan": ["具体修改步骤1", "具体修改步骤2"],
  "needs_rewrite": false
}
要求:
- 只诊断,**绝不输出修正后的完整代码**(诊断很短,不会被截断)。
- plan 要具体到能让另一个人照着改对,不要泛泛而谈。
- needs_rewrite: 仅需局部小修=false;若模块逻辑大面积错误、或修好后明显超过 %MAXMOD% 行=true。
- 保持模块原有职责与 provides 接口不变的前提下诊断。
- 只输出 JSON。
"""

# 粒度上限注入 prompt(prompt 含字面花括号,不能用 f-string,故用占位符替换)
for _ph, _v in (("%MAXMOD%", MAX_MODULE_LINES), ("%MAXMAIN%", MAX_MAIN_LINES),
                ("%MAXTEST%", MAX_TEST_LINES),
                ("%SPLITMIN%", SPLIT_MIN_SUBTASKS), ("%SPLITMAX%", SPLIT_MAX_SUBTASKS)):
    SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_ph, str(_v))
    SUBAGENT_SYS = SUBAGENT_SYS.replace(_ph, str(_v))
    SUBAGENT_SPLIT_SYS = SUBAGENT_SPLIT_SYS.replace(_ph, str(_v))
    SUBAGENT_DIAG_SYS = SUBAGENT_DIAG_SYS.replace(_ph, str(_v))


# ----------------------------------------------------------------------------
# 分层子编排器(无损 scope 分解 / hierarchical sub-orchestrator)
# ----------------------------------------------------------------------------
# 背景:当项目 scope 太大(顶层契约列表都装不下)时,串行递归的单个编排器会被
# 「契约列表膨胀」拖爆上下文。解法:把一个内聚子系统委派给【子编排器】,子编排器在
# 独立上下文里递归 delegate+assemble,只把「对外薄契约」(provides 符号)返回给父。
# 父的 SUBTASKS 里该子系统只占 1 个节点(内部模块不进父上下文),契约列表不膨胀。
# 与「递归压缩」互补:压缩是有损兜底,分层委派是无损结构分解,scope 爆优先用它。
MAX_ORCH_DEPTH = int(os.environ.get("AGENT_MAX_ORCH_DEPTH", "2"))

# 进入子编排器时,把 SUBTASKS/MAIN_CODE/SYSTEM_PROMPT 临时换成子树私有实例(用栈支持嵌套),
# 子编排器跑的就是同一套 delegate/assemble/verify 工具;退出时还原。
_ORCH_STACK = []


def _push_orch_ctx(sys_prompt: str):
    global SUBTASKS, MAIN_CODE, SYSTEM_PROMPT
    _ORCH_STACK.append((SUBTASKS, MAIN_CODE, SYSTEM_PROMPT))
    SUBTASKS = []
    MAIN_CODE = ""
    SYSTEM_PROMPT = sys_prompt


def _pop_orch_ctx():
    global SUBTASKS, MAIN_CODE, SYSTEM_PROMPT
    SUBTASKS, MAIN_CODE, SYSTEM_PROMPT = _ORCH_STACK.pop()


def _orch_depth() -> int:
    """当前嵌套深度 = 已压栈的父上下文数。顶层(未进任何子编排器)为 0。"""
    return len(_ORCH_STACK)


SUBSYS_PROMPT = """你是一个运行在用户本机、由 Ollama(本地35B)提供算力的「子系统编排器」。
你【不】是顶层编排器,而是受父编排器委派、只负责构建【一个内聚子系统】的子编排器。
职责边界:
1. 你只交付一个内聚子系统,对外只暴露父编排器指定的【薄契约】(provides 符号)。
2. 把子系统拆成若干小模块,逐个 delegate(每个 ≤%MAXMOD% 行);逻辑下沉成模块,不要堆在 main。
3. 子系统内部模块全部落在子目录 {SUBDIR}/ 里:你【必须】用 assemble 组装到子目录 "{SUBDIR}",
   main.py 只在子系统内部做 import + 编排(若子系统只需被父调用,main.py 可最小化)。
4. 内部模块之间用相对导入:`from .其他模块 import 符号`;不要用 `from 其他模块`(无点)。
5. 你【必须】确保父要求的 provides 符号在子系统中真实存在,且能被
   `from {SUBDIR} import <符号>` 导入(即该符号是某内部模块的顶层 def/class)。
6. 每次只做一步(一次工具调用),不要一口气规划完——和顶层编排器一样的串行节奏。
7. assemble 通过后直接结束(返回一句简短总结),不要写多余解释。
8. 你【禁止】再调用 delegate_subsystem(你已是子树底层,再开子编排器会无限嵌套);
   若某模块仍太大,用普通 delegate,它会自行递归拆分。
9. delegate 工具【没有】subdir 参数——不要给它传 subdir(会被拒绝);本子系统内的模块
   会自动落到 {SUBDIR}/,你只需正常 delegate 并用 assemble 组装。
"""


def sub_orchestrate(task: str, name: str, provides: list, depends_on: list,
                    subdir: str, parent_project: str, depth: int) -> str:
    """运行一个子编排器构建子系统,把成果落到磁盘并在父上下文登记一个薄契约节点。"""
    # 深度到顶:退化为普通 delegate(在父上下文直接建一个模块,不再分层)。
    if depth >= MAX_ORCH_DEPTH:
        return delegate(task, name=name, provides=provides, depends_on=depends_on, _depth=depth)
    sys_prompt = SUBSYS_PROMPT.replace("{SUBDIR}", subdir)
    _push_orch_ctx(sys_prompt)
    n_mods = 0
    try:
        query = (
            f"构建子系统「{name or task[:14]}」,目标是:{task}\n"
            f"父编排器要求你对外暴露的薄契约(provides)符号为:"
            f" {provides or '（由你自行决定,但务必内聚且可被父调用）'}\n"
            f"请把它拆成小模块逐个 delegate,最后 assemble 到子目录 '{subdir}',"
            f"并确保这些 provides 符号能被 `from {subdir} import ...` 导入。"
        )
        print(f"\n📡 进入子编排器构建子系统 '{name or task[:14]}'"
              f" (subdir={subdir}, 深度 {depth + 1}/{MAX_ORCH_DEPTH})", flush=True)
        run_agent(query)  # 子编排器在独立上下文里跑,只看到自己的子树
        n_mods = len([s for s in SUBTASKS
                      if s.get("manifest", {}).get("module") not in ("main.py", "__init__.py")])
    finally:
        _pop_orch_ctx()  # 还原父上下文(必须在读 SUBTASKS 之前)

    # 把子系统成果从 PROJECT_DIR/{subdir} 归位到 PROJECT_DIR/{parent_project}/{subdir}
    built = os.path.join(PROJECT_DIR, subdir)
    target_rel = os.path.join(parent_project, subdir) if parent_project else subdir
    target = os.path.join(PROJECT_DIR, target_rel)
    os.makedirs(os.path.dirname(target) or PROJECT_DIR, exist_ok=True)  # 父级目录兜底
    if os.path.isdir(built) and built != target:
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.move(built, target)
    # 子编排器的内部 SUBTASKS 已在 _pop 时丢弃,这里扫描落盘目录重建内部模块清单以生成桥
    os.makedirs(target, exist_ok=True)  # 首轮被截断/没真正组装时 target 可能还不存在
    built_mods = [f for f in os.listdir(target)
                  if f.endswith(".py") and f != "__init__.py"]
    if not built_mods:
        # 空子系统必须硬失败:子编排器没产出任何模块(规划被截断/只调用未组装),
        # 若照常登记薄契约节点,父 assemble 冒烟必失败并拖入数轮 5 分钟级修复。
        return (f"[{name}] 子系统构建失败:子编排器未产出任何模块({target} 为空)。"
                f"请不要登记薄契约,请重新 delegate_subsystem 或拆成普通 delegate 分别构建。")
    init_code = _build_subpkg_init_from_dir(target, provides)
    try:
        with open(os.path.join(target, "__init__.py"), "w", encoding="utf-8") as f:
            f.write(init_code)
    except Exception as e:
        return f"[{name}] 子编排器已构建但写 __init__.py 失败: {e}"

    sid = len(SUBTASKS) + 1
    SUBTASKS.append({
        "id": sid,
        "name": name,
        "is_subsystem": True,
        "subdir": target_rel,
        "task": task,
        "manifest": {
            "module": f"{target_rel}/__init__.py",
            "provides": provides or [],
            "code": init_code,  # re-export 桥,使契约校验不误报
        },
        "contract": {"provides": provides or [], "depends_on": depends_on or []},
        "model": "sub-orchestrator",
        "tier": "flash",
        "latency": 0.0,
    })
    tag = f"[{name}] " if name else ""
    return (f"{tag}子系统#{sid} 构建完成(子编排器) | subdir={target_rel} | "
            f"薄契约 provides={provides} | 内部 {n_mods} 个模块不进父上下文")


def _build_subpkg_init_from_dir(subdir_path: str, provides: list) -> str:
    """扫描子系统目录里的 .py 文件,重建内部模块 -> 顶层符号映射,生成 __init__.py 桥。"""
    mod_syms = {}
    if os.path.isdir(subdir_path):
        for fn in sorted(os.listdir(subdir_path)):
            if not fn.endswith(".py") or fn == "main.py" or fn == "__init__.py":
                continue
            try:
                with open(os.path.join(subdir_path, fn), encoding="utf-8") as f:
                    code = f.read()
            except Exception:
                continue
            mod_syms[fn[:-3]] = _defined_top_names(code)
    lines = ["# 自动生成的薄契约桥:仅 re-export 父编排器需要的符号,隐藏子系统内部实现。",
             "from __future__ import annotations", ""]
    missing = []
    for p in provides:
        sym = p.split("(")[0].split("->")[0].strip()
        if not sym:
            continue
        host = next((b for b, syms in mod_syms.items() if sym in syms), None)
        if host:
            lines.append(f"from .{host} import {sym}")
        else:
            missing.append(sym)
    lines.append("")
    if missing:
        lines.append("# 以下薄契约符号在子系统内部未找到定义(子系统未兑现);"
                     " 父 assemble / 契约校验会暴露此问题:")
        for sym in missing:
            lines.append(f"#   {sym}")
    return "\n".join(lines)


def delegate_subsystem(task: str, subdir: str, name: str = None,
                       provides: list = None, depends_on: list = None,
                       project_name: str = None) -> str:
    """把一个【子系统】(一簇内聚模块的合集)派给子编排器构建(无损 scope 分解)。
    当某簇任务体量很大,或你基于一大坨【已有代码】做改进而顶层契约列表装不下时,用本工具:
    把内聚的一簇任务打包成一个子系统,委派给子编排器递归 delegate+assemble;
    子编排器只把『对外薄契约』(provides 符号)返回给你,你这边只占 1 个节点,契约列表不膨胀。
    子编排器会把模块落到 {project}/{subdir}/ 并生成 __init__.py 桥;你后续用
    `from subdir import <符号>` 或 `import subdir` 调用它。
    project_name 须与后续 write_main/assemble 用的项目名一致(默认取当前项目或 'project')。"""
    depth = _orch_depth()
    if depth >= MAX_ORCH_DEPTH:
        return delegate(task, name=name, provides=provides, depends_on=depends_on)
    parent = project_name or _CURRENT_PROJECT or "project"
    return sub_orchestrate(task, name, provides or [], depends_on or [],
                           subdir, parent, depth)


def _seed_one_file(full: str, fn: str) -> bool:
    """载入单个已有 .py 文件为 is_existing 节点。成功返回 True。"""
    try:
        with open(full, encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"   ⚠ 读 {fn} 失败: {e}")
        return False
    if not code.strip():
        return False
    provides = sorted(_defined_top_names(code))
    sid = len(SUBTASKS) + 1
    SUBTASKS.append({
        "id": sid,
        "name": fn[:-3],
        "is_existing": True,
        "task": f"已有模块 {fn}(基于已有代码改进,保留核心职责与对外接口)",
        "manifest": {"module": fn, "code": code,
                     "provides": provides, "depends_on": []},
        "contract": None,
        "model": "existing",
        "tier": "flash",
        "latency": 0.0,
    })
    print(f"   📦 载入已有模块: {fn} | provides={provides}")
    return True


def _seed_existing_code(path: str) -> int:
    """「基于已有代码改进」入口:PATH 支持两种形态——
    · 单文件 .py:只载入这一个有代码文件(最常用的「改进这一份」);
    · 目录:扫描其顶层 .py(不递归),跳过 main.py/__init__.py。
    载入的模块作为 is_existing 节点入侧边存储,编排器可在【保留核心职责与对外接口】前提下
    delegate(repair=...) 改进,或改动面大时用 delegate_subsystem 归成子系统。非破坏式——原代码不动。
    返回载入模块数。"""
    global _CURRENT_PROJECT
    p = os.path.abspath(path)
    if os.path.isfile(p) and p.endswith(".py"):
        _CURRENT_PROJECT = os.path.basename(os.path.dirname(p).rstrip(os.sep)) or "project"
        return 1 if _seed_one_file(p, os.path.basename(p)) else 0
    if not os.path.isdir(p):
        print(f"   ⚠ --improve 路径不存在或不是 .py 文件/目录: {path}")
        return 0
    _CURRENT_PROJECT = os.path.basename(p.rstrip(os.sep)) or "project"
    count = 0
    for fn in sorted(os.listdir(p)):
        if not fn.endswith(".py") or fn in ("main.py", "__init__.py"):
            continue
        full = os.path.join(p, fn)
        if not os.path.isfile(full):
            continue
        if _seed_one_file(full, fn):
            count += 1
    return count


def _subtask_inventory() -> str:
    """把 SUBTASKS 里已载入/已交付的普通模块渲染成『模块名 → 提供符号』清单,注入编排器上下文。
    35B 若看不到真实符号名会臆造(如把 mining.py 记成 miner.py、把 hide_window 记成 get_sys_info),
    进而写错 import、把 verify/修复闭环拖入数轮空转。清单只占几十 token,物超所值。"""
    lines = []
    provider_of = {}
    for st in SUBTASKS:
        m = st.get("manifest", {}) or {}
        mod = m.get("module") or ""
        if mod == "main.py" or st.get("is_subsystem"):
            continue
        prov = [p for p in (m.get("provides") or []) if p]
        lines.append(f"- {mod} 提供: {', '.join(prov) if prov else '(无对外提供)'}")
        for p in prov:
            sym = p.split("(")[0].split("->")[0].strip()
            if sym:
                provider_of.setdefault(sym, []).append(mod)
    dup = sorted({s for s, ms in provider_of.items() if len(set(ms)) > 1})
    if dup:
        lines.append("  ⚠ 重复提供: " + ", ".join(dup) +
                     " 由多个模块提供,请合并或删除其一")
    return "\n".join(lines) if lines else "(空)"


# ----------------------------------------------------------------------------
# 工具 schema(精简,控制总 token 数)
# ----------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": "把一个子任务派给云端模型(subagent)完成。用于需要生成代码/内容/计算的子任务。返回简短指针(真实成果存入侧边存储,不占上下文)。复杂任务请先自行分解为多个子任务,逐个 delegate。注意:subagent 无法读取你的项目文件,产出是【新代码模块】而非报告——不要委派『读取/查看/打印文件』类任务;要审查已有代码,直接基于你掌握的真实符号清单 delegate 目标模块即可。",
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
            "name": "delegate_subsystem",
            "description": "把一个【子系统】(一簇内聚模块的合集)派给子编排器构建——无损 scope 分解。当某簇任务体量很大,或你基于一大坨【已有代码】做改进而顶层契约列表装不下时,用本工具:把内聚的一簇任务打包成一个子系统,委派给子编排器递归 delegate+assemble;子编排器只把『对外薄契约』(provides 符号)返回给你,你这边只占 1 个节点,契约列表不膨胀。子编排器会把模块落到 {project}/{subdir}/ 并生成 __init__.py 桥;你后续用 `from subdir import <符号>` 或 `import subdir` 调用它。project_name 须与后续 write_main/assemble 用的项目名一致(默认取当前项目或 'project')。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "该子系统的「子目标」:清晰、自包含,描述这一簇模块合起来要做什么(不要写入总目标)。"},
                    "subdir": {"type": "string", "description": "子系统落盘子目录名(如 'ingest_pipeline'),须是合法 Python 包名(小写、下划线),不要带斜杠或 .py"},
                    "name": {"type": "string", "description": "可选的子系统工程名/编号,便于后续引用"},
                    "provides": {"type": "array", "items": {"type": "string"}, "description": "该子系统须对外暴露的薄契约符号,如 ['build_index(docs) -> Index'];子编排器须确保这些符号可被父 `from subdir import` 调用。父只看到这些,内部模块不进父上下文。"},
                    "depends_on": {"type": "array", "items": {"type": "string"}, "description": "该子系统可依赖的其它(父级)模块符号契约,格式 '模块:符号(签名)'。"},
                    "project_name": {"type": "string", "description": "项目目录名(须与后续 write_main/assemble 一致);默认取当前项目或 'project'"},
                },
                "required": ["task", "subdir"],
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
            "description": "把所有已完成的子任务(侧边存储中)与 write_main 暂存的 main.py 组装成一个项目:落到同一目录(Python import 即 linker,不做接线)、做语法校验,并自动做「导入全部模块 + 调用 main 入口」的契约感知冒烟——冒烟失败(符号名/签名漂移、缺失模块、循环依赖)会启动自动修复闭环(回灌云端 subagent 修复后重新组装)。组装后还会做符号级契约校验(模块是否真定义其 provides 符号、main 是否接线各模块),如有漂移会在返回里列出,可用 delegate 修复或 verify 暴露数据形状错配。用于「写一个大项目」类任务收尾。",
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
            "description": "对已 assemble 的项目做【一片】集成校验:写一个简短 Python 片段导入生成模块并对某一个核心流程做**真实断言**(如 register 后 login 能拿到 token)。校验失败会自动把报错回灌给 subagent 修复并重新组装。请分多次调用,每次只验一个切面——一次写太长会被输出上限截断,而截断的测试会被误判成项目 bug。**防作弊硬约束**:断言必须是真实校验(校验具体行为/内容),严禁把失败测试改成恒真/弱化断言(如 `assert len(output)>10 or \"Error\" not in output`、`assert True`、只用 `len(x)>0` 验存在性不验内容)来强行通过;同一 name 重复 verify 应修正测试逻辑或修复项目,而非放宽断言。命中过弱断言的校验会被系统标记警告。",
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
    # 两阶段修复:阶段①诊断已给出病因与方案,这里注入,让本阶段"照方案改"而非重新推理
    diag = repair.get("_diag")
    if diag:
        if diag.get("cause"):
            parts.append(f"[已诊断病因]\n{diag['cause']}")
        if diag.get("plan"):
            parts.append("[修改方案(严格照此逐条修改)]\n" +
                         "\n".join(f"- {p}" for p in diag["plan"]))
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


def subagent_diag_messages(repair, contract=None):
    """两阶段修复的阶段①诊断:只让 subagent 输出短的病因+方案(不输出代码),
    因此可安全开思考——思考链 + 短诊断都装得进窗口。"""
    parts = [
        f"[原始子目标]\n{repair.get('original_task', '')}",
        f"[当前代码]\n```python\n{repair.get('current_code', '')}\n```",
        f"[报错 / 校验失败信息]\n{repair.get('error', '')}",
    ]
    if contract:
        provides = contract.get("provides") or []
        if provides:
            parts.append("[必须保持的接口 provides(诊断时须保证方案不改这些签名)]\n" +
                         "\n".join(f"- {p}" for p in provides))
    return [
        {"role": "system", "content": SUBAGENT_DIAG_SYS},
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


def parse_diag_output(text):
    """解析诊断输出,返回 {cause, plan, needs_rewrite} 或 None(解析失败→回退单阶段修复)。"""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    obj = None
    try:
        obj = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None
    cause = (obj.get("cause") or "").strip()
    plan = obj.get("plan") or []
    if isinstance(plan, str):
        plan = [plan]
    plan = [str(p).strip() for p in plan if str(p).strip()]
    if not cause and not plan:
        return None
    return {"cause": cause, "plan": plan, "needs_rewrite": bool(obj.get("needs_rewrite"))}


def _wait_printer(stop, model, label, interval=20):
    """在阻塞的网络调用期间周期性打印等待提示,避免看起来像卡死。"""
    waited = 0
    while not stop.wait(interval):
        waited += interval
        print(f"   ⌛ 仍在等待 {model} 生成子任务'{label}'… 已 {waited}s", flush=True)


def _code_lines(code):
    return len([ln for ln in (code or "").splitlines() if ln.strip()])


def _norm_module_name(name):
    """把请求的 name 规范成合法的 .py 文件名(取 basename、补 .py)。"""
    if not name:
        return None
    base = os.path.basename(str(name).strip())
    if not base:
        return None
    return base if base.endswith(".py") else base + ".py"


_STDLIB_NAMES = set(getattr(sys, "stdlib_module_names", set()))


def _shadow_warning(module_name):
    """检测模块名是否与 stdlib 或已安装第三方库同名 → 本地文件会遮蔽真库。
    典型灾难:delegate 一个 requests.py,里面 import requests 实为导入自己 → 自引用递归,
    被上层 except 吞掉后静默返回默认值。返回告警串(空串表示无冲突)。"""
    if not GUARD_STDLIB_SHADOW or not module_name:
        return ""
    top = os.path.basename(module_name)
    top = top[:-3] if top.endswith(".py") else top
    if not top or top == "main":
        return ""
    kind = None
    if top in _STDLIB_NAMES:
        kind = "标准库"
    else:
        try:
            import importlib.util
            # 屏蔽项目自身目录,避免把已交付的同名业务模块误报为第三方
            spec = importlib.util.find_spec(top)
            if spec is not None and getattr(spec, "origin", None) not in (None, "namespace"):
                origin = spec.origin or ""
                if "site-packages" in origin or "dist-packages" in origin:
                    kind = "已安装第三方库"
        except Exception:
            pass
    if not kind:
        return ""
    return (f"模块名 '{top}' 与{kind}同名,本地文件会遮蔽真库(import {top} 将导入你自己→"
            f"极易自引用递归并被上层 except 静默吞掉)。请改名(如 {top}_client.py)或直接 import 真库,"
            f"不要 delegate 同名模块")


def _grain_warning(manifest):
    """粒度 / 完整性体检。语法错误是「被截断」最确定的信号——低比特量化下,
    与其让半截代码流进 assemble 再去猜,不如当场告诉编排器:拆得更小、重来。"""
    notes = []
    code = manifest.get("code", "") or ""
    sw = _shadow_warning(manifest.get("module"))
    if sw:
        notes.append(sw)
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


def _subagent_once(router, chat_fn, tier, tmo, msgs, label, think=False, verb="拆分"):
    """发起一次 subagent 调用(带等待提示),返回 content。用于拆分/诊断等辅助调用(不做 failover)。
    think=True 仅对本地 ollama 后端开思考(云端后端思考由服务端控制,忽略此参数)。
    verb 只影响打印文案(如"拆分"/"诊断")。"""
    model = router.select(tier)
    stop = threading.Event()
    wp = threading.Thread(target=_wait_printer, args=(stop, model, label), daemon=True)
    t0 = time.perf_counter()
    try:
        print(f"   ⌛ 调用 {model} {verb}'{label}'…", flush=True)
        wp.start()
        extra = {"think": True} if (think and chat_fn is ollama_subagent_chat) else {}
        content = chat_fn(model, msgs, timeout=tmo, **extra)
        router.report(model, time.perf_counter() - t0, None)
        return content
    finally:
        stop.set(); wp.join(timeout=1)


def _diagnose(repair, contract, router, chat_fn, tier, tmo, label):
    """两阶段修复的阶段①:开思考做诊断,只输出短方案 {cause, plan, needs_rewrite}。
    成功返回 dict;失败(调用异常/无法解析)返回 None,由调用方回退单阶段修复。
    诊断输出极短(实测 120~175 字符),思考链装得进窗口,不会触发"空内容"截断。"""
    try:
        content = _subagent_once(
            router, chat_fn, tier, tmo,
            subagent_diag_messages(repair, contract), f"{label}:diag",
            think=THINK_ON_REPAIR_DIAG, verb="诊断")  # 诊断=推理步骤,输出短,安全开思考
    except Exception as e:
        print(f"   ⚠ 诊断调用失败({type(e).__name__}),回退单阶段修复", flush=True)
        return None
    diag = parse_diag_output(content)
    if not diag:
        print("   ⚠ 诊断输出无法解析,回退单阶段修复", flush=True)
        return None
    rw = "需重写" if diag.get("needs_rewrite") else "局部小修"
    print(f"   🔎 诊断: {diag.get('cause', '')[:60]} | {rw} | {len(diag.get('plan', []))} 步方案",
          flush=True)
    return diag


def _split_and_recurse(task, name, contract, prev_manifest, tier, depth,
                       router, chat_fn, tmo, replace_id=None):
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
                 _depth=MAX_DELEGATE_DEPTH,   # 深度到顶:强制出代码,聚合不再拆
                 replace_id=replace_id)       # repair 重写:聚合顶替原失败条目(保契约映射)
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
    # 两阶段修复(仅本地 ollama):先诊断(开思考·短输出),再据诊断决定"递归重写"还是"局部小修"。
    # 化解"想开思考但长输出会撑爆窗口"的假两难——诊断与修复本是两种相反性质的工作,拆开就没了。
    if repair and TWO_STAGE_REPAIR and chat_fn is ollama_subagent_chat:
        diag = _diagnose(repair, contract, router, chat_fn, tier, tmo, label)
        if diag:
            # 大改/重写 且 原地修复(replace_id) 且深度未满 → 把该模块当成一次全新 delegate,
            # 递归拆子模块 + 短聚合模块顶替原条目(保持 module/provides 契约映射不破)。
            if (diag.get("needs_rewrite") and replace_id is not None
                    and _depth < MAX_DELEGATE_DEPTH):
                rewritten = _split_and_recurse(
                    repair.get("original_task", task), name, contract,
                    {"code": repair.get("current_code", ""), "truncated": False},
                    tier, _depth, router, chat_fn, tmo, replace_id=replace_id)
                if rewritten is not None:
                    tag = f"[{name}] " if name else ""
                    return f"{tag}子任务#{replace_id} 诊断判定需重写 → {rewritten}"
            # 局部小修:把诊断方案并入 repair,循环内修复阶段关思考、照方案改
            repair = dict(repair)
            repair["_diag"] = diag
    for _ in range(tries):
        model = router.select(tier, exclude=tried)
        t0 = time.perf_counter()
        stop = threading.Event()
        wp = threading.Thread(target=_wait_printer, args=(stop, model, label), daemon=True)
        try:
            print(f"   ⌛ 调用 {model} 生成子任务'{label}'…", flush=True)
            wp.start()
            msgs = subagent_repair_messages(repair, contract) if repair else subagent_messages(task, contract)
            # 修复阶段思考策略:
            #  - 两阶段(repair 带 _diag):诊断已在阶段①开思考做完,本阶段"照方案吐长代码",关思考;
            #  - 单阶段回退(无 _diag):按 THINK_ON_REPAIR(默认关,长输出+思考有撑爆窗口风险);
            #  - 写新模块:纯长输出,恒关思考。
            if repair and chat_fn is ollama_subagent_chat:
                want_think = False if repair.get("_diag") else THINK_ON_REPAIR
            else:
                want_think = False
            extra = {"think": True} if want_think else {}
            content = chat_fn(model, msgs, timeout=tmo, **extra)
            latency = time.perf_counter() - t0
            stop.set(); wp.join(timeout=1)
            router.report(model, latency, None)
            manifest = parse_subagent_output(content)
            # 请求的 name 是权威文件名:模型自作主张改名(如请求 fetch.py 却返回 rouge_fetcher.py)
            # 会导致落盘文件名与 main 的 import 不符 → 孤儿文件 + 编排器重复 delegate 补名。
            # 强制以 name 覆盖 module,一次到位,不再产生孤儿。
            _nm = _norm_module_name(name)
            if _nm:
                manifest["module"] = _nm
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
    global MAIN_CODE, _CURRENT_PROJECT
    MAIN_CODE = main_code or ""
    _CURRENT_PROJECT = project_name or _CURRENT_PROJECT
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


def _defined_top_names(code: str) -> set:
    """模块顶层定义的 def/class/async def/赋值名字集合(契约校验用)。"""
    names = set()
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return names
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def _parse_dep(dep: str):
    """'module:symbol(args)->ret' -> (module, symbol)。无冒号则 (None, 原文)。"""
    if ":" not in (dep or ""):
        return (None, (dep or "").strip())
    mod, sym = dep.split(":", 1)
    return (mod.strip(), sym.strip())


def _check_contracts(subtasks, main_code):
    """静态契约校验(无 LLM,纯 AST):返回问题文本列表。
    覆盖:① 模块未定义其 provides 符号(孤儿/改名/影子遮蔽);② depends_on 目标符号在目标模块不存在;
    ③ 依赖的模块从未被 delegate(缺失模块);④ main.py 未 import 某已交付模块;
    ⑤ main 调用了未由任何模块提供的符号(契约未兑现/名字拼错)。
    注:模块间『数据形状』错配(如 main 平铺 dict 而消费方读嵌套)属运行时行为,需 verify 真实断言暴露。"""
    issues = []
    mod_symbols, mod_basenames, provided = {}, {}, {}
    for st in subtasks:
        m = st.get("manifest", {}) or {}
        mod = m.get("module") or ""
        if mod == "main.py":
            continue
        if st.get("is_subsystem"):  # 子系统:以包名注册,薄契约符号视为已定义(re-export 桥提供)
            sub = st.get("subdir") or ""
            base = sub.split("/")[-1] or sub
            mod_basenames[base] = mod
            prov = {p.split("(")[0].split("->")[0].strip()
                    for p in (m.get("provides") or [])}
            mod_symbols[mod] = prov
            provided[mod] = prov
            continue
        base = mod[:-3] if mod.endswith(".py") else mod
        mod_basenames[base] = mod
        mod_symbols[mod] = _defined_top_names(m.get("code", "") or "")
    for st in subtasks:
        m = st.get("manifest", {}) or {}
        mod = m.get("module") or ""
        if mod == "main.py":
            continue
        prov = set()
        for p in (m.get("provides") or []):
            prov.add(p.split("(")[0].split("->")[0].strip())
        provided[mod] = prov
        for sym in prov:                      # ① provides 未定义
            if sym and sym not in mod_symbols.get(mod, set()):
                issues.append(f"模块 {mod} 声明提供 '{sym}' 但文件中未定义该符号"
                              f"(可能改名/孤儿文件/被同名影子模块遮蔽)")
    # ⑥ 重复提供:同一符号被多个模块声明提供(重复/遗留模块,如 sys_ops.py 与 system_ops.py)
    provider_of = {}
    for mod, prov in provided.items():
        for sym in prov:
            if sym:
                provider_of.setdefault(sym, []).append(mod)
    for sym, mods in provider_of.items():
        if len(mods) > 1:
            issues.append(f"符号 '{sym}' 被多个模块重复提供: {', '.join(sorted(set(mods)))}"
                          f"(疑似重复/遗留模块,建议合并或删除其一)")
    for st in subtasks:                      # ② + ③ depends_on 解析
        m = st.get("manifest", {}) or {}
        mod = m.get("module") or ""
        if mod == "main.py" or st.get("is_subsystem"):
            continue
        for dep in (m.get("depends_on") or []):
            dmod, dsym = _parse_dep(dep)
            if not dmod:
                continue
            target = mod_basenames.get(dmod)
            if not target:
                issues.append(f"模块 {mod} 依赖 '{dep}' 但目标模块 '{dmod}' 从未被 delegate(缺失模块)")
                continue
            if dsym and dsym not in mod_symbols.get(target, set()):
                issues.append(f"模块 {mod} 依赖 {target} 的 '{dsym}',但 {target} 未定义该符号(契约漂移)")
    if main_code and main_code.strip():       # ④ + ⑤ main 接线
        try:
            mtree = ast.parse(main_code)
        except SyntaxError:
            mtree = None
        if mtree:
            imported, called = set(), set()
            for node in ast.walk(mtree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported.add((a.asname or a.name).split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported.add(node.module.split(".")[0])
                    for a in node.names:
                        imported.add(a.name)
                elif isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Name):
                        called.add((None, f.id))
                    elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                        called.add((f.value.id, f.attr))
            for base in mod_basenames:        # ④ main 未接线模块
                if base not in imported:
                    issues.append(f"main.py 未 import 模块 '{mod_basenames[base]}'(该模块已交付但组合根未接线)")
            all_provided = set()
            for s in provided.values():
                all_provided |= s
            _builtins = {"print", "len", "range", "dict", "list", "str", "int", "float", "open",
                         "set", "tuple", "bool", "enumerate", "zip", "map", "filter", "sorted",
                         "isinstance", "type", "min", "max", "sum", "abs", "Exception", "ValueError",
                         "TypeError", "KeyError", "IndexError", "RuntimeError", "AttributeError",
                         "json", "os", "sys", "re", "time", "math"}
            for modp, sym in called:          # ⑤ 调用的符号无模块提供
                if modp is None:
                    if sym in _builtins or sym in imported or sym in all_provided:
                        continue
                    issues.append(f"main.py 调用了 '{sym}()' 但无模块提供该符号(契约未兑现或名字拼错)")
                else:
                    if modp in mod_basenames:
                        target = mod_basenames[modp]
                        if sym not in mod_symbols.get(target, set()):
                            issues.append(f"main.py 调用 {target}.{sym}() 但 {target} 未定义该符号(契约漂移)")
    return issues


def assemble(project_name: str) -> str:
    global _CURRENT_PROJECT
    _CURRENT_PROJECT = project_name or _CURRENT_PROJECT
    # 子系统节点已在磁盘上(子编排器构建时落盘 + 归位),不进扁平写;只写普通业务模块
    biz_subtasks = [st for st in SUBTASKS
                    if not st.get("is_subsystem")
                    and st.get("manifest", {}).get("module") != "main.py"]
    if not biz_subtasks:
        return "[assemble] 侧边存储中没有业务子任务,请先 delegate 若干子任务(并 write_main 组合根)。"
    out_dir = os.path.join(PROJECT_DIR, project_name or "project")
    # 子系统子目录是本地包,需从 requirements.txt 聚合里排除(否则被误判为第三方依赖)
    local_pkgs = [st.get("subdir", "").split("/")[0]
                  for st in SUBTASKS if st.get("is_subsystem") and st.get("subdir")]
    try:
        # main.py 由 _get_main_code 单独落盘;业务模块排除 main.py 条目,避免双重写
        main_code = _get_main_code()
        written = smoke.assemble(out_dir, biz_subtasks, main_code, local_pkgs=local_pkgs)
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
        # 运行时契约校验(符号级,纯 AST 机械检查):暴露孤儿/改名/影子/未接线等契约漂移。
        # 与冒烟互补——冒烟只验「能跑」,契约校验验「接的对不对」。
        if CONTRACT_CHECK:
            cissues = _check_contracts(biz_subtasks, main_code)
            if cissues:
                lines.append(f"   ⚠ 契约校验发现 {len(cissues)} 处符号级漂移:")
                for c in cissues:
                    lines.append(f"     - {c}")
                lines.append("     提示:模块间『数据形状』错配(如 main 平铺 dict 而消费方读嵌套)"
                              "冒烟不报错,请用 verify 写真实断言暴露并修复。")
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
            provides=[],
            depends_on=[],
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


# 校验防作弊:低比特量化模型可能在 verify 失败后把断言改弱(如把失败测试改成
# `len(output)>10 or "Error" not in output`)来强行变绿。这里做轻量静态扫描,命中可疑
# 模式就在结果里告警,提醒编排器这可能不是真实通过——失败应去修项目/改测试逻辑,而非放宽断言。
_WEAK_ASSERT_PATTERNS = [
    (re.compile(r'or\s+["\']Error["\']\s+not\s+in', re.I),
     '"... or Error 不在输出里" 兜底:任何输出都能通过,失去校验意义'),
    (re.compile(r'\bassert\s+True\b', re.I),
     'assert True 恒真,无校验意义'),
    (re.compile(r'\bassert\s+\d+\b'),
     'assert <数字常量> 恒真,无校验意义'),
    (re.compile(r'\bassert\s+len\([^)]*\)\s*>\s*0\b'),
     '仅用 len()>0 验存在性,不校验内容:空壳/占位也能通过'),
]

def _warn_weak_assert(test_code: str) -> str:
    """扫描 test_code,若命中过弱断言模式返回告警文本(空串表示未发现)。"""
    if not test_code:
        return ""
    hits = []
    for pat, msg in _WEAK_ASSERT_PATTERNS:
        if pat.search(test_code):
            hits.append(msg)
    if not hits:
        return ""
    return ("⚠ [verify 防作弊] 检测到可能过弱的断言,请确认其确实在验证核心行为而非强行变绿:"
            + "".join(f"\n   - {h}" for h in hits)
            + "\n   若校验失败,应修正测试逻辑或修复项目,不要放宽断言。")

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
    if not os.path.isfile(os.path.join(out_dir, "main.py")):
        return (f"{tag}[verify 未执行] 项目目录里还没有 main.py,请先 write_main 并 assemble"
                f"后再校验,避免对缺失的组合根空跑修复闭环。")
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
    weak = _warn_weak_assert(test_code)  # 防作弊:扫描过弱/恒真断言
    try:
        ok, fails = _check_test(out_dir, test_code)
        if ok:
            suffix = (f"{over}\n{weak}" if weak else over)
            return f"{tag}✅ 集成校验通过({n} 行 | 项目: {project_name}){suffix}"
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
        if weak:
            lines.append(weak)
        return "\n".join(lines)
    except Exception as e:
        return f"[verify 错误] {e}"


DISPATCH = {
    "delegate": delegate,
    "delegate_subsystem": delegate_subsystem,
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
        bigger = max(base_ctx, SUB_CTX_MAX)
        if bigger > base_ctx:
            print(f"   ↻ 疑似思考链耗光上下文窗口,放大 num_ctx={bigger} 重试一次…", flush=True)
            try:
                content, reason = _ollama_complete(
                    model, messages, bigger, SUB_NUM_PREDICT, SUB_TEMPERATURE, timeout, think=think)
                if content.strip():
                    acc = content
                    if reason == "length":
                        print(f"   ⚠ 放大后仍被长度截断,将尝试从残缺 JSON 中抢救代码", flush=True)
            except Exception as e:
                # 放大 num_ctx 常在小显存上 OOM 让 Ollama 崩(HTTP 500)。这里吞掉,
                # 让流程落到"空内容"错误分支(可换更小子任务重试),而非炸穿整个 delegate。
                print(f"   ⚠ 放大 num_ctx 重试失败({type(e).__name__}),多半是显存不足;"
                      f"建议拆更小的子任务或降低 AGENT_SUB_CTX_MAX", flush=True)
    if not acc.strip():
        raise RuntimeError("本地 subagent 返回空内容")
    return acc


def _coerce_list(v):
    """把模型可能传成字符串的 provides/depends_on 归一化为 list。
    兼容 JSON 数组文本、单引号数组、逗号/换行分隔等脏形态;签名内的逗号(如
    tokenize(text, lang="en"))由引号匹配保护,不会被误拆。"""
    if v is None:
        return None
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            j = json.loads(s)  # 合法 JSON 数组文本(双引号)
            if isinstance(j, list):
                return [str(x) for x in j if str(x).strip()]
        except Exception:
            pass
        s2 = s
        if s2.startswith("[") and s2.endswith("]"):   # 剥掉包裹方括号,避免被兜底当裸项
            s2 = s2[1:-1].strip()
            if not s2:
                return []
        parts = re.findall(r"""'([^']*)'|"([^"]*)"|([^,\n]+)""", s2)
        out = []
        for a, b, c in parts:
            it = (a or b or c).strip().strip("\"'")
            if it:
                out.append(it)
        return out
    return [str(v)]


def _dispatch_tool(tc: dict) -> str:
    fn = tc.get("function", {})
    name = fn.get("name", "")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except Exception:
            args = {}
    # 模型常把 provides/depends_on 传成字符串(JSON 文本/单引号数组),归一化再派发
    if isinstance(args, dict) and name in ("delegate", "delegate_subsystem"):
        for _k in ("provides", "depends_on"):
            if _k in args:
                args[_k] = _coerce_list(args[_k])
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
def fix_common_errors_prompt(proj: str = "项目") -> str:
    """--fix-project 的内置巡检修复提示词(带项目名,强化『先读后改』与『不另起炉灶』)。"""
    return (
        f"对已载入侧边存储的这批已有模块({proj})做一轮『质量巡检与修复』,重点检查并修正常见错误:\n"
        "0. 【先读后改】逐个读取已载入的每个模块的完整代码,先理解再审查;修复必须基于这些真实代码,"
        "不要凭空新建无关的子系统/模块,不要改变对外符号名;\n"
        "1. 孤儿文件:落盘模块名与 main.py 的 import 是否一致(不一致则改 main 的 import 或补模块);\n"
        "2. 影子模块:不要命名成与标准库/已装第三方库同名(如 requests.py),避免遮蔽真库自引用递归;\n"
        "3. 重复模块:同一符号被多个模块重复提供(如 sys_ops.py 与 system_ops.py 都提供 hide_window)时,"
        "合并或删除其一;\n"
        "4. 契约错配:模块对外函数签名与 main.py 调用方式一致(参数顺序/返回值形状,字典是平铺还是嵌套);\n"
        "5. 静默吞异常:except 里吞掉错误后返回假数据(如恒 0.0)的,改为抛错或打日志;\n"
        "6. 未接线/死代码:已交付模块是否被 main.py 真正 import 并使用,重复代码清除;\n"
        "7. 相对导入:扁平淡目录下 main.py 用绝对导入(from x import ...),不要 from .x import ...;\n"
        "8. 校验:用 verify 写【真实断言】验证核心流程(禁止把失败测试改成恒真断言强行通过)。\n"
        f"组装时 write_main/assemble 的 project_name 请沿用 {proj} 或其派生名(如 {proj}_fixed),"
        "不要另起炉灶。修复后重新 assemble 并确保冒烟通过。"
    )

_MANUAL = """\
使用示例:
  python ollama_agent.py --local "写一个 py 脚本..."                # 单次任务(全本地,串行递归)
  python ollama_agent.py --local --improve 项目目录 "优化这里..."     # 目录模式:扫顶层 .py 载入侧边存储
  python ollama_agent.py --local --improve 某个文件.py "重构它"       # 单文件模式:只载入这一份
  python ollama_agent.py --local --fix-project 项目目录              # 项目回喂:巡检并修复常见错误
  python ollama_agent.py                                           # 交互模式(不传 query)

项目回喂标准流程:
  构建完成后,把项目目录传给 --fix-project(等价于 --improve 目录 + 内置"检查并修复常见错误"提示词),
  再带上 --local,让同一个 35B 对成品做一轮质量巡检与修复;也可用 query 自定义本次巡检要求。

内置巡检项(fix_common_errors_prompt):
  先读后改 / 孤儿文件 / 影子模块(遮蔽标准库与第三方库) / 重复模块 / 契约错配(签名与返回形状) /
  静默吞异常返回假数据 / 未接线死代码 / 相对导入 / verify 真实断言防作弊。

环境变量(可选):
  AGENT_NUM_CTX 编排器上下文窗口(默认8192)  AGENT_SUB_CTX_MAX 放大上限(默认24576,8GB显存勿再调大)
  AGENT_BUDGET 压缩触发预算  AGENT_CONTRACT_CHECK 契约校验开关  AGENT_GUARD_SHADOW 影子模块告警
  AGENT_MAX_ORCH_DEPTH 分层子编排器深度上限(默认2)
"""


def main():
    global DELEGATE_BACKEND
    parser = argparse.ArgumentParser(
        description="🦙 本地多智能体编排器:串行递归 + 侧边存储 + 契约感知组装(全本地 Ollama 35B)",
        epilog=_MANUAL,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="*", help="单次任务(不填则进入交互模式)")
    parser.add_argument("--local", action="store_true",
                        help="全本地模式:subagent 也用本地 Ollama(串行递归),完全不联网")
    parser.add_argument("--improve", metavar="PATH",
                        help="基于已有代码改进:PATH 可以是单个 .py 文件(改进这一份),"
                             "或目录(扫描其顶层 .py 载入侧边存储);随后执行改进任务(query)")
    parser.add_argument("--fix-project", metavar="DIR",
                        help="对已完成项目目录做一轮『巡检并修复常见错误』:等价于 --improve DIR + 内置修复提示词(可再附 query 自定义)")
    args = parser.parse_args()
    if args.local:
        DELEGATE_BACKEND = "ollama"

    # --fix-project:项目完成后回喂 = --improve 目录 + 默认巡检修复提示词
    if args.fix_project:
        if not args.improve:
            args.improve = args.fix_project
        if not args.query:
            _proj = os.path.basename(os.path.abspath(args.improve).rstrip(os.sep)) or "项目"
            args.query = [fix_common_errors_prompt(_proj)]

    # --improve:把已有代码载入侧边存储,并把「在此基础改进」的指引前置到任务里
    if args.improve:
        n = _seed_existing_code(args.improve)
        if n:
            inv = _subtask_inventory()
            note = (f"[已有代码已载入侧边存储,共 {n} 个模块。请在保留这些模块核心职责与"
                    f"对外接口的前提下做如下改进。\n"
                    f"[已载入模块清单——只准引用下列【真实】模块名与提供符号,不要臆造不存在的名字"
                    f"(如把 mining.py 记成 miner.py)]\n{inv}\n"
                    f"注意:不要 delegate『读取/查看/打印文件』类任务——subagent 产出的是新代码模块,"
                    f"不会把文件内容读回给你;要审查/修复代码,直接基于上面清单 delegate 目标模块。\n"
                    f"改进任务: ")
            args.query = [note] + list(args.query)
            print(f"   📥 已载入已有代码 {n} 个模块,将在此基础上改进(项目名={_CURRENT_PROJECT})")

    print(f"🦙 本地编排器 | 模型={MODEL} | num_ctx={NUM_CTX} | Ollama={OLLAMA_URL}")
    print(f"   compact: mode={SUMMARY_MODE if USE_SUMMARY else 'off'} | thinking={'on' if ENABLE_THINKING else 'off'} | 预算={BUDGET} token")
    _repair_desc = (f"两阶段(诊断={'on' if THINK_ON_REPAIR_DIAG else 'off'}/修复=off)"
                    if TWO_STAGE_REPAIR else f"单阶段={'on' if THINK_ON_REPAIR else 'off'}")
    print(f"   选择性推理: 顶层规划={'on' if ENABLE_THINKING else 'off'}"
          f" · 拆分决策={'on' if THINK_ON_SPLIT else 'off'}"
          f" · 修复诊断={_repair_desc}"
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
