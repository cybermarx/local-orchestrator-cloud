#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke.py —— 去 linker 后的「同目录 + 冒烟 + manifest 路由」验证器

设计前提(Python 导入即 linker):模块 = 文件 = 命名空间,把所有 subagent 交付的
模块文件与编排器写的 main.py 放到同一目录,import 在运行时自行解析,无需独立的接线步骤。
本模块只做两件必要的事(原来 linker 的接线逻辑已删除):
  1. assemble():把 SUBTASKS 里每个模块的 code + 编排器的 main_code 落到同一扁平目录。
  2. smoke_project():导入全部业务模块 + 真正调用 main 入口,捕获契约漂移
     (AttributeError / TypeError / ModuleNotFoundError / 循环依赖),并给出可路由的失败点
     (module.symbol),供 repair_loop 按 manifest 把错误回灌给正确的 subagent。

纯标准库,零依赖。可独立测试:
  python -c "import smoke, json; print(smoke.smoke_project('项目目录'))"
"""
import os
import re
import sys
import subprocess

STDLIB = set(sys.stdlib_module_names)
IMPORT_RE = re.compile(r'^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))', re.M)


def list_modules(out_dir):
    """业务 .py 模块(排除 main.py 脚手架与本模块自己写的 test_smoke.py)。"""
    if not os.path.isdir(out_dir):
        return []
    return [fn for fn in sorted(os.listdir(out_dir))
            if fn.endswith(".py") and fn != "main.py" and fn != "test_smoke.py"]


def assemble(out_dir, subtasks, main_code):
    """把所有 subagent 交付的模块文件 + 编排器写的 main.py 落到同一目录(扁平,不接线)。
    subtasks: [{"manifest": {module, code, provides, depends_on}, ...}]。返回写入的文件列表。"""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    local_mods = set()
    codes = []
    for st in subtasks:
        m = st.get("manifest", {})
        mod = m.get("module")
        code = m.get("code", "") or ""
        if not mod or not code.strip():
            continue
        with open(os.path.join(out_dir, mod), "w", encoding="utf-8") as f:
            f.write(code)
        written.append(mod)
        local_mods.add(mod[:-3] if mod.endswith(".py") else mod)
        codes.append(code)
    if main_code and main_code.strip():
        with open(os.path.join(out_dir, "main.py"), "w", encoding="utf-8") as f:
            f.write(main_code)
        written.append("main.py")
    # 轻量 requirements.txt:聚合第三方 import(排除 stdlib 与项目自有模块),供用户 pip install
    reqs = set()
    for c in codes:
        for mm in IMPORT_RE.finditer(c or ""):
            top = (mm.group(1) or mm.group(2) or "").split(".")[0]
            if top and top not in STDLIB and top not in local_mods:
                reqs.add(top)
    with open(os.path.join(out_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        if reqs:
            f.write("\n".join(sorted(reqs)) + "\n")
    return written


def _symbol_from_error(text):
    """从错误文本尽量提取 'module.symbol' 形式的问题符号,用于精确路由修复。
    覆盖:符号缺失(AttributeError)、未定义名字、签名不匹配、
    未交付模块(No module named 'X')、无法从模块导入某符号(cannot import name 'Y' from 'Z')。"""
    m = re.search(r"module '([\w\.]+)' has no attribute '([\w]+)'", text or "")
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    # 未交付/拼错的模块:No module named 'X' -> 返回 X(裸名,供路由或报告)
    m = re.search(r"No module named '([\w\.]+)'", text or "")
    if m:
        return m.group(1)
    # 无法从模块导入某符号:cannot import name 'Y' from 'Z' -> 返回 Z.Y
    m = re.search(r"cannot import name '([\w]+)' from '([\w\.]+)'", text or "")
    if m:
        return f"{m.group(2)}.{m.group(1)}"
    m = re.search(r"name '([\w]+)' is not defined", text or "")
    if m:
        return m.group(1)
    m = re.search(r"([\w\.]+)\(\) (?:missing|takes)", text or "")
    if m:
        return m.group(1)
    return None


def smoke_project(out_dir, entry="main", timeout=60):
    """导入全部业务模块 + 调用 main 入口。返回 (ok, failures)。
    failure: {"module": 业务模块名或 'main.py' 或 None, "symbol": 'module.symbol' 或 None, "error": 末段文本}
    调用入口是「契约感知」的:会真正执行 main(),从而暴露组合根里对 subagent 符号的
    名字/签名漂移(AttributeError / TypeError),而不只是导入期错误。"""
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        return False, [{"module": None, "symbol": None, "error": f"目录不存在: {out_dir}"}]
    env = dict(os.environ)
    env["_SMOKE_DIR"] = out_dir
    failures = []

    # 1) 逐个导入业务模块(捕获 ModuleNotFoundError / 循环依赖 / 语法错误)
    for fn in list_modules(out_dir):
        mod = fn[:-3]
        code = (
            "import sys, os, traceback\n"
            "sys.path.insert(0, os.environ['_SMOKE_DIR'])\n"
            "try:\n"
            f"    __import__({mod!r})\n"
            "except Exception as e:\n"
            f"    print('IMPORT_FAIL', repr({mod!r}), repr(str(e)))\n"
            "    traceback.print_exc()\n"
        )
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, cwd=out_dir, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            failures.append({"module": fn, "symbol": None, "error": f"导入超时(>{timeout}s)"})
            continue
        if ("IMPORT_FAIL" in r.stdout) or r.returncode != 0:
            err = (r.stderr or r.stdout)
            failures.append({"module": fn, "symbol": _symbol_from_error(err), "error": err[-2500:]})

    # 2) 导入 main 并真正调用入口(捕获组合根里的 AttributeError / TypeError)
    if os.path.exists(os.path.join(out_dir, "main.py")):
        code = (
            "import sys, os, traceback\n"
            "sys.path.insert(0, os.environ['_SMOKE_DIR'])\n"
            "try:\n"
            "    import main\n"
            "    if not hasattr(main, 'main'):\n"
            "        print('ENTRY_FAIL', 'main.py 未定义 main() 入口')\n"
            "    else:\n"
            "        main.main()\n"
            "except Exception as e:\n"
            f"    print('ENTRY_FAIL', repr(str(e)))\n"
            "    traceback.print_exc()\n"
        )
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, cwd=out_dir, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            failures.append({"module": "main.py", "symbol": None, "error": f"入口执行超时(>{timeout}s)"})
        else:
            if ("ENTRY_FAIL" in r.stdout) or r.returncode != 0:
                err = (r.stderr or r.stdout)
                failures.append({"module": "main.py", "symbol": _symbol_from_error(err), "error": err[-2500:]})

    # 去重(同一模块可能同时出现在导入与入口失败)
    seen, uniq = set(), []
    for f in failures:
        key = (f["module"], f["symbol"])
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return (len(uniq) == 0, uniq)


def format_report(out_dir, written, ok, failures):
    lines = [f"📦 项目已组装 -> {out_dir}", f"   文件数: {len(written)}"]
    if ok:
        lines.append("   ✅ 冒烟通过:全部模块导入正常 + main 入口可调用(可直接 python main.py 运行)")
    else:
        lines.append(f"   ⚠ 冒烟发现 {len(failures)} 个失败点:")
        for f in failures:
            loc = f["module"] or "?"
            if f["symbol"]:
                loc += f" :: {f['symbol']}"
            last = f["error"].strip().splitlines()[-1] if f["error"].strip() else "导入/执行失败"
            lines.append(f"     - {loc}: {last[:160]}")
    return "\n".join(lines)
