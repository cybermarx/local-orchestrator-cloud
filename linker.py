#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linker.py —— 大项目「动态连接器」

把多个云端 subagent 产出的 (code + manifest) 组装成一个可运行项目:
  - 解析依赖图(depends_on -> provides)
  - 检测缺失 / 循环依赖(循环依赖通过 provides 自环粗判)
  - 落盘到项目目录(正确路径)
  - 生成 requirements.txt(聚合第三方 import)
  - 生成 main.py 脚手架(懒导入,加载不崩)
  - py_compile 语法校验
  - 返回报告(项目树 / 依赖 / 未解析符号 / 校验结果)

纯标准库,零依赖。既能被 ollama_agent.py import,也能独立测试:
  python linker.py subtasks.json /path/to/out
其中 subtasks.json 是 [{"module","language","provides","depends_on","code"}, ...]
"""
import os
import re
import sys
import json
import ast
import subprocess

# 权威标准库清单:用解释器自带的 sys.stdlib_module_names,避免手写遗漏(secrets/sqlite3 等)
STDLIB = set(sys.stdlib_module_names)

IMPORT_RE = re.compile(r'^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))', re.M)


def extract_symbols(code):
    """返回模块顶层定义的符号名集合(函数/类/异步函数/模块级常量)。
    语法错误返回 None(交由 py_compile 统一报告,避免重复告警)。"""
    try:
        tree = ast.parse(code or "")
    except Exception:
        return None
    syms = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            syms.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    syms.add(t.id)
    return syms


def _uses_name(code, name):
    """代码中是否出现名为 name 的标识符(用于契约兑现检测)。语法错误返回 None。
    同时匹配裸名(Name)与属性访问(Attribute,如 db.get_user)。"""
    try:
        tree = ast.parse(code or "")
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def _norm_symbol(spec):
    """从 'get_user(id) -> User' 或 'db.py:get_user(id)' 中取出纯符号名。"""
    s = str(spec).split(":", 1)[-1] if ":" in str(spec) else str(spec)
    return s.split("(")[0].strip()


def _norm_module(name: str) -> str:
    return name.replace("\\", "/").strip()


def extract_requirements(codes, local_modules=None):
    local_modules = local_modules or set()
    reqs = set()
    for c in codes:
        for m in IMPORT_RE.finditer(c or ""):
            top = (m.group(1) or m.group(2) or "").split(".")[0]
            if top and top not in STDLIB and top not in local_modules:
                reqs.add(top)
    return sorted(reqs)


def validate_python(out_dir):
    results = []
    for root, _, fs in os.walk(out_dir):
        for fn in sorted(fs):
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                r = subprocess.run([sys.executable, "-m", "py_compile", p],
                                   capture_output=True, text=True)
                results.append({
                    "file": os.path.relpath(p, out_dir),
                    "ok": r.returncode == 0,
                    "detail": (r.stderr.strip().splitlines()[-1] if r.returncode else ""),
                })
    return results


def list_modules(out_dir):
    """返回 out_dir 下所有业务 .py 模块路径(排除 main.py 脚手架与 verify 写入的测试文件)。"""
    mods = []
    for root, _, fs in os.walk(out_dir):
        for fn in fs:
            if fn.endswith(".py") and fn != "main.py" and fn != "test_smoke.py":
                mods.append(os.path.join(root, fn))
    return sorted(mods)


def import_smoke(out_dir):
    """在隔离子进程中逐个导入生成的 .py 模块(除 main.py),捕获导入期错误。
    返回 [{"module","ok","detail"}]。导入期副作用(如建库)会落在 out_dir。
    路径通过环境变量传入,避免 Windows 反斜杠在 -c 代码里变成转义符。"""
    out_dir = os.path.abspath(out_dir)
    results = []
    env = dict(os.environ)
    env["_SMOKE_DIR"] = out_dir
    for path in list_modules(out_dir):
        fn = os.path.basename(path)
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
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, cwd=out_dir, env=env)
        ok = ("IMPORT_FAIL" not in r.stdout) and r.returncode == 0
        detail = ""
        if not ok:
            detail = (r.stderr.strip().splitlines()[-1] if r.stderr.strip()
                      else (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "import failed"))
        results.append({"module": fn, "ok": ok, "detail": detail})
    return results



def link_project(subtasks, out_dir):
    subtasks = subtasks or []
    files = {}
    provides = {}           # symbol -> module
    provides_by_mod = {}    # module -> set(symbol)
    actual = {}             # module -> set(symbol) | None(语法错误)
    issues = []
    for i, st in enumerate(subtasks):
        m = st.get("manifest", st) if isinstance(st, dict) else {}
        mod = _norm_module(m.get("module") or f"module_{i}.py")
        code = m.get("code", "") or ""
        if not code.strip():
            issues.append(f"{mod}: 无 code")
        files[mod] = code
        for p in (m.get("provides") or []):
            sym = _norm_symbol(p)
            if sym:
                provides[sym] = mod
                provides_by_mod.setdefault(mod, set()).add(sym)
        actual[mod] = extract_symbols(code)

    # ---- 契约校验(应对「subagent 只知子目标、互不知晓」的情况)----
    # A. 声明提供的符号必须真在代码里定义
    for st in subtasks:
        m = st.get("manifest", st) if isinstance(st, dict) else {}
        owner = _norm_module(m.get("module") or "?")
        a = actual.get(owner)
        if a is None:
            continue  # 语法错误交给 py_compile 统一报告
        for p in (m.get("provides") or []):
            sym = _norm_symbol(p)
            if sym and sym not in a:
                issues.append(f"{owner}: 声明 provides '{p}' 但代码中未定义该符号")

    # 模块 stem -> 文件名 映射(depends_on 常用 'user_db' 而非 'user_db.py' 引用)
    stem_map = {}
    for _m in files:
        _s = _m[:-3] if _m.endswith(".py") else _m
        stem_map[_s] = _m

    # B. 依赖的「模块:符号」必须存在且被真实提供/定义
    missing = []
    for st in subtasks:
        m = st.get("manifest", st) if isinstance(st, dict) else {}
        owner = _norm_module(m.get("module") or "?")
        for dep in (m.get("depends_on") or []):
            dep_s = str(dep)
            dep_raw = _norm_module(dep_s.split(":")[0])
            dep_mod = dep_raw if dep_raw in files else stem_map.get(
                dep_raw[:-3] if dep_raw.endswith(".py") else dep_raw)
            sym = _norm_symbol(dep_s)
            if dep_mod is None:
                missing.append({"from": owner, "needs": dep_s, "module": dep_mod,
                                "symbol": sym, "reason": "module-missing"})
                continue
            if sym and sym not in provides_by_mod.get(dep_mod, set()):
                missing.append({"from": owner, "needs": dep_s, "module": dep_mod,
                                "symbol": sym, "reason": "symbol-not-provided"})
                continue
            if sym and sym not in (actual.get(dep_mod) or set()):
                missing.append({"from": owner, "needs": dep_s, "module": dep_mod,
                                "symbol": sym, "reason": "symbol-not-defined"})
                continue
            # 软检查:声明依赖却在自身代码里从未引用该符号 -> 可能契约未兑现/符号名不一致
            used = _uses_name(files.get(owner, ""), sym)
            if used is False:
                issues.append(f"{owner}: 声明 depends_on '{dep_s}' 但代码中未引用 '{sym}'(可能契约未兑现或符号名不一致)")

    # 落盘
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for mod, code in files.items():
        path = os.path.join(out_dir, mod)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        written.append(mod)

    # requirements.txt(排除项目自有模块)
    local_mods = set()
    for mod in files:
        nm = mod.replace("\\", "/").split("/")[0]
        if nm.endswith(".py"):
            nm = nm[:-3]
        if nm:
            local_mods.add(nm)
    reqs = extract_requirements(files.values(), local_mods)
    with open(os.path.join(out_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        if reqs:
            f.write("\n".join(reqs) + "\n")

    # main.py 脚手架(懒导入,加载不崩)
    if "main.py" not in files:
        imp_lines = []
        for mod in files:
            if mod.endswith(".py") and mod != "main.py":
                nm = mod[:-3].replace("/", ".").replace("\\", ".")
                imp_lines.append(f"    try:")
                imp_lines.append(f"        import {nm}  # noqa")
                imp_lines.append(f"    except Exception as e:")
                imp_lines.append(f"        print(f'[skip {nm}] {{e}}')")
        body = (
            "def main():\n"
            + "\n".join(imp_lines) + "\n"
            + "    print('project entrypoint scaffold')\n\n"
            + "if __name__ == '__main__':\n"
            + "    main()\n"
        )
        with open(os.path.join(out_dir, "main.py"), "w", encoding="utf-8") as f:
            f.write(body)
        written.append("main.py")

    # contracts.json:导出各模块契约(声明 provides / depends_on / 实际定义符号 / 编排器预定契约)
    # 供人工核对「盲 subagent」是否兑现了接口契约
    contract_doc = {}
    for st in subtasks:
        m = st.get("manifest", st) if isinstance(st, dict) else {}
        mod = _norm_module(m.get("module") or "?")
        contract_doc[mod] = {
            "provides_declared": m.get("provides") or [],
            "depends_on": m.get("depends_on") or [],
            "symbols_defined": sorted(actual.get(mod) or []),
            "intended_contract": (st.get("contract") if isinstance(st, dict) else None),
        }
    contracts_path = os.path.join(out_dir, "contracts.json")
    with open(contracts_path, "w", encoding="utf-8") as f:
        json.dump(contract_doc, f, ensure_ascii=False, indent=2)

    validation = validate_python(out_dir)
    n_ok = sum(1 for v in validation if v["ok"])

    return {
        "out_dir": os.path.abspath(out_dir),
        "files": sorted(written),
        "n_files": len(written),
        "missing_deps": missing,
        "issues": issues,
        "validation": validation,
        "n_valid": n_ok,
        "requirements": reqs,
        "contracts": contracts_path,
    }


def format_report(rep):
    lines = []
    lines.append(f"📦 项目已链接 -> {rep['out_dir']}")
    lines.append(f"   文件数: {rep['n_files']} | 语法校验通过: {rep['n_valid']}/{len(rep['validation'])}")
    lines.append("   文件: " + ", ".join(rep["files"]))
    if rep.get("contracts"):
        lines.append(f"   契约导出: {rep['contracts']}")
    if rep["missing_deps"]:
        lines.append("   ⚠ 契约未兑现 / 缺失依赖:")
        for d in rep["missing_deps"]:
            reason = {
                "module-missing": f"模块 {d['module']} 不存在",
                "symbol-not-provided": f"模块 {d['module']} 未声明提供 '{d['symbol']}'",
                "symbol-not-defined": f"模块 {d['module']} 声明了但未定义 '{d['symbol']}'",
            }.get(d.get("reason"), f"模块 {d['module']} 不匹配")
            lines.append(f"     - {d['from']} 依赖 {d['needs']}: {reason}")
    if rep["issues"]:
        lines.append("   ⚠ 问题: " + "; ".join(rep["issues"]))
    if rep["requirements"]:
        lines.append("   requirements.txt: " + ", ".join(rep["requirements"]))
    bad = [v for v in rep["validation"] if not v["ok"]]
    if bad:
        lines.append("   校验失败:")
        for v in bad:
            lines.append(f"     - {v['file']}: {v['detail']}")
    if not bad and not rep["missing_deps"]:
        lines.append("   ✅ 链接干净,无缺失依赖、无语法错误")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python linker.py subtasks.json out_dir")
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    rep = link_project(data, sys.argv[2])
    print(format_report(rep))
