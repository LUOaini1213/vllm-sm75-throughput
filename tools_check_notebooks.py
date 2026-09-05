# -*- coding: utf-8 -*-
"""notebook 自检 —— 抓 ast.parse 查不出来的几类坑。

每条规则都对应一次真实踩坑（2026-09-05）：
  1 语法          —— 基础
  2 拼接残骸      —— 嵌套三引号把字符串吃掉，写出 `json.loads( + json.dumps(X) + )`
  3 可疑格式符    —— "%>0s" 这种运行时才 ValueError
  4 子进程源码    —— "-c" 传的那段也要能编译
  5 正则捕获组    —— 无捕获组的正则配 m.group(1) 抛 IndexError
  6 宽 except     —— 循环里 `except Exception: pass` 把 5 那种真 bug 无声吞掉

规则 2 只在**抹掉字符串字面量之后**的源码上跑，否则正则里的 (\d+) 会误报。
规则 3 逐个消费 %-转换符，%% 视为一对，否则合法的 "%.0f%%" 会误报。
"""
import ast
import io
import json
import re
import sys

# 合法的 %-转换符（含宽度/精度/标志），%% 也算一个整体
CONV = re.compile(r"%[-+ #0]*[\d*]*(?:\.[\d*]+)?[hlL]?[diouxXeEfFgGcrsa%]")


def bad_percent_spans(s):
    """返回 s 中非法 %-转换符的起始位置。%% 作为一对整体消费。"""
    out, i, n = [], 0, len(s)
    while i < n:
        j = s.find("%", i)
        if j < 0:
            break
        m = CONV.match(s, j)
        if m:
            i = m.end()
        elif j + 1 < n and s[j + 1].isspace():
            # `x % (a, b)` 的格式化运算符——当字符串里嵌着代码时会出现。
            # 真正的转换符没人写成「% 空格」，跳过，避免把运算符当成语法错。
            i = j + 1
        else:
            out.append(j)
            i = j + 1
    return out


def mask_strings(src, tree):
    """把源码里所有字符串字面量替换成空格，便于对「真代码」做文本检查。"""
    offs, pos = [0], 0
    for ln in src.split(chr(10)):
        pos += len(ln) + 1
        offs.append(pos)
    buf = list(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if getattr(node, "end_col_offset", None) is None:
                continue
            a = offs[node.lineno - 1] + node.col_offset
            b = offs[node.end_lineno - 1] + node.end_col_offset
            for k in range(a, min(b, len(buf))):
                if buf[k] != chr(10):
                    buf[k] = " "
    return "".join(buf)


def check(path):
    nb = json.load(io.open(path, encoding="utf-8"))
    problems = []

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("!"):
            continue

        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append((i, "语法错", str(e)[:110]))
            continue

        masked = mask_strings(src, tree)
        for m in re.finditer(r"\(\s*\+\s*[A-Za-z_]", masked):
            frag = src[max(0, m.start() - 40):m.start() + 40].replace(chr(10), " ")
            problems.append((i, "拼接残骸", frag))

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value
                if "%" not in s:
                    continue
                for pos in bad_percent_spans(s):
                    frag = s[max(0, pos - 15):pos + 15].replace(chr(10), " ")
                    problems.append((i, "可疑格式符", repr(frag)))

        for m in re.finditer(r'"-c",\s*("""|\'\'\')(.*?)\1', src, re.S):
            try:
                ast.parse(m.group(2))
            except SyntaxError as e:
                problems.append((i, "子进程源码语法错", str(e)[:110]))

        uses_g1 = ".group(1)" in src
        for m in re.finditer(r'\(r"([^"]+)"\s*,\s*"(\w+)"\)', src):
            pat, key = m.group(1), m.group(2)
            try:
                ngroups = re.compile(pat).groups
            except Exception as e:
                problems.append((i, "正则非法", "%s: %s" % (key, str(e)[:60])))
                continue
            if uses_g1 and ngroups < 1:
                problems.append((i, "正则缺捕获组", "%s 无捕获组但代码用了 group(1)" % key))

        wide = "except Exception:" + chr(92) + "s*" + chr(92) + "n" + chr(92) + "s*pass"
        if re.search(wide, src) and "for " in src:
            problems.append((i, "宽except吞异常", "循环里 except Exception: pass 会掩盖真 bug"))

    return problems


if __name__ == "__main__":
    files = sys.argv[1:] or ["cloud_sglang_v2.ipynb", "cloud_quant_matrix.ipynb"]
    bad_total = 0
    for f in files:
        ps = check(f)
        nb = json.load(io.open(f, encoding="utf-8"))
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        if ps:
            bad_total += len(ps)
            print("%s  (%d 个代码 cell)  发现 %d 处：" % (f, n_code, len(ps)))
            for i, kind, detail in ps:
                print("   cell %-3d %-16s %s" % (i, kind, detail))
        else:
            print("%s  (%d 个代码 cell)  通过" % (f, n_code))
    sys.exit(1 if bad_total else 0)
