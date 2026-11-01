# -*- coding: utf-8 -*-
"""把 README 里的每个数字从产物里回算一遍。纯 CPU、纯标准库，不需要 GPU。

这仓的测量全在 Colab T4 上做，GPU 那部分进不了 CI。能进 CI 的是另一半：
**日志解析 + 数字回算**。本脚本做九组检查：

  1  口径     probe.py / 旧 bench_serving.py（冻结副本）的 max_tokens 与请求数公式没被改过
  2  双来源   notebook 的真实 cell 输出 与 results_t4/*.txt 的转写逐格一致
  3  每请求   墙钟 × 吞吐 ÷ 请求数 == 128 token/请求（口径的独立验证）
  4  自洽     墙钟 ≈ (请求数/并发) × (TTFT + 127×TPOT)，逐格给偏差
  5  比值     README 第 D 节的 TPOT 表 / B÷A / C÷A / A÷C 全部回算
  6  更正     corrections_2026-09-13.txt 里的 19.12 / 26.92 / 856.8 / 3.14 回算
  7  GEMM     README 第一屏那张 T4 对照表 对 results_t4/t4_gemm_control.txt
  8  本地     README 第二节的表自洽，且 results_local/ 那份 JSON 的归属可证
  9  引用     README 里提到的每个仓内文件真的存在，notebook 表不缺行

第 4 组有三个已知离群格（A_default@4、A_default@32、D_default_backend@4），
写在 WHITELIST 里，每条都要给理由。白名单是双向的：这些格子如果哪天不再离群，
脚本同样会红 —— 免得白名单变成挡箭牌。
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

NB = "cloud_dip_confirm_executed.ipynb"
TXT = os.path.join("results_t4", "dip_causal_2026-09-08.txt")
CORR = os.path.join("results_t4", "corrections_2026-09-13.txt")
GEMM = os.path.join("results_t4", "t4_gemm_control.txt")
LOCAL_JSON = os.path.join("results_local",
                          "gtx1650_sweep_enforce_eager_2026-09-02.json")
# 历史表格是旧客户端测的。2026-10-12 起根目录的 bench_serving.py 换成了 schema v2 客户端，
# 旧客户端的原始字节冻结在这里（verify_historical.py 核它的 sha256 与 d933443 源码）。
OLD_BENCH = os.path.join("historical", "d933443", "bench_serving.py.txt")

MAX_TOKENS = 128          # probe.py / bench_serving.py 的 --max-tokens 默认值
WALL_TOL = 0.15           # |墙钟 − 自洽值| / 墙钟 的阈值
TOKREQ_TOL = 2.0          # 每请求 token 数允许的绝对偏差
RATIO_TOL = 0.02          # 比值的容差：输入是两位小数的转写值，比值带舍入残差

# 已知离群格：(组, batch) -> 理由。理由要能在仓里指到出处。
WHITELIST = {
    ("A_default", 4):
        "dip_causal_2026-09-08.txt 第六节第 3 条已记：墙钟 11.38s 但 TPOT 5.70ms 正常，"
        "与 sglang_outliers_repeat_2026-09-05.txt 的「并发 4 间歇不稳定」同一现象",
    ("D_default_backend", 4):
        "同上，dip_causal_2026-09-08.txt 第六节第 3 条已记：墙钟 24.19s、TPOT 5.82ms 正常",
    ("A_default", 32):
        "corrections_2026-09-13.txt：同一签名第三次出现，7.8s 停顿不在解码里；"
        "该格吞吐 608.0 已更正为按自洽值折算的 856.8",
}

FAILS = []
NOTES = []


def ok(cond, msg):
    if cond:
        print("    ok   " + msg)
    else:
        print("    FAIL " + msg)
        FAILS.append(msg)


def close(a, b, tol):
    return abs(a - b) <= tol


def read(rel):
    return io.open(os.path.join(HERE, rel), encoding="utf-8").read()


def floats(line):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", line)]


def n_req(conc):
    """probe.py / bench_serving.py：每档发 max(4×并发, 16) 条请求。"""
    return max(4 * conc, 16)


# ---------------------------------------------------------------- 1 口径
def check_protocol():
    print("[1] 压测口径")
    nb = json.load(io.open(os.path.join(HERE, NB), encoding="utf-8"))
    writer = ""
    for c in nb["cells"]:
        if c["cell_type"] == "code" and "probe.py" in "".join(c["source"]):
            writer = "".join(c["source"])
            break
    ok(bool(writer), "notebook 里找得到写出 probe.py 的那个 cell")
    ok(re.search(r'"--max-tokens", type=int, default=128', writer) is not None,
       "probe.py 的 --max-tokens 默认值是 128")
    ok(re.search(r"max\(c\s*\*\s*4,\s*16\)", writer) is not None,
       "probe.py 每档请求数是 max(4×并发, 16)")

    bs = read(OLD_BENCH)
    ok('ap.add_argument("--max-tokens", type=int, default=128)' in bs,
       "旧 bench_serving.py 的 --max-tokens 默认值是 128")
    ok(re.search(r"max\(c\s*\*\s*4,\s*16\)", bs) is not None,
       "旧 bench_serving.py 每档请求数是 max(4×并发, 16)")
    ok("for c in [1, 2, 4, 8, 16, 32]:" in bs,
       "旧 bench_serving.py 的并发列表是 [1, 2, 4, 8, 16, 32]")

    # SHARED_PREFIX 的长度：README 写的是 2380 字符
    m = re.search(r'SHARED_PREFIX = \(\s*(.*?)\)\s*\n', bs, re.S)
    ok(m is not None, "旧 bench_serving.py 里找得到 SHARED_PREFIX 的定义")
    if m:
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
        mult = re.search(r"\*\s*(\d+)", m.group(1))
        n = sum(len(p) for p in parts) * (int(mult.group(1)) if mult else 1)
        ok(n == 2380, "SHARED_PREFIX 长度 %d == README 写的 2380 字符" % n)


# ------------------------------------------------- 2 两个来源逐格一致
def parse_notebook():
    """从 notebook 的真实 cell 输出里取四组的 墙钟/吞吐/TTFT/TPOT。"""
    nb = json.load(io.open(os.path.join(HERE, NB), encoding="utf-8"))
    text = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain")
            if t:
                text.append("".join(t))
    text = "\n".join(text)
    out = {}
    chunks = text.split("启动 [")[1:]
    for ch in chunks:
        tag = ch.split("]")[0]
        rows = {}
        for m in re.finditer(
                r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)ms\s+([\d.]+)ms\s*$",
                ch, re.M):
            rows[int(m.group(1))] = dict(wall=float(m.group(2)),
                                         tput=float(m.group(3)),
                                         ttft=float(m.group(4)),
                                         tpot=float(m.group(5)))
        if rows:
            out[tag] = rows
    return out


def parse_txt():
    """从 dip_causal_2026-09-08.txt 的转写表里取 A/B/C 三列与 D 两行。"""
    txt = read(TXT)
    head, tail = txt.split("四、补跑成功", 1)
    abc = {}
    for label, key in [("TPOT p50 (ms)", "tpot"),
                       ("吞吐 tok/s", "tput"),
                       ("TTFT p50 (ms)", "ttft")]:
        m = re.search(r"^\s*%s\s+((?:\d+\s+)+\d+)\s*$" % re.escape(label),
                      head, re.M)
        if not m:
            FAILS.append("txt 里找不到表头 " + label)
            continue
        cols = [int(x) for x in m.group(1).split()]
        block = head[m.end():]
        for tag in ["A_default", "B_nograph", "C_maxbs32"]:
            r = re.search(r"^\s*%s\s+((?:[\d.]+\s+)+[\d.]+)\s*$" % tag,
                          block, re.M)
            if not r:
                FAILS.append("txt 里找不到 %s 的 %s 行" % (tag, label))
                continue
            vals = [float(x) for x in r.group(1).split()]
            for c, v in zip(cols, vals):
                abc.setdefault(tag, {}).setdefault(c, {})[key] = v

    d = {}
    m = re.search(r"^\s*batch\s+((?:\d+\s+)+\d+)\s*$", tail, re.M)
    if m:
        cols = [int(x) for x in m.group(1).split()]
        for label, key in [("TPOT p50 (ms)", "tpot"), ("吞吐 tok/s", "tput")]:
            r = re.search(r"^\s*%s\s+((?:[\d.]+\s+)+[\d.]+)\s*$"
                          % re.escape(label), tail, re.M)
            if not r:
                FAILS.append("txt 的 D 组里找不到 " + label)
                continue
            for c, v in zip(cols, [float(x) for x in r.group(1).split()]):
                d.setdefault(c, {})[key] = v
    else:
        FAILS.append("txt 里找不到 D 组的 batch 表头")
    abc["D_default_backend"] = d
    return abc


def check_two_sources(nbd, txtd):
    print("[2] 两个来源逐格一致（notebook 真实输出 vs txt 转写）")
    # D 组在 txt 里只转写了 TPOT 与吞吐两行，没有墙钟与 TTFT。
    keys = {"A_default": ("tput", "ttft", "tpot"),
            "B_nograph": ("tput", "ttft", "tpot"),
            "C_maxbs32": ("tput", "ttft", "tpot"),
            "D_default_backend": ("tput", "tpot")}
    n = 0
    bad = []
    for tag, ks in keys.items():
        for bs in sorted(nbd.get(tag, {})):
            for k in ks:
                a = nbd[tag][bs][k]
                b = txtd.get(tag, {}).get(bs, {}).get(k)
                n += 1
                if b is None or not close(a, b, 1e-9):
                    bad.append("%s@%d %s notebook=%s txt=%s" % (tag, bs, k, a, b))
    ok(not bad, "%d 个格子逐个相同%s" % (n, ("；不一致：" + "; ".join(bad)) if bad else ""))
    ok(sorted(nbd) == sorted(keys),
       "notebook 里正好四组：%s" % ", ".join(sorted(nbd)))


# ------------------------------------------------------------- 3/4 自洽
def check_selfconsistency(nbd):
    print("[3] 每请求输出 token 数（墙钟 × 吞吐 ÷ 请求数）")
    bad = []
    for tag in sorted(nbd):
        for bs in sorted(nbd[tag]):
            r = nbd[tag][bs]
            v = r["wall"] * r["tput"] / n_req(bs)
            if not close(v, MAX_TOKENS, TOKREQ_TOL):
                bad.append("%s@%d = %.2f" % (tag, bs, v))
    ok(not bad, "23 个格子全部落在 %d ± %.0f token/请求%s"
       % (MAX_TOKENS, TOKREQ_TOL, ("；越界：" + ", ".join(bad)) if bad else ""))

    print("[4] 墙钟自洽：墙钟 ≈ (请求数/并发) × (TTFT + 127×TPOT)")
    for tag in sorted(nbd):
        for bs in sorted(nbd[tag]):
            r = nbd[tag][bs]
            exp = (n_req(bs) / bs) * (r["ttft"] / 1e3 + 127 * r["tpot"] / 1e3)
            dev = abs(r["wall"] - exp) / r["wall"]
            wl = WHITELIST.get((tag, bs))
            flag = "离群" if dev >= WALL_TOL else "    "
            print("         %-18s bs=%-3d 预期 %7.2fs 实测 %7.2fs 偏差 %+6.1f%% %s"
                  % (tag, bs, exp, r["wall"], (r["wall"] - exp) / exp * 100, flag))
            if wl is None:
                ok(dev < WALL_TOL,
                   "%s@%d 与自洽值的差占实测墙钟 %.1f%% < %.0f%%"
                   % (tag, bs, dev * 100, WALL_TOL * 100))
            else:
                # 白名单是双向的：不再离群也要报，免得白名单永远挂着
                ok(dev >= WALL_TOL,
                   "%s@%d 仍是已知离群格（差占实测墙钟 %.1f%%）—— %s"
                   % (tag, bs, dev * 100, wl))


# ------------------------------------------------------ 5 README 的比值
def md_row(readme, label):
    # 允许 blockquote 前缀：README 顶部那张 GEMM 对照表在 "> " 引用块里
    m = re.search(r"^(?:>\s*)?\|\s*%s\s*\|(.*)\|\s*$" % re.escape(label),
                  readme, re.M)
    if not m:
        FAILS.append("README 里找不到表格行「%s」" % label)
        return None
    return floats(m.group(1))


def check_readme_ratios(readme, nbd):
    print("[5] README 第 D 节的表与比值")
    a, b, c = nbd["A_default"], nbd["B_nograph"], nbd["C_maxbs32"]
    bs6 = [4, 8, 12, 16, 24, 32]
    bs4 = [12, 16, 24, 32]

    for label, src in [("A 默认", a), ("B 关解码图", b), ("C 上限抬到 32", c)]:
        row = md_row(readme, label)
        if row is None:
            continue
        # 行尾的「捕获的 bs」列也是数字，只取前 6 个
        vals = row[:6]
        ok(all(close(v, src[k]["tpot"], 1e-9) for v, k in zip(vals, bs6)),
           "TPOT 行「%s」= %s" % (label, vals))

    # 比值的容差：转写表里的 TPOT 只有两位小数，比值会带舍入残差。
    # 例如 B4/A4 用 34.17/5.70 算是 5.99，而原始未舍入值给的是 6.00 ——
    # 差 0.01 是舍入，差 0.08（把 4.42 写成 4.50）不是，RATIO_TOL 卡在中间。
    for label, keys, num, den, key in [
            ("B ÷ A", bs6, b, a, "tpot"),
            ("吞吐比 C ÷ A", bs4, c, a, "tput"),
            ("TPOT 比 A ÷ C", bs4, a, c, "tpot")]:
        row = md_row(readme, label)
        if not row:
            continue
        exp = [round(num[k][key] / den[k][key], 2) for k in keys]
        bad = [i for i in range(min(len(row), len(exp)))
               if abs(row[i] - exp[i]) > RATIO_TOL]
        ok(len(row) == len(exp) and not bad,
           "%s 行 %s ≈ 回算 %s（容差 %.2f）" % (label, row, exp, RATIO_TOL))

    # 导航块与正文里出现的绝对值
    ok("608.0 → 2690.2 tok/s" in readme,
       "README 写出了 608.0 → 2690.2 的绝对值（面试官会 Ctrl+F 这个数）")
    ok(close(a[32]["tpot"], 35.74, 1e-9) and close(c[32]["tpot"], 10.26, 1e-9),
       "导航块里的 35.74 → 10.26 ms 与 notebook 输出一致")


# ------------------------------------------------------------ 6 更正文件
def check_corrections(readme, nbd):
    print("[6] corrections_2026-09-13.txt 的更正数")
    a, b, c = nbd["A_default"], nbd["B_nograph"], nbd["C_maxbs32"]
    r = a[32]
    exp = 4 * (r["ttft"] / 1e3 + 127 * r["tpot"] / 1e3)
    fixed = n_req(32) * MAX_TOKENS / exp
    ratio = c[32]["tput"] / fixed
    corr = read(CORR)

    ok(("%.2f" % exp) == "19.12", "A32 自洽墙钟回算 = %.4f s → 19.12" % exp)
    ok("19.12 / 26.92   +40.8%" in corr, "更正文件里的 19.12 / 26.92 / +40.8% 与回算一致")
    ok(("%.1f" % fixed) == "856.8", "A32 折算吞吐回算 = %.2f tok/s → 856.8" % fixed)
    ok("856.8 tok/s" in corr, "更正文件里写的是 856.8 tok/s")
    ok(("%.2f" % ratio) == "3.14", "C ÷ A（扣停顿）回算 = %.4f → 3.14×" % ratio)
    ok("= 3.14x" in corr, "更正文件里写的是 3.14x")
    ok(("%.2f" % (r["wall"] - exp)) == "7.80",
       "停顿时长回算 = %.3f s → 7.80 s" % (r["wall"] - exp))

    # 三条不用反推的矛盾，也要真的成立
    ok(a[32]["tput"] < a[24]["tput"],
       "矛盾一：A32 吞吐 %.1f < A24 %.1f" % (a[32]["tput"], a[24]["tput"]))
    ok(b[32]["tput"] / a[32]["tput"] > 1.4,
       "矛盾二：B32 ÷ A32 吞吐 = %.2f×，而同表 TPOT 比是 %.2f×"
       % (b[32]["tput"] / a[32]["tput"], b[32]["tpot"] / a[32]["tpot"]))
    ok(nbd["D_default_backend"][32]["tput"] > a[32]["tput"],
       "矛盾三：D32 吞吐 %.1f > A32 %.1f"
       % (nbd["D_default_backend"][32]["tput"], a[32]["tput"]))
    ok(abs(fixed - b[32]["tput"]) / b[32]["tput"] < 0.02,
       "折算值 %.1f 与同轮 B32 实测 %.1f 相差 %.1f%%"
       % (fixed, b[32]["tput"], abs(fixed - b[32]["tput"]) / b[32]["tput"] * 100))

    # README 里必须给出 TPOT 口径，且写明吞吐口径的分母有停顿
    ok("以 TPOT 口径 3.48× 为准" in readme, "README 导航块写明以 TPOT 口径 3.48× 为准")
    ok("batch 32 以 TPOT 口径 3.48× 为准" in readme, "README 第 D 节同样写明")

    # 2026-09-18 流式计数更正：旧客户端按非空内容片段计数，倍数只是历史近似统计量。
    # 更正说明必须留在第一屏，导航块与第 D 节的 3.48× 旁边都要写明不是纯优化收益。
    ok("2026-09-18 流式计数更正" in "\n".join(readme.split("\n")[:5]),
       "README 前 5 行保留 2026-09-18 流式计数更正")
    nav = readme.split("## 本仓两条主线", 1)[-1].split("\n---", 1)[0]
    ok("只是历史近似统计量，不是纯优化收益" in nav, "导航块写明 3.48× / 4.42× 不是纯优化收益")
    sec_d = readme.split("batch 32 以 TPOT 口径 3.48× 为准", 1)[-1].split("**仍未做**", 1)[0]
    ok("不能当作纯优化收益引用" in sec_d, "第 D 节写明表里比值不能当作纯优化收益引用")


# -------------------------------------------------------------- 7 GEMM 表
def check_gemm(readme):
    print("[7] README 第一屏的 GEMM 对照表 vs t4_gemm_control.txt")
    g = read(GEMM)
    m1 = re.search(r"^\s*1\s+([\d.]+)\s+1\.00×\s+1\s+([\d.]+)\s+1\.00×"
                   r"\s+1\s+([\d.]+)\s+1\.00×", g, re.M)
    m2 = re.search(r"^\s*2\s+([\d.]+)\s+([\d.]+)×\s+2\s+([\d.]+)\s+([\d.]+)×"
                   r"\s+2\s+([\d.]+)\s+([\d.]+)×", g, re.M)
    ok(m1 is not None and m2 is not None, "结果文件里解析出 fp16 三组形状的 M=1 / M=2 行")
    if not (m1 and m2):
        return
    t4_m1, t4_m2 = float(m1.group(1)), float(m2.group(1))
    shapes = [float(m2.group(2)), float(m2.group(4)), float(m2.group(6))]

    row = md_row(readme, "Tesla T4（**有** Tensor Core）")
    if row:
        ok(close(row[0], t4_m1, 1e-9) and close(row[1], t4_m2, 1e-9),
           "T4 行 %s GiB/s == 结果文件 %.1f / %.1f" % (row[:2], t4_m1, t4_m2))
        ok(close(row[2], round(t4_m2 / t4_m1, 2), 1e-9),
           "T4 行的 %.2f× == %.1f / %.1f" % (row[2], t4_m2, t4_m1))

    m = re.search(r"三组形状一致（([\d.]+)× / ([\d.]+)× / ([\d.]+)×）", readme)
    ok(m is not None, "README 里找得到「三组形状一致（…）」")
    if m:
        got = [float(x) for x in m.groups()]
        ok(got == shapes, "三组形状 %s == 结果文件 %s" % (got, shapes))

    row = md_row(readme, "GTX 1650（**无** Tensor Core）")
    if row:
        ok(("GTX 1650（无 TC）fp16 MLP gate/up：M=1 %.1f → M=2 %.1f GiB/s"
            % (row[0], row[1])) in g,
           "GTX 1650 行 %.1f → %.1f 与结果文件结论段的抄写一致" % (row[0], row[1]))


# ---------------------------------------------------------- 8 本地那半边
def check_local(readme):
    print("[8] README 第二节的表 与 results_local/ 的那份 JSON")
    body = readme.split("## 二、异常", 1)[1].split("## 三、", 1)[0]
    rows = {}
    for line in body.split("\n"):
        if not line.startswith("|") or "并发" in line or "---" in line:
            continue
        v = floats(line.replace("**", ""))
        if len(v) == 5:
            rows[int(v[0])] = dict(wall=v[1], tput=v[2], ttft=v[3], tpot=v[4])
    ok(sorted(rows) == [1, 2, 4, 8, 16, 32],
       "第二节表的并发列 %s 与旧 bench_serving.py 的扫描列表一致" % sorted(rows))
    for conc in sorted(rows):
        r = rows[conc]
        tok = r["wall"] * r["tput"] / n_req(conc)
        exp = (n_req(conc) / conc) * (r["ttft"] / 1e3 + 127 * r["tpot"] / 1e3)
        dev = abs(r["wall"] - exp) / r["wall"]
        ok(close(tok, MAX_TOKENS, TOKREQ_TOL) and dev < WALL_TOL,
           "并发 %-2d  %.1f token/请求，墙钟 %.2f vs 自洽 %.2f（%+.1f%%）"
           % (conc, tok, r["wall"], exp, (r["wall"] - exp) / exp * 100))

    data = json.load(io.open(os.path.join(HERE, LOCAL_JSON), encoding="utf-8"))
    j = {r["conc"]: r for r in data}
    ok(sorted(j) == [1, 2, 4, 8, 16, 32], "JSON 的并发列 %s" % sorted(j))
    for conc in sorted(j):
        r = j[conc]
        tok = r["wall"] * r["tput"] / n_req(conc)
        exp = (n_req(conc) / conc) * (r["ttft_p50"] + 127 * r["tpot_p50"])
        ok(close(tok, MAX_TOKENS, TOKREQ_TOL)
           and abs(r["wall"] - exp) / r["wall"] < WALL_TOL,
           "JSON 并发 %-2d %.1f token/请求，墙钟 %.2f vs 自洽 %.2f（%+.1f%%）"
           % (conc, tok, r["wall"], exp, (r["wall"] - exp) / exp * 100))

    # 原样收录：sha256 必须与 results_local/README.md 里记的那一行一致
    import hashlib
    blob = io.open(os.path.join(HERE, LOCAL_JSON), "rb").read()
    got = hashlib.sha256(blob).hexdigest()
    prov = read(os.path.join("results_local", "README.md"))
    m = re.search(r"sha256 = ([0-9a-f]{64})", prov)
    ok(m is not None and m.group(1) == got,
       "JSON 的 sha256 %s 与 results_local/README.md 记的一致" % got[:16])

    # 归属：这份 JSON 是 --enforce-eager 那一轮，不是第二节那张表
    eager_tpot = j[1]["tpot_p50"] * 1e3
    m = re.search(r"7\.20 vs ([\d.]+) ms", readme)
    ok(m is not None, "README 第三节里找得到「7.20 vs xx ms」那条")
    if m:
        ok(("%.2f" % eager_tpot) == m.group(1),
           "JSON 并发 1 的 TPOT %.3f ms == README 里的 %s ms（这就是归属依据）"
           % (eager_tpot, m.group(1)))
    ok(not close(j[1]["tput"], rows[1]["tput"], 1.0),
       "JSON 并发 1 吞吐 %.1f ≠ 第二节表的 %.1f —— 两轮不同，README 里已写明"
       % (j[1]["tput"], rows[1]["tput"]))


# ------------------------------------------------------------- 9 文件引用
# README 里合法出现、但本来就不在本仓的文件名（上游源码 / 运行时生成物）
EXTERNAL = {
    "python/sglang/srt/arg_groups/memory_hook.py": "SGLang v0.5.19 上游源码",
    "marlin_utils.py": "vLLM v0.28.0 上游源码",
    "sweep_results.json": "旧 bench_serving.py 每次运行覆盖写出的产物，仓里只留 results_local/ 那一份",
    "sweep_results_v2.json": "2026-10-12 修订后的 bench_serving.py 运行时写出的产物，不入库",
    "probe.py": "cloud_dip_confirm_executed.ipynb 里由 cell 现写出来的压测脚本（第 1 组从 notebook 里核它）",
}


def check_refs(readme):
    print("[9] README 提到的仓内文件都存在")
    names = sorted(set(re.findall(
        r"`([A-Za-z0-9_./-]+\.(?:py|sh|ipynb|json|txt|md|yml))`", readme)))
    missing, n_ext = [], 0
    for n in names:
        if n in EXTERNAL:
            NOTES.append("%s —— %s" % (n, EXTERNAL[n]))
            n_ext += 1
            continue
        if not os.path.exists(os.path.join(HERE, n)):
            missing.append(n)
    ok(not missing, "%d 个反引号文件名全部存在（另有 %d 个已声明在仓外）%s"
       % (len(names) - n_ext, n_ext,
          ("；缺：" + ", ".join(missing)) if missing else ""))

    import urllib.parse
    bad = []
    for link in set(re.findall(r"\]\((?!https?:)([^)#]+)\)", readme)):
        if not os.path.exists(os.path.join(HERE, urllib.parse.unquote(link))):
            bad.append(link)
    ok(not bad, "所有仓内相对链接可达%s" % (("；断链：" + ", ".join(bad)) if bad else ""))

    have = sorted(f for f in os.listdir(HERE) if re.match(r"cloud_.*\.ipynb$", f))
    listed = sorted(set(re.findall(r"^\| `(cloud_[A-Za-z0-9_]+\.ipynb)` \|",
                                   readme, re.M)))
    ok(have == listed, "notebook 表列了全部 %d 份%s"
       % (len(have), ("；漏：" + ", ".join(set(have) - set(listed))) if have != listed else ""))
    m = re.search(r"仓内共 (\d+) 份 notebook", readme)
    ok(m is not None and int(m.group(1)) == len(have),
       "README 写的份数与实际 %d 份一致" % len(have))


def main():
    print("check_claims.py —— 从产物回算 README 里的数字（纯 CPU）")
    print("=" * 72)
    readme = read("README.md")
    check_protocol()
    nbd = parse_notebook()
    txtd = parse_txt()
    check_two_sources(nbd, txtd)
    check_selfconsistency(nbd)
    check_readme_ratios(readme, nbd)
    check_corrections(readme, nbd)
    check_gemm(readme)
    check_local(readme)
    check_refs(readme)
    print("=" * 72)
    for n in NOTES:
        print("    note " + n)
    if FAILS:
        print("%d 处对不上：" % len(FAILS))
        for f in FAILS:
            print("  - " + f)
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
