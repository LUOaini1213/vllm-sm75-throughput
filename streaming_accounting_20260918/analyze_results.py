"""Verify evidence and render a compact report without inventing missing runs.

`python analyze_results.py` rewrites evidence/summary.json and RESULTS.md.
`python analyze_results.py --check` rebuilds both in memory from the raw evidence
and exits 1 if either committed file differs (CI runs this; stdlib only).
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build():
    runs =[(p, read(p)) for p in sorted(EVIDENCE.glob("local_*/manifest.json"))]
    complete = [(p, m) for p, m in runs if m.get("completed")]
    if not complete:
        raise RuntimeError("No completed real run; no performance report created")
    path, manifest = complete[-1]
    root = path.parent
    equivalence = read(root / "generate_equivalence_repetition1.json")
    assert equivalence["passed"] == equivalence["total"] == 4
    summaries = []
    checks = []
    artifacts = []
    for label in ["fp16", "fp32"]:
        sweep = read(root / f"{label}_sweep.json")
        reliability = read(root / f"{label}_reliability.json")
        quality = read(root / f"{label}_quality.json")
        assert len(sweep["batches"]) == 6
        records = [r for batch in sweep["batches"] for r in batch["records"]]
        assert len(records) == 48
        assert all(r["status"] == "success" for r in records)
        assert all(r["stream_token_count"] == r["usage_completion_tokens"] == 32 for r in records)
        assert all(r["done"] and r["single_token_events"] for r in records)
        assert reliability["passed"] == reliability["total"] == 5
        assert quality["successes"] == quality["total"] == 16
        checks.append({"label": label, "successful_benchmark_requests": len(records),
                       "validated_generated_tokens": sum(r["stream_token_count"] for r in records),
                       "reliability_passed": reliability["passed"], "quality_exact_matches": quality["exact_matches"],
                       "quality_total": quality["total"]})
        for concurrency in [1, 2, 4]:
            batches = [b for b in sweep["batches"] if b["concurrency"] == concurrency]
            rs = [r for b in batches for r in b["records"]]
            summaries.append({"label": label, "concurrency": concurrency, "requests": len(rs),
                "tokens_per_s": sum(r["stream_token_count"] for r in rs) / sum(b["summary"]["wall_s"] for b in batches),
                "two_run_tokens_per_s": [b["summary"]["successful_tokens_per_s"] for b in batches],
                "ttft_median_ms": statistics.median(r["ttft_ms"] for r in rs),
                "tpot_median_ms": statistics.median(r["tpot_ms"] for r in rs),
                "ttft_1s_slo_success_count": sum(r["ttft_ms"] <= 1000 for r in rs)})
    for p in sorted(root.glob("*")):
        if p.is_file():
            artifacts.append({"path": p.relative_to(HERE).as_posix(), "bytes": p.stat().st_size,
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    report = {"run_directory": root.relative_to(HERE).as_posix(), "backend": manifest["backend"],
              "hardware": manifest["gpu"], "checks": checks, "summary": summaries,
              "reference_generate_equivalence": {"passed": equivalence["passed"], "total": equivalence["total"]},
              "previous_incomplete_runs": [{"directory": p.parent.relative_to(HERE).as_posix(), "backend": m.get("backend", "vllm"),
                  "errors": [e.get("error") for e in m["experiments"] if e.get("error")]} for p, m in runs if not m.get("completed")],
              "artifacts": artifacts,
              "scope": "GPU Transformers serialized reference service; not vLLM/continuous-batching performance; no new INT4 quality data"}
    summary_text = json.dumps(report, ensure_ascii=False, indent=2)
    lines = ["# 实跑结果与未覆盖范围", "", f"实跑证据：`{report['run_directory']}`。", "",
             "GTX 1650 4GB，WSL Ubuntu，PyTorch 2.13.0+cu130 / Transformers 5.16.1，缓存 Qwen2.5-0.5B-Instruct，512上下文。",
             "**实际成功的是 Transformers 串行参考服务；vLLM 本次未启动成功。** 两种精度从同一缓存权重转换，不是新量化实验。", "",
             "| 精度 | 客户端并发 | 请求数（两轮） | 成功输出 token/s | TTFT中位 ms | TPOT中位 ms | TTFT≤1s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['label']} | {row['concurrency']} | {row['requests']} | {row['tokens_per_s']:.2f} | {row['ttft_median_ms']:.1f} | {row['tpot_median_ms']:.2f} | {row['ttft_1s_slo_success_count']}/{row['requests']} |")
    lines += ["", "吞吐按两轮成功 token 总数 / 两轮墙钟总和合并，TTFT/TPOT 按两轮请求合并取中位数。",
              "TTFT含服务内部排队，不含客户端并发信号量等待；原始JSON另存客户端等待。闭环8请求/档、两轮的小样本没有生产容量或统计显著性含义。",
              "由于后端串行执行，增加客户端并发主要增加服务器排队，而不产生 continuous batching 收益。不能把这张表用于声称 vLLM、SGLang 或量化优劣。", "",
              "## 验收", ""]
    for check in checks:
        lines.append(f"- {check['label']}：48/48 压测请求成功，1536 个真实输出 token 的 ID 计数与 usage 一致；5/5 异常及后续恢复探测通过；固定16题精确匹配 {check['quality_exact_matches']}/16。")
    lines += ["", "五类探测：模型不存在、上下文溢出、非法长度、客户端1ms超时、生成3个token后取消；每项后发真实恢复请求。测试的是服务仍能响应，不是生产级故障自愈或资源回收证明。",
              "16题是预写的小型算术、数据读取和阈值判断冒烟任务，仅可检测端点行为，不是公开能力基准，也不能证明量化无损。两种精度均仅3/16严格匹配：部分输出虽算对但未遵守仅返回答案格式，也确有查找与阈值判断错误。结果不因分数低而改题或放宽主评分。",
              "另用独立 Transformers.generate API 对算术、查找、规则与计数4题交叉核验，4/4输出 token 序列与参考服务一致。这个检查支持采集实现正确，不能把错误答案解释成模型可靠。", "",
              "## 失败也保留", "",
              "- vLLM 默认V2路径日志报 FlashAttention 2 不支持 sm_75，随后进程初始化失败；该日志本身不证明完整根因。",
              "- 显式 V1 + TRITON_ATTN 后引擎 SIGBUS；未修改驱动/环境、未假称修复。",
              "- 参考服务初次因 Transformers 5 的聊天模板返回值结构导致张量转换失败；显式 return_dict=False 后本次完整实验通过。",
              "- 两次vLLM日志与参考服务初次失败均在各自时间戳目录，不覆盖旧失败。", "",
              "## 一句话摘要", "",
              "补建基于 token IDs 与 usage 交叉核验的流式测量工具，分别记录首 token、客户端排队、逐 token 间隔和失败请求；在本机 GPU 的 Qwen 0.5B 参考服务完成两种精度、三档并发共96次请求及10组异常恢复检查，保留兼容性失败和原始证据。", "",
              "这段是本次 AI 辅助实现的新增练习。原有 T4 CUDA Graph 结果仍按历史片段计数口径说明，不写4.42倍干净收益。", "",
              "## 尚未补齐", "",
              "多GPU/分布式训练、集群调度、量化算法或QAT、公开大规模准确率基准、长上下文、高可用和生产SLO仍无新实证；本目录只覆盖单机部署排障、基准测量与验证。"]
    outputs = {EVIDENCE / "summary.json": summary_text.encode("utf-8"),
               HERE / "RESULTS.md": ("\n".join(lines) + "\n").encode("utf-8")}
    return report, checks, summaries, outputs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    report, checks, summaries, outputs = build()
    if "--check" in argv:
        stale = [p.relative_to(HERE).as_posix() for p, data in outputs.items()
                 if not p.is_file() or p.read_bytes() != data]
        if stale:
            print("FAIL: stale, regenerate with python analyze_results.py: " + ", ".join(stale))
            return 1
        print("PASS: evidence/summary.json and RESULTS.md match the raw evidence")
        return 0
    for path, data in outputs.items():
        path.write_bytes(data)  # LF on every platform, so --check is byte-stable
    print(json.dumps({"backend": report["backend"], "checks": checks, "summary": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
