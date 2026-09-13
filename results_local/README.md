# 本地 GTX 1650 那半边的原始产物

本目录只有一个文件，收得晚（2026-09-13），先把它是什么、不是什么说清楚。

## `gtx1650_sweep_enforce_eager_2026-09-02.json`

- **它是什么**：`bench_serving.py` 的 `sweep()` 直接 `json.dump` 出来的原始结果，
  不是手抄、不是整理稿。原位置 `~/vllmbench/sweep_results.json`（WSL2 Ubuntu-24.04），
  文件时间 **2026-09-02 17:20:17**，1382 字节，
  `sha256 = 54dff497ad6004fc8547dd07a3349905614117febfffe3e3bda6bd959ee17b44`。
  本仓原样收录，一个字节没改；`check_claims.py` 每次 CI 都重算这个 sha256 并与本行比对，
  `.gitattributes` 里给这个路径加了 `-text`，免得 checkout 换行尾把校验和弄坏。
- **它是哪一轮**：`--enforce-eager` 对照组（`serve_eager.sh`）。依据有两条，都在文件里：
  并发 1 的 `tpot_p50` = **19.13 ms**，正是 README 第三节表里
  「CUDA Graph 只在 batch=1 有用，7.20 vs **19.13** ms」的那个 19.13；
  并发 2/4/8/16/32 的 `tpot_p50` 是 184.5 / 182.3 / 183.0 / 185.7 / 186.3 ms，
  与同一行「batch≥2 两者都是 ~183 ms」一致。
- **它不是什么**：**不是 README 第二节那张表的来源。**
  第二节并发 1 是 119.5 tok/s / TPOT 7.20 ms，本文件是 47.47 tok/s / 19.13 ms。
  第二节那一轮（CUDA Graph 正常开启）的 JSON 没有留下来 ——
  `bench_serving.py:82` 把结果固定写到同名的 `sweep_results.json`，文件名写死、不带时间戳，
  后跑的这一轮把前一轮覆盖了。第二节那张表目前只有 README 里的抄写版。

## 这半边还缺什么

第一～五节的其余数字（`gemm_cliff.py` 的 GEMM 表、`isolate.py` 的稳态突发表、
`--dtype float32` 那一轮、以及启动日志里的 `GPU KV cache size: 148,576 tokens` /
`Maximum concurrency ... 72.55x`）在仓里都只有 README 中的表格，没有对应产物。
`/tmp` 下的服务日志随 WSL 重启已清空，要补齐得在同一张卡上重跑一轮服务。

对照之下，T4 那半边有 `results_t4/` 的 11 份结果文件，外加
`cloud_dip_confirm_executed.ipynb` 这份带真实 cell 输出的 notebook。

结论层面这个缺口是可补的：`gemm_cliff.py` 在任何一张无 Tensor Core 的 `sm_75` 卡上都能复跑，
归因本身也已经在 T4 上做过带产物的受控对照（`results_t4/t4_gemm_control.txt`）。
不可回查的是 **2026-09-02 那一次运行本身**。
