#!/usr/bin/env python3
"""并发基准（批量参数矩阵版）：测试多种 (duration, steps) 配置的并发生成速度。

用法示例：
  # 单配置：4 个并发 t2v，duration=5s, steps=10
  python3 bench_concurrent.py --n 4 --duration 5 --steps 10

  # 多配置批量：每个配置跑 N 个并发，所有任务一次全部提交后等齐
  python3 bench_concurrent.py --n 2 --configs '5/10,8/10,10/10,5/20' --api http://127.0.0.1:8000

  # 自动下载生成的 mp4 到 ./out_bench/
  python3 bench_concurrent.py --n 2 --duration 5 --steps 10 --download

输出：
  1) 提交阶段：每任务提交耗时、worker 分配、分配均衡度
  2) 等待阶段：每个任务状态变化时间线
  3) 汇总表：每个 (duration, steps) 组的单任务推理耗时 + 该组端到端 wall time
  4) 全局汇总：worker 分配、并发实测吞吐（tasks/min）
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request
import urllib.error


def post_json(api, path, payload, timeout=30):
    req = urllib.request.Request(
        f"{api}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_json(api, path, timeout=15):
    with urllib.request.urlopen(f"{api}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def download_file(api, url, out_path, timeout=120):
    """下载视频到本地。"""
    full = url if url.startswith("http") else f"{api}{url}"
    with urllib.request.urlopen(full, timeout=timeout) as r:
        with open(out_path, "wb") as f:
            f.write(r.read())


PROMPT = (
    "一只蓝色的小鸟站在绿色的树枝上，背景是蓝天白云，阳光透过树叶洒下，画面安静祥和。"
)


def parse_configs(s):
    """解析 '5/10,8/10,10/10' -> [(5.0, 10), (8.0, 10), (10.0, 10)]"""
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        d, st = tok.split("/")
        out.append((float(d), int(st)))
    return out


def submit(api, idx, cfg_idx, duration, steps, width, height, seed):
    """提交单个任务。"""
    t0 = time.time()
    body = {
        "prompt": PROMPT,
        "duration": duration,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed + idx + cfg_idx * 1000,  # 不同 cfg 用不同 seed 段
    }
    try:
        r = post_json(api, "/api/v1/generate", body)
    except urllib.error.HTTPError as e:
        body_err = e.read().decode("utf-8", "replace")
        return {"cfg_idx": cfg_idx, "idx": idx, "error": f"HTTP {e.code}: {body_err}"}
    return {
        "cfg_idx": cfg_idx,
        "idx": idx,
        "submit_s": round(time.time() - t0, 3),
        "task_id": r.get("task_id"),
        "worker": r.get("worker"),
        "frame_count": r.get("frame_count"),
        "duration": duration,
        "steps": steps,
    }


def poll(api, task_id, deadline_s, quiet=False):
    """轮询直到 success / failed / 超时。"""
    t0 = time.time()
    last = None
    while time.time() - t0 < deadline_s:
        d = get_json(api, f"/api/v1/task/{task_id}")
        st = d.get("status")
        if st != last and not quiet:
            print(f"  [{task_id[:8]}] {st} @ {round(time.time()-t0,1)}s")
            last = st
        if st in ("success", "failed"):
            return {
                "task_id": task_id,
                "elapsed_s": round(time.time() - t0, 1),
                "status": st,
                "worker": d.get("worker"),
                "videos": d.get("videos", []),
                "error": d.get("error"),
            }
        time.sleep(3)
    return {"task_id": task_id, "elapsed_s": round(time.time() - t0, 1),
            "status": "timeout", "worker": None, "videos": []}


def fmt_table(rows, headers):
    """简易表格打印。"""
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000",
                    help="API base URL（默认 127.0.0.1:8000，远程可写 http://<ip>:8000）")
    ap.add_argument("--n", type=int, default=4,
                    help="每个 (duration, steps) 配置跑几个并发")
    ap.add_argument("--duration", type=float, default=None,
                    help="单配置模式：视频时长（秒）")
    ap.add_argument("--steps", type=int, default=10,
                    help="单配置模式：采样步数")
    ap.add_argument("--configs", type=str, default=None,
                    help="批量模式：'5/10,8/10,10/10,5/20' 形式（duration/steps 列表）")
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=1800, help="单任务超时秒数")
    ap.add_argument("--global-timeout", type=int, default=7200, help="全批次总超时秒数")
    ap.add_argument("--download", action="store_true", help="自动下载生成的 mp4")
    ap.add_argument("--out-dir", default="./out_bench", help="下载目录")
    ap.add_argument("--quiet-poll", action="store_true", help="轮询时不打印状态变化")
    args = ap.parse_args()

    # 配置矩阵
    if args.configs:
        configs = parse_configs(args.configs)
    elif args.duration is not None:
        configs = [(args.duration, args.steps)]
    else:
        configs = [(5.0, 10)]
    print(f"配置矩阵: {configs}，每配置并发数={args.n}，API={args.api}")
    total_jobs = len(configs) * args.n
    print(f"总任务数: {total_jobs}\n")

    t_start = time.time()

    # ===== 阶段1：并发提交全部任务 =====
    print(f"=== 阶段1: 提交 {total_jobs} 个任务 ===")
    jobs = [(cfg_idx, idx, dur, st)
            for cfg_idx, (dur, st) in enumerate(configs)
            for idx in range(args.n)]
    with cf.ThreadPoolExecutor(max_workers=total_jobs) as ex:
        futs = [ex.submit(submit, args.api, idx, cfg_idx, dur, st,
                          args.width, args.height, args.seed)
                for cfg_idx, idx, dur, st in jobs]
        subs = [f.result() for f in cf.as_completed(futs)]
    submit_phase_s = time.time() - t_start
    ok_subs = [s for s in subs if s.get("task_id")]
    print(f"提交阶段耗时: {submit_phase_s:.2f}s（成功 {len(ok_subs)}/{len(subs)}）\n")

    # 提交结果
    print("提交详情:")
    fmt_table(
        rows=[[s["cfg_idx"], s["idx"], s["duration"], s["steps"],
               s.get("submit_s"), s.get("worker"),
               s.get("frame_count"), (s.get("task_id") or "")[:8],
               s.get("error") or ""] for s in sorted(subs, key=lambda x: (x["cfg_idx"], x["idx"]))],
        headers=["cfg", "n", "dur", "steps", "sub_s", "wkr", "frames", "tid", "err"],
    )
    worker_dist = {}
    for s in ok_subs:
        w = s.get("worker")
        worker_dist[w] = worker_dist.get(w, 0) + 1
    print(f"\nworker 分配: {worker_dist}\n")

    if not ok_subs:
        print("无任务可等待，退出。")
        return

    # ===== 阶段2：并发轮询所有任务 =====
    print(f"=== 阶段2: 轮询 {len(ok_subs)} 个任务 ===")
    deadline = args.timeout
    t_poll = time.time()
    with cf.ThreadPoolExecutor(max_workers=len(ok_subs)) as ex:
        futs = {ex.submit(poll, args.api, s["task_id"], deadline, args.quiet_poll): s
                for s in ok_subs}
        results = []
        for f in cf.as_completed(futs):
            results.append(f.result())
    poll_phase_s = time.time() - t_poll
    total_s = time.time() - t_start

    # ===== 阶段3：下载（可选） =====
    if args.download:
        print(f"\n=== 阶段3: 下载视频 ===")
        os.makedirs(args.out_dir, exist_ok=True)
        for r in results:
            if r.get("status") != "success":
                continue
            for v_url in r.get("videos", []):
                fname = v_url.split("?")[0].split("/")[-1]
                out = os.path.join(args.out_dir, fname)
                try:
                    download_file(args.api, v_url, out)
                    print(f"  {fname} -> {out} ({os.path.getsize(out)//1024} KB)")
                except Exception as e:
                    print(f"  FAIL {fname}: {e}")

    # ===== 汇总报告 =====
    print(f"\n=== 汇总 ===")
    print(f"提交阶段: {submit_phase_s:.2f}s")
    print(f"等待阶段: {poll_phase_s:.2f}s")
    print(f"端到端 total: {total_s:.2f}s = {total_s/60:.1f} min")
    success_n = sum(1 for r in results if r["status"] == "success")
    fail_n = sum(1 for r in results if r["status"] != "success")
    print(f"成功 {success_n}/{len(results)}，失败/超时 {fail_n}")
    throughput = success_n / (total_s / 60) if total_s > 0 else 0
    print(f"实测吞吐: {throughput:.2f} tasks/min")

    # 按配置分组
    by_cfg = {}
    for s, r in zip(ok_subs, results):
        k = (s["duration"], s["steps"])
        by_cfg.setdefault(k, []).append((s, r))
    rows = []
    for (dur, steps), pairs in by_cfg.items():
        succ = [r for _, r in pairs if r["status"] == "success"]
        if not succ:
            rows.append([dur, steps, len(pairs), "-", "-", "-", "-"])
            continue
        # 每个任务自己的端到端耗时（从提交起算，包含队列等待）
        per_task_e2e = [r["elapsed_s"] + s["submit_s"] for s, r in pairs]
        # 单任务从开始 running 到 success 的耗时 = elapsed_s（poll 自提交后开始计时，减去 polling 启动到 running 之间的间隔）
        # 这里用 r["elapsed_s"] 作为参考（submit 后到 success 的总时长）
        e2e_avg = sum(per_task_e2e) / len(per_task_e2e)
        e2e_max = max(per_task_e2e)
        e2e_min = min(per_task_e2e)
        rows.append([dur, steps, len(pairs),
                     f"{e2e_min:.1f}", f"{e2e_avg:.1f}", f"{e2e_max:.1f}",
                     f"{len(succ)}/{len(pairs)}"])
    print("\n按 (duration, steps) 分组耗时（端到端，submit 到 success）:")
    fmt_table(rows, headers=["dur(s)", "steps", "n", "min(s)", "avg(s)", "max(s)", "succ"])


if __name__ == "__main__":
    main()