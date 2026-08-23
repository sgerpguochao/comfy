#!/usr/bin/env python3
"""并发跑 run_7cases_http.py 的 case 1-6（远程 API）。

调用现有 run_7cases_http.py 的 case 提交逻辑，但替换成并发提交 + 并发轮询 + 并发下载，
看 6 个 case 同时跑、worker 分配、整体吞吐。

用法：
  python3 bench_6cases_concurrent.py --api http://162.14.110.145:8000
  python3 bench_6cases_concurrent.py --api http://127.0.0.1:8000 --download
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import requests as _req  # 用于 multipart 上传

# ---- 复用 run_7cases_http.py 里的素材路径 ----
import importlib.util
spec = importlib.util.spec_from_file_location("rc", "/home/ubuntu/minmax/comfy/playground/run_7cases_http.py")
rc = importlib.util.module_from_spec(spec)
sys.modules["rc"] = rc
spec.loader.exec_module(rc)

CASES = list(range(1, 7))  # case 1-6


def http_post(api, path, payload, timeout=30):
    req = urllib.request.Request(f"{api}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post_multipart(api, path, files, timeout=120):
    """files: list[(field, filename, content_bytes)]，走 requests 实现。"""
    files_payload = [(f[0], (f[1], f[2])) for f in files]
    r = _req.post(f"{api}{path}", files=files_payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def http_get(api, path, timeout=60):
    with urllib.request.urlopen(f"{api}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_get_bytes(api, url, timeout=300):
    full = url if url.startswith("http") else f"{api}{url}"
    with urllib.request.urlopen(full, timeout=timeout) as r:
        return r.read()


def upload(api, path, kind):
    print(f"  [upload] {kind}: {path} ({os.path.getsize(path)} B) -> POST", flush=True)
    t0 = time.time()
    with open(path, "rb") as f:
        data = f.read()
    fn = http_post_multipart(api, "/api/v1/upload",
        [("file", os.path.basename(path), data)])["filename"]
    print(f"  [upload] {kind} -> {fn} ({time.time()-t0:.2f}s)", flush=True)
    return fn


def ensure_materials(api):
    """准备/上传测试素材，返回 {sunset_fn, ocean_fn, bgm_fn}。"""
    refs = {}
    rc.log("[素材] 准备并上传...")
    sunset_img = "/home/ubuntu/minmax/output/test_sunset.png"
    ocean_img = "/home/ubuntu/minmax/output/test_ocean.png"
    bgm = "/home/ubuntu/minmax/bgm_test.mp3"

    # 如不存在则生成（与 run_7cases_http.py 一致）
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(sunset_img), exist_ok=True)
    if not os.path.exists(sunset_img):
        img = Image.new('RGB', (1024, 768), (220, 120, 80))
        ImageDraw.Draw(img).rectangle([(100, 100), (900, 700)], outline=(255, 255, 255), width=10)
        img.save(sunset_img)
    if not os.path.exists(ocean_img):
        img = Image.new('RGB', (1024, 768), (40, 80, 160))
        ImageDraw.Draw(img).ellipse([(300, 200), (700, 600)], outline=(255, 255, 255), width=10)
        img.save(ocean_img)

    refs['sunset_fn'] = upload(api, sunset_img, "sunset")
    refs['ocean_fn'] = upload(api, ocean_img, "ocean")
    if os.path.exists(bgm):
        refs['bgm_fn'] = upload(api, bgm, "bgm")
    else:
        refs['bgm_fn'] = None
        rc.log(f"  [WARN] BGM 不存在: {bgm}")
    return refs


def build_body(case_id, refs, base_seed, steps=10):
    """按 case_id 返回 (name, body)。"""
    bgm = refs.get("bgm_fn") or ""
    sunset = refs.get("sunset_fn", "")
    ocean = refs.get("ocean_fn", "")
    cases = {
        1: ("t2v 基础", {
            "prompt": "海浪拍打礁石，清晨金色阳光，航拍，写实风格",
            "width": 1344, "height": 768, "duration": 5,
            "seed": base_seed + 1, "steps": steps}),
        2: ("t2v + 静音", {
            "prompt": "城市夜景延时摄影，霓虹灯光流转，电影感",
            "width": 1344, "height": 768, "duration": 5,
            "seed": base_seed + 2, "steps": steps, "no_audio": True}),
        3: ("t2i2v + 中文口播", {
            "prompt": "夕阳缓缓落下，海面波光粼粼，云霞渐变色调，唯美写实",
            "first_frame": sunset, "duration": 5,
            "seed": base_seed + 3, "steps": steps,
            "voiceover": "夕阳无限好，只是近黄昏。让我们静静欣赏这一刻的美好。",
            "voice": "zh-CN-XiaoxiaoNeural"}),
        4: ("i2v + 静音", {
            "prompt": "保持画面主体和构图不变，让场景自然流畅地动起来，画面连贯稳定",
            "first_frame": ocean, "duration": 5,
            "seed": base_seed + 4, "steps": steps, "no_audio": True}),
        5: ("首末帧 + 静音 + BGM", {
            "first_frame": sunset, "last_frame": ocean,
            "prompt": "画面从夕阳黄昏平滑过渡到夜晚海洋，色调由暖橙渐变为深蓝",
            "duration": 5, "seed": base_seed + 5, "steps": steps,
            "no_audio": True, "bgm": bgm, "bgm_volume": 0.5}),
        6: ("t2v + 原声 + BGM", {
            "prompt": "森林中微风拂过树叶，阳光透过枝叶斑驳洒落，自然声音",
            "width": 1344, "height": 768, "duration": 5,
            "seed": base_seed + 6, "steps": steps,
            "no_audio": False, "bgm": bgm, "bgm_volume": 0.3}),
    }
    return cases[case_id]


def submit_case(api, case_id, refs, base_seed, steps=10):
    t0 = time.time()
    name, body = build_body(case_id, refs, base_seed, steps)
    try:
        r = http_post(api, "/api/v1/generate", body)
        return {"case_id": case_id, "name": name, "submit_s": round(time.time()-t0, 3),
                "task_id": r["task_id"], "worker": r["worker"],
                "frame_count": r["frame_count"], "duration": r["duration"], "error": None}
    except urllib.error.HTTPError as e:
        body_err = e.read().decode("utf-8", "replace")
        return {"case_id": case_id, "name": name, "submit_s": round(time.time()-t0, 3),
                "task_id": None, "worker": None, "frame_count": None, "duration": None,
                "error": f"HTTP {e.code}: {body_err[:200]}"}


def poll_case(api, case_id, task_id, deadline_s, quiet=True):
    t0 = time.time()
    last = None
    while time.time() - t0 < deadline_s:
        try:
            d = http_get(api, f"/api/v1/task/{task_id}")
        except Exception as e:
            time.sleep(3)
            continue
        st = d.get("status")
        if st != last and not quiet:
            print(f"  [case{case_id}] {st} @ {round(time.time()-t0,1)}s", flush=True)
            last = st
        if st in ("success", "failed", "error"):
            return {"case_id": case_id, "task_id": task_id,
                    "elapsed_s": round(time.time()-t0, 1), "status": st,
                    "videos": d.get("videos", []), "error": d.get("error"),
                    "audio_processed": d.get("audio_processed")}
        time.sleep(3)
    return {"case_id": case_id, "task_id": task_id,
            "elapsed_s": round(time.time()-t0, 1), "status": "timeout",
            "videos": [], "error": "polling timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://162.14.110.145:8000")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out-dir", default=None, help="下载目录（默认按 steps 输出到 output/step_<steps>/）")
    ap.add_argument("--steps", type=int, default=10, help="每个 case 的采样步数（默认 10）")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = f"/home/ubuntu/minmax/output/step_{args.steps}"

    print(f"=== bench_6cases_concurrent: API={args.api} steps={args.steps} out_dir={args.out_dir} ===")
    print(f"[0] 健康检查...", flush=True)
    health = http_get(args.api, "/health", timeout=10)
    print(f"  {health}\n")

    print(f"[1] 准备素材并上传...")
    refs = ensure_materials(args.api)
    print(f"  refs: {refs}\n")

    base_seed = int(time.time()) % 100000
    t_start = time.time()

    # ===== 阶段1: 并发提交 6 个 case =====
    print(f"[2] 并发提交 6 个 case...")
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(submit_case, args.api, cid, refs, base_seed, args.steps) for cid in range(1, 7)]
        subs = [f.result() for f in cf.as_completed(futs)]
    submit_s = time.time() - t_start
    ok = [s for s in subs if s.get("task_id")]
    print(f"  提交阶段耗时: {submit_s:.2f}s（成功 {len(ok)}/{len(subs)}）\n")

    print("提交详情:")
    print(f"  {'case':<6}{'name':<24}{'submit_s':<10}{'worker':<8}{'frames':<8}{'tid':<10}{'err'}")
    for s in sorted(subs, key=lambda x: x["case_id"]):
        print(f"  {s['case_id']:<6}{s['name']:<24}{s.get('submit_s', '-'):<10}"
              f"{str(s.get('worker', '-')):<8}{str(s.get('frame_count', '-')):<8}"
              f"{(s.get('task_id') or '')[:8]:<10}{s.get('error') or ''}")
    worker_dist = {}
    for s in ok:
        w = s.get("worker")
        worker_dist[w] = worker_dist.get(w, 0) + 1
    print(f"\n  worker 分配: {worker_dist}\n")

    if not ok:
        print("无任务可等，退出")
        return

    # ===== 阶段2: 并发轮询 =====
    print(f"[3] 并发轮询 {len(ok)} 个任务...")
    t_poll = time.time()
    with cf.ThreadPoolExecutor(max_workers=len(ok)) as ex:
        futs = {ex.submit(poll_case, args.api, s["case_id"], s["task_id"], args.timeout): s
                for s in ok}
        results = []
        for f in cf.as_completed(futs):
            results.append(f.result())
            r = results[-1]
            print(f"  case {r['case_id']:>2}: status={r['status']} elapsed={r['elapsed_s']}s")
    poll_s = time.time() - t_poll
    total_s = time.time() - t_start

    # ===== 阶段3: 下载 =====
    if args.download:
        print(f"\n[4] 下载视频到 {args.out_dir}...")
        os.makedirs(args.out_dir, exist_ok=True)
        for r in results:
            if r["status"] != "success":
                continue
            for v_url in r.get("videos", []):
                fname = f"case{r['case_id']}_{os.path.basename(v_url.split('?')[0])}"
                try:
                    data = http_get_bytes(args.api, v_url)
                    out = os.path.join(args.out_dir, fname)
                    with open(out, "wb") as f:
                        f.write(data)
                    print(f"  case {r['case_id']}: {fname} ({len(data)//1024} KB)")
                except Exception as e:
                    print(f"  case {r['case_id']}: FAIL {e}")

    # ===== 汇总 =====
    print(f"\n=== 汇总 ===")
    print(f"提交阶段: {submit_s:.2f}s")
    print(f"等待阶段: {poll_s:.2f}s")
    print(f"端到端 total: {total_s:.2f}s = {total_s/60:.1f} min")
    succ = sum(1 for r in results if r["status"] == "success")
    print(f"成功 {succ}/{len(results)}")
    print(f"实测吞吐: {succ / (total_s/60):.2f} tasks/min\n")

    print("按 case 耗时（端到端 submit→success，队列等待+推理+后处理）:")
    print(f"  {'case':<6}{'name':<24}{'e2e(s)':<10}{'status':<10}{'audio':<10}")
    for r in sorted(results, key=lambda x: x.get("case_id")):
        # 找到对应 submit_s
        sub = next((s for s in ok if s["task_id"] == r["task_id"]), None)
        e2e = round(r["elapsed_s"] + (sub["submit_s"] if sub else 0), 1)
        audio = "yes" if r.get("audio_processed") else "-"
        print(f"  {r['case_id']:<6}{build_body(r['case_id'], refs, base_seed, args.steps)[0]:<24}"
              f"{e2e:<10}{r['status']:<10}{audio:<10}")


if __name__ == "__main__":
    main()