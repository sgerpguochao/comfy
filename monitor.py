"""监控指定任务直到完成，统计耗时（含 ComfyUI 侧精确执行时间）。"""
import json
import sys
import time
import urllib.request

task_id = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
WORKERS = ["http://127.0.0.1:8188", "http://127.0.0.1:8189"]


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


t0 = time.time()
while True:
    st = get(f"{base}/api/v1/task/{task_id}")
    print(f"[poll] {time.time()-t0:6.0f}s status={st['status']}", flush=True)
    if st["status"] in ("success", "error"):
        break
    time.sleep(15)

# 从 ComfyUI history 提取精确执行起止时间
for w in WORKERS:
    try:
        h = get(f"{w}/history/{task_id}")
    except Exception:
        continue
    if task_id in h:
        msgs = h[task_id].get("status", {}).get("messages", [])
        ts = {}
        for m in msgs:
            if isinstance(m, list) and len(m) == 2 and isinstance(m[1], dict):
                key = m[0]
                if key.startswith("execution_start"):
                    ts["start"] = m[1].get("timestamp", 0)
                elif key.startswith("execution_success"):
                    ts["success"] = m[1].get("timestamp", 0)
                elif key.startswith("execution_error"):
                    ts["error"] = m[1].get("timestamp", 0)
        if "start" in ts:
            end = ts.get("success") or ts.get("error") or int(time.time() * 1000)
            print(f"[history] worker={w} 执行时长={ (end - ts['start']) / 1000:.1f}s "
                  f"(start={time.strftime('%H:%M:%S', time.localtime(ts['start']/1000))}, "
                  f"end={time.strftime('%H:%M:%S', time.localtime(end/1000))})", flush=True)

print(f"[done] status={st['status']} api_轮询期间={time.time()-t0:.1f}s", flush=True)
print(json.dumps(st, ensure_ascii=False)[:600], flush=True)
