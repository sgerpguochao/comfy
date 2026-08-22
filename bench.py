"""双 GPU 视频生成基准测试：并发提交多个任务，统计每个任务的耗时与分配的 worker。"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 832
HEIGHT = int(sys.argv[3]) if len(sys.argv) > 3 else 480
DURATION = float(sys.argv[4]) if len(sys.argv) > 4 else 4
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 20
N = int(sys.argv[6]) if len(sys.argv) > 6 else 2
PROMPT = (sys.argv[7] if len(sys.argv) > 7 else
          "海浪拍打礁石，清晨金色阳光，航拍，写实风格，无文字")


def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


tasks = []
t0 = time.time()
seed_base = int(time.time()) % 10 ** 9  # 每次运行取随机种子，避免触发 ComfyUI 执行缓存
for i in range(N):
    resp = post(f"{BASE}/api/v1/generate", {
        "prompt": f"{PROMPT} (第{i+1}段)", "width": WIDTH, "height": HEIGHT,
        "duration": DURATION, "seed": seed_base + i, "steps": STEPS})
    tasks.append({"id": resp["task_id"], "worker": resp["worker"],
                  "submit": time.time()})
    print(f"[submit] task{i+1} -> worker{resp['worker']} id={resp['task_id'][:8]}",
          flush=True)

done = [False] * N
while not all(done):
    time.sleep(5)
    for i, t in enumerate(tasks):
        if done[i]:
            continue
        st = get(f"{BASE}/api/v1/task/{t['id']}")
        if st["status"] in ("success", "error"):
            done[i] = True
            t["elapsed"] = time.time() - t["submit"]
            print(f"[done] task{i+1} worker{t['worker']} status={st['status']} "
                  f"elapsed={t['elapsed']:.1f}s", flush=True)
            if st.get("videos"):
                print(f"       video: {st['videos'][0]}", flush=True)

print(f"\n总耗时(全部完成): {time.time() - t0:.1f}s")
