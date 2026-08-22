"""双 GPU 并行测试：提交两个视频生成任务并轮询，完成后下载到本地 output 目录。"""
import json
import os
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
OUT_DIR = "/home/ubuntu/minmax/output"
os.makedirs(OUT_DIR, exist_ok=True)

TASKS = [
    {"prompt": "航拍海浪拍打礁石，清晨金色阳光洒在海面，写实风格，自然声音，无文字",
     "width": 1344, "height": 768, "duration": 5.0, "seed": 20260817, "steps": 10,
     "label": "横屏-海浪"},
    {"prompt": "一只橘猫趴在窗台晒太阳，慵懒眨眼，特写镜头，柔和的午后光线，温馨氛围，无文字",
     "width": 768, "height": 1344, "duration": 5.0, "seed": 20260818, "steps": 10,
     "label": "竖屏-橘猫"},
]


def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def download(url, path):
    with urllib.request.urlopen(url) as r:
        with open(path, "wb") as f:
            f.write(r.read())


def main():
    jobs = []
    for t in TASKS:
        payload = {k: v for k, v in t.items() if k != "label"}
        resp = post(f"{BASE}/api/v1/generate", payload)
        jobs.append({"task_id": resp["task_id"], "worker": resp["worker"],
                     "label": t["label"], "done": False, "file": None})
        print(f"[submit] {t['label']} -> worker{resp['worker']} task={resp['task_id'][:8]}", flush=True)

    while not all(j["done"] for j in jobs):
        time.sleep(15)
        for j in jobs:
            if j["done"]:
                continue
            st = get(f"{BASE}/api/v1/task/{j['task_id']}")
            print(f"  [{j['label']}] status={st['status']}", flush=True)
            if st["status"] in ("success", "error"):
                j["done"] = True
                if st["status"] == "success" and st.get("videos"):
                    j["file"] = st["videos"][0]
                else:
                    j["error"] = st.get("error", "unknown")
        sys.stdout.flush()

    print("\n===== 结果 =====", flush=True)
    for j in jobs:
        if j["file"]:
            fname = f"{j['label']}-{time.strftime('%H%M%S')}.mp4"
            path = os.path.join(OUT_DIR, fname)
            download(j["file"], path)
            print(f"[{j['label']}] OK -> {path}", flush=True)
        else:
            print(f"[{j['label']}] FAILED: {j.get('error')}", flush=True)


if __name__ == "__main__":
    main()
