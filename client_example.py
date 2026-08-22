"""MiniMax-H3 视频生成 API 调用示例（Python）。

用法（服务器地址已默认指向公网，无需输入）：
    python client_example.py "<提示词>" [宽度] [高度] [时长秒]

示例：
    python client_example.py "一只橘猫在夕阳下漫步"
    python client_example.py "竖屏广告片段" 768 1344 15
    python client_example.py "提示词" 768 1344 15 "http://其他地址:8000"
"""
import json
import sys
import time
import urllib.request

# 服务器公网地址（默认；如需其他服务器，作为最后一个参数传入）
DEFAULT_BASE = "http://117.50.216.253:8000"


def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "海浪拍打礁石，清晨金色阳光，航拍，写实风格，无文字"

    # 解析参数：URL 任意位置均可，其余数字依次为 宽 高 时长
    base = DEFAULT_BASE
    nums = []
    for a in sys.argv[2:]:
        if a.lower().startswith("http"):
            base = a
        else:
            nums.append(float(a))
    width = int(nums[0]) if len(nums) > 0 else 1344
    height = int(nums[1]) if len(nums) > 1 else 768
    duration = nums[2] if len(nums) > 2 else 5

    # 1. 提交生成任务
    resp = post(f"{base}/api/v1/generate", {
        "prompt": prompt,
        "width": width,
        "height": height,
        "duration": duration,
        "seed": 42,
        "steps": 20,
    })
    task_id = resp["task_id"]
    print(f"任务已提交: {task_id}  ({width}x{height}, 预计 {resp['duration']} 秒视频)")

    # 2. 轮询任务状态（生成约需几分钟）
    while True:
        time.sleep(15)
        task = get(f"{base}/api/v1/task/{task_id}")
        print(f"  状态: {task['status']}", flush=True)
        if task["status"] in ("success", "error"):
            break

    # 3. 输出视频地址
    if task["status"] == "success":
        for url in task.get("videos", []):
            print(f"视频地址: {url}")
    else:
        print("生成失败:", task.get("error"))
