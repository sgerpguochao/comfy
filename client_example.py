"""MiniMax-H3 视频生成 API 调用示例（Python）：支持文生视频、图生视频（首帧/尾帧）与口播混流。

用法：
    python client_example.py "<提示词>" [宽度] [高度] [时长秒] [服务器地址] [--first 首帧图片] [--last 尾帧图片] [--voiceover 口播文案] [--voice 音色ID]

示例：
    python client_example.py "一只橘猫在夕阳下漫步"
    python client_example.py "让这只猫动起来" --first cat.png
    python client_example.py "无缝转场" --first a.png --last b.png 768 1344 5
    python client_example.py "矿泉水瓶滑落" --first a.png --last b.png --voiceover "清凉一夏，就喝迎驾山泉！" --voice zh-CN-YunxiNeural
    python client_example.py "海浪拍打礁石" --no-audio
"""
import base64
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


def file_to_data_uri(path: str) -> str:
    """本地图片 -> data: URI，无需额外上传接口。"""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else "png"
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def main():
    args = sys.argv[1:]
    prompt = args[0] if args else "海浪拍打礁石，清晨金色阳光，航拍，写实风格，无文字"

    base = DEFAULT_BASE
    nums = []
    first = last = voiceover = voice = None
    no_audio = False
    i = 1
    while i < len(args):
        a = args[i]
        if a == "--first":
            first = args[i + 1]
            i += 2
        elif a == "--last":
            last = args[i + 1]
            i += 2
        elif a == "--voiceover":
            voiceover = args[i + 1]
            i += 2
        elif a == "--voice":
            voice = args[i + 1]
            i += 2
        elif a == "--no-audio":
            no_audio = True
            i += 1
        elif a.lower().startswith("http"):
            base = a
            i += 1
        else:
            nums.append(float(a))
            i += 1

    body = {"prompt": prompt, "duration": nums[2] if len(nums) > 2 else 5,
            "seed": 42, "steps": 20}
    if len(nums) > 0:
        body["width"] = int(nums[0])
    if len(nums) > 1:
        body["height"] = int(nums[1])
    # 不传宽高 + 传图 => 服务端按图片比例自适应
    if first:
        body["first_frame"] = first if first.startswith(("http", "data:")) else file_to_data_uri(first)
    if last:
        body["last_frame"] = last if last.startswith(("http", "data:")) else file_to_data_uri(last)
    if voiceover:
        body["voiceover"] = voiceover
    if voice:
        body["voice"] = voice
    if no_audio:
        body["no_audio"] = True

    resp = post(f"{base}/api/v1/generate", body)
    task_id = resp["task_id"]
    mode = "图生视频+口播" if (first or last) and voiceover else (
        "图生视频" if (first or last) else ("文生视频+口播" if voiceover else "文生视频"))
    print(f"任务已提交({mode}): {task_id}  "
          f"宽高: {body.get('width', '自适应')}x{body.get('height', '自适应')}  "
          f"时长: {body['duration']}s")

    # 轮询任务状态（生成约需几分钟；口播混流在完成后自动执行）
    while True:
        time.sleep(15)
        task = get(f"{base}/api/v1/task/{task_id}")
        print(f"  状态: {task['status']}", flush=True)
        if task["status"] in ("success", "error"):
            break

    if task["status"] == "success":
        for url in task.get("videos", []):
            print(f"视频地址: {url}")
        if task.get("voiceover"):
            print("（已混入口播）")
    else:
        print("生成失败:", task.get("error"))


if __name__ == "__main__":
    main()
