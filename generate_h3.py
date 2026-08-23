"""MiniMax-H3 视频生成脚本（完整版）。

调用本地 H3 API（FastAPI，默认 http://127.0.0.1:8000，对应 PM2 进程 minimax-h3-api）
生成视频，支持三种模式与完整音频配置：

生成模式（--mode）：
  t2v     文生视频：仅提示词
  t2i2v   文图生视频：首帧图片 + 提示词
  i2v     图生视频：仅首帧图片（提示词可省略）
首尾帧生成视频：--mode i2v/t2i2v 时额外传 --last-frame 尾帧图片即可（首尾帧都需提供）

音频配置：
  口播（--voiceover on）：任务完成后自动用 edge-tts 合成配音混流，
        文案用 --voiceover-text 自定义，音色用 --voice 自定义（默认 zh-CN-XiaoxiaoNeural）
  音效（--sfx off）：关掉 H3 原生音效（静音，可与口播/BGM 组合）
  BGM（--bgm on）：混入背景音乐，文件用 --bgm-file 自定义（本地路径 / http(s) URL），
        音量用 --bgm-volume 调节（默认 0.3，BGM 自动循环并对齐视频时长、结尾淡出）

图片/BGM 引用：本地文件自动上传到服务端（/api/v1/upload），也可直接传 http(s) URL、
data: URI 或服务器上已存在的文件名。

用法：
  python generate_h3.py --mode t2v --prompt "一只橘猫在夕阳下漫步" --duration 5
  python generate_h3.py --mode i2v --image cat.png --sfx off --bgm on --bgm-file bgm.mp3
  python generate_h3.py --mode t2i2v --image poster.png --prompt "画面缓缓拉近" \
      --voiceover on --voiceover-text "欢迎观看本期视频" --voice zh-CN-YunxiNeural \
      --bgm on --bgm-file bgm.mp3 --bgm-volume 0.4
  python generate_h3.py --mode i2v --image start.png --last-frame end.png \
      --prompt "从第一张平滑过渡到最后一张" --duration 5
"""

import argparse
import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

# 本地 H3 服务地址（可在 .env 的 LOCAL_BASE_URL 中修改）
DEFAULT_BASE = os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1344, 768
DEFAULT_DURATION = 5
DEFAULT_STEPS = 10
POLL_INTERVAL = 15  # 秒；本地生成约需数分钟

# i2v 模式未给提示词时的默认提示词
IMAGE_DEFAULT_PROMPT = "保持画面主体和构图不变，让场景自然流畅地动起来，画面连贯稳定"


def upload_asset(base: str, path: str) -> str:
    """把本地文件上传到服务端（/api/v1/upload），返回裸文件名供引用。"""
    with open(path, "rb") as f:
        resp = requests.post(
            f"{base}/api/v1/upload",
            files={"file": (os.path.basename(path), f)},
            timeout=120)
    resp.raise_for_status()
    return resp.json()["filename"]


def resolve_asset_ref(base: str, ref: str, kind: str) -> str:
    """把引用归一化为服务端可用的形式：本地文件 -> 上传；URL/data URI/裸文件名 -> 原样。"""
    if ref and os.path.exists(ref) and os.path.isfile(ref):
        print(f"  上传{kind}文件: {ref}")
        return upload_asset(base, ref)
    return ref


def submit_generate(base: str, body: dict) -> str:
    """提交生成任务，返回 task_id。"""
    resp = requests.post(f"{base}/api/v1/generate", json=body, timeout=60)
    if resp.status_code != 202:
        raise SystemExit(f"提交失败（{resp.status_code}）: {resp.text[:500]}")
    return resp.json()["task_id"]


def poll_task(base: str, task_id: str) -> dict:
    """按 task_id 轮询任务状态，成功/失败时返回任务信息。"""
    while True:
        time.sleep(POLL_INTERVAL)
        task = requests.get(f"{base}/api/v1/task/{task_id}", timeout=60).json()
        print(f"  状态: {task['status']}", flush=True)
        if task["status"] in ("success", "error"):
            return task


def download_video(url: str, output_path: str) -> None:
    """下载成片到本地。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"视频已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="通过本地 MiniMax-H3 服务生成视频（文生/文图生/图生 + 口播/音效/BGM）")
    parser.add_argument("--mode", choices=["t2v", "t2i2v", "i2v"], default="t2v",
                        help="生成模式：t2v 文生、t2i2v 文图生、i2v 图生，默认 t2v")
    parser.add_argument("--prompt", default=None,
                        help="视频提示词（画面 + 运镜）；i2v 可省略")
    parser.add_argument("--image", default=None,
                        help="首帧图片：本地路径 / http(s) URL / data URI；i2v、t2i2v 需要")
    parser.add_argument("--last-frame", default=None,
                        help="尾帧图片：本地路径 / http(s) URL / data URI；与 --image 一起提供即为首尾帧生成视频")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help=f"视频时长（秒），默认 {DEFAULT_DURATION}")
    parser.add_argument("--size", default=None,
                        help="分辨率，如 1344x768；缺省时文生视频用 1344x768，图生视频按图片自适应")
    parser.add_argument("--width", type=int, default=None, help="宽度（覆盖 --size）")
    parser.add_argument("--height", type=int, default=None, help="高度（覆盖 --size）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"采样步数，默认 {DEFAULT_STEPS}")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help=f"本地 H3 服务地址，默认 {DEFAULT_BASE}")
    parser.add_argument("--output", default=None,
                        help="输出文件路径，默认 output/h3_<模式>_<时间戳>.mp4")

    # ---- 音频配置 ----
    parser.add_argument("--voiceover", choices=["on", "off"], default="off",
                        help="是否添加口播配音，默认 off")
    parser.add_argument("--voiceover-text", default=None,
                        help="口播文案（--voiceover on 时必填）")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural",
                        help="口播音色 ID，默认 zh-CN-XiaoxiaoNeural")
    parser.add_argument("--sfx", choices=["on", "off"], default="on",
                        help="是否保留 H3 原生音效，默认 on；off 表示静音（可与口播/BGM 组合）")
    parser.add_argument("--bgm", choices=["on", "off"], default="off",
                        help="是否添加 BGM 背景音乐，默认 off")
    parser.add_argument("--bgm-file", default=None,
                        help="BGM 音频文件：本地路径 / http(s) URL（--bgm on 时必填）")
    parser.add_argument("--bgm-volume", type=float, default=0.3,
                        help="BGM 相对音量 0~1，默认 0.3")
    args = parser.parse_args()

    # ---- 参数校验 ----
    if args.mode in ("i2v", "t2i2v") and not args.image:
        raise SystemExit(f"--mode {args.mode} 需要 --image 指定首帧图片")
    if args.last_frame and not args.image:
        raise SystemExit("--last-frame 需要 --image 一起提供首帧图片（首尾帧模式）")
    if args.voiceover == "on" and not args.voiceover_text:
        raise SystemExit("--voiceover on 需要 --voiceover-text 提供口播文案")
    if args.bgm == "on" and not args.bgm_file:
        raise SystemExit("--bgm on 需要 --bgm-file 指定 BGM 音频文件")
    if args.mode == "i2v" and not args.prompt:
        args.prompt = IMAGE_DEFAULT_PROMPT

    width, height = args.width, args.height
    if not width and not height and args.size and "x" in args.size:
        w, h = args.size.split("x", 1)
        try:
            width, height = int(w), int(h)
        except ValueError:
            raise SystemExit(f"--size 格式应为 宽x高，例如 1344x768，收到: {args.size}")

    # ---- 音频选项 -> 请求体 ----
    sfx_off = args.sfx == "off"
    voiceover_text = args.voiceover_text if args.voiceover == "on" else None
    bgm_ref = resolve_asset_ref(args.base, args.bgm_file, "BGM") if args.bgm == "on" else None

    body = {
        "prompt": args.prompt,
        "duration": args.duration,
        "seed": args.seed,
        "steps": args.steps,
        "no_audio": sfx_off,
        "voiceover": voiceover_text,
        "voice": args.voice,
        "bgm": bgm_ref,
        "bgm_volume": max(0.0, min(1.0, args.bgm_volume)),
    }
    if width:
        body["width"] = width
    if height:
        body["height"] = height

    first_frame = None
    last_frame = None
    if args.image:
        first_frame = resolve_asset_ref(args.base, args.image, "图片")
        body["first_frame"] = first_frame
    if args.last_frame:
        last_frame = resolve_asset_ref(args.base, args.last_frame, "尾帧图片")
        body["last_frame"] = last_frame

    # 打印配置摘要
    print("=" * 60)
    print(f"模式: {args.mode}  时长: {args.duration}s  steps: {args.steps}  seed: {args.seed}")
    print(f"分辨率: {width or '按图自适应'}x{height or ''}")
    print(f"提示词: {args.prompt}")
    if first_frame:
        print(f"首帧图片: {first_frame}")
    if last_frame:
        print(f"尾帧图片: {last_frame}")
    print(f"口播: {'on（' + voiceover_text + '，音色 ' + args.voice + '）' if voiceover_text else 'off'}")
    print(f"原生音效: {'off（静音）' if sfx_off else 'on'}")
    print(f"BGM: {('on（' + (bgm_ref or '') + '，音量 ' + str(body['bgm_volume']) + '）') if bgm_ref else 'off'}")
    print("=" * 60)

    # 提交任务
    task_id = submit_generate(args.base, body)
    print(f"任务已提交: {task_id}")

    # 轮询任务状态
    task = poll_task(args.base, task_id)
    if task["status"] != "success":
        raise SystemExit(f"生成失败: {task.get('error')}")

    # 下载成片
    videos = task.get("videos", [])
    if not videos:
        raise SystemExit("任务成功但未返回视频地址")
    url = videos[0]
    if task.get("audio_processed"):
        print("音频后处理完成（口播/静音/BGM 已混入）")
    print(f"视频地址: {url}")

    output = args.output or os.path.join(
        "output", f"h3_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    download_video(url, output)


if __name__ == "__main__":
    main()
