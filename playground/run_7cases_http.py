"""comfy/playground/run_7cases_http.py

通过 start_services.sh 启动的 MiniMax-H3 服务，**直接走 HTTP 接口** 跑 7 个子任务测试。

调用入口（与 api_server.py 一一对应）：
  POST /api/v1/upload       上传本地图片/BGM 到共享 input 目录
  POST /api/v1/generate     提交生成任务（异步 202）
  GET  /api/v1/task/{id}    轮询任务状态与视频 URL
  GET  /health              健康检查
  GET  /api/v1/video/{fn}   下载产物视频

所有视频时长 ≤5 秒，覆盖：
  Case 1: 文生视频（t2v）
  Case 2: 文生视频 + 静音（no_audio=true）
  Case 3: 文图生视频（first_frame）+ 中文口播（voiceover）
  Case 4: 图生视频（first_frame）+ 静音
  Case 5: 首末帧生成（first_frame + last_frame）+ 静音 + BGM
  Case 6: 文生视频 + 原生音效 + BGM（三轨混流）
  Case 7: 参数校验 6 个场景（422 / 400 / 字段缺失）

用法：
    python run_7cases_http.py                  # 跑全部
    python run_7cases_http.py --only 1 3 5     # 只跑指定 case
    python run_7cases_http.py --quick          # 只跑校验类（不消耗 GPU）

产物：
    /home/ubuntu/minmax/output/playground/case{N}_{name}.mp4
    /home/ubuntu/minmax/comfy/playground/run_7cases_http_<时间戳>.log
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import requests

# ---- 路径配置 ----
COMFY_DIR = "/home/ubuntu/minmax/comfy"
START_SCRIPT = os.path.join(COMFY_DIR, "start_services.sh")
OUTPUT_DIR = "/home/ubuntu/minmax/output/playground"
LOG_FILE = os.path.join(COMFY_DIR, "playground",
                        f"run_7cases_http_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# ---- 测试素材 ----
SUNSET_IMG = "/home/ubuntu/minmax/output/test_sunset.png"
OCEAN_IMG = "/home/ubuntu/minmax/output/test_ocean.png"
BGM = "/home/ubuntu/minmax/bgm_test.mp3"

# ---- API 基址（来自 api_server.py:974 uvicorn.run(app, host="0.0.0.0", port=8000)）----
BASE = "http://127.0.0.1:8000"
POLL_INTERVAL = 10   # 秒；本地生成约 1-3 分钟
TIMEOUT = 900        # 单任务最长等待（秒）


def log(msg: str):
    """同时打印到 stdout 和日志文件。"""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def ensure_service() -> bool:
    """通过 /health 确认服务在线，必要时调用 start_services.sh 启动。"""
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        if r.status_code == 200 and r.json().get("status") == "ok":
            return True
    except Exception:
        pass

    log("服务未就绪，尝试 start_services.sh start ...")
    r = subprocess.run(["bash", START_SCRIPT, "start"],
                       cwd=COMFY_DIR, capture_output=True, text=True, timeout=60)
    log(f"  start stdout: {r.stdout.strip()[:200]}")
    time.sleep(8)  # 等 worker 起来
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        if r.status_code == 200 and r.json().get("status") == "ok":
            return True
    except Exception:
        pass
    return False


def ensure_materials() -> dict:
    """确保测试素材存在，返回引用映射（路径 → 已上传文件名/原始路径）。

    HTTP 上传接口：POST /api/v1/upload（multipart，字段名 file），返回 {"filename": "i2v_xxx.png"}
    """
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(SUNSET_IMG), exist_ok=True)

    refs = {}

    if not os.path.exists(SUNSET_IMG):
        img = Image.new('RGB', (1024, 768), (220, 120, 80))
        d = ImageDraw.Draw(img)
        d.rectangle([(100, 100), (900, 700)], outline=(255, 255, 255), width=10)
        d.text((400, 360), 'SUNSET', fill=(255, 255, 255))
        img.save(SUNSET_IMG)
        log(f"  生成首帧图: {SUNSET_IMG}")

    if not os.path.exists(OCEAN_IMG):
        img = Image.new('RGB', (1024, 768), (40, 80, 160))
        d = ImageDraw.Draw(img)
        d.ellipse([(300, 200), (700, 600)], outline=(255, 255, 255), width=10)
        d.text((400, 360), 'OCEAN', fill=(255, 255, 255))
        img.save(OCEAN_IMG)
        log(f"  生成尾帧图: {OCEAN_IMG}")

    if not os.path.exists(BGM):
        log(f"⚠️  BGM 文件不存在: {BGM}（部分 case 需要）")

    # 上传图片到共享 input 目录，得到裸文件名（API 文档：4 种引用形式之一）
    refs['sunset_filename'] = upload_file(SUNSET_IMG, "首帧图")
    refs['ocean_filename'] = upload_file(OCEAN_IMG, "尾帧图")
    if os.path.exists(BGM):
        refs['bgm_filename'] = upload_file(BGM, "BGM")

    log(f"  引用映射: {json.dumps(refs, ensure_ascii=False)}")
    return refs


def upload_file(path: str, kind: str) -> str:
    """POST /api/v1/upload，返回裸文件名。"""
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/v1/upload",
            files={"file": (os.path.basename(path), f)},
            timeout=120,
        )
    r.raise_for_status()
    name = r.json()["filename"]
    log(f"  上传{kind} ({path}) → {name}")
    return name


def submit_generate(body: dict) -> dict:
    """POST /api/v1/generate，返回 {task_id, worker, status, frame_count, duration}。"""
    r = requests.post(f"{BASE}/api/v1/generate", json=body, timeout=60)
    if r.status_code != 202:
        raise RuntimeError(f"提交失败 {r.status_code}: {r.text[:300]}")
    return r.json()


def poll_task(task_id: str) -> dict:
    """GET /api/v1/task/{id}，循环直到 success/error。"""
    t0 = time.time()
    last_status = None
    while True:
        elapsed = time.time() - t0
        if elapsed > TIMEOUT:
            raise TimeoutError(f"轮询超时（>{TIMEOUT}s）")
        time.sleep(POLL_INTERVAL)
        r = requests.get(f"{BASE}/api/v1/task/{task_id}", timeout=60)
        r.raise_for_status()
        task = r.json()
        status = task.get("status")
        if status != last_status:
            log(f"    [{elapsed:>5.0f}s] status={status}")
            last_status = status
        if status in ("success", "error"):
            return task


def download_video(url: str, save_path: str) -> int:
    """下载产物视频，返回字节数。"""
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    size = 0
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            size += len(chunk)
    log(f"  下载完成: {save_path} ({size // 1024} KB)")
    return size


def run_video_case(case_id: int, name: str, body: dict, save_path: str) -> tuple:
    """执行一个功能类用例：提交→轮询→下载。"""
    log(f"{'=' * 70}")
    log(f"Case {case_id}: {name}")
    log(f"  请求体: {json.dumps(body, ensure_ascii=False, indent=2)}")
    log(f"  保存到: {save_path}")
    log(f"{'=' * 70}")

    t0 = time.time()
    try:
        resp = submit_generate(body)
        log(f"  ✅ 提交成功 task_id={resp['task_id']} worker={resp['worker']} "
            f"frames={resp['frame_count']} duration={resp['duration']}s")

        task = poll_task(resp["task_id"])
        elapsed = time.time() - t0

        if task["status"] != "success":
            log(f"  ❌ 任务失败 status={task['status']} error={task.get('error')}")
            return False, elapsed, None

        videos = task.get("videos", [])
        if not videos:
            log(f"  ❌ 任务成功但未返回视频地址")
            return False, elapsed, None

        url = videos[0]
        log(f"  视频 URL: {url}")
        if task.get("audio_processed"):
            log(f"  音频后处理: 已执行")
        size = download_video(url, save_path)
        log(f"  ✅ 完成 耗时={elapsed:.1f}s 大小={size // 1024}KB")
        return True, elapsed, save_path

    except Exception as e:
        elapsed = time.time() - t0
        log(f"  ❌ 异常: {e}")
        return False, elapsed, None


def run_validation_case(case_id: int, name: str, fn) -> tuple:
    """执行一个校验类用例：fn() 应抛 HTTPError 且包含预期错误。"""
    log(f"{'=' * 70}")
    log(f"Case {case_id}: {name}")
    log(f"{'=' * 70}")

    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        log(f"  ❌ 期望失败但实际成功")
        return False, elapsed
    except requests.HTTPError as e:
        elapsed = time.time() - t0
        body_preview = ""
        try:
            body_preview = e.response.text[:200]
        except Exception:
            pass
        log(f"  ✅ 正确拒绝 status={e.response.status_code} 耗时={elapsed:.1f}s")
        log(f"  响应体: {body_preview}")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        log(f"  ❌ 异常类型: {type(e).__name__}: {e}")
        return False, elapsed


def main():
    global BASE
    parser = argparse.ArgumentParser(description="通过 HTTP API 跑 7 个子任务测试")
    parser.add_argument("--api", default=BASE,
                        help=f"API 基址（默认 {BASE}）")
    parser.add_argument("--only", nargs="+", type=int, default=None,
                        help="只跑指定 case 编号，如 --only 1 3 5")
    parser.add_argument("--quick", action="store_true",
                        help="只跑校验类（不消耗 GPU）")
    args = parser.parse_args()

    BASE = args.api

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    log(f"日志文件: {LOG_FILE}")
    log(f"输出目录: {OUTPUT_DIR}")
    log(f"API 基址: {BASE}")

    # 0. 健康检查
    log("\n[Step 0] 健康检查 / 启动服务")
    if not ensure_service():
        log(f"❌ 服务不可用，请手动执行: bash {START_SCRIPT} start")
        sys.exit(1)
    log(f"  ✅ 服务就绪")

    # 0.1 准备素材 + 上传
    log("\n[Step 0.1] 准备测试素材并上传")
    refs = ensure_materials()
    sunset_fn = refs.get("sunset_filename", SUNSET_IMG)
    ocean_fn = refs.get("ocean_filename", OCEAN_IMG)
    bgm_fn = refs.get("bgm_filename", BGM)

    base_seed = int(time.time()) % 100000

    # 1. 定义所有用例
    # ---- 功能类（调用 POST /api/v1/generate）----
    cases = [
        (1, "文生视频 t2v 基础（POST /api/v1/generate）",
         lambda: run_video_case(1, "文生视频 t2v 基础", {
             "prompt": "海浪拍打礁石，清晨金色阳光，航拍，写实风格",
             "width": 1344, "height": 768, "duration": 5,
             "seed": base_seed + 1, "steps": 10,
         }, os.path.join(OUTPUT_DIR, "case1_t2v.mp4"))),

        (2, "文生视频 t2v + 静音（no_audio=true）",
         lambda: run_video_case(2, "文生视频 t2v + 静音", {
             "prompt": "城市夜景延时摄影，霓虹灯光流转，电影感",
             "width": 1344, "height": 768, "duration": 5,
             "seed": base_seed + 2, "steps": 10,
             "no_audio": True,
         }, os.path.join(OUTPUT_DIR, "case2_t2v_mute.mp4"))),

        (3, "文图生视频 t2i2v + 中文口播（first_frame + voiceover）",
         lambda: run_video_case(3, "文图生视频 t2i2v + 中文口播", {
             "prompt": "夕阳缓缓落下，海面波光粼粼，云霞渐变色调，唯美写实",
             "first_frame": sunset_fn,
             "duration": 5, "seed": base_seed + 3, "steps": 10,
             "voiceover": "夕阳无限好，只是近黄昏。让我们静静欣赏这一刻的美好。",
             "voice": "zh-CN-XiaoxiaoNeural",
         }, os.path.join(OUTPUT_DIR, "case3_t2i2v_voice.mp4"))),

        (4, "图生视频 i2v + 静音（first_frame + 描述动效的 prompt）",
         # v2: HTTP 接口 prompt 必填，所以这里传一个描述动效的提示词
         # （等价于 generate_h3.py CLI 的 IMAGE_DEFAULT_PROMPT 行为）
         lambda: run_video_case(4, "图生视频 i2v + 静音（带 prompt）", {
             "prompt": "保持画面主体和构图不变，让场景自然流畅地动起来，画面连贯稳定",
             "first_frame": ocean_fn,
             "duration": 5, "seed": base_seed + 4, "steps": 10,
             "no_audio": True,
         }, os.path.join(OUTPUT_DIR, "case4_i2v_mute.mp4"))),

        (5, "首末帧生成 + 静音 + BGM（最复杂组合）",
         lambda: run_video_case(5, "首末帧生成 + 静音 + BGM", {
             "first_frame": sunset_fn, "last_frame": ocean_fn,
             "prompt": "画面从夕阳黄昏平滑过渡到夜晚海洋，色调由暖橙渐变为深蓝",
             "duration": 5, "seed": base_seed + 5, "steps": 10,
             "no_audio": True,
             "bgm": bgm_fn, "bgm_volume": 0.5,
         }, os.path.join(OUTPUT_DIR, "case5_f2f_bgm.mp4"))),

        (6, "文生视频 t2v + 原声 + BGM（三轨混流）",
         lambda: run_video_case(6, "文生视频 t2v + 原声 + BGM", {
             "prompt": "森林中微风拂过树叶，阳光透过枝叶斑驳洒落，自然声音",
             "width": 1344, "height": 768, "duration": 5,
             "seed": base_seed + 6, "steps": 10,
             "no_audio": False,
             "bgm": bgm_fn, "bgm_volume": 0.3,
         }, os.path.join(OUTPUT_DIR, "case6_t2v_bgm.mp4"))),

        # ---- 校验类 ----
        (71, "校验: prompt 字段缺失（POST /api/v1/generate）",
         lambda: run_validation_case(71, "prompt 字段缺失", lambda: (
             requests.post(f"{BASE}/api/v1/generate",
                           json={"duration": 5}, timeout=10).raise_for_status()
         ))),

        (72, "校验: duration 超出范围（>20）",
         lambda: run_validation_case(72, "duration=25 越界", lambda: (
             requests.post(f"{BASE}/api/v1/generate",
                           json={"prompt": "test", "duration": 25}, timeout=10).raise_for_status()
         ))),

        (73, "校验: duration 越界（<1）",
         lambda: run_validation_case(73, "duration=0 越界", lambda: (
             requests.post(f"{BASE}/api/v1/generate",
                           json={"prompt": "test", "duration": 0}, timeout=10).raise_for_status()
         ))),

        (74, "校验: width 超出范围（>2048）",
         lambda: run_validation_case(74, "width=4096 越界", lambda: (
             requests.post(f"{BASE}/api/v1/generate",
                           json={"prompt": "test", "width": 4096, "height": 768},
                           timeout=10).raise_for_status()
         ))),

        (75, "校验: first_frame 引用无法识别（随机字符串）",
         lambda: run_validation_case(75, "first_frame=invalid_ref", lambda: (
             requests.post(f"{BASE}/api/v1/generate",
                           json={"prompt": "test", "first_frame": "not-a-url-or-data-uri",
                                 "duration": 5}, timeout=10).raise_for_status()
         ))),

        (76, "校验: 查询不存在的 task_id（API 设计：返回 queued 而非 404）",
         lambda: run_validation_case(76, "GET /api/v1/task/<unknown>", lambda: (
             requests.get(f"{BASE}/api/v1/task/00000000-0000-0000-0000-000000000000",
                          timeout=10).raise_for_status()
             # API 实际行为：200 + {"status":"queued"}（不视为错误），所以此 case FAIL
         ))),

        (77, "校验: 上传非图片/音频文件类型",
         lambda: run_validation_case(77, "上传 .txt 当图片", lambda: (
             # 上传一个 .txt 看接口是否仅校验大小（实际接口不限 MIME，只校验大小）
             # 此处预期会成功（HTTP 200），不算校验通过；标记期望失败
             requests.post(f"{BASE}/api/v1/upload",
                           files={"file": ("fake.txt", b"hello world")},
                           timeout=10).raise_for_status()
         ))),
    ]

    # Case 76/77 是文档化 API 行为的探针（不是期望被拦截）
    # 我们把它们的预期标记为"已文档化"——失败也算合理发现

    # 选择执行的 case
    if args.quick:
        selected = [c for c in cases if c[0] >= 70]
        log(f"\n[模式] quick：只跑 {len(selected)} 个校验类用例")
    elif args.only:
        selected = [c for c in cases if c[0] in args.only]
        log(f"\n[模式] 只跑指定用例 {args.only}，共 {len(selected)} 个")
    else:
        selected = cases
        func_count = sum(1 for c in cases if c[0] < 70)
        val_count = sum(1 for c in cases if c[0] >= 70)
        log(f"\n[模式] 跑全部 {len(selected)} 个用例（{func_count} 功能 + {val_count} 校验）")
        log(f"预计耗时约 {func_count * 200 // 60} 分钟（仅功能类）")

    log("")

    results = []
    t_start = time.time()
    for case_id, name, fn in selected:
        ok, elapsed, *_ = fn()
        results.append((case_id, name, ok, elapsed))
        log("")  # 空行分隔

    total_elapsed = time.time() - t_start

    # ---- 汇总 ----
    log("=" * 70)
    log("汇总")
    log("=" * 70)
    log(f"{'Case':<6} {'结果':<8} {'耗时':<10} {'名称'}")
    log("-" * 70)
    for cid, name, ok, elapsed in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        log(f"{cid:<6} {status:<8} {elapsed:>6.1f}s    {name}")
    log("-" * 70)

    passed = sum(1 for _, _, ok, _ in results if ok)
    failed = len(results) - passed
    log(f"通过: {passed}/{len(results)}    失败: {failed}")
    log(f"总耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} 分钟)")
    log(f"日志: {LOG_FILE}")
    log(f"产物目录: {OUTPUT_DIR}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
