"""MiniMax-H3 视频生成 API 封装服务。

包装多个 ComfyUI 实例（每个实例绑定一张 GPU）的 /prompt API，对外提供简洁的 HTTP 接口：
  POST /api/v1/generate  提交文生视频任务（自动分发给队列最短的 GPU 实例）
  GET  /api/v1/task/{id} 查询任务状态
  GET  /health           健康检查

多 GPU 部署：通过环境变量 COMFY_HOSTS 指定所有 ComfyUI 实例（逗号分隔的 host:port），
默认值为两个 worker（127.0.0.1:8188,127.0.0.1:8189），每个实例由 --cuda-device 绑定一张 GPU。
"""

import asyncio
import base64
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import uuid
from typing import Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field

# ---- 配置 ----
COMFY_WORKERS = [w.strip() for w in
                 os.environ.get("COMFY_HOSTS", "127.0.0.1:8188,127.0.0.1:8189").split(",")
                 if w.strip()]
COMFY_BASE = f"http://{COMFY_WORKERS[0]}"  # 兼容旧引用

# 模型文件（完整 INT8 版）
DIFFUSION_MODEL = "minimax_h3_fl2va_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

DEFAULT_STEPS = 10  # 10 步为速度/质量折中（20 步约慢 1.7 倍）
DEFAULT_FPS = 24
MAX_DURATION = 10  # 模型训练区间 ~5-15s；本地实测 10s 不 OOM（显存 98.2%），与 generate_h3 的 --duration 上限保持一致

# 口播（edge-tts + ffmpeg 混流）
VOICE_DEFAULT = "zh-CN-XiaoxiaoNeural"  # 默认音色（女声，新闻/小说）
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI", "output")
OUTPUT_DIRS = [BASE_OUTPUT_DIR, os.path.join(BASE_OUTPUT_DIR, "gpu1")]  # 与 COMFY_WORKERS 对应
MUX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "muxed")
# 任务元数据（口播文案/音色、静音标记、worker 分配）持久化文件，服务重启后仍可恢复
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_state.json")


def frame_count_for_duration(duration: float) -> int:
    """duration(秒) -> 帧数，对齐到模型 17k+5 帧网格。"""
    n = max(5, round(duration * DEFAULT_FPS))
    return n + (5 - n % 17) % 17


def adapt_canvas_size(width: int, height: int) -> tuple:
    """短边 768、面积上限 768*1344、逐轴对齐 32 的画布（与 H3 节点 adapt_canvas 一致）。"""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = 768 * ratio, 768
    else:
        nom_w, nom_h = 768, 768 / ratio
    if nom_w * nom_h > 768 * 1344:
        s = math.sqrt(768 * 1344 / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return max(32, round(nom_w / 32) * 32), max(32, round(nom_h / 32) * 32)


def image_size(data: bytes) -> tuple:
    """从图片字节读取原始宽高。"""
    with Image.open(io.BytesIO(data)) as im:
        return im.size


_UPLOAD_EXT = ("png", "jpg", "jpeg", "webp",
               "mp3", "wav", "ogg", "m4a", "aac", "flac", "mp4", "mov")


def image_upload_name(ref: str) -> str:
    """从 URL / data URI 派生一个合法的上传文件名（uuid 前缀避免跨任务缓存冲突）。

    图片保留图片扩展名，音频（BGM）保留音频扩展名——扩展名会被 ffmpeg 用于选择
    解复用器，BGM 被改名成 .png 会导致 ffmpeg 按 image2 解复用而混流失败。
    """
    ext = "png"
    if ref.startswith("data:"):
        mime = ref[5:ref.find(";")].lower() if ";" in ref[:64] else "image/png"
        ext = {"image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp",
               "audio/mpeg": "mp3", "audio/wav": "wav", "audio/wave": "wav",
               "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/aac": "aac"}.get(mime, "png")
    else:
        ext = os.path.splitext(urllib.parse.urlparse(ref).path)[1].lstrip(".").lower()
        if ext not in _UPLOAD_EXT:
            ext = "png"
    return f"i2v_{uuid.uuid4().hex[:12]}.{ext}"


def build_prompt(prompt: str, width: int, height: int, duration: float,
                 seed: int, steps: int,
                 first_frame: Optional[str] = None,
                 last_frame: Optional[str] = None,
                 prefix: Optional[str] = None) -> dict:
    """构造 ComfyUI API 格式的 prompt（与官方 T2V 工作流等价，使用完整 INT8 模型）。

    first_frame/last_frame 为已上传到该 worker input 目录的图片文件名；
    传图片即图生视频（fl2va），不传则为文生视频（t2va）。
    prefix 为每个任务唯一的输出文件名前缀，避免双 worker 序号碰撞导致覆盖。
    """
    length = frame_count_for_duration(duration)
    graph = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "5": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                         "width": width, "height": height, "length": length}},
        "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "7": {"class_type": "BasicScheduler",
              "inputs": {"model": ["1", 0], "scheduler": "simple",
                         "steps": steps, "denoise": 1.0}},
        "8": {"class_type": "BasicGuider",
              "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["9", 0], "guider": ["8", 0], "sampler": ["6", 0],
                          "sigmas": ["7", 0], "latent_image": ["5", 1]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo",
               "inputs": {"images": ["11", 0], "audio": ["12", 0],
                          "fps": DEFAULT_FPS, "bit_depth": 8}},
        "14": {"class_type": "SaveVideo",
               "inputs": {"video": ["13", 0],
                          "filename_prefix": f"video/minimax_{prefix or 'h3'}",
                          "format": "mp4", "codec": "auto"}},
    }
    node = graph["5"]["inputs"]
    if first_frame:
        graph["15"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        node["first_frame"] = ["15", 0]
    if last_frame:
        graph["16"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
        node["last_frame"] = ["16", 0]
    return graph


def build_ref2v_prompt(prompt: str, width: int, height: int, duration: float,
                       seed: int, steps: int,
                       ref_images: list,
                       ref_image_size: str = "max",
                       prefix: Optional[str] = None) -> dict:
    """构造 ComfyUI API 格式的 ref2va prompt（MiniMaxH3ReferenceToVideo）。

    参考图以 <Picture i> 形式参与 conditioning，是人物/商品跨镜头一致性的身份锚点；
    故事版风格引用 @ImageN / @图片N 会被重写为 <Picture N>。ref_images 为已上传到
    worker input 目录的图片文件名列表（≤9 张，节点上限）。
    """
    length = frame_count_for_duration(duration)
    graph = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "5": {"class_type": "MiniMaxH3ReferenceToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
                         "prompt": prompt, "width": width, "height": height,
                         "length": length, "ref_image_size": ref_image_size,
                         "ref_images": {f"ref_image_{i}": [str(15 + i), 0]
                                        for i in range(len(ref_images))}}},
        "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "7": {"class_type": "BasicScheduler",
              "inputs": {"model": ["1", 0], "scheduler": "simple",
                         "steps": steps, "denoise": 1.0}},
        "8": {"class_type": "BasicGuider",
              "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["9", 0], "guider": ["8", 0], "sampler": ["6", 0],
                          "sigmas": ["7", 0], "latent_image": ["5", 1]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo",
               "inputs": {"images": ["11", 0], "audio": ["12", 0],
                          "fps": DEFAULT_FPS, "bit_depth": 8}},
        "14": {"class_type": "SaveVideo",
               "inputs": {"video": ["13", 0],
                          "filename_prefix": f"video/minimax_{prefix or 'h3'}",
                          "format": "mp4", "codec": "auto"}},
    }
    for i, name in enumerate(ref_images):
        graph[str(15 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
    return graph


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="视频提示词（含画面 + 运镜 + 音频描述）")
    width: Optional[int] = Field(None, ge=32, le=2048, description="视频宽度；图生视频缺省时按图片自适应")
    height: Optional[int] = Field(None, ge=32, le=2048, description="视频高度；图生视频缺省时按图片自适应")
    duration: float = Field(5.0, ge=1.0, le=MAX_DURATION, description="视频时长（秒）")
    seed: int = Field(0, ge=0, le=2 ** 64 - 1)
    steps: int = Field(DEFAULT_STEPS, ge=1, le=100)
    first_frame: Optional[str] = Field(None, description="首帧图片：http(s) URL 或 data: URI")
    last_frame: Optional[str] = Field(None, description="尾帧图片：http(s) URL 或 data: URI")
    image: Optional[str] = Field(None, description="单图快捷方式（等价于 first_frame）")
    voiceover: Optional[str] = Field(None, description="口播文案，非空时任务完成后自动合成配音混流")
    voice: Optional[str] = Field(None, description="口播音色 ID，默认 zh-CN-XiaoxiaoNeural")
    no_audio: bool = Field(False, description="为 true 时剥掉 H3 原生音效（静音），可与口播/BGM 组合")
    bgm: Optional[str] = Field(None, description="BGM 背景音乐：本地路径、http(s) URL 或已上传文件名，非空时任务完成后自动混入")
    bgm_volume: float = Field(0.3, ge=0.0, le=1.0, description="BGM 相对音量，默认 0.3")


app = FastAPI(title="MiniMax-H3 视频生成 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_workers = []        # 每个 ComfyUI 实例一个 aiohttp 会话，下标与 COMFY_WORKERS 对应
_worker_assign = {}  # task_id -> worker 索引
_video_worker = {}   # 输出文件名 -> worker 索引
_rr = 0              # 兜底轮询游标
# 并发改造：去掉 _submit_lock，改为 _local_load 预留式调度
# _local_load[i] = 已选 worker i 但 /prompt 尚未返回的"在飞任务"数
# 选 worker 时按 (远程队列长度 + 本地在飞) 之和最小选，避免并发都冲到同一个 worker
_local_load = []
_pick_lock = asyncio.Lock()  # 仅保护 _pick_worker 内的"读 /queue + 写 _local_load"原子性
_task_voiceover = {}  # task_id -> (口播文案, 音色)
_task_noaudio = set() # task_id -> 静音（剥掉原声）
_task_bgm = {}        # task_id -> (BGM 引用, 音量)
_muxed = {}           # 混流后文件名 -> 本地路径
_vo_done = set()      # 后处理已终结的 task_id（成功或 ffmpeg 失败；源文件缺失不记入）
_post_enqueued = set()  # 已入后台队列、避免每次轮询重复 put
_post_failed = set()    # ffmpeg 已失败，不再重试

# 启动预热：异步提交 1 个 dummy 任务把模型加载到显存，避免业务任务摊 7.5 分钟冷启
_warmed_workers: set = set()       # 已完成 warmup 的 worker 索引
_warmup_tasks: dict = {}           # worker 索引 -> asyncio.Task（用于状态查询/防重入）

# 后处理后台队列：避免 mux 阻塞 /api/v1/task 轮询响应
_post_queue: asyncio.Queue = None   # 启动时初始化
_post_workers: list = []            # 后台 mux 处理协程列表


def _save_state():
    """把任务元数据写入磁盘，服务重启后可恢复，避免口播等后处理静默丢失。"""
    st = {
        "voiceover": {tid: [text, voice] for tid, (text, voice) in _task_voiceover.items()},
        "noaudio": sorted(_task_noaudio),
        "bgm": {tid: [ref, vol] for tid, (ref, vol) in _task_bgm.items()},
        "assign": _worker_assign,
    }
    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(st, f, ensure_ascii=False)
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass


def _index_muxed_dir():
    """把 muxed/ 里已有的 muted_/muxed_ 文件登记进 _muxed，重启后查询仍能命中。"""
    if not os.path.isdir(MUX_DIR):
        return
    for fn in os.listdir(MUX_DIR):
        if fn.startswith(("muted_", "muxed_")) and fn.endswith(".mp4"):
            _muxed[fn] = os.path.join(MUX_DIR, fn)


def _load_state():
    """启动时从磁盘恢复任务元数据。"""
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
    except (OSError, ValueError):
        _index_muxed_dir()
        return
    for tid, tv in st.get("voiceover", {}).items():
        if isinstance(tv, list) and len(tv) == 2 and isinstance(tv[0], str):
            _task_voiceover[tid] = (tv[0], tv[1])
    _task_noaudio.update(st.get("noaudio", []))
    for tid, bv in st.get("bgm", {}).items():
        if isinstance(bv, list) and len(bv) == 2 and isinstance(bv[0], str):
            _task_bgm[tid] = (bv[0], float(bv[1]) if isinstance(bv[1], (int, float)) else 0.3)
    for tid, idx in st.get("assign", {}).items():
        try:
            _worker_assign[tid] = int(idx)
        except (TypeError, ValueError):
            pass
    _index_muxed_dir()


def _worker_url(i: int) -> str:
    return f"http://{COMFY_WORKERS[i]}"


@app.on_event("startup")
async def startup():
    global _workers, _local_load, _post_queue
    _workers = [aiohttp.ClientSession() for _ in COMFY_WORKERS]
    _local_load = [0] * len(COMFY_WORKERS)  # 并发调度：每个 worker 的"已预留"计数
    _load_state()  # 恢复任务元数据（口播/静音/worker 分配），防止重启丢失

    # 后处理后台队列：把 mux 从 /api/v1/task 轮询链路剥离，避免阻塞客户端轮询
    import asyncio as _aio
    _post_queue = _aio.Queue()
    # CPU 端 mux 开 2 个 worker 足够（ffmpeg + edge-tts 各占 CPU 不重）
    for _ in range(2):
        _post_workers.append(_aio.create_task(_mux_worker_loop()))

    # 启动预热：后台异步加载模型，不阻塞 /health
    _aio.create_task(_warmup_all_workers())


async def _warmup_all_workers():
    """给每张卡并发提交 1 个 dummy 任务，把 H3 模型加载到显存。

    不阻塞 startup：_warmed_workers 在 /health 中反映 warm 状态，
    客户端/测试脚本可以据此判断"可以提交业务任务"。
    """
    import asyncio as _aio
    tasks = [_aio.create_task(_warmup_one(i)) for i in range(len(COMFY_WORKERS))]
    await _aio.gather(*tasks, return_exceptions=True)


async def _warmup_one(idx: int):
    """单卡 warmup：提交 1 帧 1 step 的最小任务，触发 UNet/CLIP/VAE 加载。"""
    if idx in _warmed_workers:
        return
    _warmup_tasks[idx] = _warmup_one  # 占位，避免重复触发
    session, base = _workers[idx], _worker_url(idx)
    try:
        graph = build_prompt(
            prompt="warmup", width=128, height=128,
            duration=1, seed=0, steps=1,
            first_frame=None, last_frame=None, prefix="warmup")
        async with session.post(f"{base}/prompt",
            json={"prompt": graph, "client_id": "warmup"}) as r:
            await r.read()
        _warmed_workers.add(idx)
        print(f"[warmup] worker{idx} 模型已加载", flush=True)
    except Exception as e:
        print(f"[warmup] worker{idx} 失败: {e}", flush=True)
    finally:
        _warmup_tasks.pop(idx, None)


async def _mux_worker_loop():
    """后台消费 _post_queue，调用 _post_process；与 /task 轮询完全解耦。"""
    while True:
        item = await _post_queue.get()
        try:
            task_id, idx, src_fn, subfolder = item
            await _post_process(task_id, idx, src_fn, subfolder)
        except Exception as e:
            if item:
                _post_enqueued.discard(item[0])
            print(f"[mux-bg] task={item[0] if item else '?'} 失败: {e}", flush=True)
        finally:
            _post_queue.task_done()


@app.on_event("shutdown")
async def shutdown():
    for s in _workers:
        await s.close()


async def _pick_worker() -> int:
    """选择（远程队列长度 + 本地在飞）最小的 worker；全部不可达时按轮询兜底。

    并发改造点：
    1. 读各 worker 的 /queue + 本地 _local_load 之和选最小
    2. 选中的 worker 在 _local_load 上 +1（"预留"），避免后续并发请求都选同一个
    3. 调用方在 /prompt 返回（成功或失败）后调用 _release_worker(idx) 释放预留
    4. _pick_lock 保护"读 + 写 _local_load"的原子性，但只锁这一小段，请求主体不阻塞

    注意：_local_load 是预估值（"已选未提交"），不是真实负载；ComfyUI 实际队列长度可能滞后
    """
    global _rr
    async with _pick_lock:
        best, best_load = 0, 10 ** 9
        for i in range(len(COMFY_WORKERS)):
            try:
                async with _workers[i].get(f"{_worker_url(i)}/queue") as resp:
                    data = await resp.json()
                remote = len(data.get("queue_running", [])) + len(data.get("queue_pending", []))
            except Exception:
                remote = 10 ** 9
            load = remote + _local_load[i]
            if load < best_load:
                best, best_load = i, load
        if best_load == 10 ** 9:
            best = _rr % len(COMFY_WORKERS)
        _rr += 1
        _local_load[best] += 1
        return best


async def _release_worker(idx: int):
    """释放 worker 预留。/prompt 返回后调用（成功或失败都调）。"""
    async with _pick_lock:
        if 0 <= idx < len(_local_load):
            _local_load[idx] = max(0, _local_load[idx] - 1)


# ComfyUI 共享 input 目录（两个 worker 共用）
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI", "input")


def _is_uploaded_ref(ref: Optional[str]) -> bool:
    """裸文件名（如 /api/v1/upload 返回的）视为已上传到 ComfyUI input 目录，直接引用。"""
    return bool(ref) and not ref.startswith(("http://", "https://", "data:")) and "/" not in ref


async def _resolve_image(session, ref: str) -> bytes:
    """图片引用（http(s) URL 或 data: URI）-> 原始字节。"""
    if ref.startswith("data:"):
        return base64.b64decode(ref.split(",", 1)[1])
    if not ref.startswith(("http://", "https://")):
        raise HTTPException(status_code=400,
                            detail="无法识别的图片引用，应为 http(s) URL、data: URI、"
                                   "/api/v1/upload 返回的文件名或服务器本地文件路径")
    async with session.get(ref, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        if resp.status != 200:
            raise HTTPException(status_code=400, detail=f"无法下载图片（{resp.status}）: {ref[:100]}")
        return await resp.read()


async def _fetch_frame_bytes(session, ref: Optional[str]) -> tuple:
    """解析图片引用 -> (直接使用的文件名, 待上传字节)，两者至多其一非 None。

    裸文件名（已上传）直接返回；服务器本地已存在的路径（绝对或 input 相对）读取字节；
    http(s) URL / data URI 下载字节。
    """
    if not ref:
        return None, None
    if _is_uploaded_ref(ref):
        # 已上传的裸文件名：若文件在共享 input 目录，顺带读取字节用于画布自适应（不回传重新上传）
        for p in [os.path.join(INPUT_DIR, ref)]:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return ref, f.read()
        return ref, None
    candidates = [ref]
    if "/" in ref and not os.path.isabs(ref):
        candidates.append(os.path.join(INPUT_DIR, ref))
    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return None, f.read()
    return None, await _resolve_image(session, ref)


async def _upload_image(session, worker_url: str, name: str, data: bytes) -> str:
    """上传图片到 ComfyUI worker 的 input 目录，返回 LoadImage 可引用的文件名。"""
    ext = os.path.splitext(name)[1].lower()
    ctype = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "webp": "image/webp"}.get(ext, "image/png")
    form = aiohttp.FormData()
    form.add_field("image", data, filename=name, content_type=ctype)
    async with session.post(f"{worker_url}/upload/image", data=form) as resp:
        if resp.status != 200:
            raise HTTPException(status_code=502, detail=await resp.text())
        info = await resp.json()
    return info["name"]


@app.post("/api/v1/upload")
async def upload_image(request: Request):
    """上传自定义图片（multipart，字段名 file）到共享 input 目录，返回文件名供 first_frame/last_frame 引用。"""
    form = await request.form()
    file = form.get("file")
    if file is None or not getattr(file, "filename", None):
        raise HTTPException(status_code=400, detail="缺少 file 字段（multipart/form-data 上传）")
    data = await file.read()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片超过 30MB 限制")
    name = await _upload_image(_workers[0], _worker_url(0),
                               image_upload_name(file.filename), data)
    return {"filename": name}


async def _submit_job(prompt: str, width: Optional[int], height: Optional[int],
                      duration: float, seed: int, steps: int,
                      first_frame: Optional[str] = None,
                      last_frame: Optional[str] = None,
                      voiceover: Optional[str] = None,
                      voice: Optional[str] = None,
                      no_audio: bool = False,
                      bgm: Optional[str] = None,
                      bgm_volume: float = 0.3) -> tuple:
    """选择 worker、读取/上传首尾帧、提交图生/文生任务，返回 (task_id, worker_idx)。

    first_frame/last_frame 支持四种引用：http(s) URL、data: URI、
    /api/v1/upload 返回的裸文件名、服务器本地图片路径（绝对或 input 相对）。
    voiceover 非空时，任务完成后自动用 edge-tts 合成口播并 ffmpeg 混流；
    no_audio=True 时剥掉 H3 原生音效；bgm 非空时自动混入背景音乐（可组合）。
    """
    # 图片读取/下载放在锁外，避免大图/慢链接阻塞其他任务；裸文件名直接使用
    ff_ref, ff_data = await _fetch_frame_bytes(_workers[0], first_frame)
    lf_ref, lf_data = await _fetch_frame_bytes(_workers[0], last_frame)
    if (ff_data or lf_data) and (width is None or height is None):
        width, height = adapt_canvas_size(*image_size(ff_data or lf_data))
    if width is None or height is None:
        width, height = 1344, 768

    task_id = str(uuid.uuid4())
    if voiceover:
        _task_voiceover[task_id] = (voiceover, voice or VOICE_DEFAULT)
    if no_audio:
        _task_noaudio.add(task_id)
    if bgm:
        _task_bgm[task_id] = (bgm, bgm_volume)
    # 并发改造：先在外层预留 worker idx（在 _pick_lock 内已 _local_load[idx]+1），
    # 再只在"上传 + 提交"这段短暂持锁；上传失败也必须释放预留，避免 _local_load 单调累加
    idx = await _pick_worker()
    try:
        session, base = _workers[idx], _worker_url(idx)
        ff_name = (ff_ref or
                   (await _upload_image(session, base, image_upload_name(first_frame), ff_data)
                    if ff_data else None))
        lf_name = (lf_ref or
                   (await _upload_image(session, base, image_upload_name(last_frame), lf_data)
                    if lf_data else None))
        prompt_graph = build_prompt(prompt, width, height, duration, seed, steps,
                                    ff_name, lf_name, prefix=task_id[:12])
        payload = {"prompt": prompt_graph, "client_id": "api-wrapper", "prompt_id": task_id}
        async with session.post(f"{base}/prompt", json=payload) as resp:
            body = await resp.json()
            if resp.status != 200:
                raise HTTPException(status_code=502, detail=body)
    finally:
        # /prompt 已成功入队（或失败抛出），释放预留；HTTPException 会冒泡给上层
        await _release_worker(idx)
    _worker_assign[task_id] = idx
    _save_state()  # 持久化任务元数据，重启后仍可完成后处理
    return task_id, idx


async def _submit_ref2v_job(prompt: str, width: Optional[int], height: Optional[int],
                            duration: float, seed: int, steps: int,
                            ref_images: list,
                            ref_image_size: str = "match",
                            voiceover: Optional[str] = None,
                            voice: Optional[str] = None,
                            no_audio: bool = False,
                            bgm: Optional[str] = None,
                            bgm_volume: float = 0.3) -> tuple:
    """上传参考图并提交 ref2va 任务，返回 (task_id, worker_idx)。

    ref_images 支持与 first_frame/last_frame 相同的四种引用（http(s) URL、data: URI、
    /api/v1/upload 返回的裸文件名、服务器本地路径），节点上限 9 张：保留前 9 张
    （定妆照排在最前，其次为各分镜关键帧），超出部分直接丢弃。
    """
    refs = ref_images[:9]
    # 图片读取/下载放在锁外，避免大图/慢链接阻塞其他任务
    refs_data = []
    for ref in refs:
        name, data = await _fetch_frame_bytes(_workers[0], ref)
        refs_data.append((ref, name, data))
    if (width is None or height is None) and refs_data:
        # 参考图缺省时按第一张有内容的图自适应画布
        for _, _, data in refs_data:
            if data:
                width, height = adapt_canvas_size(*image_size(data))
                break
    if width is None or height is None:
        width, height = 1344, 768

    task_id = str(uuid.uuid4())
    if voiceover:
        _task_voiceover[task_id] = (voiceover, voice or VOICE_DEFAULT)
    if no_audio:
        _task_noaudio.add(task_id)
    if bgm:
        _task_bgm[task_id] = (bgm, bgm_volume)
    idx = await _pick_worker()
    try:
        session, base = _workers[idx], _worker_url(idx)
        names = []
        for ref, name, data in refs_data:
            if name:
                names.append(name)
            else:
                names.append(await _upload_image(session, base, image_upload_name(ref), data))
        prompt_graph = build_ref2v_prompt(prompt, width, height, duration, seed, steps,
                                          names, ref_image_size, prefix=task_id[:12])
        payload = {"prompt": prompt_graph, "client_id": "api-wrapper", "prompt_id": task_id}
        async with session.post(f"{base}/prompt", json=payload) as resp:
            body = await resp.json()
            if resp.status != 200:
                raise HTTPException(status_code=502, detail=body)
    finally:
        await _release_worker(idx)
    _worker_assign[task_id] = idx
    _save_state()
    return task_id, idx


async def _run_cmd(cmd: list) -> tuple:
    """异步运行子进程，返回 (returncode, stderr)。"""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    return proc.returncode, err


async def _resolve_src(task_id: str, idx: int, src_filename: str, subfolder: str) -> Optional[str]:
    """定位生成视频的本地路径；磁盘缺失时回退从 worker HTTP 拉取。"""
    os.makedirs(MUX_DIR, exist_ok=True)
    src = os.path.join(OUTPUT_DIRS[idx], subfolder or "", src_filename)
    if os.path.exists(src):
        return src
    try:
        async with _workers[idx].get(
                f"{_worker_url(idx)}/view?filename={src_filename}&type=output"
                + (f"&subfolder={subfolder}" if subfolder else "")) as resp:
            if resp.status == 200:
                src = os.path.join(MUX_DIR, f"src_{task_id}.mp4")
                with open(src, "wb") as f:
                    f.write(await resp.read())
                return src
    except Exception:
        pass
    return None


async def _probe_duration(path: str) -> float:
    """用 ffprobe 读取视频时长（秒），失败时返回 0。"""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate()
    try:
        return float(out.decode().strip())
    except (ValueError, UnicodeDecodeError):
        return 0.0


async def _resolve_bgm(session, ref: str) -> Optional[str]:
    """BGM 引用 -> 本地 ffmpeg 可读路径。

    支持：服务器本地绝对路径、input 目录相对路径（含 /api/v1/upload 返回的裸文件名）、
    http(s) URL（下载到 muxed 目录）。
    """
    if not ref:
        return None
    if ref.startswith(("http://", "https://")):
        os.makedirs(MUX_DIR, exist_ok=True)
        dest = os.path.join(MUX_DIR, f"bgm_{uuid.uuid4().hex[:12]}.mp3")
        try:
            async with session.get(ref, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    return None
                with open(dest, "wb") as f:
                    f.write(await resp.read())
            return dest
        except Exception:
            return None
    if os.path.isabs(ref) and os.path.exists(ref):
        return ref
    p = os.path.join(INPUT_DIR, ref)
    if os.path.exists(p):
        return p
    return None


def _needs_post_process(task_id: str) -> bool:
    """该任务是否需要口播 / 静音 / BGM 后处理。"""
    return (task_id in _task_voiceover
            or task_id in _task_noaudio
            or task_id in _task_bgm)


def _lookup_processed(task_id: str) -> Optional[str]:
    """查找已完成的静音/混流文件（内存 + muxed/ 磁盘）。优先 muted_，其次 muxed_。"""
    for name in (f"muted_{task_id}.mp4", f"muxed_{task_id}.mp4"):
        path = _muxed.get(name) or os.path.join(MUX_DIR, name)
        if os.path.isfile(path):
            _muxed[name] = path
            return name
    return None


def _output_phase(task_id: str) -> tuple:
    """返回 (processed_name_or_None, phase)。

    phase:
      skip    — 不需要后处理，用原片
      ready   — muted_/muxed_ 已就绪
      pending — 需要后处理但尚未完成
      failed  — ffmpeg 已失败，不再重试
    """
    if not _needs_post_process(task_id):
        return None, "skip"
    name = _lookup_processed(task_id)
    if name:
        return name, "ready"
    if task_id in _post_failed or task_id in _vo_done:
        return None, "failed"
    return None, "pending"


def _enqueue_post(task_id: str, idx: int, src_fn: str, subfolder: str) -> None:
    """入后台后处理队列；同一 task 只 put 一次，直到源文件缺失被允许重试。"""
    if task_id in _post_enqueued or task_id in _vo_done or task_id in _post_failed:
        return
    if _lookup_processed(task_id):
        return
    if _post_queue is None:
        return
    _post_enqueued.add(task_id)
    _post_queue.put_nowait((task_id, idx, src_fn, subfolder or ""))


def _collect_video_files(entry: dict) -> list:
    video_files = []
    for node_out in entry.get("outputs", {}).values():
        for out in node_out.get("images", []) + node_out.get("video", []):
            if isinstance(out, dict) and out.get("filename"):
                video_files.append(out)
    return video_files


def _public_base(request: Request) -> str:
    host = request.headers.get("host", "localhost:8000")
    return f"http://{host}"


def _video_url(base: str, filename: str, subfolder: str = "") -> str:
    url = f"{base}/api/v1/video/{filename}"
    if subfolder:
        url += f"?subfolder={subfolder}"
    return url


async def _post_process(task_id: str, idx: int, src_filename: str, subfolder: str) -> Optional[str]:
    """任务完成后按需后处理（口播混流 / 静音 / BGM），返回处理后文件名或 None。

    组合规则（no_audio=True 表示剥掉 H3 原生音效）：
      no_audio=True 且无口播且无 BGM -> 纯静音（剥掉原声）
      有口播 / 有 BGM -> 混流：原声（除非 no_audio）+ 口播 + BGM（按音量叠加、循环、淡出）

    源文件尚未落盘时不记入 _vo_done，下次轮询可重试；ffmpeg 失败才终结。
    """
    existing = _lookup_processed(task_id)
    if existing:
        _vo_done.add(task_id)
        return existing
    if task_id in _vo_done or task_id in _post_failed:
        return None
    text = voice = None
    if task_id in _task_voiceover:
        text, voice = _task_voiceover[task_id]
    mute = task_id in _task_noaudio
    bgm_ref, bgm_volume = _task_bgm.get(task_id, (None, 0.3))
    if not text and not mute and not bgm_ref:
        return None

    src = await _resolve_src(task_id, idx, src_filename, subfolder)
    if not src:
        # 原片可能刚落盘，放开入队标记以便下次轮询重试
        _post_enqueued.discard(task_id)
        return None

    inputs = ["ffmpeg", "-y", "-i", src]
    map_args = ["-map", "0:v", "-c:v", "copy"]

    if text:
        tts_path = os.path.join(MUX_DIR, f"tts_{task_id}.mp3")
        code, err = await _run_cmd([sys.executable, "-m", "edge_tts",
                                    "--text", text, "--voice", voice,
                                    "--write-media", tts_path])
        if code != 0 or not os.path.exists(tts_path):
            text = None  # TTS 失败则退化为不处理口播
            tts_path = None
    else:
        tts_path = None

    bgm_path = await _resolve_bgm(_workers[0], bgm_ref) if bgm_ref else None

    if mute and not tts_path and not bgm_path:
        # 纯静音：剥掉音轨
        cmd = inputs + map_args + ["-an"]
    else:
        # 至少一路音频（口播 / BGM / 原声），按需混流
        audio_sources, labels = [], []
        if not mute:
            audio_sources.append("[0:a]aformat=sample_rates=44100:channel_layouts=stereo")
            labels.append("[a0]")
        if tts_path:
            inputs += ["-i", tts_path]
            audio_sources.append("[1:a]aformat=sample_rates=44100:channel_layouts=stereo,adelay=400|400")
            labels.append("[a1]")
        if bgm_path:
            inputs += ["-stream_loop", "-1", "-i", bgm_path]
            bgm_idx = 1 + (1 if tts_path else 0)  # bgm 输入在 inputs 中的实际下标
            dur = await _probe_duration(src) or 10.0
            fade = min(2.0, dur * 0.2)
            audio_sources.append(
                f"[{bgm_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                f"volume={bgm_volume},atrim=0:{dur:.3f},asetpts=N/SR/TB,"
                f"afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade:.3f}")
            labels.append(f"[a{bgm_idx}]")
        mix_labels = "".join(labels)
        filter_cmd = (f"{';'.join(s + l for s, l in zip(audio_sources, labels))};"
                      f"{mix_labels}amix=inputs={len(labels)}"
                      f":duration={'longest' if bgm_path else 'first'}"
                      f":dropout_transition=0[aout]")
        out_label = "[aout]"
        if bgm_path:
            # BGM/口播可能长于视频，最终裁齐到视频时长，避免音轨拖尾
            filter_cmd += f";[aout]atrim=0:{dur:.3f},asetpts=N/SR/TB[fin]"
            out_label = "[fin]"
        cmd = (inputs + ["-filter_complex", filter_cmd,
                         "-map", "0:v", "-map", out_label,
                         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"])

    out_name = f"{'muted' if mute and not tts_path and not bgm_path else 'muxed'}_{task_id}.mp4"
    out_path = os.path.join(MUX_DIR, out_name)
    code, err = await _run_cmd(cmd + [out_path])
    if code != 0 or not os.path.exists(out_path):
        print(f"[post-process] {task_id} 后处理失败: rc={code} {err.decode(errors='replace')[:300]}",
              flush=True)
        _vo_done.add(task_id)
        _post_failed.add(task_id)
        return None
    _muxed[out_name] = out_path
    _vo_done.add(task_id)
    return out_name


async def _resolve_result_videos(task_id: str, idx: int, video_files: list, request: Request):
    """根据后处理阶段决定任务终态。

    返回 (status, extra)：
      processing + audio_processing — 需要后处理但尚未完成，不交原片 URL
      error — ffmpeg 已失败
      success + videos — 原片或不需等待的成片；ready 时带 audio_processed
    """
    if not video_files:
        return "success", {}

    first = video_files[0]
    src_fn = first.get("filename")
    sub = first.get("subfolder", "") or ""
    _video_worker[src_fn] = idx

    mux_name, phase = _output_phase(task_id)
    if phase == "pending":
        if _post_queue is not None:
            _enqueue_post(task_id, idx, src_fn, sub)
        else:
            mux_name = await _post_process(task_id, idx, src_fn, sub)
            if mux_name:
                phase = "ready"
            elif task_id in _post_failed:
                phase = "failed"

    if phase == "pending":
        return "processing", {"audio_processing": True}
    if phase == "failed":
        return "error", {"error": "audio post-process failed"}

    base = _public_base(request)
    if phase == "ready":
        urls = [_video_url(base, mux_name)]
        extra = {"videos": urls, "audio_processed": True}
    else:
        urls = [_video_url(base, src_fn, sub)]
        extra = {"videos": urls}
    for f in video_files[1:]:
        _video_worker[f.get("filename")] = idx
        extra["videos"].append(_video_url(base, f.get("filename"), f.get("subfolder", "") or ""))
    return "success", extra


@app.get("/health")
async def health():
    states = []
    for i in range(len(COMFY_WORKERS)):
        try:
            async with _workers[i].get(f"{_worker_url(i)}/system_stats") as resp:
                ok = resp.status == 200
        except Exception:
            ok = False
        states.append({
            "worker": i, "url": _worker_url(i), "ok": ok,
            "warmed": i in _warmed_workers,
        })
    all_ok = all(s["ok"] for s in states)
    all_warm = all(s["warmed"] for s in states)
    if not all_ok:
        status = "degraded"
    elif all_warm:
        status = "ok"
    else:
        status = "warming"
    return {"status": status, "workers": states}


@app.get("/api/v1/video/{filename}")
async def video_proxy(filename: str, subfolder: str = ""):
    """从产出该视频的 ComfyUI worker 代理输出视频文件，避免外部调用者直接访问 worker 端口。"""
    # 口播/静音后处理文件直接伺服本地磁盘（含重启后尚未登记进 _muxed 的）
    if filename in _muxed:
        return FileResponse(_muxed[filename], media_type="video/mp4")
    disk = os.path.join(MUX_DIR, filename)
    if filename.startswith(("muted_", "muxed_")) and os.path.isfile(disk):
        _muxed[filename] = disk
        return FileResponse(disk, media_type="video/mp4")
    from fastapi.responses import Response
    order = ([_video_worker[filename]] if filename in _video_worker
             else list(range(len(COMFY_WORKERS))))
    params = f"filename={filename}&type=output"
    if subfolder:
        params += f"&subfolder={subfolder}"
    for idx in order:
        async with _workers[idx].get(f"{_worker_url(idx)}/view?{params}") as resp:
            if resp.status == 200:
                content = await resp.read()
                return Response(content, media_type=resp.headers.get("Content-Type", "video/mp4"))
    raise HTTPException(status_code=404, detail="video not found")


@app.post("/api/v1/generate", status_code=202)
async def generate(req: GenerateRequest):
    task_id, idx = await _submit_job(req.prompt, req.width, req.height,
                                     req.duration, req.seed, req.steps,
                                     req.first_frame or req.image, req.last_frame,
                                     req.voiceover, req.voice, req.no_audio,
                                     req.bgm, req.bgm_volume)
    return {"task_id": task_id,
            "status": "queued",
            "worker": idx,
            "frame_count": frame_count_for_duration(req.duration),
            "duration": req.duration}


@app.get("/api/v1/task/{task_id}")
async def task_status(task_id: str, request: Request):
    idx = _worker_assign.get(task_id, 0)
    async with _workers[idx].get(f"{_worker_url(idx)}/history/{task_id}") as resp:
        history = await resp.json()
    entry = history.get(task_id)
    if entry is None:
        return {"task_id": task_id, "status": "queued"}

    status = entry.get("status", {})
    completed = status.get("completed")
    status_str = "success" if completed else "error"

    result = {
        "task_id": task_id,
        "status": status_str,
        "progress": None,
    }
    if completed:
        video_files = _collect_video_files(entry)
        if video_files:
            phase, extra = await _resolve_result_videos(task_id, idx, video_files, request)
            result["status"] = phase
            result.update(extra)
    if status.get("messages"):
        for msg in status["messages"]:
            if isinstance(msg, list) and len(msg) == 2 and isinstance(msg[1], dict):
                if msg[0] == "execution_error":
                    result["status"] = "error"
                    result["error"] = msg[1].get("message", str(msg[1]))
    return result


# ==================== OpenAI 兼容端点 ====================
# ClipForge 的 CustomOpenAIProvider 期望以下端点（OpenAI 兼容格式）：
#   GET  /models              -> { data: [{ id, name }] }
#   POST /videos              -> { id, status }（异步）
#   POST /videos/generations  -> 同上（兼容旧版路径）
#   GET  /videos/glm_5.2_ark_toC        -> { id, status, data: [{ url }] }
#   GET  /videos/glm_5.2_ark_toC/content -> 302 重定向到视频文件
# 这些端点与上面的 /api/v1/* 端点共存，内部复用相同的 ComfyUI 提交/查询逻辑。

RESOLUTION_MAP = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
}


@app.get("/models")
async def openai_list_models():
    """OpenAI 兼容的模型列表端点。"""
    return {"data": [{"id": "minimax-h3", "name": "MiniMax-H3", "type": "video"}]}


@app.post("/images/generations")
async def openai_image_generate():
    """MiniMax-H3 是视频模型，不支持图片生成。"""
    raise HTTPException(status_code=400, detail="MiniMax-H3 是视频生成模型，不支持图片生成")


@app.post("/videos")
@app.post("/videos/generations")
async def openai_video_generate(request: Request):
    """OpenAI 兼容的视频生成端点，接受灵活的请求体。

    支持图生视频：input 数组（OpenAI 风格，第 1 个为首帧、第 2 个为尾帧），
    或 first_frame / last_frame / image 字段，值为 http(s) URL 或 data: URI。
    传 reference_images（数组）时走 ref2va 路径：多张参考图全程参与 conditioning
    （人物定妆照 + 分镜关键帧的身份锚定），prompt 中的 @ImageN / @图片N 引用会被
    自动重写为 ref2va 的 <Picture N> 语法。
    """
    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # 首帧/尾帧解析：input 数组支持字符串（OpenAI 风格）与 {type:image_url} 对象，
    # 第 1 个为首帧、第 2 个为尾帧；也可用 first_frame / last_frame / image 字段
    inputs = body.get("input")
    first_frame = body.get("first_frame") or body.get("image")
    last_frame = body.get("last_frame")
    if isinstance(inputs, list):
        urls = []
        for u in inputs:
            if isinstance(u, str):
                urls.append(u)
            elif (isinstance(u, dict) and u.get("type") == "image_url"
                  and isinstance(u.get("image_url"), dict)
                  and isinstance(u["image_url"].get("url"), str)
                  and u["image_url"]["url"]):
                urls.append(u["image_url"]["url"])
        if urls and not first_frame:
            first_frame = urls[0]
        if len(urls) > 1 and not last_frame:
            last_frame = urls[1]
    if not isinstance(first_frame, str):
        first_frame = None
    if not isinstance(last_frame, str):
        last_frame = None

    # OpenAI 兼容补丁：openai 适配器图生视频发 input_reference.image_url（值是
    # data URI 字符串；也接受 {url: "..."} 对象风格）。_fetch_frame_bytes 原生
    # 支持 http(s) URL / data URI，这里只做字段搬运，不重复校验。
    input_ref = body.get("input_reference")
    if isinstance(input_ref, dict):
        ref_url = input_ref.get("image_url")
        if isinstance(ref_url, dict):
            ref_url = ref_url.get("url")
        if isinstance(ref_url, str) and ref_url.strip() and not first_frame:
            first_frame = ref_url.strip()

    # 多参考图（ref2va 路径）：定妆照 + 各分镜关键帧作为人物/商品身份锚点。
    # 与 input 不同，它们不是首尾帧，而是全程参与 conditioning（<Picture N> 引用）。
    reference_images = body.get("reference_images") or body.get("referenceImageUrls")
    refs = []
    if isinstance(reference_images, list):
        refs = [u for u in reference_images if isinstance(u, str) and u.strip()][:9]

    # 分辨率解析：优先 width/height，其次 resolution token；图生视频缺省时由图片自适应
    w = h = None
    if body.get("width") and body.get("height"):
        w, h = int(body["width"]), int(body["height"])
    elif body.get("resolution") and body["resolution"] in RESOLUTION_MAP:
        w, h = RESOLUTION_MAP[body["resolution"]]
    # OpenAI 兼容补丁：适配器发 sora 白名单 size（"720x1280"/"1280x720"/
    # "1024x1792"/"1792x1024"）。解析出 w/h 后交给下方 adapt_canvas_size 对齐 32
    # 倍数（H3 latent 硬约束：720/32=22.5 不可整除，直接用会 RuntimeError）。
    size_str = body.get("size")
    if (isinstance(size_str, str) and "x" in size_str.lower()
            and not (body.get("width") and body.get("height"))):
        try:
            sw, sh = size_str.lower().split("x", 1)
            w, h = int(sw), int(sh)
        except ValueError:
            pass
    if w and h:
        # H3 的 latent 要求画布尺寸能被 32 整除（patchify 按 patch 2 切分），而标准
        # 1080p/720p/480p（1920x1080 / 1280x720 / 854x480）在短边都不满足：
        # 1080/32=33.75、720/32=22.5 → SamplerCustomAdvanced 直接 RuntimeError。
        # 与原生 /api/v1/generate 一致，先套用 adapt_canvas_size 对齐到兼容画布。
        w, h = adapt_canvas_size(w, h)

    # OpenAI 兼容补丁：适配器发 seconds（档位 4/8/12，客户端 durationSeconds 已被
    # 舍入——传 1~6.4 发 4，6.5~10.4 发 8，>=10.5 发 12）。本工作区标准 5s、
    # 上限 10s，故做档位重映射：4→5（标准）、8→8、12→10（clamp 到 MAX_DURATION）。
    # duration 字段直传时按原值处理（仍受 clamp 保护），仅 OpenAI 档位值走映射表。
    OPENAI_SECONDS_REMAP = {4: 5.0, 8: 8.0, 12: 10.0}
    raw_duration = body.get("duration", body.get("seconds", 5.0))
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        duration = 5.0
    if "duration" not in body and duration in OPENAI_SECONDS_REMAP:
        duration = OPENAI_SECONDS_REMAP[duration]
    duration = max(1.0, min(duration, MAX_DURATION))
    seed = int(body.get("seed", 0))
    steps = int(body.get("steps", DEFAULT_STEPS))
    voiceover = body.get("voiceover")
    voice = body.get("voice")
    if not isinstance(voiceover, str) or not voiceover.strip():
        voiceover = None
    if not isinstance(voice, str):
        voice = None
    # OpenAI 兼容补丁：/videos 端点默认静音——CF 流水线（faceless 等 skill）配音/
    # BGM 全在后期自己做，H3 原生音效混进去是噪音。要原声显式传 no_audio: false。
    # 仅改本端点默认值，/api/v1/generate 原生入口保持 no_audio 默认 False 不变。
    no_audio = bool(body.get("no_audio", True))
    bgm = body.get("bgm")
    if not isinstance(bgm, str) or not bgm.strip():
        bgm = None
    try:
        bgm_volume = float(body.get("bgm_volume", 0.3))
    except (TypeError, ValueError):
        bgm_volume = 0.3
    bgm_volume = max(0.0, min(1.0, bgm_volume))

    if refs:
        # 故事版整片 prompt 用 @ImageN / @图片N 引用参考图（Seedance 风格），ref2va
        # 的引用语法是 <Picture N>——按参考图顺序原位重写，让身份锚定真正生效
        prompt = re.sub(r"@(?:Image|图片)(\d+)",
                        lambda m: f"<Picture {int(m.group(1))}>", prompt)
        # 身份保真优先：默认按 2048px 短边高保真编码参考图（更贴近定妆照，但慢数倍）；
        # 客户端可传 "match"（按生成画布像素面积等比缩放，更快）覆盖
        ref_image_size = body.get("ref_image_size")
        if ref_image_size not in ("match", "max"):
            ref_image_size = "max"
        task_id, _ = await _submit_ref2v_job(prompt, w, h, duration, seed, steps, refs,
                                             ref_image_size, voiceover, voice, no_audio,
                                             bgm, bgm_volume)
    else:
        task_id, _ = await _submit_job(prompt, w, h, duration, seed, steps,
                                       first_frame, last_frame, voiceover, voice, no_audio,
                                       bgm, bgm_volume)
    # OpenAI 兼容补丁：回显档位信息（秒数 + 帧数），便于客户端/冒烟核对
    # 档位重映射是否生效（seconds 4 → duration 5 → 124 帧 @24fps）。
    return {"id": task_id, "status": "queued",
            "duration": duration,
            "seconds": body.get("seconds"),
            "frame_count": frame_count_for_duration(duration)}


@app.get("/videos/{task_id}")
async def openai_video_status(task_id: str, request: Request):
    """OpenAI 兼容的任务状态查询。"""
    idx = _worker_assign.get(task_id, 0)
    try:
        async with _workers[idx].get(f"{_worker_url(idx)}/history/{task_id}") as resp:
            history = await resp.json()
    except Exception:
        return {"id": task_id, "status": "queued"}

    entry = history.get(task_id)
    if entry is None:
        return {"id": task_id, "status": "queued"}

    status = entry.get("status", {})
    completed = status.get("completed")

    if completed:
        video_files = _collect_video_files(entry)
        if video_files:
            phase, extra = await _resolve_result_videos(task_id, idx, video_files, request)
            if phase == "processing":
                return {"id": task_id, "status": "processing"}
            if phase == "error":
                # OpenAI 兼容：error 用 {message} 对象结构，openai 适配器读
                # error.message 才能拿到失败明细（纯字符串会显示通用文案）
                return {"id": task_id, "status": "failed",
                        "error": {"message": extra.get("error", "audio post-process failed")}}
            result = {"id": task_id, "status": "completed"}
            if extra.get("videos"):
                result["data"] = [{"url": u} for u in extra["videos"]]
            if extra.get("audio_processed"):
                result["audio_processed"] = True
            return result
        return {"id": task_id, "status": "completed"}

    # 检查是否有执行错误
    if status.get("messages"):
        for msg in status["messages"]:
            if isinstance(msg, list) and len(msg) == 2 and isinstance(msg[1], dict):
                if msg[0] == "execution_error":
                    return {"id": task_id, "status": "failed",
                            "error": {"message": msg[1].get("message", str(msg[1]))}}

    # ComfyUI 中任务存在但未完成 -> 仍在处理
    return {"id": task_id, "status": "processing"}


@app.get("/videos/{task_id}/content")
async def openai_video_content(task_id: str, request: Request):
    """OpenAI 兼容的视频下载端点：302 重定向到实际视频文件。"""
    idx = _worker_assign.get(task_id, 0)
    async with _workers[idx].get(f"{_worker_url(idx)}/history/{task_id}") as resp:
        history = await resp.json()
    entry = history.get(task_id)
    if not entry or not entry.get("status", {}).get("completed"):
        raise HTTPException(status_code=404, detail="video not ready")

    video_files = _collect_video_files(entry)
    if not video_files:
        raise HTTPException(status_code=404, detail="video not found")

    first = video_files[0]
    _video_worker[first["filename"]] = idx
    processed = _lookup_processed(task_id)
    if processed:
        return RedirectResponse(url=_video_url(_public_base(request), processed),
                                status_code=302)
    if _needs_post_process(task_id):
        _enqueue_post(task_id, idx, first.get("filename"), first.get("subfolder", "") or "")
        raise HTTPException(status_code=404, detail="video not ready")

    return RedirectResponse(
        url=_video_url(_public_base(request), first["filename"], first.get("subfolder", "") or ""),
        status_code=302)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
