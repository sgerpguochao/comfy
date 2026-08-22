"""MiniMax-H3 视频生成 API 封装服务。

包装多个 ComfyUI 实例（每个实例绑定一张 GPU）的 /prompt API，对外提供简洁的 HTTP 接口：
  POST /api/v1/generate  提交文生视频任务（自动分发给队列最短的 GPU 实例）
  GET  /api/v1/task/{id} 查询任务状态
  GET  /health           健康检查

多 GPU 部署：通过环境变量 COMFY_HOSTS 指定所有 ComfyUI 实例（逗号分隔的 host:port），
默认值为两个 worker（127.0.0.1:8188,127.0.0.1:8189），每个实例由 --cuda-device 绑定一张 GPU。
"""

import asyncio
import json
import os
import time
import uuid

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
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
MAX_DURATION = 20  # 模型训练区间 ~5-15s；>15s 未经验证，但节点支持到更长，按需放开


def frame_count_for_duration(duration: float) -> int:
    """duration(秒) -> 帧数，对齐到模型 17k+5 帧网格。"""
    n = max(5, round(duration * DEFAULT_FPS))
    return n + (5 - n % 17) % 17


def build_prompt(prompt: str, width: int, height: int, duration: float,
                 seed: int, steps: int) -> dict:
    """构造 ComfyUI API 格式的 prompt（与官方 T2V 工作流等价，使用完整 INT8 模型）。"""
    length = frame_count_for_duration(duration)
    return {
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
               "inputs": {"video": ["13", 0], "filename_prefix": "video/minimax_h3",
                          "format": "mp4", "codec": "auto"}},
    }


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="视频提示词（含画面 + 运镜 + 音频描述）")
    width: int = Field(1344, ge=32, le=2048)
    height: int = Field(768, ge=32, le=2048)
    duration: float = Field(5.0, ge=1.0, le=MAX_DURATION, description="视频时长（秒）")
    seed: int = Field(0, ge=0, le=2 ** 64 - 1)
    steps: int = Field(DEFAULT_STEPS, ge=1, le=100)


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
_submit_lock = asyncio.Lock()  # 串行化“选 worker + 提交”，避免并发提交时队列长度读到旧值


def _worker_url(i: int) -> str:
    return f"http://{COMFY_WORKERS[i]}"


@app.on_event("startup")
async def startup():
    global _workers
    _workers = [aiohttp.ClientSession() for _ in COMFY_WORKERS]


@app.on_event("shutdown")
async def shutdown():
    for s in _workers:
        await s.close()


async def _pick_worker() -> int:
    """选择队列（运行中+排队中）最短的 worker；全部不可达时按轮询兜底。"""
    global _rr
    best, best_load = 0, 10 ** 9
    for i in range(len(COMFY_WORKERS)):
        try:
            async with _workers[i].get(f"{_worker_url(i)}/queue") as resp:
                data = await resp.json()
            load = len(data.get("queue_running", [])) + len(data.get("queue_pending", []))
        except Exception:
            load = 10 ** 9
        if load < best_load:
            best, best_load = i, load
    if best_load == 10 ** 9:
        best = _rr % len(COMFY_WORKERS)
    _rr += 1
    return best


@app.get("/health")
async def health():
    states = []
    for i in range(len(COMFY_WORKERS)):
        try:
            async with _workers[i].get(f"{_worker_url(i)}/system_stats") as resp:
                states.append({"worker": i, "url": _worker_url(i), "ok": resp.status == 200})
        except Exception:
            states.append({"worker": i, "url": _worker_url(i), "ok": False})
    return {"status": "ok" if all(s["ok"] for s in states) else "degraded", "workers": states}


@app.get("/api/v1/video/{filename}")
async def video_proxy(filename: str, subfolder: str = ""):
    """从产出该视频的 ComfyUI worker 代理输出视频文件，避免外部调用者直接访问 worker 端口。"""
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
    prompt_graph = build_prompt(req.prompt, req.width, req.height,
                                req.duration, req.seed, req.steps)
    task_id = str(uuid.uuid4())
    async with _submit_lock:
        idx = await _pick_worker()
        payload = {"prompt": prompt_graph, "client_id": "api-wrapper", "prompt_id": task_id}
        async with _workers[idx].post(f"{_worker_url(idx)}/prompt", json=payload) as resp:
            body = await resp.json()
            if resp.status != 200:
                raise HTTPException(status_code=502, detail=body)
    _worker_assign[task_id] = idx
    return {"task_id": body.get("prompt_id", task_id),
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

    video_files = []
    if completed:
        for node_out in entry.get("outputs", {}).values():
            for out in node_out.get("images", []) + node_out.get("video", []):
                if isinstance(out, dict) and out.get("filename"):
                    video_files.append(out)

    result = {
        "task_id": task_id,
        "status": status_str,
        "progress": None,
    }
    if video_files:
        host = request.headers.get("host", "localhost:8000")
        base = f"http://{host}"
        result["videos"] = []
        for f in video_files:
            _video_worker[f.get("filename")] = idx
            url = f"{base}/api/v1/video/{f.get('filename')}"
            if f.get("subfolder"):
                url += f"?subfolder={f['subfolder']}"
            result["videos"].append(url)
    if status.get("messages"):
        for msg in status["messages"]:
            if isinstance(msg, list) and len(msg) == 2 and isinstance(msg[1], dict):
                if msg[0] == "execution_error":
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
    """OpenAI 兼容的视频生成端点，接受灵活的请求体。"""
    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # 分辨率解析：优先 width/height，其次 resolution token，最后默认值
    if body.get("width") and body.get("height"):
        w, h = int(body["width"]), int(body["height"])
    elif body.get("resolution") and body["resolution"] in RESOLUTION_MAP:
        w, h = RESOLUTION_MAP[body["resolution"]]
    else:
        w, h = 1344, 768

    duration = float(body.get("duration", 5.0))
    duration = max(1.0, min(duration, MAX_DURATION))
    seed = int(body.get("seed", 0))
    steps = int(body.get("steps", DEFAULT_STEPS))

    prompt_graph = build_prompt(prompt, w, h, duration, seed, steps)
    task_id = str(uuid.uuid4())
    async with _submit_lock:
        idx = await _pick_worker()
        payload = {"prompt": prompt_graph, "client_id": "api-wrapper", "prompt_id": task_id}
        async with _workers[idx].post(f"{_worker_url(idx)}/prompt", json=payload) as resp:
            resp_body = await resp.json()
            if resp.status != 200:
                raise HTTPException(status_code=502, detail=resp_body)
    _worker_assign[task_id] = idx
    return {"id": resp_body.get("prompt_id", task_id), "status": "queued"}


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
        result = {"id": task_id, "status": "completed"}
        video_files = []
        for node_out in entry.get("outputs", {}).values():
            for out in node_out.get("images", []) + node_out.get("video", []):
                if isinstance(out, dict) and out.get("filename"):
                    video_files.append(out)

        if video_files:
            host = request.headers.get("host", "localhost:8000")
            base = f"http://{host}"
            data = []
            for f in video_files:
                _video_worker[f.get("filename")] = idx
                url = f"{base}/api/v1/video/{f.get('filename')}"
                if f.get("subfolder"):
                    url += f"?subfolder={f['subfolder']}"
                data.append({"url": url})
            result["data"] = data
        return result

    # 检查是否有执行错误
    if status.get("messages"):
        for msg in status["messages"]:
            if isinstance(msg, list) and len(msg) == 2 and isinstance(msg[1], dict):
                if msg[0] == "execution_error":
                    return {"id": task_id, "status": "failed",
                            "error": msg[1].get("message", str(msg[1]))}

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

    for node_out in entry.get("outputs", {}).values():
        for out in node_out.get("images", []) + node_out.get("video", []):
            if isinstance(out, dict) and out.get("filename"):
                _video_worker[out["filename"]] = idx
                host = request.headers.get("host", "localhost:8000")
                url = f"http://{host}/api/v1/video/{out['filename']}"
                if out.get("subfolder"):
                    url += f"?subfolder={out['subfolder']}"
                return RedirectResponse(url=url, status_code=302)

    raise HTTPException(status_code=404, detail="video not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
