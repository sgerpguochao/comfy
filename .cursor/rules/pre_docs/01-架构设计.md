# 架构 Spec — 分层与模块职责

## 1. 分层视图

```
┌─────────────────────────────────────────────────┐
│ L4 调用层    client_example.py / ClipForge / bench│
├─────────────────────────────────────────────────┤
│ L3 网关层    api_server.py (FastAPI :8000)        │
│             · 路由 /api/v1/* + OpenAI 兼容        │
│             · Pydantic 请求校验                   │
│             · CORS                               │
├─────────────────────────────────────────────────┤
│ L2 编排层    _pick_worker / _submit_lock /        │
│             _worker_assign / _video_worker        │
├─────────────────────────────────────────────────┤
│ L1 引擎层    ComfyUI × N (每实例一 GPU)           │
│             /prompt /queue /history /view /system_stats │
├─────────────────────────────────────────────────┤
│ L0 资源层    models/ 63GB · venv · wheels · logs  │
└─────────────────────────────────────────────────┘
```

## 2. api_server.py 模块解剖

### 2.1 配置常量

| 常量 | 值 | 说明 |
|---|---|---|
| `COMFY_WORKERS` | env `COMFY_HOSTS`，默认 `127.0.0.1:8188,127.0.0.1:8189` | worker 地址列表 |
| `DIFFUSION_MODEL` | `minimax_h3_fl2va_int8_convrot.safetensors` | 32 GB INT8 DiT |
| `TEXT_ENCODER` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 26 GB 文本编码器 |
| `VIDEO_VAE` / `AUDIO_VAE` | fp16 4.9GB / fp32 578MB | 视频/音频 VAE |
| `DEFAULT_STEPS` | 10 | 速度/质量折中（20 步慢 ~1.7×） |
| `DEFAULT_FPS` | 24 | 与节点 `FPS=24` 一致 |
| `MAX_DURATION` | 20 s | 训练区间 5–15s 的保守外推 |

### 2.2 请求模型（Pydantic）

```python
GenerateRequest:
  prompt:    str          # 必填
  width:     int = 1344   # 32..2048
  height:    int = 768    # 32..2048
  duration:  float = 5.0  # 1.0..20.0
  seed:      int = 0      # 0..2^64-1
  steps:     int = 10     # 1..100
```

### 2.3 运行时状态（进程内，非持久化）

| 变量 | 类型 | 生命周期 | 用途 |
|---|---|---|---|
| `_workers` | `list[aiohttp.ClientSession]` | startup 创建，shutdown 关闭 | 与 COMFY_WORKERS 一一对应 |
| `_worker_assign` | `dict[task_id → int]` | 进程内存 | 查询状态时定位 worker；**重启即失** |
| `_video_worker` | `dict[filename → int]` | 进程内存 | 视频代理时定位 worker |
| `_rr` | `int` | 进程内存 | 全 worker 不可达时的轮询兜底 |
| `_submit_lock` | `asyncio.Lock` | 全局 | 串行化「读队列 + 提交」，避免竞态 |

> ⚠️ **单点约束**：`_worker_assign`/`_video_worker` 是内存字典，API 进程重启后旧 task_id 将 fallback 到 worker 0 查询（`api_server.py:192` `idx = _worker_assign.get(task_id, 0)`），跨 worker 的旧任务会误报 "queued"。生产化需外置 Redis。

### 2.4 负载均衡算法

```python
async def _pick_worker() -> int:
    for i in workers:                      # 并发查 /queue
        load = running + pending
    选 load 最小的
    若全部不可达 → _rr 轮询兜底
```

- 粒度：**实例级**（不感知 GPU 显存，只看队列长度）
- 失败语义：单 worker 挂掉时 load 视为 ∞，自然被绕开
- 提交路径全程持 `_submit_lock`，代价是提交吞吐 ≤ 1/RTT

### 2.5 两套 API 的关系

| | `/api/v1/*`（原生） | `/videos*`、`/models`（OpenAI 兼容） |
|---|---|---|
| 请求校验 | Pydantic 严格校验 | 手动 `request.json()` 宽松解析 |
| 分辨率 | 显式 width/height | width/height 或 `resolution ∈ {1080p,720p,480p}` |
| 状态语义 | `queued/success/error` | `queued/processing/completed/failed` |
| 返回结构 | `{task_id, status, worker, frame_count, duration}` | `{id, status[, data:[{url}]]}` |
| 视频下载 | `GET /api/v1/video/{filename}` | `GET /videos/{id}/content` → 302 → 上述端点 |

OpenAI 兼容层复用 `build_prompt()` + `_pick_worker()` + `_submit_lock`，仅是**协议适配层**，无独立状态。

## 3. ComfyUI 侧关键扩展（`ComfyUI/comfy_extras/`）

### 3.1 `nodes_minimax_h3.py` — H3 专用节点

| 节点 | 作用 |
|---|---|
| `MiniMaxH3ImageToVideo` | t2va/fl2va：prompt(+可选首尾帧) → conditioning + AV latent |
| `MiniMaxH3ReferenceToVideo` | ref2va：`<Picture i>/<Video k>/<Audio j>` 引用式生成 |
| `EmptyMiniMaxH3LatentAV` | 空 AV latent（纯工作流编辑用） |

硬约束（`nodes_minimax_h3.py:25-30`）：

```
CANVAS_MULTIPLE = 32          # 宽高对齐
BASE_SHORT_EDGE = 768         # 短边基准
MAX_PIXELS      = 768*1344    # 画布面积上限
FPS             = 24
AUDIO_LATENT_FPS = 40
帧网格: length % 17 == 5      # align_frame_count()
视频 latent: [B,24,T,H/16,W/16]   音频 latent: [B,32,2,T40]
```

### 3.2 `nodes_video.py` / `nodes_audio.py`

- `CreateVideo`：images + audio → video 对象（fps、bit_depth）
- `SaveVideo`：`format=mp4, codec=auto`，写 `output/video/minimax_h3*.mp4`
- `VAEDecodeAudio`：从 AV latent 解出音轨

## 4. 数据流时序（成功路径）

```
Client          API(:8000)          Worker(:8188)        GPU
  │ POST /api/v1/generate │                │              │
  ├──────────────►│       │                │              │
  │               │ lock + pick_worker     │              │
  │               ├──GET /queue──────────►│              │
  │               │◄───running+pending─────┤              │
  │               ├──POST /prompt─────────►│ 入队          │
  │ 202 {task_id}│◄───prompt_id────────────┤              │
  │◄──────────────┤       │                │  DiT 采样     │
  │  (poll)       │       │                │  (10 steps)  │
  ├─GET /task/id─►│       │                │              │
  │               ├──GET /history/{id}────►│              │
  │               │◄──outputs{filename}────┤              │
  │ {status:success, videos:[url]}          │              │
  │◄──────────────┤       │                │              │
  │ GET /api/v1/video/{filename}            │              │
  ├──────────────►├──GET /view─────────────►│ 读 mp4        │
  │◄──video/mp4───┤◄───────────────────────┤              │
```

## 5. 故障路径

| 故障 | 现象 | 当前行为 |
|---|---|---|
| worker 全挂 | `_pick_worker` 全 ∞ | 轮询兜底 → POST 失败 → 502 |
| 单 worker 挂 | 其 load=∞ | 自动绕开，不影响提交 |
| 节点执行错 | history `messages` 含 `execution_error` | 状态 `error` + 透传 message |
| 提交 4xx/5xx | `/prompt` 非 200 | HTTPException 502 + ComfyUI 原始 body |
| API 重启 | `_worker_assign` 清空 | 旧任务查询 fallback worker 0（见 2.3 ⚠️） |
