# API Reference Spec — HTTP 端点契约

Base URL：`http://<host>:8000`（本机 `127.0.0.1:8000`，公网默认 `117.50.216.253:8000`）

## 1. 原生端点 `/api/v1/*`

### 1.1 `POST /api/v1/generate` — 提交文生视频任务

**请求体**（`application/json`）：

| 字段 | 类型 | 必填 | 默认 | 约束 | 说明 |
|---|---|---|---|---|---|
| `prompt` | string | ✅ | — | — | 提示词（画面+运镜+音频描述） |
| `width` | int | — | 1344 | 32–2048 | 会被对齐到 32 的倍数 |
| `height` | int | — | 768 | 32–2048 | 同上 |
| `duration` | float | — | 5.0 | 1.0–20.0 | 秒；帧数对齐 17k+5 网格 |
| `seed` | int | — | 0 | 0–2⁶⁴⁻¹ | 0 时由 ComfyUI 随机 |
| `steps` | int | — | 10 | 1–100 | 采样步数 |

**响应 202**：

```json
{
  "task_id": "uuid4",
  "status": "queued",
  "worker": 0,
  "frame_count": 124,
  "duration": 5.0
}
```

**错误**：
- `422` Pydantic 校验失败（字段越界/缺 prompt）
- `502` ComfyUI `/prompt` 拒绝（body 为 ComfyUI 原始错误）

**curl 示例**：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"海浪拍打礁石，清晨金色阳光","width":832,"height":480,"duration":4,"steps":10}'
```

### 1.2 `GET /api/v1/task/{task_id}` — 查询任务状态

**响应**（状态机：`queued → success | error`）：

```json
// 排队/执行中
{"task_id": "...", "status": "queued", "progress": null}

// 成功
{
  "task_id": "...",
  "status": "success",
  "progress": null,
  "videos": ["http://<request-host>/api/v1/video/minimax_h3_00001.mp4"]
}

// 失败
{"task_id": "...", "status": "error", "progress": null, "error": "OOM ..."}
```

> `videos[].url` 的 host 取自**请求头 Host**，所以经反向代理时需保证 Host 正确传递，或改用配置化 base URL（当前实现的局限）。
> `progress` 字段恒为 `null`——预留位，尚未接 ComfyUI 的 `/ws` 进度事件。

### 1.3 `GET /api/v1/video/{filename}` — 视频文件代理

| Query | 说明 |
|---|---|
| `subfolder` | 可选，输出子目录（如 `video`） |

行为：优先查 `_video_worker[filename]` 命中的 worker，miss 时遍历全部 worker 的 `/view`。命中返回 `200 video/mp4`（Content-Type 透传），全 miss 返回 `404`。

### 1.4 `GET /health` — 健康检查

```json
{
  "status": "ok" | "degraded",
  "workers": [
    {"worker": 0, "url": "http://127.0.0.1:8188", "ok": true},
    {"worker": 1, "url": "http://127.0.0.1:8189", "ok": true}
  ]
}
```

探测方式：并发 GET 各 worker `/system_stats`，全部 200 → `ok`。

## 2. OpenAI 兼容端点

> 面向 ClipForge `CustomOpenAIProvider`。与原生端点共存、复用同一编排内核。

### 2.1 `GET /models`

```json
{"data": [{"id": "minimax-h3", "name": "MiniMax-H3", "type": "video"}]}
```

### 2.2 `POST /videos` | `POST /videos/generations`

**请求体**（宽松解析，非 Pydantic）：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | string | 必填 | — |
| `width`/`height` | int | — | 优先级最高 |
| `resolution` | string | — | `1080p`(1920×1080) / `720p`(1280×720) / `480p`(854×480) |
| `duration` | number | 5.0 | 自动 clamp 到 [1, 20] |
| `seed` | int | 0 | — |
| `steps` | int | 10 | — |

分辨率解析优先级：`width+height` > `resolution` > 默认 `1344×768`。

**响应 200**：`{"id": "<prompt_id>", "status": "queued"}`

### 2.3 `GET /videos/{task_id}` — 状态查询

状态机：`queued → processing → completed | failed`

```json
{"id": "...", "status": "processing"}
{"id": "...", "status": "completed", "data": [{"url": "http://.../api/v1/video/xxx.mp4"}]}
{"id": "...", "status": "failed", "error": "..."}
```

### 2.4 `GET /videos/{task_id}/content` — 视频下载

- 完成 → `302` 重定向到 `/api/v1/video/{filename}`
- 未完成/不存在 → `404 {"detail": "video not ready"}`

### 2.5 `POST /images/generations` — 显式不支持

恒返回 `400 {"detail": "MiniMax-H3 是视频生成模型，不支持图片生成"}`。

## 3. 上游 ComfyUI API（网关内部依赖）

网关调用 worker 的端点（供排障时直连 worker 用）：

| 端点 | 方法 | 用途 |
|---|---|---|
| `/prompt` | POST | 提交 graph `{prompt, client_id, prompt_id}` |
| `/queue` | GET | `queue_running` + `queue_pending` 长度 → 负载 |
| `/history/{prompt_id}` | GET | 任务状态与输出文件 |
| `/view?filename=&type=output[&subfolder=]` | GET | 读输出文件 |
| `/system_stats` | GET | 健康探测 |

`/history` 响应中网关关心的字段：

```
history[task_id].status.completed            # bool，完成标志
history[task_id].status.messages[]           # ["execution_error", {...}] 等
history[task_id].outputs.*.images[]/.video[] # {filename, subfolder, ...}
```

## 4. 横切约定

| 项 | 现状 |
|---|---|
| 鉴权 | 无（任何可达者可提交/拉视频） |
| CORS | `allow_origins=["*"]` |
| 限流 | 无 |
| 幂等 | 无（同 prompt 重复提交会触发 ComfyUI 缓存去重，seed 不同则完全独立） |
| 超时 | aiohttp 默认（5min total）；`/prompt` 无显式超时 |
| 日志 | uvicorn access log（stdout → PM2/tmux log） |
| 错误格式 | FastAPI 默认 `{"detail": ...}` |
