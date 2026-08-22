# ComfyUI Workflow Spec — 14 节点 MiniMax-H3 T2V Graph

> 源码：`api_server.py::build_prompt()`（与 `test_h3.py::build_prompt()` 等价，仅 steps 默认值不同）

## 1. Graph 总览

```
UNETLoader(1) ─┬─► BasicScheduler(7) ─┐
               └─► BasicGuider(8) ────┤
CLIPLoader(2) ──► MiniMaxH3 ─┬────────┤► SamplerCustomAdvanced(10) ─┬─► VAEDecode(11) ─┐
VAELoader(3) ───► ImageTo(5) ├────────┘                              │                  ├─► CreateVideo(13) ─► SaveVideo(14)
VAELoader(4) ─────────────────┘                                      └─► VAEDecodeAudio(12) ┘
                              RandomNoise(9) ────────────────────────┘
```

## 2. 逐节点定义

| # | class_type | 关键 inputs | 输出流向 |
|---|---|---|---|
| 1 | `UNETLoader` | `unet_name=minimax_h3_fl2va_int8_convrot.safetensors`, `weight_dtype=default` | → 7, 8 |
| 2 | `CLIPLoader` | `clip_name=qwen3vl_32b_minimax_h3_int8_convrot.safetensors`, `type=minimax`, `device=default` | → 5 |
| 3 | `VAELoader` | `vae_name=minimax_h3_video_vae_fp16.safetensors` | → 5, 11 |
| 4 | `VAELoader` | `vae_name=minimax_h3_audio_vae_fp32.safetensors` | → 12 |
| 5 | `MiniMaxH3ImageToVideo` | `clip=[2,0]`, `vae=[3,0]`, `prompt`, `width`, `height`, `length` | cond→8, latent→10 |
| 6 | `KSamplerSelect` | `sampler_name=res_multistep` | → 10 |
| 7 | `BasicScheduler` | `model=[1,0]`, `scheduler=simple`, `steps`, `denoise=1.0` | → 10 |
| 8 | `BasicGuider` | `model=[1,0]`, `conditioning=[5,0]` | → 10 |
| 9 | `RandomNoise` | `noise_seed=<seed>` | → 10 |
| 10 | `SamplerCustomAdvanced` | `noise=[9,0]`, `guider=[8,0]`, `sampler=[6,0]`, `sigmas=[7,0]`, `latent_image=[5,1]` | → 11, 12 |
| 11 | `VAEDecode` | `samples=[10,0]`, `vae=[3,0]` | → 13 |
| 12 | `VAEDecodeAudio` | `samples=[10,0]`, `vae=[4,0]` | → 13 |
| 13 | `CreateVideo` | `images=[11,0]`, `audio=[12,0]`, `fps=24`, `bit_depth=8` | → 14 |
| 14 | `SaveVideo` | `video=[13,0]`, `filename_prefix=video/minimax_h3`, `format=mp4`, `codec=auto` | （落盘） |

## 3. 参数换算规则

### 3.1 duration → length（帧数）

```python
n = max(5, round(duration * 24))       # 24 fps
length = n + (5 - n % 17) % 17          # 向上对齐到 17k+5 网格
```

| duration(s) | n | length(帧) | 实际秒 |
|---|---|---|---|
| 1.0 | 24 | 39 | 1.625 |
| 5.0 | 120 | 124 | 5.17 |
| 10.0 | 240 | 244 | 10.17 |
| 15.0 | 360 | 364 | 15.17 |
| 20.0 | 480 | 484 | 20.17 |

> 网格约束来自 `ComfyUI/comfy_extras/nodes_minimax_h3.py:34` `align_frame_count()`。
> 训练区间 124–362 帧（~5–15s）；API 允许到 484 帧（20s）属保守外推，质量未验证。

### 3.2 分辨率约束（引擎侧自动执行）

`MiniMaxH3ImageToVideo` 节点内部（`nodes_minimax_h3.py:49 adapt_canvas()`）：

```
短边基准 768px，画布面积上限 768×1344=1,032,192 px²
宽高各自 round 到 32 的倍数，下限 32
```

因此请求 `width=832,height=480` 会被引擎重整为短边 768 的等效画布。**API 层不做此换算**，直接透传。

## 4. Latent 结构（AV 联合）

```python
# nodes_minimax_h3.py:70 _empty_av_latent()
video = zeros[B, 24, T, H/16, W/16]   # T = ((帧数-5)//17)*5 + 2
audio = zeros[B, 32, 2, T40]           # T40 = round(秒数 × 40)
NestedTensor((video, audio))
```

- 视频 latent 时间压缩比 17 帧→5 步（除首 5 帧），空间 1/16
- 音频 latent 40 fps 独立时间轴，双声道 (2) × 32 通道
- **参考帧/keyframe latent 每步重注入，不参与去噪**（fl2va/ref2va 语义）

## 5. 模型文件清单（`ComfyUI/models/`，共 ~63 GB）

| 类别 | 文件 | 大小 |
|---|---|---|
| diffusion_models/ | `minimax_h3_fl2va_int8_convrot.safetensors` | 32 GB |
| text_encoders/ | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 26 GB |
| vae/ | `minimax_h3_video_vae_fp16.safetensors` | 4.9 GB |
| vae/ | `minimax_h3_audio_vae_fp32.safetensors` | 578 MB |

> INT8 量化（convrot）版本；显存占用：DiT+TE+VAE 全载约 40+ GB，单张 4090 49GB 可跑。

## 6. 输出物

- 路径：`ComfyUI/output/video/minimax_h3_<seq:05d>.mp4`（worker 1 在 `output/gpu1/video/`）
- 编码：mp4 容器，codec=auto（PyAV 按平台选 h264），24fps，bit_depth 8
- `/history` 中 outputs 出现于 `outputs["14"].video[] = {filename, subfolder, type}`

## 7. 变体工作流（引擎已支持，API 暂未暴露）

| 节点 | 用途 | 暴露状态 |
|---|---|---|
| `MiniMaxH3ImageToVideo` + `first_frame`/`last_frame` | 首尾帧锚定（fl2va） | ❌ API 未传该参数 |
| `MiniMaxH3ReferenceToVideo` | `<Picture i>/<Video k>/<Audio j>` 引用生成（ref2va） | ❌ API 未暴露 |
| `EmptyMiniMaxH3LatentAV` | 纯工作流编辑 | — |

如需开放 i2v/ref2va，需在 `build_prompt()` 增加可选参数并把图像经 ComfyUI `/upload/image` 预传。
