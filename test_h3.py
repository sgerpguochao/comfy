"""MiniMax-H3 文生视频测试脚本：提交任务并轮询结果。"""
import json
import sys
import time
import urllib.request


def build_prompt(prompt, width, height, duration, seed, steps=10):
    n = max(5, round(duration * 24))
    length = n + (5 - n % 17) % 17
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "minimax_h3_fl2va_int8_convrot.safetensors",
                         "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                         "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "5": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                         "width": width, "height": height, "length": length}},
        "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "7": {"class_type": "BasicScheduler",
              "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "8": {"class_type": "BasicGuider",
              "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["9", 0], "guider": ["8", 0], "sampler": ["6", 0],
                          "sigmas": ["7", 0], "latent_image": ["5", 1]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo",
               "inputs": {"images": ["11", 0], "audio": ["12", 0], "fps": 24, "bit_depth": 8}},
        "14": {"class_type": "SaveVideo",
               "inputs": {"video": ["13", 0], "filename_prefix": "video/minimax_h3",
                          "format": "mp4", "codec": "auto"}},
    }


def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    prompt_text = (sys.argv[1] if len(sys.argv) > 1 else
                   "一只橘猫在夕阳下的城市天台漫步，电影感镜头，微风拂过毛发，背景有金色落日和飞鸟，画面唯美写实，无文字")
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 832
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    duration = float(sys.argv[4]) if len(sys.argv) > 4 else 4
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    # 可选第 6 个参数指定目标 ComfyUI worker（默认 GPU 0，即 8188；GPU 1 为 8189）
    COMFY = sys.argv[6] if len(sys.argv) > 6 else "http://127.0.0.1:8188"

    graph = build_prompt(prompt_text, width, height, duration, seed)
    resp = post(f"{COMFY}/prompt", {"prompt": graph})
    pid = resp["prompt_id"]
    print(f"[submit] comfy={COMFY} task_id={pid} node_errors={resp.get('node_errors')}", flush=True)

    while True:
        time.sleep(10)
        hist = get(f"{COMFY}/history/{pid}")
        entry = hist.get(pid)
        if not entry:
            print("[poll] still queued/running...", flush=True)
            continue
        st = entry.get("status", {})
        if st.get("completed"):
            outs = entry.get("outputs", {})
            print("[done] outputs:", json.dumps(outs, ensure_ascii=False)[:500], flush=True)
            break
        if st.get("status_str") == "error" or any(
                isinstance(m, list) and m and m[0] == "execution_error"
                for m in st.get("messages", [])):
            print("[error]", json.dumps(st, ensure_ascii=False)[:800], flush=True)
            break
