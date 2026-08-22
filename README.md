# comfy

Multi-GPU ComfyUI orchestration service.

This repository hosts the **glue code** that runs alongside an upstream
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) checkout. It does **not**
vendor ComfyUI itself — the engine lives in the sibling `ComfyUI/` directory,
which is its own git checkout pulled from upstream.

## Layout

```
.
├── api_server.py        # FastAPI wrapper exposing /api/v1/generate over N ComfyUI workers
├── bench.py             # Load test / benchmark utility
├── client_example.py    # Sample client calling the API
├── ecosystem.config.cjs # PM2 process manifest (2 ComfyUI workers + api_server)
├── gen.sh               # Generator helpers
├── monitor.py           # Process / GPU monitoring
├── start_services.sh    # Bring everything up under PM2
├── test_h3.py           # Integration tests against the API
├── prompt.txt           # Sample prompts used by the benchmark
├── .cursor/rules/       # Project conventions (commit rules etc.)
├── ComfyUI/             # Upstream ComfyUI clone (git ignored here)
├── venv/                # Local Python virtualenv (git ignored)
└── wheels/              # Prebuilt pip wheels (git ignored)
```

## ComfyUI — where it lives

`ComfyUI/` is **excluded from this repo on purpose**:

- It contains the official upstream engine source (see `ComfyUI/.git` for its
  own history).
- It weighs ~2 GB of source plus the `models/` directory (~63 GB) which holds
  downloaded model weights and **must not be committed**.
- You update ComfyUI by running `git -C ComfyUI pull` against its upstream
  remote `https://github.com/comfyanonymous/ComfyUI.git`.

To bootstrap ComfyUI on a fresh machine:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
python3 -m venv venv && source venv/bin/activate
pip install -r ComfyUI/requirements.txt
```

## Running the service

```bash
# 1. Make sure ComfyUI is set up (see above) and models are downloaded
# 2. Start the workers + API server
./start_services.sh
# PM2 will manage comfy-gpu0, comfy-gpu1, and minimax-h3-api
```

Verify:

```bash
curl http://127.0.0.1:8188/       # ComfyUI worker 0
curl http://127.0.0.1:8189/       # ComfyUI worker 1
curl http://127.0.0.1:8188/health # Service health
```

## Submitting prompts

See `prompt.txt` for example prompts and `client_example.py` for a working
client. The API contract is documented at the top of `api_server.py`.
