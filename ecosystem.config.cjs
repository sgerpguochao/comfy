const fs = require("fs");
const path = require("path");

// load .env (same logic as openrouter-relay)
const envFile = path.join(__dirname, ".env");
const env = {};
if (fs.existsSync(envFile)) {
  for (const line of fs.readFileSync(envFile, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !line.trim().startsWith("#")) env[m[1]] = m[2];
  }
}

const VENV = "/home/ubuntu/minmax/comfy/venv/bin/python";
const COMFY_DIR = "/home/ubuntu/minmax/comfy/ComfyUI";
const FAST = "--fast autotune cublas_ops";

module.exports = {
  apps: [
    {
      name: "comfy-gpu0",
      script: `${COMFY_DIR}/main.py`,
      args: `--listen 0.0.0.0 --port 8188 --cuda-device 0 ${FAST}`,
      interpreter: VENV,
      cwd: COMFY_DIR,
      autorestart: true,
      max_restarts: 5,
      env: {
        ...env,
        // 清除沙箱可能注入的 LD_LIBRARY_PATH，避免 CUDA 初始化失败
        LD_LIBRARY_PATH: "",
      },
    },
    {
      name: "comfy-gpu1",
      script: `${COMFY_DIR}/main.py`,
      args: `--listen 0.0.0.0 --port 8189 --cuda-device 1 --output-directory ${COMFY_DIR}/output/gpu1 --user-directory ${COMFY_DIR}/user/gpu1 --database-url sqlite:////home/ubuntu/minmax/comfy/ComfyUI/user/gpu1/comfyui.db ${FAST}`,
      interpreter: VENV,
      cwd: COMFY_DIR,
      autorestart: true,
      max_restarts: 5,
      env: {
        ...env,
        LD_LIBRARY_PATH: "",
      },
    },
    {
      name: "minimax-h3-api",
      script: "api_server.py",
      interpreter: VENV,
      cwd: __dirname,
      autorestart: true,
      max_restarts: 5,
      env: {
        ...env,
        COMFY_HOSTS: "127.0.0.1:8188,127.0.0.1:8189",
        LD_LIBRARY_PATH: "",
      },
    },
  ],
};
