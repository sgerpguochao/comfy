#!/bin/bash
# MiniMax-H3 视频生成服务启动脚本（双 GPU，tmux 常驻）
# 用法: ./start_services.sh start | stop | status | attach
#
# 每张 GPU 跑一个 ComfyUI worker（--cuda-device 隔离），API 服务按队列长度
# 把任务分发给最空闲的 GPU。进程跑在 tmux 会话中，不受终端退出影响。
# --fast autotune cublas_ops: 启用 cuDNN benchmark 与 cuBLAS，加速推理。

COMFY_DIR=/home/ubuntu/comfy/ComfyUI
VENV=/home/ubuntu/comfy/venv/bin/python
API=/home/ubuntu/comfy/api_server.py
LOG_DIR=/home/ubuntu/comfy/logs
SESSION=comfy-multi
API_PORT=8000
W0_PORT=8188
W1_PORT=8189
# worker 1 使用独立输出/用户目录，避免两实例的文件计数器与 comfyui.db 冲突
OUT_DIR=/home/ubuntu/comfy/ComfyUI/output
USER_DIR=/home/ubuntu/comfy/ComfyUI/user
FAST="--fast autotune cublas_ops"

start() {
    mkdir -p "$LOG_DIR" "$USER_DIR/gpu1"
    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION" -x 220 -y 50
    # worker 0 (GPU 0)
    tmux send-keys -t "$SESSION:0" \
        "env -u LD_LIBRARY_PATH $VENV $COMFY_DIR/main.py --listen 0.0.0.0 --port $W0_PORT --cuda-device 0 $FAST 2>&1 | tee $LOG_DIR/comfy_gpu0.log" Enter
    # worker 1 (GPU 1)
    tmux new-window -t "$SESSION" \
        "env -u LD_LIBRARY_PATH $VENV $COMFY_DIR/main.py --listen 0.0.0.0 --port $W1_PORT --cuda-device 1 --output-directory $OUT_DIR/gpu1 --user-directory $USER_DIR/gpu1 $FAST 2>&1 | tee $LOG_DIR/comfy_gpu1.log"
    # API
    tmux new-window -t "$SESSION" \
        "env -u LD_LIBRARY_PATH COMFY_HOSTS=127.0.0.1:$W0_PORT,127.0.0.1:$W1_PORT $VENV $API 2>&1 | tee $LOG_DIR/api.log"
    sleep 3
    status
}

stop() {
    tmux kill-session -t "$SESSION" 2>/dev/null
    echo "stopped"
}

status() {
    curl -s http://127.0.0.1:$API_PORT/health && echo
}

case "$1" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    attach) tmux attach -t "$SESSION" ;;
    *) echo "usage: $0 start|stop|status|attach" ;;
esac
