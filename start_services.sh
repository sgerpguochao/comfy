#!/bin/bash
# MiniMax-H3 视频生成服务启动脚本（双 GPU，统一走 PM2）
# 用法: ./start_services.sh start | stop | restart | status | logs
#
# 由 PM2 托管三个进程：
#   - comfy-gpu0    (ComfyUI worker, GPU 0, 端口 8188)
#   - comfy-gpu1    (ComfyUI worker, GPU 1, 端口 8189，独立 output/user/db 目录)
#   - minimax-h3-api (FastAPI 调度层，端口 8000，含启动预热 + 后台 mux)
#
# 实际配置见 ecosystem.config.cjs；本脚本只是 PM2 的便捷包装。

ECOSYSTEM=/home/ubuntu/comfy/ecosystem.config.cjs
API_PORT=8000

start() {
    if ! command -v pm2 >/dev/null 2>&1; then
        echo "ERR: pm2 not installed, run: npm install -g pm2"
        exit 1
    fi
    if [ ! -f "$ECOSYSTEM" ]; then
        echo "ERR: ecosystem config not found: $ECOSYSTEM"
        exit 1
    fi
    echo "[start] pm2 start $ECOSYSTEM ..."
    pm2 start "$ECOSYSTEM"
    sleep 3
    status
}

stop() {
    # 复用 stop_services.sh 的逻辑（PM2 + tmux 兜底）
    bash /home/ubuntu/comfy/stop_services.sh
}

restart() {
    echo "[restart] stop then start ..."
    stop
    sleep 2
    start
}

status() {
    echo "--- pm2 ---"
    pm2 list 2>/dev/null | grep -E 'comfy-gpu|minimax-h3' || echo "  no matching apps"
    echo "--- /health ---"
    curl -s -m 5 "http://127.0.0.1:$API_PORT/health" && echo
}

logs() {
    # 用法: ./start_services.sh logs [app_name]
    pm2 logs "${1:-}" --lines 80 --nostream 2>/dev/null || pm2 logs --lines 80 --nostream
}

case "$1" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    logs)    shift; logs "$@" ;;
    *) echo "usage: $0 start|stop|restart|status|logs [app_name]" ;;
esac
