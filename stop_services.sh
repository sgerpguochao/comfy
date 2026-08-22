#!/bin/bash
# MiniMax-H3 服务停止脚本
#
# 用法:
#   ./stop_services.sh            # 优雅停止（推荐）
#   ./stop_services.sh --force    # 强制停止（连同残留的 python 进程也一起清掉）
#
# 行为:
#   1. 通过 tmux 杀 comfy-multi 会话（优雅停 API + 两个 ComfyUI worker）
#   2. 检查 8000/8188/8189 端口是否真正释放
#   3. --force 模式下额外扫一遍残留的 api_server.py / ComfyUI/main.py 进程并 kill
#
# 退出码:
#   0 全部停止成功
#   1 有端口仍被占用

set -e

SESSION="comfy-multi"
PORTS=(8000 8188 8189)
FORCE=0

[ "$1" = "--force" ] && FORCE=1

echo "[1/3] kill tmux session '$SESSION' ..."
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "  -> killed"
else
    echo "  -> session not running"
fi

if [ "$FORCE" = "1" ]; then
    echo "[2/3] force kill residual python processes ..."
    # 残留 api_server.py / ComfyUI/main.py 进程
    pkill -TERM -f "api_server\.py" 2>/dev/null && echo "  -> killed api_server.py" || echo "  -> no api_server.py"
    pkill -TERM -f "ComfyUI/main\.py" 2>/dev/null && echo "  -> killed ComfyUI/main.py" || echo "  -> no ComfyUI/main.py"
    sleep 2
    # 还有顽固的就 SIGKILL
    pkill -KILL -f "api_server\.py" 2>/dev/null || true
    pkill -KILL -f "ComfyUI/main\.py" 2>/dev/null || true
else
    echo "[2/3] skip force kill (use --force to enable)"
fi

echo "[3/3] verify ports freed ..."
RC=0
for p in "${PORTS[@]}"; do
    # -z 表示空端口返回 0（没人监听）；非0 表示端口仍被占
    if ss -ltn "sport = :$p" 2>/dev/null | tail -n +2 | grep -q .; then
        echo "  port $p: STILL LISTENING"
        RC=1
    else
        echo "  port $p: ok"
    fi
done

if [ "$RC" = "0" ]; then
    echo ""
    echo "stopped (use 'bash start_services.sh start' to restart)"
    exit 0
else
    echo ""
    echo "WARN: some ports still busy, retry with --force"
    exit 1
fi