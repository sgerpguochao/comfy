#!/bin/bash
# MiniMax-H3 服务停止脚本
#
# 用法:
#   ./stop_services.sh            # 优雅停止（推荐）
#   ./stop_services.sh --force    # 强制停止（连同残留的 python 进程也一起清掉）
#
# 行为（按检测顺序）:
#   1. 优先通过 PM2 停 minimax-h3-api / comfy-gpu0 / comfy-gpu1
#   2. 若 tmux 会话 'comfy-multi' 存在（start_services.sh 拉的备用方案），一并杀掉
#   3. --force 模式下额外扫一遍残留的 api_server.py / ComfyUI/main.py 进程并 kill
#   4. 检查 8000/8188/8189 端口是否真正释放
#
# 退出码:
#   0 全部停止成功
#   1 有端口仍被占用

set -e

SESSION="comfy-multi"
PORTS=(8000 8188 8189)
FORCE=0

[ "$1" = "--force" ] && FORCE=1

echo "[1/4] stop via PM2 ..."
if command -v pm2 >/dev/null 2>&1; then
    # 找出所有跟项目相关的进程名（避免误停无关服务）
    PM2_TARGETS=$(pm2 jlist 2>/dev/null | \
        python3 -c "
import json, sys
apps = json.load(sys.stdin)
targets = []
for a in apps:
    name = a.get('name', '')
    if name in ('minimax-h3-api', 'comfy-gpu0', 'comfy-gpu1'):
        targets.append(name)
print('\n'.join(targets))
" 2>/dev/null)
    if [ -n "$PM2_TARGETS" ]; then
        echo "$PM2_TARGETS" | while read -r name; do
            [ -z "$name" ] && continue
            pm2 stop "$name" 2>/dev/null && echo "  -> pm2 stopped $name" \
                                       || echo "  -> pm2 stop $name failed"
        done
        sleep 2
        echo "$PM2_TARGETS" | while read -r name; do
            [ -z "$name" ] && continue
            pm2 delete "$name" 2>/dev/null && echo "  -> pm2 deleted $name"
        done
    else
        echo "  -> no matching PM2 apps (minimax-h3-api / comfy-gpu0 / comfy-gpu1)"
    fi
else
    echo "  -> pm2 not installed, skipped"
fi

echo "[2/4] kill tmux session '$SESSION' ..."
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "  -> killed"
else
    echo "  -> session not running"
fi

if [ "$FORCE" = "1" ]; then
    echo "[3/4] force kill residual python processes ..."
    # 残留 api_server.py / ComfyUI/main.py 进程
    pkill -TERM -f "api_server\.py" 2>/dev/null && echo "  -> killed api_server.py" || echo "  -> no api_server.py"
    pkill -TERM -f "ComfyUI/main\.py" 2>/dev/null && echo "  -> killed ComfyUI/main.py" || echo "  -> no ComfyUI/main.py"
    sleep 2
    # 还有顽固的就 SIGKILL
    pkill -KILL -f "api_server\.py" 2>/dev/null || true
    pkill -KILL -f "ComfyUI/main\.py" 2>/dev/null || true
else
    echo "[3/4] skip force kill (use --force to enable)"
fi

echo "[4/4] verify ports freed ..."
RC=0
for p in "${PORTS[@]}"; do
    # IPv4 + IPv6 都查，避免 IPv6-only 监听漏检
    if ss -tlnH "( sport = :$p )" 2>/dev/null | grep -q .; then
        echo "  port $p: STILL LISTENING"
        RC=1
    else
        echo "  port $p: ok"
    fi
done

if [ "$RC" = "0" ]; then
    echo ""
    echo "stopped (use 'pm2 start ecosystem.config.cjs' to restart)"
    exit 0
else
    echo ""
    echo "WARN: some ports still busy, retry with --force"
    exit 1
fi