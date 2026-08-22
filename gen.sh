#!/bin/bash
# MiniMax-H3 视频生成一键脚本
# 用法: bash gen.sh <提示词文件> [宽度] [高度] [时长秒]
# 示例: bash gen.sh                    （默认用 /home/ubuntu/minmax/comfy/prompt.txt，768x1344 竖屏 15 秒）
#       bash gen.sh my.txt 1344 768 5  （横屏 5 秒）

PROMPT_FILE="$1"
[ -z "$PROMPT_FILE" ] && PROMPT_FILE=/home/ubuntu/minmax/comfy/prompt.txt
WIDTH="${2:-768}"
HEIGHT="${3:-1344}"
DURATION="${4:-15}"

# Ubuntu 默认无 python 命令，用 python3；若存在 python 则优先
PY="python3"
command -v python >/dev/null 2>&1 && PY="python"

"$PY" /home/ubuntu/minmax/comfy/client_example.py "$(cat "$PROMPT_FILE")" "$WIDTH" "$HEIGHT" "$DURATION"
