# 测试与基准 Spec

## 1. 测试脚本矩阵

| 脚本 | 直连对象 | 用途 | 前置 |
|---|---|---|---|
| `test_h3.py` | ComfyUI worker（绕过 API） | 最小闭环：提交 graph → 轮询 history | worker 已起、模型已就位 |
| `client_example.py` | API `:8000` | 客户端视角的端到端示例 | API + ≥1 worker |
| `bench.py` | API `:8000` | 并发基准：N 任务并发提交，统计各自耗时与 worker 分配 | 同上 |
| `monitor.py` | API + workers | 单任务精确计时（含 ComfyUI 侧时间戳） | 已有 task_id |
| `gen.sh` | （包装 client_example） | 一键 `prompt.txt` → 视频 | 同 client_example |

## 2. test_h3.py — 引擎层集成测试

```bash
python3 test_h3.py [prompt] [w] [h] [dur] [seed] [comfy_url]
# 默认: 橘猫提示词 832x480 4s seed=42 http://127.0.0.1:8188
# 指定第二张卡: ... http://127.0.0.1:8189
```

- 自带 `build_prompt()`（与 api_server 等价），**不经网关**，用于隔离"API 层 bug vs 引擎层 bug"
- 输出 `node_errors`（graph 校验错误立即暴露）与 outputs JSON 前 500 字符
- 轮询间隔 10s

**判定**：`[done] outputs:` 出现且含 `filename` → 引擎链路 OK。

## 3. client_example.py — 端到端示例

```bash
python3 client_example.py "<prompt>" [w] [h] [duration] [base_url]
# 默认 base = http://117.50.216.253:8000（公网），本地调试传 http://127.0.0.1:8000
# 参数顺序宽松：URL 可在任意位置，其余数字依次为 宽 高 时长
python3 client_example.py "一只橘猫在夕阳下漫步"                 # 1344x768 5s
python3 client_example.py "竖屏广告" 768 1344 15                 # 竖屏 15s
```

固定 `seed=42, steps=20`。轮询 15s。**注意**：该脚本写死公网 IP，在仓库内联存在泄露面（见 overview 待改进项）。

## 4. bench.py — 并发基准

```bash
python3 bench.py [base] [w] [h] [dur] [steps] [N] [prompt]
# 默认: 127.0.0.1:8000, 832x480, 4s, steps=20, N=2
```

行为：
1. 顺序提交 N 个任务（`seed = time()%1e9 + i` **强制绕开 ComfyUI 执行缓存**）
2. 每 5s 轮询，逐任务打印 `[done] taskK workerW status= elapsed=`
3. 汇总 `总耗时(全部完成)`

**双卡吞吐验证建议**：

```bash
# N=2 应分到两张卡（观察 worker 字段），总耗时 ≈ 单任务耗时
python3 bench.py http://127.0.0.1:8000 832 480 4 10 2

# N=4 每卡排队 2 个，总耗时 ≈ 2× 单任务
python3 bench.py http://127.0.0.1:8000 832 480 4 10 4
```

## 5. monitor.py — 精确计时

```bash
python3 monitor.py <task_id> [base]
```

- API 轮询（15s 间隔）到终态后，再从**两个 worker 的 `/history`** 提取
  `execution_start` / `execution_success` / `execution_error` 毫秒时间戳
- 输出：`[history] worker=... 执行时长=X.Xs (start=HH:MM:SS, end=HH:MM:SS)`
- 用途：区分「排队等待」与「GPU 实际执行」，校验负载均衡效果

## 6. gen.sh — 一键生成

```bash
bash gen.sh                        # prompt.txt, 768x1344, 15s
bash gen.sh my.txt 1344 768 5      # 自定义提示词文件 + 横屏 5s
```

`prompt.txt` 格式约定：`旁白：...` / `画面：...` 交替的多段文案（竖屏 9:16 广告风格）。

## 7. 基准参考值（832×480, steps=10, RTX 4090）

> 以下为部署后应实测校准的占位目标；当前无历史数据。

| 场景 | 预期 |
|---|---|
| 单任务 4s/10steps | 待实测（经验量级：1–3 min） |
| 双卡并发 N=2 | ≈ 单任务耗时（各占一卡） |
| 双卡并发 N=4 | ≈ 2× 单任务耗时 |
| steps 10→20 | 耗时 ×~1.7 |

## 8. 冒烟清单（每次部署后）

```bash
# 1. 进程
pm2 list                          # 3 个进程 online（单卡期 gpu1 可 stopped）

# 2. 健康
curl -s localhost:8000/health | jq .status    # "ok"（单卡期 "degraded" 属预期）

# 3. 引擎闭环（~2min）
python3 test_h3.py "" 832 480 4 42 http://127.0.0.1:8188

# 4. API 闭环（~2min）
python3 client_example.py "冒烟测试" 832 480 4 http://127.0.0.1:8000

# 5. OpenAI 兼容面
curl -s localhost:8000/models
curl -s -X POST localhost:8000/videos -H 'Content-Type: application/json' \
     -d '{"prompt":"冒烟","resolution":"480p","duration":4}'
```

## 9. 测试注意事项

- **种子策略**：固定 seed 会命中 ComfyUI 执行缓存（直接回放历史输出，秒回）。压测必须变 seed（bench.py 已内置）。
- **队列污染**：长任务未完成时提交新任务会排队，测试前 `curl -X POST http://127.0.0.1:8188/queue` 检查（或 `Interrupt` 清空）。
- **输出膨胀**：每次测试产生 mp4，注意 `output/` 磁盘（见 deployment §9）。
- **公网回调**：`client_example.py` 默认公网地址，本机测试务必显式传 `http://127.0.0.1:8000`，否则测的是公网链路。
