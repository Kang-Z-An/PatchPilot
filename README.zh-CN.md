# PatchPilot

[English](README.md) · [使用与评测文档](docs/usage.md) · [后训练数据流程](docs/post_training.md)

PatchPilot 是一个可追踪的代码修复 Agent：接收 issue，检查仓库、修改代码、
运行测试、审查 diff，并保存完整的模型与工具交互轨迹。项目使用显式 Python
循环实现，便于理解 Agent 如何决策，以及为什么某次修复成功或失败。

目前已实现 Agent、SWE-bench 批量评测和训练数据整理流程。
RAG、SFT/LoRA 和强化学习训练尚未完成。

## 快速体验

需要 Python 3.11+、Git 和 uv。克隆项目后运行离线演示：

```bash
git clone https://github.com/Kang-Z-An/PatchPilot.git
cd PatchPilot
uv sync --locked
uv run python examples/offline_demo.py
```

演示用预设动作修复一个算术错误，实际执行工具和测试，打印最终补丁，
在 `.patchpilot/demo/` 保存轨迹。不需要 API key、Docker 或 GPU；
这是执行流程演示，不代表真实模型能力。临时仓库会在结束后自动清理。

运行离线测试：

```bash
uv run python -m unittest discover -s tests -v
```

## 使用真实模型

复制 `.env.example` 为 `.env`，填写：

```dotenv
PATCHPILOT_API_KEY=replace-with-your-api-key
PATCHPILOT_BASE_URL=https://your-provider.example/v1
PATCHPILOT_MODEL=your-model-name
```

先准备一个可丢弃的本地 Git checkout，并安装目标项目的测试依赖，然后运行：

```bash
uv run patchpilot \
  --repo /目标仓库副本的绝对路径 \
  --issue "描述实际错误和期望行为" \
  --test-command "python -m unittest discover -s tests -q" \
  --protect-test-files \
  --max-steps 20
```

Agent 会直接修改这个目录；模型请求可能产生费用，并会把仓库片段和测试输出
发送给所配置的模型服务。若目标项目使用 pytest，需要先安装其依赖，再改用
`--test-command "python -m pytest -q"`。环境变量优先于 `.env`，
`--model` 优先于 `PATCHPILOT_MODEL`。

## 当前能力与证据

已支持文件检索、静态继承分析、精确文本替换、unified diff、测试文件保护、
固定测试命令、Docker 探针、可恢复的 SWE-bench 批量评测，以及严格的 SFT/
偏好数据导出。双环境探针会在原始和修改后 checkout 上运行同一段断言。

| 实验 | 官方结果 | 使用边界 |
| --- | --- | --- |
| 历史 Verified 小样本 | 7/8 | 仅 8 道选择题目，其中 3 个补丁修改过测试；现作为开发集 |
| 冻结 train 分区 | 7/16 | 训练数据采集结果，无测试修改，不是独立测试分数 |
| 双环境探针 smoke | 1/2 | 两道 train 失败题，新增严格 SFT 样本 0 条 |

纠错后累计有 8 条通过审计的训练轨迹和 1 对有效偏好数据。仓库只公开紧凑的
[实验报告](experiments/training_pool_v1_deepseek/README.md)，不附带原始对话、
模型权重或 Docker 镜像。尚无后训练提升结论。

`agent_verified` 表示通过本地配置的检查，`resolved` 才是官方评测结果；
两者可能不一致。小样本结果也不能当作完整 SWE-bench 排行榜成绩。

## 下一步与参与

接下来优先审计公共测试与官方测试的预期差异，改善行为覆盖和轨迹质量，
再做检索与固定条件下的基座/LoRA 对比。

详细命令见[使用文档](docs/usage.md)，代码入口与目录图见[英文首页](README.md)。
贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，执行边界见 [SECURITY.md](SECURITY.md)。
项目原创代码采用 [MIT](LICENSE)，第三方 benchmark 材料保留
[上游来源和条款](THIRD_PARTY_NOTICES.md)。
