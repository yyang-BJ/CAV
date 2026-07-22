# CAV 环境安装说明（发给协作者）

你只需要这套安装方式，在 **CAV 源码目录** 里从零搭环境。  
**不要**复用机器上已有的 conda / 其它项目的 `verl`。

## 你需要有什么

1. **CAV 源码**（完整仓库，含 `pyproject.toml`、`src/`、`scripts/`）
2. 本目录里的：
   - `install_env.sh`（安装脚本）
   - 本 `README.md`

若对方只给了这两个文件：把它们放进 CAV 仓库的 `share/` 目录（或任意位置，安装时传入 CAV 路径）。

## 机器要求

- Linux + NVIDIA GPU 驱动
- 建议 **Python 3.11**
- 能访问外网（装 PyTorch / vLLM，并 clone GitHub 上的 veRL）
- 磁盘建议预留 **≥ 20GB**（虚拟环境很大）

## 一键安装

```bash
# 进入 CAV 仓库根目录
cd /path/to/CAV

# 推荐：用仓库自带入口（与 share 脚本等价）
bash scripts/setup_env.sh

# 或者直接用本分享脚本：
bash share/install_env.sh
# 若当前不在 CAV 根目录：
# bash /path/to/share/install_env.sh /path/to/CAV
```

脚本会：

1. 在 `CAV/.venv` 创建**独立**虚拟环境  
2. 安装固定版本：`torch==2.7.1+cu126`、`vllm==0.10.1`、`ray==2.50.0` 等  
3. 把官方 veRL 克隆到 `CAV/third_party/verl`（默认 `v0.5.0`）  
4. `pip install -e` 安装本仓库的 `verl` 与 `cav-rl`  
5. 做一次导入校验  

可选：

```bash
SKIP_FLASH_ATTN=1 bash share/install_env.sh   # flash_attn 编译失败时跳过
PYTHON_BIN=python3.11 bash share/install_env.sh
```

## 每次训练前

```bash
cd /path/to/CAV
source scripts/use_env.sh
```

终端里应看到类似：

```text
[CAV] using .../CAV/.venv
verl=... @ .../CAV/third_party/verl/...   # 或 .../CAV/.venv/lib/...
```

若 `verl @` 指向其它项目路径，**停止**，说明环境串了。

## 跑训练前还要改路径

安装脚本**不管**数据和模型。请自行设置，例如：

```bash
export DATA_DIR=/your/path/gsm8k_baseline          # 含 train.parquet / test.parquet
export BASE_MODEL=/your/path/Qwen2.5-xxx           # 或 hf_merged 权重目录
export WANDB_MODE=online                           # 或 offline
```

然后再跑对应脚本，例如：

```bash
bash scripts/train_baseline_gsm8k.sh      # PPO
bash scripts/train_grpo_gsm8k.sh          # GRPO
bash scripts/train_grpo_correct_gsm8k.sh  # GRPO-correct
bash scripts/train_cav_gsm8k.sh           # CAV PPO
```

## 重要注意事项

1. **不要** `conda activate` 其它环境后再跑；只用 `source scripts/use_env.sh`。  
2. **不要**设置 `ALLOW_LEGACY_T3=1`（那是原作者机器遗留开关）。  
3. 脚本默认**不会**使用你电脑上其它目录里的 verl。  
4. 需要联网；纯内网需自行准备 PyTorch/vLLM wheel 与 veRL 源码。  
5. `flash_attn` 失败一般可跳过；若训练报错再针对性安装。

## 版本一览（与当前跑通配置对齐）

| 组件 | 版本 |
|------|------|
| Python | 3.11（推荐） |
| PyTorch | 2.7.1+cu126 |
| vLLM | 0.10.1 |
| Ray | 2.50.0 |
| Transformers | 4.55.4 |
| veRL | v0.5.0（`third_party/verl`） |
