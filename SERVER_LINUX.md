# Linux服务器运行说明

本说明面向把项目放在`/data/4t/lwk/`并通过MobaXterm管理的场景。服务器只需上传源码包，历史`results/`、Windows虚拟环境、Windows编译器和Windows可执行文件都不需要上传。

## 1. 上传和解压

在Windows项目根目录生成服务器包：

```powershell
.\.venv\Scripts\python.exe scripts\package_linux_server.py
```

把`exports/compress_DRL_EDCA_linux_source.tar.gz`拖到服务器`/data/4t/lwk/`，然后执行：

```bash
cd /data/4t/lwk
tar -xzf compress_DRL_EDCA_linux_source.tar.gz
cd compress_DRL_EDCA
```

该压缩包包含Python/C++源码、配置、测试、固定拓扑和未编译的ns-3.44源码；不包含历史结果、checkpoint、`.venv`、`.tools`、Windows `.exe`、ns-3旧build目录或报告图片。

## 2. 首次安装和编译

Ubuntu/Debian且有sudo权限时，先安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git python3 python3-venv python3-pip
```

随后执行一次：

```bash
cd /data/4t/lwk/compress_DRL_EDCA
bash scripts/setup_linux_server.sh
```

脚本会创建Linux自己的`.venv`、安装Python依赖、给固定ns-3.44源码应用研究补丁、编译TGax控制后端并运行测试。不要把Windows的`.venv`、`.tools`、`build/`或`.exe`复制到Linux。

如果服务器没有sudo权限，需由管理员提供GCC/G++、CMake、Ninja、Python 3和`python3-venv`；之后仍运行同一个setup脚本。

## 3. 修改唯一配置

日常只编辑：

```text
configs/mac_cmdp_experiment.yaml
```

可以在Windows上改好后单独拖到服务器同一路径覆盖。主要可修改：

- `traffic`：各业务模型、方向、AC和并发数量；
- `cmdp.minimums`：VO/VI及时率、公平性、BE/BK满足度下限；
- `simulation.training_seeds`和`evaluation_seeds`：两组必须互不重叠；
- `training`：训练阶段、学习率、Replay、epsilon和Target更新；
- `training.parallel_workers`：同时运行的独立训练seed数量；
- `output.root`：默认结果目录。

每套新配置应使用新的`experiment.id`和输出目录，避免覆盖旧运行。

## 4. 启动训练、评估和绘图

推荐用命令行明确指定输出位置：

```bash
cd /data/4t/lwk/compress_DRL_EDCA
source .venv/bin/activate
python scripts/run_mac_cmdp_experiment.py train \
  --config configs/mac_cmdp_experiment.yaml \
  --output /data/4t/lwk/results/mac_cmdp_run_001
```

`train`会按顺序完成输入生成、所有训练seed、每个final checkpoint在独立评估seed上的评估，以及最终CSV/PNG绘图。中断后用完全相同的命令重启，会从已有完整checkpoint继续；不要更换输出目录来伪装续训。

只重新汇总和绘图：

```bash
python scripts/run_mac_cmdp_experiment.py plot \
  --config configs/mac_cmdp_experiment.yaml \
  --output /data/4t/lwk/results/mac_cmdp_run_001
```

后台运行并保存终端日志：

```bash
mkdir -p /data/4t/lwk/logs
nohup .venv/bin/python scripts/run_mac_cmdp_experiment.py train \
  --config configs/mac_cmdp_experiment.yaml \
  --output /data/4t/lwk/results/mac_cmdp_run_001 \
  > /data/4t/lwk/logs/mac_cmdp_run_001.log 2>&1 &
echo $! > /data/4t/lwk/logs/mac_cmdp_run_001.pid
```

查看进度：

```bash
tail -f /data/4t/lwk/logs/mac_cmdp_run_001.log
cat /data/4t/lwk/results/mac_cmdp_run_001/pipeline_status.json
find /data/4t/lwk/results/mac_cmdp_run_001 -name status.json -print
```

## 5. GPU说明

当前版本不会使用GPU。ns-3的离散事件仿真占绝大多数墙钟时间，而且每个ns-3进程主要占用一个CPU核；TCN/DDQN网络较小，当前训练器也在CPU上运行。把PyTorch换成CUDA版不会明显加速ns-3。

一键入口支持安全的seed级并行。例如三个训练seed可设置：

```yaml
training:
  parallel_workers: 3
```

每个worker拥有独立的`seed_<N>`目录、checkpoint、日志和最终评估目录；共享输入和配置只读。所有seed结束后，主进程才统一汇总绘图。不要同时启动两个指向同一`--output`目录的总控命令。

并行度建议不超过训练seed数量，也不要超过服务器可用CPU核和内存。三个worker通常至少需要三个空闲CPU核；若服务器同时还有其他任务，可改为1或2。
