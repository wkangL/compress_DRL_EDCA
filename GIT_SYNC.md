# GitHub同步说明

远端仓库：`https://github.com/wkangL/compress_DRL_EDCA.git`

Git只同步源码、配置、测试和Markdown文档。以下内容不进入Git：`results/`、`output/`、`exports/`、虚拟环境、`vendor/ns-3.44`、构建产物、Checkpoint和恢复包。

## 本地Windows

```powershell
cd D:\compress_DRL_EDCA
git status
git pull --rebase origin main
git add -A
git commit -m "描述本次修改"
git push origin main
```

首次推送前必须先运行 `git fetch origin`，确认远端是否已有历史。禁止使用 `git push --force`。

## Linux服务器首次部署

若服务器目录尚无需要保留的本地修改，推荐把现有目录改名留作备份，再clone：

```bash
cd /data/4t/lwk
mv compress_DRL_EDCA compress_DRL_EDCA_before_git
git clone https://github.com/wkangL/compress_DRL_EDCA.git
cd compress_DRL_EDCA
```

然后从备份目录只复制服务器专属且被Git忽略的内容，例如虚拟环境外的本地ns-3安装路径或必要结果。不要把旧源码整目录覆盖回clone目录。

已有Git clone日常更新：

```bash
cd /data/4t/lwk/compress_DRL_EDCA
git status
git pull --rebase origin main
```

服务器产生的源码修改需要回传时：

```bash
git switch -c server/<简短名称>
git add <明确的源码或配置文件>
git commit -m "说明服务器修改"
git push -u origin HEAD
```

再在本地审查并合并该分支。不要提交 `results/`、Checkpoint、虚拟环境或ns-3构建目录。

## GitHub认证

HTTPS push需要GitHub Personal Access Token，不能使用账户密码。也可在两端配置SSH key后把远端切换为：

```bash
git remote set-url origin git@github.com:wkangL/compress_DRL_EDCA.git
```

检查命令：

```bash
git remote -v
git branch --show-current
git status --short
```
