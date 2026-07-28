# 远端 JupyterLab 故障恢复手册

适用远端：`ssh -p 20338 root@connect.bjb1.seetacloud.com`

## 1. 故障现象

AutoDL 提示：

```text
dial tcp 127.0.0.1:8888: connect: connection refused
```

## 2. 故障定位

远端系统 Python `/usr/bin/python3` 没有 pip，直接执行 `jupyter` 也找不到命令；但 `/root/miniconda3/bin/` 中存在残留启动脚本，说明 JupyterLab 相关包处于不完整状态。后续操作必须使用 `/root/miniconda3` 环境。

```bash
export PATH=/root/miniconda3/bin:$PATH
command -v python
python -m pip --version
```

## 3. 卸载并重装

```bash
export PATH=/root/miniconda3/bin:$PATH

python -m pip uninstall -y \
  jupyterlab notebook jupyter-server jupyterlab-server \
  jupyterlab_widgets jupyterlab-pygments nbclassic

python -m pip install --no-cache-dir --upgrade jupyterlab
```

本次验证版本：

- `jupyterlab==4.6.1`
- `jupyter-server==2.20.0`

## 4. AutoDL 服务配置

AutoDL 容器通过 `/init/jupyter/jupyter_config.py` 自动拉起官方 Jupyter 实例，关键配置为：

```python
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.base_url = '/jupyter/'
c.ServerApp.root_dir = '/root'
```

## 5. 服务验证

```bash
export PATH=/root/miniconda3/bin:$PATH
jupyter server list
ps -ef | grep '[j]upyter-lab'
curl -sS -D - -o /dev/null http://127.0.0.1:8888/jupyter/lab
```

本次验证结果：

- Jupyter 进程正常运行
- `8888` 端口正常监听
- 未带 token 时 `/jupyter/lab` 返回 `302` 登录重定向
- 使用配置文件中的 token 请求 `/jupyter/lab` 返回 `HTTP 200`
- 正确访问前缀是 `/jupyter/`，不是根路径 `/lab`

## 6. 注意事项

1. 不要再次手动启动第二个 Jupyter 实例占用 `8889`；AutoDL 官方实例已经负责 `8888`。
2. 不要使用系统 Python 安装 Jupyter，必须先加入 `/root/miniconda3/bin`。
3. 重装后若日志提示 `nbclassic` 或 `jupyter_server_ydoc` 可选扩展缺失，只要页面和终端可用即可暂不处理。
4. 故障恢复时不要终止 Coco-LIC、FAST-LIVO2 或 rosbag 实验进程；如需清场，按实验复现手册执行。
