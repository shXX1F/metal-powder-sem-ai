# Metal Powder SEM GUI 便携版使用说明

## 给使用者

1. 解压 `MetalPowderSEM_GUI.zip`。
2. 双击 `start_gui.bat`。启动器会在后台运行服务，健康检查通过后再打开浏览器。
3. 浏览器会自动打开本地页面：`http://127.0.0.1:8501`。
4. 上传 SEM 图像，输入比例尺参数，然后点击界面中的识别按钮。
5. 分析结果会保存在软件目录下的 `runs/gui/时间戳/` 文件夹中。

空心粉页面需要分别选择 particle 和 hollow 权重。权重文件由管理员单独提供，不放进通用发布包。

如果需要让同一局域网内的其他电脑访问，可由管理员运行 `start_gui_lan.bat`，并在 Windows 防火墙中允许 TCP 8501。使用完毕后双击 `stop_gui.bat`。

运行日志保存在 `runs/gui_server.log`。遇到页面无法访问时，先重新双击 `start_gui.bat`；启动器会检测已有服务，不会重复启动。请不要删除 `.runtime`、`tools`、`hollow_version0/code`、`metal_powder_sem_ai`、`app_streamlit.py` 等文件。

## 给开发者

在项目根目录执行：

```powershell
.\build_release.ps1
```

脚本会生成：

```text
dist/
  MetalPowderSEM_GUI/
  MetalPowderSEM_GUI.zip
```

把 `dist/MetalPowderSEM_GUI.zip` 发给用户即可。用户不需要安装 Python，也不需要运行代码。

如果构建时下载依赖很慢，可以先配置国内 pip 镜像，例如：

```powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

然后重新执行：

```powershell
.\build_release.ps1
```
